//! Contracts for registered and retained data type codec support.

use std::any::{Any, TypeId};
use std::borrow::Cow;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};

use unsafe_cell_slice::UnsafeCellSlice;
use zarrs::array::codec::BytesCodec;
use zarrs::array::{ArrayBytes, ArrayBytesFixedDisjointView, ArraySubset, data_type};
use zarrs_codec::{
    CodecOptions, CodecSpecificOptions, ComputeResources, UnboundArrayToBytesCodecTraits,
};
use zarrs_data_type::codec_traits::bytes::{
    BytesCodecEndiannessMissingError, BytesDataTypeExt, BytesDataTypePlugin, BytesDataTypeTraits,
};
use zarrs_data_type::{
    CowBytes, DataType, DataTypeCodecError, DataTypeFillValueError, DataTypeFillValueMetadataError,
    DataTypeTraits, FillValue,
};
use zarrs_metadata::{Configuration, DataTypeSize, Endianness, FillValueMetadata};
use zarrs_plugin::{ExtensionName, ZarrVersion};

/// A custom data type whose `bytes` codec support XORs bytes with its mask.
#[derive(Debug, Default)]
struct Xor;

/// A custom data type whose `bytes` codec support adds its mask to bytes.
#[derive(Debug, Default)]
struct Add;

/// A custom data type registered with the `bytes` codec but not castable to its support.
#[derive(Debug, Default)]
struct Uncastable;

#[derive(Debug, Default)]
struct Custom<K> {
    mask: u8,
    decode_lengths: Mutex<Vec<usize>>,
    kind: PhantomData<K>,
}

impl<K> Custom<K> {
    fn new(mask: u8) -> Self {
        Self {
            mask,
            decode_lengths: Mutex::default(),
            kind: PhantomData,
        }
    }
}

impl<K> ExtensionName for Custom<K> {
    fn name(&self, _: ZarrVersion) -> Option<Cow<'static, str>> {
        Some(Cow::Borrowed("test.bytes_support"))
    }
}

impl<K: Debug + Send + Sync + 'static> DataTypeTraits for Custom<K> {
    fn configuration(&self, _: ZarrVersion) -> Configuration {
        serde_json::Map::from_iter([("mask".into(), self.mask.into())]).into()
    }

    fn size(&self) -> DataTypeSize {
        DataTypeSize::Fixed(2)
    }

    fn fill_value(
        &self,
        metadata: &FillValueMetadata,
        version: ZarrVersion,
    ) -> Result<FillValue, DataTypeFillValueMetadataError> {
        data_type::uint16().fill_value(metadata, version)
    }

    fn metadata_fill_value(
        &self,
        fill_value: &FillValue,
    ) -> Result<FillValueMetadata, DataTypeFillValueError> {
        data_type::uint16().metadata_fill_value(fill_value)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl BytesDataTypeTraits for Custom<Xor> {
    fn encode<'a>(
        &self,
        bytes: CowBytes<'a>,
        _: Option<Endianness>,
    ) -> Result<CowBytes<'a>, BytesCodecEndiannessMissingError> {
        Ok(bytes
            .iter()
            .map(|byte| byte ^ self.mask)
            .collect::<Vec<_>>()
            .into())
    }

    fn decode<'a>(
        &self,
        bytes: CowBytes<'a>,
        endian: Option<Endianness>,
    ) -> Result<CowBytes<'a>, BytesCodecEndiannessMissingError> {
        self.decode_lengths.lock().unwrap().push(bytes.len());
        self.encode(bytes, endian)
    }
}

impl BytesDataTypeTraits for Custom<Add> {
    fn encode<'a>(
        &self,
        bytes: CowBytes<'a>,
        _: Option<Endianness>,
    ) -> Result<CowBytes<'a>, BytesCodecEndiannessMissingError> {
        Ok(bytes
            .iter()
            .map(|byte| byte.wrapping_add(self.mask))
            .collect::<Vec<_>>()
            .into())
    }

    fn decode<'a>(
        &self,
        bytes: CowBytes<'a>,
        _: Option<Endianness>,
    ) -> Result<CowBytes<'a>, BytesCodecEndiannessMissingError> {
        Ok(bytes
            .iter()
            .map(|byte| byte.wrapping_sub(self.mask))
            .collect::<Vec<_>>()
            .into())
    }
}

zarrs_data_type::register_data_type_extension_codec!(
    Custom<Xor>,
    BytesDataTypePlugin,
    BytesDataTypeTraits
);

zarrs_data_type::register_data_type_extension_codec!(
    Custom<Add>,
    BytesDataTypePlugin,
    BytesDataTypeTraits
);

inventory::submit! {
    BytesDataTypePlugin {
        type_id: TypeId::of::<Custom<Uncastable>>(),
        caster: |_| None,
        arc_caster: |_| None,
    }
}

/// A custom codec's support trait, defined outside the data type crate.
pub trait ProbeDataTypeTraits {
    /// Read instance-specific configuration through the registered cast.
    fn mask(&self) -> u8;
}

zarrs_data_type::define_data_type_support!(Probe);

impl ProbeDataTypeTraits for Custom<Xor> {
    fn mask(&self) -> u8 {
        self.mask
    }
}

zarrs_data_type::register_data_type_extension_codec!(
    Custom<Xor>,
    ProbeDataTypePlugin,
    ProbeDataTypeTraits
);

#[test]
fn arc_support_matches_extension_lookup() {
    for dtype in [
        data_type::bool(),
        data_type::uint8(),
        data_type::uint16(),
        data_type::int32(),
        data_type::float32(),
        data_type::complex64(),
        data_type::raw_bits(3),
    ] {
        let arc = dtype.codec_bytes_arc().unwrap();
        assert!(std::ptr::addr_eq(Arc::as_ptr(&arc), dtype.as_any()));
        let input: Vec<u8> = (0..dtype.fixed_size().unwrap() * 4)
            .map(|index| u8::try_from(index).unwrap())
            .collect();
        for endian in [None, Some(Endianness::Little), Some(Endianness::Big)] {
            let retained = &*arc;
            let direct = dtype.codec_bytes().unwrap();
            assert_eq!(
                retained.is_decode_passthrough(endian),
                direct.is_decode_passthrough(endian)
            );
            assert_eq!(
                retained
                    .encode(CowBytes::from(input.as_slice()), endian)
                    .map_err(|e| e.to_string()),
                direct
                    .encode(CowBytes::from(input.as_slice()), endian)
                    .map_err(|e| e.to_string())
            );
            assert_eq!(
                retained
                    .decode(CowBytes::from(input.as_slice()), endian)
                    .map_err(|e| e.to_string()),
                direct
                    .decode(CowBytes::from(input.as_slice()), endian)
                    .map_err(|e| e.to_string())
            );
        }
    }
}

#[test]
fn arc_support_retains_the_configured_instance() {
    for mask in [0xA5, 0x5A] {
        let owner = Arc::new(Custom::<Xor>::new(mask));
        let weak = Arc::downgrade(&owner);
        let dtype = DataType::from(owner);
        assert_eq!(dtype.codec_probe().unwrap().mask(), mask);
        assert_eq!(dtype.codec_probe_arc().unwrap().mask(), mask);
        let arc = dtype.codec_bytes_arc().unwrap();
        assert!(std::ptr::addr_eq(Arc::as_ptr(&arc), dtype.as_any()));
        let cloned = arc.clone();
        drop(arc);
        drop(dtype);
        assert!(weak.upgrade().is_some());
        for _ in 0..3 {
            assert_eq!(
                cloned
                    .encode(CowBytes::from(&[0, 1][..]), None)
                    .unwrap()
                    .as_ref(),
                [mask, mask ^ 1]
            );
        }
        drop(cloned);
        assert!(weak.upgrade().is_none());
    }
}

#[test]
fn each_data_type_uses_its_own_support() {
    let input = [0, 1, 2, 3];
    let shape = [NonZeroU64::new(2).unwrap()];
    let options = CodecOptions::default();
    let resources = ComputeResources::default();
    let xor = DataType::new(Custom::<Xor>::new(0x0F));
    let add = DataType::new(Custom::<Add>::new(0x0F));
    for (dtype, expected) in [
        (&xor, input.map(|byte| byte ^ 0x0F)),
        (&add, input.map(|byte| byte + 0x0F)),
    ] {
        let arc = dtype.codec_bytes_arc().unwrap();
        assert!(std::ptr::addr_eq(Arc::as_ptr(&arc), dtype.as_any()));
        assert_eq!(
            arc.encode(CowBytes::from(&input[..]), None)
                .unwrap()
                .as_ref(),
            expected
        );
        let codec = BytesCodec::little()
            .with_context(dtype.clone(), None, &CodecSpecificOptions::default())
            .unwrap();
        let encoded = codec
            .encode(
                ArrayBytes::new_flen(input.to_vec()),
                &shape,
                &options,
                &resources,
            )
            .unwrap();
        assert_eq!(encoded.as_ref(), expected);
        assert_eq!(
            codec
                .decode(encoded, &shape, &options, &resources)
                .unwrap()
                .into_fixed()
                .unwrap()
                .as_ref(),
            input
        );
    }
}

#[test]
fn unsupported_or_invalid_registered_casts_return_errors() {
    for dtype in [
        data_type::bytes(),
        data_type::string(),
        data_type::uint8().to_optional(),
        DataType::new(Custom::<Uncastable>::default()),
    ] {
        for error in [
            dtype.codec_bytes_arc().err().unwrap(),
            dtype.codec_bytes().err().unwrap(),
        ] {
            let DataTypeCodecError::UnsupportedDataType {
                data_type,
                codec_name,
            } = error;
            assert_eq!(data_type, dtype);
            assert_eq!(codec_name, "Bytes");
        }
        assert!(
            BytesCodec::little()
                .with_context(dtype, None, &CodecSpecificOptions::default())
                .is_err()
        );
    }
}

#[test]
fn bound_custom_support_decodes_into_output_views() {
    let custom = Arc::new(Custom::<Xor>::new(0xA5));
    let codec = BytesCodec::little()
        .with_context(
            DataType::from(Arc::clone(&custom)),
            None,
            &CodecSpecificOptions::default(),
        )
        .unwrap();
    let shape = [NonZeroU64::new(2).unwrap(); 2];
    let input = [0, 1, 2, 3, 4, 5, 6, 7];
    let options = CodecOptions::default();
    let resources = ComputeResources::default();
    assert!(!codec.is_decode_passthrough(&shape).unwrap());
    let encoded = codec
        .encode(
            ArrayBytes::new_flen(input.to_vec()),
            &shape,
            &options,
            &resources,
        )
        .unwrap();
    assert_eq!(encoded.as_ref(), input.map(|byte| byte ^ 0xA5));
    assert_eq!(
        codec
            .decode(encoded.clone(), &shape, &options, &resources)
            .unwrap()
            .into_fixed()
            .unwrap()
            .as_ref(),
        input
    );
    for columns in [2, 4] {
        let mut output = vec![0xCC; 2 * columns * 2];
        let array_shape = [2, u64::try_from(columns).unwrap()];
        {
            // SAFETY: this is the only view of output and is used synchronously.
            let view = unsafe {
                ArrayBytesFixedDisjointView::new(
                    UnsafeCellSlice::new(&mut output),
                    2,
                    &array_shape,
                    ArraySubset::new_with_shape([2, 2]),
                )
            }
            .unwrap();
            codec
                .decode_into(
                    encoded.clone(),
                    &shape,
                    &mut view.into(),
                    &options,
                    &resources,
                )
                .unwrap();
        }
        let mut expected = vec![0xCC; output.len()];
        expected[..4].copy_from_slice(&input[..4]);
        expected[columns * 2..columns * 2 + 4].copy_from_slice(&input[4..]);
        assert_eq!(output, expected);
    }
    // Decode the contiguous and strided inputs through the bound support.
    assert_eq!(*custom.decode_lengths.lock().unwrap(), [8, 8, 8]);
}
