//! Contracts for registered and retained data type codec support.

use std::any::Any;
use std::borrow::Cow;
use std::num::NonZeroU64;
use std::sync::Arc;

use unsafe_cell_slice::UnsafeCellSlice;
use zarrs::array::codec::BytesCodec;
use zarrs::array::{ArrayBytes, ArrayBytesFixedDisjointView, ArraySubset, data_type};
use zarrs_codec::{
    ArrayBytesDecodeIntoInput, CodecOptions, CodecSpecificOptions, ComputeResources,
    UnboundArrayToBytesCodecTraits,
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
#[derive(Debug)]
struct Custom {
    mask: u8,
}

impl ExtensionName for Custom {
    fn name(&self, _: ZarrVersion) -> Option<Cow<'static, str>> {
        Some(Cow::Borrowed("test.bytes_support"))
    }
}

impl DataTypeTraits for Custom {
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

impl BytesDataTypeTraits for Custom {
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
        self.encode(bytes, endian)
    }
}

zarrs_data_type::register_data_type_extension_codec!(
    Custom,
    BytesDataTypePlugin,
    BytesDataTypeTraits
);

/// A custom codec's support trait, defined outside the data type crate.
pub trait ProbeDataTypeTraits {
    /// Read instance-specific configuration through the registered cast.
    fn mask(&self) -> u8;
}

zarrs_data_type::define_data_type_support!(Probe);

impl ProbeDataTypeTraits for Custom {
    fn mask(&self) -> u8 {
        self.mask
    }
}

zarrs_data_type::register_data_type_extension_codec!(
    Custom,
    ProbeDataTypePlugin,
    ProbeDataTypeTraits
);

#[test]
fn arc_support_retains_the_configured_instance() {
    for mask in [0xA5, 0x5A] {
        let owner = Arc::new(Custom { mask });
        let weak = Arc::downgrade(&owner);
        let dtype = DataType::from(owner);
        let probe = dtype.codec_probe_arc().unwrap();
        let arc = dtype.codec_bytes_arc().unwrap();
        assert!(std::ptr::addr_eq(Arc::as_ptr(&arc), dtype.as_any()));
        drop(dtype);
        assert_eq!(probe.mask(), mask);
        drop(probe);
        assert!(weak.upgrade().is_some());
        assert_eq!(
            arc.encode(CowBytes::from(&[0, 1][..]), None)
                .unwrap()
                .as_ref(),
            [mask, mask ^ 1]
        );
        drop(arc);
        assert!(weak.upgrade().is_none());
    }
}

#[test]
fn unsupported_arc_lookup_returns_an_error() {
    let dtype = data_type::string();
    let DataTypeCodecError::UnsupportedDataType {
        data_type,
        codec_name,
    } = dtype.codec_bytes_arc().unwrap_err();
    assert_eq!(data_type, dtype);
    assert_eq!(codec_name, "Bytes");
}

#[test]
fn bound_custom_support_decodes_into_output_views() {
    let codec = BytesCodec::little()
        .with_context(
            DataType::new(Custom { mask: 0xA5 }),
            None,
            &CodecSpecificOptions::default(),
        )
        .unwrap();
    let shape = [NonZeroU64::new(2).unwrap(); 2];
    let input = [0, 1, 2, 3, 4, 5, 6, 7];
    let options = CodecOptions::default();
    let resources = ComputeResources::default();
    let encoded = codec
        .encode(
            ArrayBytes::new_flen(input.to_vec()),
            &shape,
            &options,
            &resources,
        )
        .unwrap();
    assert_eq!(encoded.as_ref(), input.map(|byte| byte ^ 0xA5));
    let mut output = [0xCC; 16];
    {
        // SAFETY: this is the only view of output and is used synchronously.
        let view = unsafe {
            ArrayBytesFixedDisjointView::new(
                UnsafeCellSlice::new(&mut output),
                2,
                &[2, 4],
                ArraySubset::new_with_shape([2, 2]),
            )
        }
        .unwrap();
        codec
            .decode_into(
                ArrayBytesDecodeIntoInput::Bytes(encoded),
                &shape,
                &mut view.into(),
                &options,
                &resources,
            )
            .unwrap();
    }
    assert_eq!(
        output,
        [
            0, 1, 2, 3, 0xCC, 0xCC, 0xCC, 0xCC, 4, 5, 6, 7, 0xCC, 0xCC, 0xCC, 0xCC
        ]
    );
}
