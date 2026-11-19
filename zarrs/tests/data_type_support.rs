//! Contracts for registered and retained data type codec support.

use std::any::Any;
use std::borrow::Cow;
use std::num::NonZeroU64;
use std::sync::Arc;

use unsafe_cell_slice::UnsafeCellSlice;
use zarrs::array::codec::BytesCodec;
use zarrs::array::{ArrayBytes, ArrayBytesFixedDisjointView, ArraySubset, data_type};
use zarrs_codec::{
    BytesDecodeInput, CodecOptions, CodecSpecificOptions, ComputeResources, Resources,
    UnboundArrayToBytesCodecTraits,
};
use zarrs_data_type::codec_traits::bytes::{
    BytesCodecEndiannessMissingError, BytesDataTypeExt, BytesDataTypePlugin, BytesDataTypeTraits,
};
use zarrs_data_type::{
    CowBytes, DataType, DataTypeCodecError, DataTypeFillValueError, DataTypeFillValueMetadataError,
    DataTypeTraits, DataTypeTraitsV3, FillValue, OptionalDataType, OptionalDataTypeTraits,
};
use zarrs_metadata::{Configuration, Endianness, FillValueMetadata};
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

impl zarrs_data_type::FixedDataTypeTraits for Custom {
    fn size(&self) -> usize {
        2
    }
}

impl DataTypeTraits for Custom {
    fn configuration(&self, _: ZarrVersion) -> Configuration {
        serde_json::Map::from_iter([("mask".into(), self.mask.into())]).into()
    }

    fn into_data_type(self: std::sync::Arc<Self>) -> zarrs_data_type::DataType {
        zarrs_data_type::DataType::Fixed(self)
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

#[derive(Debug)]
struct CustomOptional(DataType);

zarrs_plugin::impl_extension_aliases!(CustomOptional, v3: "test.optional");

impl DataTypeTraits for CustomOptional {
    fn into_data_type(self: Arc<Self>) -> DataType {
        DataType::Optional(self)
    }

    fn configuration(&self, version: ZarrVersion) -> Configuration {
        OptionalDataType::new(self.0.clone()).configuration(version)
    }

    fn fill_value(
        &self,
        metadata: &FillValueMetadata,
        version: ZarrVersion,
    ) -> Result<FillValue, DataTypeFillValueMetadataError> {
        self.0.to_optional().fill_value(metadata, version)
    }

    fn metadata_fill_value(
        &self,
        fill_value: &FillValue,
    ) -> Result<FillValueMetadata, DataTypeFillValueError> {
        self.0.to_optional().metadata_fill_value(fill_value)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl OptionalDataTypeTraits for CustomOptional {
    fn data_type(&self) -> &DataType {
        &self.0
    }
}

impl DataTypeTraitsV3 for CustomOptional {
    fn create(
        metadata: &zarrs_metadata::v3::MetadataV3,
    ) -> Result<DataType, zarrs_plugin::PluginCreateError> {
        let optional = OptionalDataType::create(metadata)?;
        Ok(DataType::new(Self(
            optional.optional_inner().unwrap().clone(),
        )))
    }
}

inventory::submit! {
    zarrs_data_type::DataTypePluginV3::new::<CustomOptional>()
}

#[test]
fn external_optional_supports_nested_array_operations() {
    use zarrs::array::{ArrayBuilder, ArraySubset};
    use zarrs::storage::store::MemoryStore;

    let owner = Arc::new(CustomOptional(DataType::new(CustomOptional(
        data_type::uint16(),
    ))));
    let erased: Arc<dyn DataTypeTraits> = owner.clone();
    let dtype = DataType::from(erased);
    assert!(dtype.is_optional());
    assert!(!dtype.is_fixed());
    assert!(!dtype.is_variable());
    assert!(Arc::ptr_eq(
        &owner,
        &dtype.downcast_arc::<CustomOptional>().unwrap()
    ));
    assert_eq!(dtype.as_optional().unwrap().data_type(), &owner.0);
    assert!(dtype.downcast_arc::<OptionalDataType>().is_none());

    for fill in [None, Some(None), Some(Some(42u16))] {
        let array = ArrayBuilder::new(vec![4], vec![2], dtype.clone(), FillValue::from(fill))
            .build(Arc::new(MemoryStore::new()), "/")
            .unwrap();
        let initial: Vec<Option<Option<u16>>> = array
            .retrieve_array_subset(&array.subset_all(), &Resources::default())
            .unwrap();
        assert_eq!(initial, vec![fill; 4]);
        let values: [Option<Option<u16>>; 4] = [None, Some(None), Some(Some(7)), Some(Some(8))];
        array
            .store_array_subset(
                &array.subset_all(),
                values.as_slice(),
                &Resources::default(),
            )
            .unwrap();
        array
            .store_array_subset(
                &ArraySubset::new_with_ranges(&[1..3]),
                &[Some(Some(9u16)), None],
                &Resources::default(),
            )
            .unwrap();
        let actual: Vec<Option<Option<u16>>> = array
            .retrieve_array_subset(&array.subset_all(), &Resources::default())
            .unwrap();
        assert_eq!(actual, [None, Some(Some(9)), None, Some(Some(8))]);
        array.store_metadata().unwrap();
        let reopened = zarrs::array::Array::open(array.storage().clone(), "/").unwrap();
        assert_eq!(reopened.data_type(), &dtype);
    }
}

#[test]
fn array_bytes_operations_check_layouts_at_every_layer() {
    use zarrs::array::{ArrayBytesOffsets, ArraySubset, update_array_bytes};

    let types = [
        data_type::uint8(),
        data_type::string(),
        data_type::uint8().to_optional(),
    ];
    let bytes = [
        ArrayBytes::new_flen(vec![7]),
        ArrayBytes::new_vlen(vec![7], ArrayBytesOffsets::new(vec![0u32, 1]).unwrap()).unwrap(),
        ArrayBytes::new_flen(vec![7]).with_optional_mask(vec![true]),
    ];
    let subset = ArraySubset::new_with_shape([1]);
    let runs = subset.contiguous_linearised_indices(&[1]).unwrap();
    let indexer = vec![vec![0]];
    for (i, dtype) in types.iter().enumerate() {
        for (j, bytes) in bytes.iter().enumerate() {
            assert_eq!(bytes.validate(1, dtype).is_ok(), i == j);
            assert_eq!(bytes.extract_runs(&runs, dtype).is_ok(), i == j);
            assert_eq!(
                bytes.extract_array_subset(&indexer, &[1], dtype).is_ok(),
                i == j
            );
            for (k, update) in bytes_for_layouts().iter().enumerate() {
                assert_eq!(
                    update_array_bytes(bytes.clone(), &[1], &subset, update, dtype).is_ok(),
                    i == j && i == k
                );
                assert_eq!(
                    update_array_bytes(bytes.clone(), &[1], &indexer, update, dtype).is_ok(),
                    i == j && i == k
                );
            }
        }
    }
    let nested_type = types[2].to_optional();
    for wrong in [
        bytes[2].clone(),
        bytes[1]
            .clone()
            .with_optional_mask(vec![true])
            .with_optional_mask(vec![true]),
    ] {
        assert!(wrong.validate(1, &nested_type).is_err());
        assert!(wrong.extract_runs(&runs, &nested_type).is_err());
    }

    fn bytes_for_layouts() -> [ArrayBytes<'static>; 3] {
        [
            ArrayBytes::new_flen(vec![8]),
            ArrayBytes::new_vlen(vec![8], ArrayBytesOffsets::new(vec![0u32, 1]).unwrap()).unwrap(),
            ArrayBytes::new_flen(vec![8]).with_optional_mask(vec![true]),
        ]
    }
}

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
                BytesDecodeInput::Bytes(encoded),
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
