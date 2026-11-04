//! The `numcodecs.packbits` array to array codec.
//!
//! Packs `bool` elements into a one-dimensional `uint8` array, eight to a byte, as in `numcodecs.PackBits`.
//! Later array-to-array codecs and the array-to-bytes codec operate on the packed `uint8` array.
//!
//! Each byte is filled from its most significant bit, as in `numpy.packbits`, and is preceded by a byte holding the number of padding bits in the last byte.
//! This is the `first_byte` padding encoding of the [`packbits`](crate::array::codec::array_to_bytes::packbits) codec, but that fills each byte from its least significant bit.
//!
//! This is a legacy codec for migrating Zarr V2 data to Zarr V3. Prefer the `packbits` codec for new data.
//! See <https://github.com/zarr-developers/zarr-extensions/issues/85>.
//!
//! ### Compatible Implementations
//! This codec is compatible with the `numcodecs.packbits` codec in `zarr-python`.
//!
//! ### Codec `name` Aliases (Zarr V3)
//! - `numcodecs.packbits`
//!
//! ### Codec `id` Aliases (Zarr V2)
//! - `packbits`
//!
//! ### Codec `configuration` Example
//! This codec has no configuration. An `id` of `packbits` written by older Python is accepted.
//! ```json
//! {}
//! ```

use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs_chunk_grid::ChunkGridCreateError;
use zarrs_codec::{
    ArrayBytes, ArrayCodecTraits, ArrayToArrayCodecSubchunkingTraits, ArrayToArrayCodecTraits,
    ArrayToBytesCodecTraits, ChunkGridDecoded, ChunkGridDecodedRef, ChunkGridEncoded,
    ChunkGridEncodedRef, Codec, CodecCreateError, CodecError, CodecMetadataOptions, CodecOptions,
    CodecPluginV2, CodecPluginV3, CodecSpecificOptions, CodecTraits, CodecTraitsV2, CodecTraitsV3,
    UnboundArrayToArrayCodecTraits,
};

use zarrs_metadata::Configuration;
use zarrs_metadata::v2::MetadataV2;
use zarrs_metadata::v3::MetadataV3;
use zarrs_plugin::ZarrVersion;

use crate::array::codec::array_to_bytes::packbits::{
    PackBitsBitOrder, PackBitsCodec, PackBitsCodecBound,
};
use crate::array::data_type::{self, BoolDataType};
use crate::array::{ChunkShape, DataType, FillValue};
use zarrs_metadata_ext::codec::packbits::PackBitsPaddingEncoding;

/// A `numcodecs.packbits` codec implementation.
#[derive(Debug, Clone, Default)]
pub struct NumcodecsPackBitsCodec;

impl NumcodecsPackBitsCodec {
    /// Create a new `numcodecs.packbits` codec.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

zarrs_plugin::impl_extension_aliases!(NumcodecsPackBitsCodec, v3: "numcodecs.packbits", v2: "packbits");

inventory::submit! {
    CodecPluginV3::new::<NumcodecsPackBitsCodec>()
}
inventory::submit! {
    CodecPluginV2::new::<NumcodecsPackBitsCodec>()
}

impl CodecTraitsV3 for NumcodecsPackBitsCodec {
    fn create(metadata: &MetadataV3) -> Result<Codec, CodecCreateError> {
        if metadata.configuration().is_some_and(|configuration| {
            configuration
                .iter()
                .any(|(key, value)| key != "id" || value.as_str() != Some("packbits"))
        }) {
            return Err(CodecCreateError::Other(
                "the numcodecs packbits codec has no configuration other than an optional id of packbits"
                    .to_string(),
            ));
        }
        Ok(Codec::ArrayToArray(Arc::new(Self)))
    }
}

impl CodecTraitsV2 for NumcodecsPackBitsCodec {
    fn create(metadata: &MetadataV2) -> Result<Codec, CodecCreateError> {
        if !metadata.configuration().is_empty() {
            return Err(CodecCreateError::Other(
                "the numcodecs packbits codec has no configuration".to_string(),
            ));
        }
        Ok(Codec::ArrayToArray(Arc::new(Self)))
    }
}

impl CodecTraits for NumcodecsPackBitsCodec {
    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        Some(Configuration::default())
    }
}

impl UnboundArrayToArrayCodecTraits for NumcodecsPackBitsCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn UnboundArrayToArrayCodecTraits> {
        self
    }

    fn with_context(
        &self,
        data_type: DataType,
        fill_value: Option<FillValue>,
        _codec_specific_options: &CodecSpecificOptions,
    ) -> Result<Arc<dyn ArrayToArrayCodecTraits>, CodecCreateError> {
        if !data_type.is::<BoolDataType>() {
            return Err(CodecCreateError::UnsupportedDataType(
                data_type,
                "numcodecs.packbits".to_string(),
            ));
        }
        // The packed bytes of elements that are all the fill value, so that the encoded fill value of a missing chunk
        // unpacks to the fill value without checking whether the chunk exists
        let encoded_fill_value = fill_value.as_ref().map(|fill_value| {
            FillValue::from(if fill_value.as_ne_bytes() == [0] {
                0u8
            } else {
                u8::MAX
            })
        });
        let packbits = PackBitsCodec::new(PackBitsPaddingEncoding::FirstByte, None, None)?.bind(
            data_type,
            fill_value,
            PackBitsBitOrder::MsbFirst,
        )?;
        Ok(Arc::new(NumcodecsPackBitsCodecBound {
            packbits: Arc::new(packbits),
            encoded_data_type: data_type::uint8(),
            encoded_fill_value,
        }))
    }
}

/// A `numcodecs.packbits` codec implementation bound to a data type and fill value.
#[derive(Debug)]
struct NumcodecsPackBitsCodecBound {
    packbits: Arc<PackBitsCodecBound>,
    encoded_data_type: DataType,
    encoded_fill_value: Option<FillValue>,
}

impl NumcodecsPackBitsCodecBound {}

impl ArrayCodecTraits for NumcodecsPackBitsCodecBound {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn data_type(&self) -> &DataType {
        self.packbits.data_type()
    }

    fn fill_value(&self) -> Option<&FillValue> {
        self.packbits.fill_value()
    }
}

impl ArrayToArrayCodecSubchunkingTraits for NumcodecsPackBitsCodecBound {
    fn encoded_chunk_grid(
        &self,
        _decoded_chunk_grid: ChunkGridDecodedRef<'_>,
    ) -> Result<ChunkGridEncoded, ChunkGridCreateError> {
        Ok(ChunkGridEncoded::ChunkLocal)
    }

    fn decoded_subchunk_grid(
        &self,
        _decoded_chunk_grid: ChunkGridDecodedRef<'_>,
        _encoded_subchunk_grid: ChunkGridEncodedRef<'_>,
    ) -> Result<ChunkGridDecoded, ChunkGridCreateError> {
        Ok(ChunkGridDecoded::None)
    }
    fn encoded_coordinates(
        &self,
        _shape: &[NonZeroU64],
        _coordinates: &[u64],
    ) -> Result<zarrs_chunk_grid::ArrayIndices, CodecError> {
        Err(CodecError::UnsupportedEncodedSubchunk)
    }
}

impl ArrayToArrayCodecTraits for NumcodecsPackBitsCodecBound {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToArrayCodecTraits> {
        self
    }

    fn plan_decode_request<'a>(
        &self,
        shape: &[NonZeroU64],
        request: zarrs_codec::plan::DecodeRequest<'a>,
    ) -> Result<zarrs_codec::plan::ArrayToArrayDecodePlan<'a>, CodecError> {
        request.indexer.validate(shape)?;
        Ok(zarrs_codec::plan::ArrayToArrayDecodePlan::Whole(request))
    }

    fn encoded_data_type(&self) -> &DataType {
        &self.encoded_data_type
    }

    fn encoded_fill_value(&self) -> Option<&FillValue> {
        self.encoded_fill_value.as_ref()
    }

    fn encoded_shape(&self, decoded_shape: &[NonZeroU64]) -> Result<ChunkShape, CodecError> {
        let size = self.packbits.encoded_size(decoded_shape)?;
        Ok(vec![
            NonZeroU64::new(size).expect("packed elements are at least one byte"),
        ])
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
        resources: &zarrs_codec::ComputeResources,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        self.packbits
            .encode(bytes, shape, options, resources)
            .map(ArrayBytes::Fixed)
    }

    fn decode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
        resources: &zarrs_codec::ComputeResources,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        self.packbits
            .decode(bytes.into_fixed()?, shape, options, resources)
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;
    use std::sync::Arc;
    use zarrs_codec::{ComputeResources, Resources};

    use zarrs_codec::{
        Codec, CodecCreateError, CodecMetadataOptions, CodecOptions, CodecSpecificOptions,
        UnboundArrayToArrayCodecTraits,
    };
    use zarrs_metadata::v2::MetadataV2;
    use zarrs_metadata::v3::MetadataV3;
    use zarrs_plugin::ExtensionName;

    use super::NumcodecsPackBitsCodec;

    use crate::array::element::Element;
    use crate::array::{ArraySubset, FillValue, data_type};

    /// Bytes are filled from the most significant bit, as in `numcodecs.PackBits`.
    #[test]
    fn codec_numcodecs_packbits_bit_order() -> Result<(), Box<dyn std::error::Error>> {
        let options = CodecOptions::default();
        let codec = Arc::new(NumcodecsPackBitsCodec::new()).with_context(
            data_type::bool(),
            Some(FillValue::from(false)),
            &CodecSpecificOptions::default(),
        )?;
        let shape = [NonZeroU64::new(4).unwrap()];
        let elements = vec![true, false, false, true];
        let bytes = bool::to_array_bytes(&data_type::bool(), &elements)?;
        let encoded = codec.encode(
            bytes.clone(),
            &shape,
            &options,
            &ComputeResources::default(),
        )?;
        // The `numcodecs.PackBits` docstring example
        assert_eq!(encoded.clone().into_fixed()?.as_slice(), [4, 0b1001_0000]);
        assert_eq!(
            codec.decode(encoded, &shape, &options, &ComputeResources::default())?,
            bytes
        );
        Ok(())
    }

    #[test]
    fn codec_numcodecs_packbits_v2() -> Result<(), Box<dyn std::error::Error>> {
        let metadata: MetadataV2 = serde_json::from_str(r#"{"id":"packbits"}"#)?;
        let Codec::ArrayToArray(codec) = Codec::from_metadata(&metadata)? else {
            panic!("expected an array to array codec");
        };
        let options = CodecMetadataOptions::default();
        assert_eq!(codec.name_v2().as_deref(), Some("packbits"));
        assert!(codec.configuration_v2(&options).unwrap().is_empty());
        assert_eq!(codec.name_v3().as_deref(), Some("numcodecs.packbits"));
        assert!(codec.configuration_v3(&options).unwrap().is_empty());
        let bound = codec.with_context(
            data_type::bool(),
            Some(FillValue::from(false)),
            &CodecSpecificOptions::default(),
        )?;
        assert_eq!(bound.encoded_data_type(), &data_type::uint8());

        for data_type in [data_type::uint8(), data_type::int16(), data_type::float32()] {
            let result = codec.with_context(
                data_type.clone(),
                Some(FillValue::from(false)),
                &CodecSpecificOptions::default(),
            );
            assert!(matches!(
                result,
                Err(CodecCreateError::UnsupportedDataType(rejected, _)) if rejected == data_type
            ));
        }

        for json in [
            r#"{"id":"packbits","first_bit":0}"#,
            r#"{"id":"packbits","last_bit":0}"#,
            r#"{"id":"packbits","padding_encoding":"first_byte"}"#,
            r#"{"id":"packbits","unknown":false}"#,
        ] {
            let invalid_metadata: MetadataV2 = serde_json::from_str(json)?;
            assert!(Codec::from_metadata(&invalid_metadata).is_err());
        }
        Ok(())
    }

    /// Filters after the Zarr V2 `packbits` filter operate on the packed `uint8` array, as in numcodecs.
    #[test]
    fn codec_numcodecs_packbits_v2_later_filters() -> Result<(), Box<dyn std::error::Error>> {
        use crate::array::{Array, ArrayMetadata};
        use zarrs_metadata::v2::ArrayMetadataV2;
        use zarrs_storage::store::MemoryStore;
        use zarrs_storage::{ReadableStorageTraits, StoreKey};

        let metadata: ArrayMetadataV2 = serde_json::from_value(serde_json::json!({
            "zarr_format": 2,
            "shape": [10],
            "chunks": [10],
            "dtype": "|b1",
            "fill_value": false,
            "order": "C",
            "compressor": null,
            "filters": [
                {"id": "packbits"},
                {"id": "fixedscaleoffset", "offset": 0, "scale": 2, "dtype": "|u1", "astype": "|u1"},
            ],
        }))?;
        let elements = vec![
            false, false, true, true, false, false, false, true, false, false,
        ];
        // `numpy.packbits` with the number of padding bits first, then scaled by 2
        let expected: &[u8] = &[6 * 2, 0b0011_0001 * 2, 0];

        let metadata_v3 = crate::convert::array_metadata_v2_to_v3(&metadata)?;
        let names: Vec<_> = metadata_v3.codecs.iter().map(MetadataV3::name).collect();
        assert_eq!(
            names,
            ["numcodecs.packbits", "numcodecs.fixedscaleoffset", "bytes"]
        );

        for metadata in [ArrayMetadata::V2(metadata), ArrayMetadata::V3(metadata_v3)] {
            let store = Arc::new(MemoryStore::new());
            let array = Array::new_with_metadata(store.clone(), "/", metadata)?;
            array.store_array_subset(
                &array.subset_all(),
                elements.clone(),
                &Resources::default(),
            )?;
            assert_eq!(store.get(&StoreKey::new("0")?)?.unwrap(), expected);
            assert_eq!(
                array.retrieve_array_subset::<Vec<bool>>(
                    &array.subset_all(),
                    &Resources::default()
                )?,
                elements
            );
            assert_eq!(
                array.retrieve_array_subset::<Vec<bool>>(
                    &ArraySubset::new_with_ranges(&[2..9]),
                    &Resources::default()
                )?,
                elements[2..9]
            );
        }
        Ok(())
    }

    #[test]
    fn codec_numcodecs_packbits_v3() -> Result<(), Box<dyn std::error::Error>> {
        for json in [
            r#"{"name":"numcodecs.packbits"}"#,
            r#"{"name":"numcodecs.packbits","configuration":{}}"#,
            r#"{"name":"numcodecs.packbits","configuration":{"id":"packbits"}}"#,
        ] {
            let metadata: MetadataV3 = serde_json::from_str(json)?;
            let Codec::ArrayToArray(codec) = Codec::from_metadata(&metadata)? else {
                panic!("expected an array to array codec");
            };
            let options = CodecMetadataOptions::default();
            assert_eq!(codec.name_v3().as_deref(), Some("numcodecs.packbits"));
            assert!(codec.configuration_v3(&options).unwrap().is_empty());
            assert_eq!(codec.name_v2().as_deref(), Some("packbits"));
            assert!(codec.configuration_v2(&options).unwrap().is_empty());
            let bound = codec.with_context(
                data_type::bool(),
                Some(FillValue::from(false)),
                &CodecSpecificOptions::default(),
            )?;
            assert_eq!(bound.encoded_data_type(), &data_type::uint8());
            for (elements, packed) in [(1, 2), (8, 2), (20, 4)] {
                let shape = [NonZeroU64::new(elements).unwrap()];
                assert_eq!(
                    bound.encoded_shape(&shape)?,
                    [NonZeroU64::new(packed).unwrap()]
                );
            }
            let overflowing_shape = [NonZeroU64::new(u64::MAX).unwrap(); 2];
            assert!(bound.encoded_shape(&overflowing_shape).is_err());
            for data_type in [data_type::uint8(), data_type::int16(), data_type::float32()] {
                assert!(matches!(
                    codec.with_context(
                        data_type.clone(),
                        Some(FillValue::from(false)),
                        &CodecSpecificOptions::default(),
                    ),
                    Err(CodecCreateError::UnsupportedDataType(rejected, _)) if rejected == data_type
                ));
            }
        }
        for configuration in [
            serde_json::json!({"first_bit": 0}),
            serde_json::json!({"padding_encoding": "first_byte"}),
            serde_json::json!({"id": "other"}),
        ] {
            let metadata: MetadataV3 = serde_json::from_value(serde_json::json!({
                "name": "numcodecs.packbits",
                "configuration": configuration,
            }))?;
            assert!(Codec::from_metadata(&metadata).is_err());
        }
        Ok(())
    }

    #[test]
    fn codec_numcodecs_packbits_planned_partial_decode() -> Result<(), Box<dyn std::error::Error>> {
        use crate::array::{ArrayBuilder, ArraySubset, Resources};
        let store = Arc::new(zarrs_storage::store::MemoryStore::new());
        let mut builder = ArrayBuilder::new(
            vec![17],
            vec![17],
            data_type::bool(),
            FillValue::from(false),
        );
        builder.array_to_array_codecs(vec![Arc::new(NumcodecsPackBitsCodec::new())]);
        let array = builder.build(store, "/")?;
        let elements: Vec<bool> = (0..17).map(|i| i % 3 == 0).collect();
        array.store_chunk(&[0], &elements, &Resources::default())?;
        for range in [0..17, 1..9, 8..17] {
            assert_eq!(
                array.retrieve_partial_chunk::<Vec<bool>>(
                    &[0],
                    &ArraySubset::new_with_ranges(std::slice::from_ref(&range)),
                    &Resources::default()
                )?,
                elements[range.start as usize..range.end as usize]
            );
        }
        Ok(())
    }
}
