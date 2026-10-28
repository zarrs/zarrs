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

mod numcodecs_packbits_partial;

use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs_chunk_grid::ChunkGridCreateError;
use zarrs_codec::{
    ArrayBytes, ArrayCodecTraits, ArrayPartialDecoderTraits, ArrayPartialEncoderTraits,
    ArrayToArrayCodecSubchunkingTraits, ArrayToArrayCodecTraits, ArrayToBytesCodecTraits,
    ChunkGridDecoded, ChunkGridDecodedRef, ChunkGridEncoded, ChunkGridEncodedRef, Codec,
    CodecCreateError, CodecError, CodecMetadataOptions, CodecOptions, CodecPluginV2, CodecPluginV3,
    CodecSpecificOptions, CodecTraits, CodecTraitsV2, CodecTraitsV3, PartialDecoderCapability,
    PartialEncoderCapability, RecommendedConcurrency, UnboundArrayToArrayCodecTraits,
};
#[cfg(feature = "async")]
use zarrs_codec::{AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits};
use zarrs_metadata::Configuration;
use zarrs_metadata::v2::MetadataV2;
use zarrs_metadata::v3::MetadataV3;
use zarrs_plugin::ZarrVersion;

use numcodecs_packbits_partial::NumcodecsPackBitsPartial;

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

    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        PartialDecoderCapability {
            partial_read: true,
            partial_decode: true,
        }
    }

    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        PartialEncoderCapability {
            partial_encode: false,
        }
    }
}

impl UnboundArrayToArrayCodecTraits for NumcodecsPackBitsCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn UnboundArrayToArrayCodecTraits> {
        self
    }

    fn with_context(
        &self,
        data_type: DataType,
        fill_value: FillValue,
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
        let encoded_fill_value = FillValue::from(if fill_value.as_ne_bytes() == [0] {
            0u8
        } else {
            u8::MAX
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
    encoded_fill_value: FillValue,
}

impl NumcodecsPackBitsCodecBound {
    /// Adapt `input_handle` to the packed `uint8` array of an array with `shape` to bytes.
    fn packed_handle<T: ?Sized>(
        &self,
        input_handle: Arc<T>,
        shape: &[NonZeroU64],
    ) -> Result<Arc<NumcodecsPackBitsPartial<T>>, CodecError> {
        Ok(Arc::new(NumcodecsPackBitsPartial::new(
            input_handle,
            self.packbits.encoded_size(shape)?,
        )))
    }
}

impl ArrayCodecTraits for NumcodecsPackBitsCodecBound {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn data_type(&self) -> &DataType {
        self.packbits.data_type()
    }

    fn fill_value(&self) -> &FillValue {
        self.packbits.fill_value()
    }

    fn recommended_concurrency(
        &self,
        shape: &[NonZeroU64],
    ) -> Result<RecommendedConcurrency, CodecError> {
        self.packbits.recommended_concurrency(shape)
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
}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl ArrayToArrayCodecTraits for NumcodecsPackBitsCodecBound {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToArrayCodecTraits> {
        self
    }

    fn encoded_data_type(&self) -> &DataType {
        &self.encoded_data_type
    }

    fn encoded_fill_value(&self) -> &FillValue {
        &self.encoded_fill_value
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
    ) -> Result<ArrayBytes<'a>, CodecError> {
        self.packbits
            .encode(bytes, shape, options)
            .map(ArrayBytes::Fixed)
    }

    fn decode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        self.packbits.decode(bytes.into_fixed()?, shape, options)
    }

    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn ArrayPartialDecoderTraits>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        let input_handle = self.packed_handle(input_handle, shape)?;
        self.packbits
            .clone()
            .partial_decoder(input_handle, shape, options)
    }

    fn partial_encoder(
        self: Arc<Self>,
        input_handle: Arc<dyn ArrayPartialEncoderTraits>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, CodecError> {
        let input_handle = self.packed_handle(input_handle, shape)?;
        self.packbits
            .clone()
            .partial_encoder(input_handle, shape, options)
    }

    #[cfg(feature = "async")]
    async fn async_partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncArrayPartialDecoderTraits>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        let input_handle = self.packed_handle(input_handle, shape)?;
        self.packbits
            .clone()
            .async_partial_decoder(input_handle, shape, options)
            .await
    }

    #[cfg(feature = "async")]
    async fn async_partial_encoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncArrayPartialEncoderTraits>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, CodecError> {
        let input_handle = self.packed_handle(input_handle, shape)?;
        self.packbits
            .clone()
            .async_partial_encoder(input_handle, shape, options)
            .await
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;
    use std::sync::Arc;

    use zarrs_codec::{
        BytesPartialDecoderTraits, Codec, CodecCreateError, CodecMetadataOptions, CodecOptions,
        CodecSpecificOptions, CowBytes, UnboundArrayToArrayCodecTraits,
        UnboundArrayToBytesCodecTraits,
    };
    use zarrs_metadata::v2::MetadataV2;
    use zarrs_metadata::v3::MetadataV3;
    use zarrs_plugin::ExtensionName;

    use super::NumcodecsPackBitsCodec;
    use crate::array::codec::BytesCodec;
    use crate::array::element::{Element, ElementOwned};
    use crate::array::{ArraySubset, FillValue, data_type};

    /// Bytes are filled from the most significant bit, as in `numcodecs.PackBits`.
    #[test]
    fn codec_numcodecs_packbits_bit_order() -> Result<(), Box<dyn std::error::Error>> {
        let options = CodecOptions::default();
        let codec = Arc::new(NumcodecsPackBitsCodec::new()).with_context(
            data_type::bool(),
            FillValue::from(false),
            &CodecSpecificOptions::default(),
        )?;
        let shape = [NonZeroU64::new(4).unwrap()];
        let elements = vec![true, false, false, true];
        let bytes = bool::to_array_bytes(&data_type::bool(), &elements)?;
        let encoded = codec.encode(bytes.clone(), &shape, &options)?;
        // The `numcodecs.PackBits` docstring example
        assert_eq!(encoded.clone().into_fixed()?.as_slice(), [4, 0b1001_0000]);
        assert_eq!(codec.decode(encoded, &shape, &options)?, bytes);
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
            FillValue::from(false),
            &CodecSpecificOptions::default(),
        )?;
        assert_eq!(bound.encoded_data_type(), &data_type::uint8());

        for data_type in [data_type::uint8(), data_type::int16(), data_type::float32()] {
            let result = codec.with_context(
                data_type.clone(),
                FillValue::from(false),
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
            array.store_array_subset(&array.subset_all(), elements.clone())?;
            assert_eq!(store.get(&StoreKey::new("0")?)?.unwrap(), expected);
            assert_eq!(
                array.retrieve_array_subset::<Vec<bool>>(&array.subset_all())?,
                elements
            );
            assert_eq!(
                array.retrieve_array_subset::<Vec<bool>>(&ArraySubset::new_with_ranges(&[2..9]))?,
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
                FillValue::from(false),
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
                        FillValue::from(false),
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

    /// Counts the calls to [`BytesPartialDecoderTraits::partial_decode_many`] and [`BytesPartialDecoderTraits::exists`].
    struct CountingPartialDecoder(
        CowBytes<'static>,
        std::sync::atomic::AtomicUsize,
        std::sync::atomic::AtomicUsize,
    );

    impl BytesPartialDecoderTraits for CountingPartialDecoder {
        fn exists(&self) -> Result<bool, zarrs_storage::StorageError> {
            self.2.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(true)
        }

        fn size_held(&self) -> usize {
            0
        }

        fn partial_decode_many(
            &self,
            decoded_regions: zarrs_storage::byte_range::ByteRangeIterator,
            options: &CodecOptions,
        ) -> Result<Option<Vec<CowBytes<'_>>>, zarrs_codec::CodecError> {
            self.1.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.0.partial_decode_many(decoded_regions, options)
        }

        fn supports_partial_decode(&self) -> bool {
            true
        }
    }

    /// A partial read of separate rows of `numcodecs.packbits` retrieves the packed bytes together,
    /// without checking that the chunk exists.
    #[test]
    fn codec_numcodecs_packbits_partial_decode_batched() -> Result<(), Box<dyn std::error::Error>> {
        let options = CodecOptions::default();
        let shape = [NonZeroU64::new(8).unwrap(), NonZeroU64::new(9).unwrap()];
        let Codec::ArrayToArray(codec) =
            Codec::from_metadata(&MetadataV3::new("numcodecs.packbits"))?
        else {
            panic!("expected an array to array codec");
        };
        let codec = codec.with_context(
            data_type::bool(),
            FillValue::from(false),
            &CodecSpecificOptions::default(),
        )?;
        let elements: Vec<bool> = (0..72).map(|i| i * 7 % 11 < 4).collect();
        let encoded = codec
            .encode(
                bool::to_array_bytes(&data_type::bool(), &elements)?,
                &shape,
                &options,
            )?
            .into_fixed()?
            .into_static();
        let encoded_shape = codec.encoded_shape(&shape)?;
        let input = Arc::new(CountingPartialDecoder(encoded, 0.into(), 0.into()));
        let input_handle = Arc::new(BytesCodec::default())
            .with_context(
                data_type::uint8(),
                FillValue::from(0u8),
                &CodecSpecificOptions::default(),
            )?
            .partial_decoder(input.clone(), &encoded_shape, &options)?;
        let partial_decoder = codec.partial_decoder(input_handle, &shape, &options)?;
        let decoded = partial_decoder
            .partial_decode(&ArraySubset::new_with_ranges(&[2..6, 1..5]), &options)?;
        let expected: Vec<bool> = (2..6)
            .flat_map(|i| elements[i * 9 + 1..i * 9 + 5].iter().copied())
            .collect();
        assert_eq!(
            bool::from_array_bytes(&data_type::bool(), decoded)?,
            expected
        );
        assert_eq!(input.1.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(input.2.load(std::sync::atomic::Ordering::Relaxed), 0);
        Ok(())
    }
}
