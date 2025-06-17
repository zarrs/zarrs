//! The `pcodec` array to bytes codec (Experimental).
//!
//! <div class="warning">
//! This codec is experimental and may be incompatible with other Zarr V3 implementations.
//! </div>
//!
//! [Pcodec](https://github.com/mwlon/pcodec) (or Pco, pronounced "pico") losslessly compresses and decompresses numerical sequences with high compression ratio and fast speed.
//!
//! This codec requires the `pcodec` feature, which is disabled by default.
//!
//! ### Compatible Implementations:
//! This codec is fully compatible with the `numcodecs.pcodec` codec in `zarr-python`.
//!
//! ### Specification
//! - <https://github.com/zarr-developers/zarr-extensions/tree/numcodecs/codecs/numcodecs.pcodec>
//!
//! ### Codec `name` Aliases (Zarr V3)
//! - `numcodecs.pcodec`
//! - `https://codec.zarrs.dev/array_to_bytes/pcodec`
//!
//! ### Codec `id` Aliases (Zarr V2)
//! - `pcodec`
//!
//! ### Codec `configuration` Example - [`PcodecCodecConfiguration`]:
//! ```rust
//! # let JSON = r#"
//! {
//!     "level": 5,
//!     "mode_spec": "auto",
//!     "delta_spec": "auto",
//!     "paging_spec": "equal_pages_up_to",
//!     "delta_encoding_order": null,
//!     "equal_pages_up_to": 262144
//! }
//! # "#;
//! # use zarrs_metadata_ext::codec::pcodec::PcodecCodecConfiguration;
//! # serde_json::from_str::<PcodecCodecConfiguration>(JSON).unwrap();
//! ```

mod pcodec_codec;
mod pcodec_partial_decoder;

use std::sync::Arc;

pub use zarrs_metadata_ext::codec::pcodec::{
    PcodecCodecConfiguration, PcodecCodecConfigurationV1, PcodecCompressionLevel,
    PcodecDeltaEncodingOrder,
};

pub use pcodec_codec::PcodecCodec;

use crate::{
    array::codec::{Codec, CodecPlugin},
    metadata::v3::MetadataV3,
    plugin::{PluginCreateError, PluginMetadataInvalidError},
};
use zarrs_registry::codec::PCODEC;

// Register the codec.
inventory::submit! {
    CodecPlugin::new(PCODEC, is_identifier_pcodec, create_codec_pcodec)
}

fn is_identifier_pcodec(identifier: &str) -> bool {
    identifier == PCODEC
}

pub(crate) fn create_codec_pcodec(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
    let configuration = metadata
        .to_configuration()
        .map_err(|_| PluginMetadataInvalidError::new(PCODEC, "codec", metadata.to_string()))?;
    let codec = Arc::new(PcodecCodec::new_with_configuration(&configuration)?);
    Ok(Codec::ArrayToBytes(codec))
}

macro_rules! unsupported_dtypes {
    // TODO: Add support for all int/float types?
    // TODO: Add support for extensions?
    () => {
        DataType::Bool
            | DataType::Int2
            | DataType::Int4
            | DataType::Int8
            | DataType::UInt2
            | DataType::UInt4
            | DataType::UInt8
            | DataType::Float4E2M1FN
            | DataType::Float6E2M3FN
            | DataType::Float6E3M2FN
            | DataType::Float8E3M4
            | DataType::Float8E4M3
            | DataType::Float8E4M3B11FNUZ
            | DataType::Float8E4M3FNUZ
            | DataType::Float8E5M2
            | DataType::Float8E5M2FNUZ
            | DataType::Float8E8M0FNU
            | DataType::BFloat16
            | DataType::ComplexBFloat16
            | DataType::ComplexFloat4E2M1FN
            | DataType::ComplexFloat6E2M3FN
            | DataType::ComplexFloat6E3M2FN
            | DataType::ComplexFloat8E3M4
            | DataType::ComplexFloat8E4M3
            | DataType::ComplexFloat8E4M3B11FNUZ
            | DataType::ComplexFloat8E4M3FNUZ
            | DataType::ComplexFloat8E5M2
            | DataType::ComplexFloat8E5M2FNUZ
            | DataType::ComplexFloat8E8M0FNU
            | DataType::RawBits(_)
            | DataType::String
            | DataType::Bytes
            | DataType::Extension(_)
    };
}
use unsupported_dtypes;

#[cfg(test)]
mod tests {
    use std::{num::NonZeroU64, sync::Arc};

    use crate::{
        array::{
            codec::{ArrayToBytesCodecTraits, CodecOptions},
            transmute_to_bytes_vec, ArrayBytes, ChunkRepresentation, ChunkShape, DataType,
            FillValue,
        },
        array_subset::ArraySubset,
    };

    use super::*;

    const JSON_VALID: &str = r#"{
        "level": 8,
        "delta_encoding_order": 2,
        "mode_spec": "auto",
        "equal_pages_up_to": 262144
    }"#;

    #[test]
    fn codec_pcodec_configuration() {
        let codec_configuration: PcodecCodecConfiguration =
            serde_json::from_str(JSON_VALID).unwrap();
        let _ = PcodecCodec::new_with_configuration(&codec_configuration);
    }

    fn codec_pcodec_round_trip_impl(
        codec: &PcodecCodec,
        data_type: DataType,
        fill_value: FillValue,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let chunk_shape = vec![NonZeroU64::new(10).unwrap(), NonZeroU64::new(10).unwrap()];
        let chunk_representation =
            ChunkRepresentation::new(chunk_shape, data_type, fill_value).unwrap();
        let size = chunk_representation.num_elements_usize()
            * chunk_representation.data_type().fixed_size().unwrap();
        let bytes: Vec<u8> = (0..size).map(|s| s as u8).collect();
        let bytes: ArrayBytes = bytes.into();

        let max_encoded_size = codec.encoded_representation(&chunk_representation)?;
        let encoded = codec.encode(
            bytes.clone(),
            &chunk_representation,
            &CodecOptions::default(),
        )?;
        assert!((encoded.len() as u64) <= max_encoded_size.size().unwrap());
        let decoded = codec
            .decode(encoded, &chunk_representation, &CodecOptions::default())
            .unwrap();
        assert_eq!(bytes, decoded);
        Ok(())
    }

    #[test]
    fn codec_pcodec_round_trip_u16() {
        codec_pcodec_round_trip_impl(
            &PcodecCodec::new_with_configuration(&serde_json::from_str(JSON_VALID).unwrap())
                .unwrap(),
            DataType::UInt16,
            FillValue::from(0u16),
        )
        .unwrap();
    }

    #[test]
    fn codec_pcodec_round_trip_u32() {
        codec_pcodec_round_trip_impl(
            &PcodecCodec::new_with_configuration(&serde_json::from_str(JSON_VALID).unwrap())
                .unwrap(),
            DataType::UInt32,
            FillValue::from(0u32),
        )
        .unwrap();
    }

    #[test]
    fn codec_pcodec_round_trip_u64() {
        codec_pcodec_round_trip_impl(
            &PcodecCodec::new_with_configuration(&serde_json::from_str(JSON_VALID).unwrap())
                .unwrap(),
            DataType::UInt64,
            FillValue::from(0u64),
        )
        .unwrap();
    }

    #[test]
    fn codec_pcodec_round_trip_i16() {
        codec_pcodec_round_trip_impl(
            &PcodecCodec::new_with_configuration(&serde_json::from_str(JSON_VALID).unwrap())
                .unwrap(),
            DataType::Int16,
            FillValue::from(0i16),
        )
        .unwrap();
    }

    #[test]
    fn codec_pcodec_round_trip_i32() {
        codec_pcodec_round_trip_impl(
            &PcodecCodec::new_with_configuration(&serde_json::from_str(JSON_VALID).unwrap())
                .unwrap(),
            DataType::Int32,
            FillValue::from(0i32),
        )
        .unwrap();
    }

    #[test]
    fn codec_pcodec_round_trip_i64() {
        codec_pcodec_round_trip_impl(
            &PcodecCodec::new_with_configuration(&serde_json::from_str(JSON_VALID).unwrap())
                .unwrap(),
            DataType::Int64,
            FillValue::from(0i64),
        )
        .unwrap();
    }

    #[test]
    fn codec_pcodec_round_trip_f16() {
        codec_pcodec_round_trip_impl(
            &PcodecCodec::new_with_configuration(&serde_json::from_str(JSON_VALID).unwrap())
                .unwrap(),
            DataType::Float16,
            FillValue::from(half::f16::from_f32(0.0)),
        )
        .unwrap();
    }

    #[test]
    fn codec_pcodec_round_trip_f32() {
        codec_pcodec_round_trip_impl(
            &PcodecCodec::new_with_configuration(&serde_json::from_str(JSON_VALID).unwrap())
                .unwrap(),
            DataType::Float32,
            FillValue::from(0f32),
        )
        .unwrap();
    }

    #[test]
    fn codec_pcodec_round_trip_f64() {
        codec_pcodec_round_trip_impl(
            &PcodecCodec::new_with_configuration(&serde_json::from_str(JSON_VALID).unwrap())
                .unwrap(),
            DataType::Float64,
            FillValue::from(0f64),
        )
        .unwrap();
    }

    #[test]
    fn codec_pcodec_round_trip_complex_float16() {
        codec_pcodec_round_trip_impl(
            &PcodecCodec::new_with_configuration(&serde_json::from_str(JSON_VALID).unwrap())
                .unwrap(),
            DataType::ComplexFloat16,
            FillValue::from(num::complex::Complex::<half::f16>::new(
                half::f16::from_f32(0f32),
                half::f16::from_f32(0f32),
            )),
        )
        .unwrap();
    }

    #[test]
    fn codec_pcodec_round_trip_complex_float32() {
        codec_pcodec_round_trip_impl(
            &PcodecCodec::new_with_configuration(&serde_json::from_str(JSON_VALID).unwrap())
                .unwrap(),
            DataType::ComplexFloat32,
            FillValue::from(num::complex::Complex::<f32>::new(0f32, 0f32)),
        )
        .unwrap();
    }

    #[test]
    fn codec_pcodec_round_trip_complex_float64() {
        codec_pcodec_round_trip_impl(
            &PcodecCodec::new_with_configuration(&serde_json::from_str(JSON_VALID).unwrap())
                .unwrap(),
            DataType::ComplexFloat64,
            FillValue::from(num::complex::Complex::<f64>::new(0f64, 0f64)),
        )
        .unwrap();
    }

    #[test]
    fn codec_pcodec_round_trip_complex64() {
        codec_pcodec_round_trip_impl(
            &PcodecCodec::new_with_configuration(&serde_json::from_str(JSON_VALID).unwrap())
                .unwrap(),
            DataType::Complex64,
            FillValue::from(num::complex::Complex32::new(0f32, 0f32)),
        )
        .unwrap();
    }

    #[test]
    fn codec_pcodec_round_trip_complex128() {
        codec_pcodec_round_trip_impl(
            &PcodecCodec::new_with_configuration(&serde_json::from_str(JSON_VALID).unwrap())
                .unwrap(),
            DataType::Complex128,
            FillValue::from(num::complex::Complex64::new(0f64, 0f64)),
        )
        .unwrap();
    }

    #[test]
    fn codec_pcodec_round_trip_u8() {
        assert!(codec_pcodec_round_trip_impl(
            &PcodecCodec::new_with_configuration(&serde_json::from_str(JSON_VALID).unwrap())
                .unwrap(),
            DataType::UInt8,
            FillValue::from(0u8),
        )
        .is_err());
    }

    #[test]
    fn codec_pcodec_partial_decode() {
        let chunk_shape: ChunkShape = vec![4, 4].try_into().unwrap();
        let chunk_representation = ChunkRepresentation::new(
            chunk_shape.to_vec(),
            DataType::UInt32,
            FillValue::from(0u32),
        )
        .unwrap();
        let elements: Vec<u32> = (0..chunk_representation.num_elements() as u32).collect();
        let bytes = transmute_to_bytes_vec(elements);
        let bytes: ArrayBytes = bytes.into();

        let codec = Arc::new(
            PcodecCodec::new_with_configuration(&serde_json::from_str(JSON_VALID).unwrap())
                .unwrap(),
        );

        let encoded = codec
            .encode(
                bytes.clone(),
                &chunk_representation,
                &CodecOptions::default(),
            )
            .unwrap();
        let decoded_regions = [ArraySubset::new_with_ranges(&[1..3, 0..1])];
        let input_handle = Arc::new(std::io::Cursor::new(encoded));
        let partial_decoder = codec
            .partial_decoder(
                input_handle,
                &chunk_representation,
                &CodecOptions::default(),
            )
            .unwrap();
        let decoded_partial_chunk = partial_decoder
            .partial_decode(&decoded_regions, &CodecOptions::default())
            .unwrap();

        let decoded_partial_chunk: Vec<u8> = decoded_partial_chunk
            .into_iter()
            .map(|bytes| bytes.into_fixed().unwrap().into_owned())
            .flatten()
            .collect::<Vec<_>>()
            .chunks(size_of::<u8>())
            .map(|b| u8::from_ne_bytes(b.try_into().unwrap()))
            .collect();
        let answer: Vec<u32> = vec![4, 8];
        assert_eq!(transmute_to_bytes_vec(answer), decoded_partial_chunk);
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn codec_pcodec_async_partial_decode() {
        let chunk_shape: ChunkShape = vec![4, 4].try_into().unwrap();
        let chunk_representation = ChunkRepresentation::new(
            chunk_shape.to_vec(),
            DataType::UInt32,
            FillValue::from(0u32),
        )
        .unwrap();
        let elements: Vec<u32> = (0..chunk_representation.num_elements() as u32).collect();
        let bytes = transmute_to_bytes_vec(elements);
        let bytes: ArrayBytes = bytes.into();

        let codec = Arc::new(
            PcodecCodec::new_with_configuration(&serde_json::from_str(JSON_VALID).unwrap())
                .unwrap(),
        );

        let encoded = codec
            .encode(
                bytes.clone(),
                &chunk_representation,
                &CodecOptions::default(),
            )
            .unwrap();
        let decoded_regions = [ArraySubset::new_with_ranges(&[1..3, 0..1])];
        let input_handle = Arc::new(std::io::Cursor::new(encoded));
        let partial_decoder = codec
            .async_partial_decoder(
                input_handle,
                &chunk_representation,
                &CodecOptions::default(),
            )
            .await
            .unwrap();
        let decoded_partial_chunk = partial_decoder
            .partial_decode(&decoded_regions, &CodecOptions::default())
            .await
            .unwrap();

        let decoded_partial_chunk: Vec<u8> = decoded_partial_chunk
            .into_iter()
            .map(|bytes| bytes.into_fixed().unwrap().into_owned())
            .flatten()
            .collect::<Vec<_>>()
            .chunks(size_of::<u8>())
            .map(|b| u8::from_ne_bytes(b.try_into().unwrap()))
            .collect();
        let answer: Vec<u32> = vec![4, 8];
        assert_eq!(transmute_to_bytes_vec(answer), decoded_partial_chunk);
    }
}
