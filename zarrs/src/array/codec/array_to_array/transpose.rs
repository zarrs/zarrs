//! The `transpose` array to array codec (Core).
//!
//! Permutes the dimensions of arrays.
//!
//! ### Compatible Implementations
//! This is a core codec and should be compatible with all Zarr V3 implementations that support it.
//!
//! ### Specification
//! - <https://zarr-specs.readthedocs.io/en/latest/v3/codecs/transpose/index.html>
//! - <https://github.com/zarr-developers/zarr-extensions/tree/main/codecs/transpose>
//!
//! ### Codec `name` Aliases (Zarr V3)
//! - `transpose`
//!
//! ### Codec `id` Aliases (Zarr V2)
//! None
//!
//! ### Codec `configuration` Example - [`TransposeCodecConfiguration`]:
//! ```rust
//! # let JSON = r#"
//! {
//!     "order": [2, 1, 0]
//! }
//! # "#;
//! # use zarrs_metadata_ext::codec::transpose::TransposeCodecConfiguration;
//! # let configuration: TransposeCodecConfiguration = serde_json::from_str(JSON).unwrap();
//! ```

mod transpose_codec;
mod transpose_partial_decoder;

use std::sync::Arc;

pub use transpose_codec::TransposeCodec;
pub use zarrs_metadata_ext::codec::transpose::{
    TransposeCodecConfiguration, TransposeCodecConfigurationV1, TransposeOrder, TransposeOrderError,
};
use zarrs_registry::codec::TRANSPOSE;

use crate::{
    array::{
        array_bytes::RawBytesOffsets,
        codec::{Codec, CodecPlugin},
        ArrayBytes, RawBytes,
    },
    metadata::v3::MetadataV3,
    plugin::{PluginCreateError, PluginMetadataInvalidError},
};

// Register the codec.
inventory::submit! {
    CodecPlugin::new(TRANSPOSE, is_identifier_transpose, create_codec_transpose)
}

fn is_identifier_transpose(identifier: &str) -> bool {
    identifier == TRANSPOSE
}

pub(crate) fn create_codec_transpose(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
    let configuration: TransposeCodecConfiguration = metadata
        .to_configuration()
        .map_err(|_| PluginMetadataInvalidError::new(TRANSPOSE, "codec", metadata.to_string()))?;
    let codec = Arc::new(TransposeCodec::new_with_configuration(&configuration)?);
    Ok(Codec::ArrayToArray(codec))
}

fn calculate_order_encode(order: &TransposeOrder, array_dimensions: usize) -> Vec<usize> {
    assert_eq!(order.0.len(), array_dimensions);
    let mut permutation_encode = Vec::<usize>::with_capacity(array_dimensions + 1);
    permutation_encode.extend(&order.0);
    permutation_encode.push(array_dimensions);
    permutation_encode
}

fn calculate_order_decode(order: &TransposeOrder, array_dimensions: usize) -> Vec<usize> {
    assert_eq!(order.0.len(), array_dimensions);
    let mut permutation_decode = vec![0; array_dimensions + 1];
    for (i, val) in order.0.iter().enumerate() {
        permutation_decode[*val] = i;
    }
    permutation_decode[array_dimensions] = array_dimensions;
    permutation_decode
}

fn transpose_array(
    transpose_order: &[usize],
    untransposed_shape: &[u64],
    bytes_per_element: usize,
    data: &[u8],
) -> Result<Vec<u8>, ndarray::ShapeError> {
    // Create an array view of the data
    let mut shape_n = Vec::with_capacity(untransposed_shape.len() + 1);
    for size in untransposed_shape {
        shape_n.push(usize::try_from(*size).unwrap());
    }
    shape_n.push(bytes_per_element);
    let array = ndarray::ArrayViewD::<u8>::from_shape(shape_n, data)?;

    // Transpose the data
    let array_transposed = array.permuted_axes(transpose_order);
    #[allow(deprecated)]
    if array_transposed.is_standard_layout() {
        Ok(array_transposed.to_owned().into_raw_vec())
    } else {
        Ok(array_transposed
            .as_standard_layout()
            .into_owned()
            .into_raw_vec())
    }
}

fn permute<T: Copy>(v: &[T], order: &[usize]) -> Vec<T> {
    let mut vec = Vec::<T>::with_capacity(v.len());
    for axis in order {
        vec.push(v[*axis]);
    }
    vec
}

fn transpose_vlen<'a>(
    bytes: &RawBytes,
    offsets: &RawBytesOffsets,
    shape: &[usize],
    order: Vec<usize>,
) -> ArrayBytes<'a> {
    debug_assert_eq!(shape.len(), order.len());

    // Get the transposed element indices
    let ndarray_indices =
        ndarray::ArrayD::from_shape_vec(shape, (0..shape.iter().product()).collect()).unwrap();
    let ndarray_indices_transposed = ndarray_indices.permuted_axes(order);

    // Collect the new bytes/offsets
    let mut bytes_new = Vec::with_capacity(bytes.len());
    let mut offsets_new = Vec::with_capacity(offsets.len());
    for idx in &ndarray_indices_transposed {
        offsets_new.push(bytes_new.len());
        let curr = offsets[*idx];
        let next = offsets[idx + 1];
        bytes_new.extend_from_slice(&bytes[curr..next]);
    }
    offsets_new.push(bytes_new.len());
    let offsets_new = unsafe {
        // SAFETY: The offsets are monotonically increasing.
        RawBytesOffsets::new_unchecked(offsets_new)
    };
    let array_bytes = unsafe {
        // SAFETY: The last offset is equal to the length of the bytes
        ArrayBytes::new_vlen_unchecked(bytes_new, offsets_new)
    };
    array_bytes
}

#[cfg(test)]
mod tests {
    use std::{num::NonZeroU64, sync::Arc};

    use crate::{
        array::{
            codec::{ArrayToArrayCodecTraits, ArrayToBytesCodecTraits, BytesCodec, CodecOptions},
            ArrayBytes, ChunkRepresentation, DataType, FillValue,
        },
        array_subset::ArraySubset,
    };

    use super::*;

    fn codec_transpose_round_trip_impl(
        json: &str,
        data_type: DataType,
        fill_value: impl Into<FillValue>,
    ) {
        let chunk_representation = ChunkRepresentation::new(
            vec![
                NonZeroU64::new(2).unwrap(),
                NonZeroU64::new(2).unwrap(),
                NonZeroU64::new(3).unwrap(),
            ],
            data_type,
            fill_value,
        )
        .unwrap();
        let size = chunk_representation.num_elements_usize()
            * chunk_representation.data_type().fixed_size().unwrap();
        let bytes: Vec<u8> = (0..size).map(|s| s as u8).collect();
        let bytes: ArrayBytes = bytes.into();

        let configuration: TransposeCodecConfiguration = serde_json::from_str(json).unwrap();
        let codec = TransposeCodec::new_with_configuration(&configuration).unwrap();

        let encoded = codec
            .encode(
                bytes.clone(),
                &chunk_representation,
                &CodecOptions::default(),
            )
            .unwrap();
        let decoded = codec
            .decode(encoded, &chunk_representation, &CodecOptions::default())
            .unwrap();
        assert_eq!(bytes, decoded);

        // let array = ndarray::ArrayViewD::from_shape(array_representation.shape(), &bytes).unwrap();
        // let array_representation_transpose =
        //     ArrayRepresentation::new(vec![2, 3, 2], data_type.clone(), fill_value.clone()).unwrap();
        // let encoded_array = ndarray::ArrayViewD::from_shape(
        //     array_representation_transpose.shape().to_vec(),
        //     &encoded,
        // )
        // .unwrap();
        // let decoded_array =
        //     ndarray::ArrayViewD::from_shape(array_representation.shape(), &decoded).unwrap();
    }

    #[test]
    fn codec_transpose_round_trip_array1() {
        const JSON: &str = r#"{
            "order": [0, 2, 1]
        }"#;
        codec_transpose_round_trip_impl(JSON, DataType::UInt8, 0u8);
    }

    #[test]
    fn codec_transpose_round_trip_array2() {
        const JSON: &str = r#"{
            "order": [2, 1, 0]
        }"#;
        codec_transpose_round_trip_impl(JSON, DataType::UInt16, 0u16);
    }

    #[test]
    fn codec_transpose_partial_decode() {
        let codec = Arc::new(TransposeCodec::new(TransposeOrder::new(&[1, 0]).unwrap()));

        let elements: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let chunk_representation = ChunkRepresentation::new(
            vec![NonZeroU64::new(4).unwrap(), NonZeroU64::new(4).unwrap()],
            DataType::Float32,
            0.0f32,
        )
        .unwrap();
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes: ArrayBytes = bytes.into();

        let encoded = codec
            .encode(bytes, &chunk_representation, &CodecOptions::default())
            .unwrap();
        let decoded_regions = [
            ArraySubset::new_with_ranges(&[0..4, 0..4]),
            ArraySubset::new_with_ranges(&[1..3, 1..4]),
            ArraySubset::new_with_ranges(&[2..4, 0..2]),
        ];
        let input_handle = Arc::new(encoded.into_fixed().unwrap());
        let bytes_codec = Arc::new(BytesCodec::default());
        let input_handle = bytes_codec
            .partial_decoder(
                input_handle,
                &chunk_representation,
                &CodecOptions::default(),
            )
            .unwrap();
        let partial_decoder = codec
            .partial_decoder(
                input_handle.clone(),
                &chunk_representation,
                &CodecOptions::default(),
            )
            .unwrap();
        assert_eq!(partial_decoder.size(), input_handle.size()); // transpose partial decoder does not hold bytes
        let decoded_partial_chunk = partial_decoder
            .partial_decode(&decoded_regions, &CodecOptions::default())
            .unwrap();
        let decoded_partial_chunk = decoded_partial_chunk
            .into_iter()
            .map(|bytes| {
                crate::array::convert_from_bytes_slice::<f32>(&bytes.into_fixed().unwrap())
            })
            .collect::<Vec<_>>();
        let answer: &[Vec<f32>] = &[
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0,
            ],
            vec![5.0, 6.0, 7.0, 9.0, 10.0, 11.0],
            vec![8.0, 9.0, 12.0, 13.0],
        ];
        assert_eq!(answer, decoded_partial_chunk);
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn codec_transpose_async_partial_decode() {
        let codec = Arc::new(TransposeCodec::new(TransposeOrder::new(&[1, 0]).unwrap()));

        let elements: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let chunk_representation = ChunkRepresentation::new(
            vec![NonZeroU64::new(4).unwrap(), NonZeroU64::new(4).unwrap()],
            DataType::Float32,
            0.0f32,
        )
        .unwrap();
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes: ArrayBytes = bytes.into();

        let encoded = codec
            .encode(
                bytes.clone(),
                &chunk_representation,
                &CodecOptions::default(),
            )
            .unwrap();
        let decoded_regions = [
            ArraySubset::new_with_ranges(&[0..4, 0..4]),
            ArraySubset::new_with_ranges(&[1..3, 1..4]),
            ArraySubset::new_with_ranges(&[2..4, 0..2]),
        ];
        let input_handle = Arc::new(encoded.into_fixed().unwrap());
        let bytes_codec = Arc::new(BytesCodec::default());
        let input_handle = bytes_codec
            .async_partial_decoder(
                input_handle,
                &chunk_representation,
                &CodecOptions::default(),
            )
            .await
            .unwrap();
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
        let decoded_partial_chunk = decoded_partial_chunk
            .into_iter()
            .map(|bytes| {
                crate::array::transmute_from_bytes_vec::<f32>(
                    bytes.into_fixed().unwrap().into_owned(),
                )
            })
            .collect::<Vec<_>>();
        let answer: &[Vec<f32>] = &[
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0,
            ],
            vec![5.0, 6.0, 7.0, 9.0, 10.0, 11.0],
            vec![8.0, 9.0, 12.0, 13.0],
        ];
        assert_eq!(answer, decoded_partial_chunk);
    }
}
