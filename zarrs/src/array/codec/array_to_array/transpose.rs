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
//! # use zarrs::metadata_ext::codec::transpose::TransposeCodecConfiguration;
//! # let configuration: TransposeCodecConfiguration = serde_json::from_str(JSON).unwrap();
//! ```

mod transpose_codec;
mod transpose_codec_partial;

use std::sync::Arc;

pub use transpose_codec::TransposeCodec;
use zarrs_metadata::v3::MetadataV3;
use zarrs_plugin::ExtensionAliasesV3;

use crate::array::array_bytes::{ArrayBytesOffsets, ArrayBytesVariableLength};
use crate::array::codec::{Codec, CodecError, CodecPluginV3};
use crate::array::{
    ArrayBytes, ArrayBytesRaw, ArraySubset, ArraySubsetTraits, DataType, Indexer, IndexerError,
};
use crate::metadata::DataTypeSize;
pub use crate::metadata_ext::codec::transpose::{
    TransposeCodecConfiguration, TransposeCodecConfigurationV1, TransposeOrder, TransposeOrderError,
};
use crate::plugin::{PluginConfigurationInvalidError, PluginCreateError};

zarrs_plugin::impl_extension_aliases!(TransposeCodec, v3: "transpose");

// Register the V3 codec.
inventory::submit! {
    CodecPluginV3::new::<TransposeCodec>(create_codec_transpose_v3)
}

pub(crate) fn create_codec_transpose_v3(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
    let configuration: TransposeCodecConfiguration = metadata
        .to_configuration()
        .map_err(|_| PluginConfigurationInvalidError::new(metadata.to_string()))?;
    let codec = Arc::new(TransposeCodec::new_with_configuration(&configuration)?);
    Ok(Codec::ArrayToArray(codec))
}

/// Compute the inverse permutation order.
///
/// For a permutation `p`, returns the inverse permutation `p_inv` such that
/// `p_inv[p[i]] = i` for all `i`.
pub(crate) fn inverse_permutation(order: &[usize]) -> Vec<usize> {
    let mut inverse = vec![0; order.len()];
    for (i, &val) in order.iter().enumerate() {
        inverse[val] = i;
    }
    inverse
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
    if array_transposed.is_standard_layout() {
        Ok(array_transposed.to_owned().into_raw_vec_and_offset().0)
    } else {
        Ok(array_transposed
            .as_standard_layout()
            .into_owned()
            .into_raw_vec_and_offset()
            .0)
    }
}

fn permute<T: Copy>(v: &[T], order: &[usize]) -> Option<Vec<T>> {
    if v.len() == order.len() {
        let mut vec = Vec::<T>::with_capacity(v.len());
        for axis in order {
            vec.push(v[*axis]);
        }
        Some(vec)
    } else {
        None
    }
}

fn transpose_vlen<'a>(
    bytes: &ArrayBytesRaw,
    offsets: &ArrayBytesOffsets,
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
        ArrayBytesOffsets::new_unchecked(offsets_new)
    };
    unsafe {
        // SAFETY: The last offset is equal to the length of the bytes
        ArrayBytes::new_vlen_unchecked(bytes_new, offsets_new)
    }
}

fn get_transposed_array_subset(
    order: &[usize],
    decoded_region: &dyn ArraySubsetTraits,
) -> Result<ArraySubset, CodecError> {
    if decoded_region.dimensionality() != order.len() {
        return Err(IndexerError::new_incompatible_dimensionality(
            decoded_region.dimensionality(),
            order.len(),
        )
        .into());
    }

    let start = permute(&decoded_region.start(), order).expect("matching dimensionality");
    let size = permute(&decoded_region.shape(), order).expect("matching dimensionality");
    let ranges = start.iter().zip(size).map(|(&st, si)| st..(st + si));
    Ok(ArraySubset::from(ranges))
}

fn get_transposed_indexer(
    order: &[usize],
    indexer: &dyn Indexer,
) -> Result<impl Indexer, CodecError> {
    indexer
        .iter_indices()
        .map(|indices| permute(&indices, order))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            IndexerError::new_incompatible_dimensionality(indexer.dimensionality(), order.len())
                .into()
        })
}

/// Apply a transpose permutation to array bytes.
///
/// # Arguments
/// * `bytes` - The input array bytes to transpose
/// * `input_shape` - The shape of the input array
/// * `permutation` - The permutation order to apply
/// * `data_type` - The data type of the array elements
///
/// The output shape will be `permute(input_shape, permutation)`.
pub(crate) fn apply_permutation<'a>(
    bytes: &ArrayBytes<'a>,
    input_shape: &[u64],
    permutation: &[usize],
    data_type: &DataType,
) -> Result<ArrayBytes<'a>, CodecError> {
    if input_shape.len() != permutation.len() {
        return Err(IndexerError::new_incompatible_dimensionality(
            input_shape.len(),
            permutation.len(),
        )
        .into());
    }

    let num_elements = input_shape.iter().product();
    bytes.validate(num_elements, data_type)?;

    match (bytes, data_type.size()) {
        (
            ArrayBytes::Variable(ArrayBytesVariableLength { bytes, offsets }),
            DataTypeSize::Variable,
        ) => {
            let shape: Vec<usize> = input_shape
                .iter()
                .map(|s| usize::try_from(*s).unwrap())
                .collect();
            Ok(transpose_vlen(bytes, offsets, &shape, permutation.to_vec()))
        }
        (ArrayBytes::Fixed(bytes), DataTypeSize::Fixed(data_type_size)) => {
            // For fixed-size types, add an extra dimension for the element bytes
            let mut order_with_bytes = permutation.to_vec();
            order_with_bytes.push(permutation.len());
            let bytes = transpose_array(&order_with_bytes, input_shape, data_type_size, bytes)
                .map_err(|_| CodecError::Other("transpose_array error".to_string()))?;
            Ok(ArrayBytes::from(bytes))
        }
        (ArrayBytes::Optional(..), _) => Err(CodecError::UnsupportedDataType(
            data_type.clone(),
            TransposeCodec::aliases_v3().default_name.to_string(),
        )),
        (_, _) => Err(CodecError::Other(
            "dev error: transpose data type mismatch".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;
    use std::sync::Arc;

    use super::*;
    use crate::array::codec::{
        ArrayToArrayCodecTraits, ArrayToBytesCodecTraits, BytesCodec, CodecOptions,
    };
    use crate::array::data_type::DataTypeExt;
    use crate::array::{ArrayBytes, ArraySubset, ChunkShapeTraits, DataType, FillValue, data_type};

    fn codec_transpose_round_trip_impl(
        json: &str,
        data_type: DataType,
        fill_value: impl Into<FillValue>,
    ) {
        let shape = vec![
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(3).unwrap(),
        ];
        let fill_value = fill_value.into();
        let size = shape.num_elements_usize() * data_type.fixed_size().unwrap();
        let bytes: Vec<u8> = (0..size).map(|s| s as u8).collect();
        let bytes: ArrayBytes = bytes.into();

        let configuration: TransposeCodecConfiguration = serde_json::from_str(json).unwrap();
        let codec = TransposeCodec::new_with_configuration(&configuration).unwrap();

        let encoded = codec
            .encode(
                bytes.clone(),
                &shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
            .unwrap();
        let decoded = codec
            .decode(
                encoded,
                &shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
            .unwrap();
        assert_eq!(bytes, decoded);
    }

    #[test]
    fn codec_transpose_round_trip_array1() {
        const JSON: &str = r#"{
            "order": [0, 2, 1]
        }"#;
        codec_transpose_round_trip_impl(JSON, data_type::uint8(), 0u8);
    }

    #[test]
    fn codec_transpose_round_trip_array2() {
        const JSON: &str = r#"{
            "order": [2, 1, 0]
        }"#;
        codec_transpose_round_trip_impl(JSON, data_type::uint16(), 0u16);
    }

    #[test]
    fn codec_transpose_round_trip_vlen_string() {
        use crate::array::Element;

        // Create a 2x3 array of strings
        let shape = vec![NonZeroU64::new(2).unwrap(), NonZeroU64::new(3).unwrap()];
        let data_type = data_type::string();
        let fill_value = FillValue::from("");

        // Create test data: 6 strings in row-major order
        let strings: Vec<&str> = vec!["s00", "s01a", "s02ab", "s10abc", "s11abcd", "s12abcde"];
        let bytes = Element::into_array_bytes(&data_type::string(), strings).unwrap();

        // Create transpose codec with order [1, 0] (swap axes)
        let codec = TransposeCodec::new(TransposeOrder::new(&[1, 0]).unwrap());

        let encoded = codec
            .encode(
                bytes.clone(),
                &shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
            .unwrap();
        let decoded = codec
            .decode(
                encoded,
                &shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
            .unwrap();

        assert_eq!(bytes, decoded);
    }

    #[test]
    fn apply_permutation_vlen_string() {
        use crate::array::Element;

        // Test apply_permutation with vlen data (used by partial encode/decode)
        // This tests a non-square shape to catch shape mismatch bugs
        // Original shape: 2x3, Transposed shape: 3x2
        let order = TransposeOrder::new(&[1, 0]).unwrap();

        // Create test data: 6 strings in row-major order for shape [2, 3]
        // [[s00, s01, s02], [s10, s11, s12]]
        let strings: Vec<&str> = vec!["s00", "s01a", "s02ab", "s10abc", "s11abcd", "s12abcde"];
        let original = Element::into_array_bytes(&data_type::string(), strings).unwrap();

        // Encode: apply transpose order [1, 0] to get shape [3, 2]
        // Transposed should be: [[s00, s10], [s01, s11], [s02, s12]]
        let transposed_strings: Vec<&str> =
            vec!["s00", "s10abc", "s01a", "s11abcd", "s02ab", "s12abcde"];
        let expected_transposed =
            Element::into_array_bytes(&data_type::string(), transposed_strings).unwrap();

        // Test encoding (forward permutation)
        let encoded =
            apply_permutation(&original, &[2, 3], &order.0, &data_type::string()).unwrap();
        assert_eq!(encoded, expected_transposed);

        // Test decoding (inverse permutation)
        // Inverse of [1, 0] is [1, 0]
        let order_decode = [1, 0];
        let decoded =
            apply_permutation(&encoded, &[3, 2], &order_decode, &data_type::string()).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn codec_transpose_partial_decode() {
        let codec = Arc::new(TransposeCodec::new(TransposeOrder::new(&[1, 0]).unwrap()));

        let elements: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let shape = vec![NonZeroU64::new(4).unwrap(), NonZeroU64::new(4).unwrap()];
        let data_type = data_type::float32();
        let fill_value = FillValue::from(0.0f32);
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes: ArrayBytes = bytes.into();

        let encoded = codec
            .encode(
                bytes,
                &shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
            .unwrap();
        let input_handle = Arc::new(encoded.into_fixed().unwrap());
        let bytes_codec = Arc::new(BytesCodec::default());
        let input_handle = bytes_codec
            .partial_decoder(
                input_handle,
                &shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
            .unwrap();
        let partial_decoder = codec
            .partial_decoder(
                input_handle.clone(),
                &shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
            .unwrap();
        assert_eq!(partial_decoder.size_held(), input_handle.size_held()); // transpose partial decoder does not hold bytes
        let decoded_regions = [
            ArraySubset::new_with_ranges(&[0..4, 0..4]),
            ArraySubset::new_with_ranges(&[1..3, 1..4]),
            ArraySubset::new_with_ranges(&[2..4, 0..2]),
        ];
        let answer: &[Vec<f32>] = &[
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0,
            ],
            vec![5.0, 6.0, 7.0, 9.0, 10.0, 11.0],
            vec![8.0, 9.0, 12.0, 13.0],
        ];
        for (decoded_region, expected) in decoded_regions.into_iter().zip(answer.iter()) {
            let decoded_partial_chunk = partial_decoder
                .partial_decode(&decoded_region, &CodecOptions::default())
                .unwrap();
            let decoded_partial_chunk = crate::array::convert_from_bytes_slice::<f32>(
                &decoded_partial_chunk.into_fixed().unwrap(),
            );
            assert_eq!(expected, &decoded_partial_chunk);
        }
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn codec_transpose_async_partial_decode() {
        let codec = Arc::new(TransposeCodec::new(TransposeOrder::new(&[1, 0]).unwrap()));

        let elements: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let shape = vec![NonZeroU64::new(4).unwrap(), NonZeroU64::new(4).unwrap()];
        let data_type = data_type::float32();
        let fill_value = FillValue::from(0.0f32);
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes: ArrayBytes = bytes.into();

        let encoded = codec
            .encode(
                bytes.clone(),
                &shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
            .unwrap();
        let input_handle = Arc::new(encoded.into_fixed().unwrap());
        let bytes_codec = Arc::new(BytesCodec::default());
        let input_handle = bytes_codec
            .async_partial_decoder(
                input_handle,
                &shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
            .await
            .unwrap();
        let partial_decoder = codec
            .async_partial_decoder(
                input_handle,
                &shape,
                &data_type,
                &fill_value,
                &CodecOptions::default(),
            )
            .await
            .unwrap();
        let decoded_regions = [
            ArraySubset::new_with_ranges(&[0..4, 0..4]),
            ArraySubset::new_with_ranges(&[1..3, 1..4]),
            ArraySubset::new_with_ranges(&[2..4, 0..2]),
        ];
        let answer: &[Vec<f32>] = &[
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0,
            ],
            vec![5.0, 6.0, 7.0, 9.0, 10.0, 11.0],
            vec![8.0, 9.0, 12.0, 13.0],
        ];
        for (decoded_region, answer) in decoded_regions.into_iter().zip(answer.iter()) {
            let decoded_partial_chunk = partial_decoder
                .partial_decode(&decoded_region, &CodecOptions::default())
                .await
                .unwrap();
            let decoded_partial_chunk = crate::array::convert_from_bytes_slice::<f32>(
                &decoded_partial_chunk.into_fixed().unwrap(),
            );
            assert_eq!(answer, &decoded_partial_chunk);
        }
    }
}
