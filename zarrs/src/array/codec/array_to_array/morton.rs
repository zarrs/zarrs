//! The `zarrs.morton` array to array codec (Experimental).
//!
//! Stores each chunk as a dense 1D array in anisotropic Morton order.
//!
//! ## Abstract
//!
//! The `zarrs.morton` codec is an `array -> array` codec that reorders each
//! decoded chunk from its decoded multidimensional C-order layout into a
//! one-dimensional encoded array whose elements are sorted by an anisotropic
//! Morton key. Decoding applies the inverse ordering.
//!
//! This codec does not compress or transform element values. It only changes
//! element order within the chunk.
//!
//! <div class="warning">
//! This codec is experimental and may be incompatible with other Zarr V3 implementations.
//! </div>
//!
//! ## Status of this document
//!
//! This is an experimental `zarrs` codec specification. It is not registered in
//! `zarr-extensions`.
//!
//! ## Document conventions
//!
//! Conformance requirements are expressed using RFC 2119 terminology. The words
//! "MUST", "MUST NOT", "SHOULD", "SHOULD NOT", and "MAY" are to be interpreted
//! as described in RFC 2119.
//!
//! ### Compatible Implementations
//! None
//!
//! ### Specification
//! - `https://codec.zarrs.dev/array_to_array/morton`
//!
//! ### Codec `name` Aliases (Zarr V3)
//! - `zarrs.morton`
//!
//! ### Codec `id` Aliases (Zarr V2)
//! None
//!
//! ### Codec `configuration` Example - [`MortonCodecConfiguration`]:
//! ```rust
//! # let JSON = r#"
//! {}
//! # "#;
//! # use zarrs::metadata_ext::codec::morton::MortonCodecConfiguration;
//! # let configuration: MortonCodecConfiguration = serde_json::from_str(JSON).unwrap();
//! ```
//!
//! ## Configuration
//!
//! The codec configuration MUST be an empty JSON object:
//!
//! ```json
//! {}
//! ```
//!
//! The JSON schema is:
//!
//! ```json
//! {
//!   "type": "object",
//!   "additionalProperties": false
//! }
//! ```
//!
//! ## Encoded Representation
//!
//! Let `D = [d0, d1, ..., dN-1]` be the decoded chunk shape. Each `di` MUST be
//! positive. The encoded chunk shape is:
//!
//! ```text
//! [product(D)]
//! ```
//!
//! The encoded data type and fill value are identical to the decoded data type
//! and fill value.
//!
//! ## Morton Key
//!
//! For each axis `i`, define:
//!
//! ```text
//! padded_i = next_power_of_two(di)
//! width_i  = log2(padded_i)
//! ```
//!
//! Equivalently, `width_i` is the number of bits required to address positions
//! in the padded extent of axis `i`.
//!
//! For a decoded coordinate `c = [c0, c1, ..., cN-1]`, where `0 <= ci < di`,
//! the Morton key is the bit string formed by scanning bit planes from most
//! significant to least significant. Within each bit plane, axes are considered
//! in original axis order. An axis contributes a bit at bit plane `b` only if
//! `width_i > b`.
//!
//! In pseudocode:
//!
//! ```text
//! key = []
//! for b in reverse(0 .. max(width_i)):
//!     for i in 0 .. N:
//!         if width_i > b:
//!             key.push(bit b of ci)
//! ```
//!
//! The padded domain is used only to define the bit-plane order. Coordinates
//! outside the decoded shape MUST be skipped and MUST NOT consume encoded
//! elements. Therefore, the encoded array contains exactly `product(D)` elements.
//!
//! ## Encoding
//!
//! To encode a decoded chunk:
//!
//! 1. Compute the Morton key for every valid decoded coordinate.
//! 2. Sort valid decoded coordinates by Morton key.
//! 3. Write decoded elements to the one-dimensional encoded array in that sorted
//!    order.
//!
//! ## Decoding
//!
//! To decode an encoded chunk:
//!
//! 1. Compute the same sorted valid coordinate order from the decoded chunk
//!    shape.
//! 2. Read the one-dimensional encoded array in order.
//! 3. Place each encoded element at the corresponding decoded coordinate.
//!
//! ## Examples
//!
//! For decoded shape `[2, 4, 8]`, axis bit widths are `[1, 2, 3]`. The bit-plane
//! sequence is:
//!
//! ```text
//! z2, y1, z1, x0, y0, z0
//! ```
//!
//! For decoded shape `[8, 2, 4]`, axis bit widths are `[3, 1, 2]`. The bit-plane
//! sequence is:
//!
//! ```text
//! x2, x1, z1, x0, y0, z0
//! ```
//!
//! For decoded shape `[3, 3]`, the padded shape is `[4, 4]`. If decoded elements
//! are labelled by their C-order linear index:
//!
//! ```text
//! decoded:
//!   0  1  2
//!   3  4  5
//!   6  7  8
//!
//! padded Morton order, with invalid padded coordinates shown as xx:
//!   0, 1, 3, 4, 2, xx, 5, xx, 6, 7, xx, xx, 8, xx, xx, xx
//!
//! compact encoded order:
//!   0, 1, 3, 4, 2, 5, 6, 7, 8
//! ```
//!
//! ## Partial Decoding and Encoding
//!
//! For a decoded subset, implementations SHOULD map the requested decoded
//! coordinates to their one-dimensional encoded positions and coalesce adjacent
//! encoded positions into runs. Implementations SHOULD request exactly those
//! encoded runs from the downstream partial decoder or encoder.
//!
//! Partial decoding MUST return elements in the output order specified by the
//! decoded indexer. Partial encoding MUST interpret input elements in the output
//! order specified by the decoded indexer before writing them to encoded Morton
//! positions.
//!
//! ## Interoperability and Compatibility
//!
//! This codec uses the experimental `zarrs.morton` name and has no Zarr V2
//! alias. Implementations that do not recognize this codec name will be unable
//! to decode arrays that use it.

mod morton_codec;

use std::sync::Arc;

pub use morton_codec::MortonCodec;
use zarrs_metadata::v3::MetadataV3;

use crate::array::{ArrayIndices, ArraySubset, Indexer, IndexerError};
use zarrs_codec::{Codec, CodecError, CodecPluginV3, CodecTraitsV3};
pub use zarrs_metadata_ext::codec::morton::{MortonCodecConfiguration, MortonCodecConfigurationV0};
use zarrs_plugin::PluginCreateError;

zarrs_plugin::impl_extension_aliases!(MortonCodec,
  v3: "zarrs.morton", []
);

// Register the V3 codec.
inventory::submit! {
    CodecPluginV3::new::<MortonCodec>()
}

impl CodecTraitsV3 for MortonCodec {
    fn create(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
        crate::warn_experimental_extension(metadata.name(), "codec");
        let configuration: MortonCodecConfiguration = metadata.to_typed_configuration()?;
        let codec = Arc::new(MortonCodec::new_with_configuration(&configuration)?);
        Ok(Codec::ArrayToArray(codec))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct MortonElement {
    morton_key: Vec<u8>,
    decoded_linear_index: u64,
    decoded_indices: ArrayIndices,
}

type MortonPartialSelection = (Vec<ArraySubset>, Vec<u64>, Vec<u64>);

fn shape_u64(shape: &[std::num::NonZeroU64]) -> Vec<u64> {
    shape.iter().map(|dim| dim.get()).collect()
}

fn morton_bit_widths(shape: &[u64]) -> Vec<u32> {
    shape
        .iter()
        .map(|&dim| {
            debug_assert!(dim > 0);
            dim.next_power_of_two().trailing_zeros()
        })
        .collect()
}

fn morton_key(indices: &[u64], bit_widths: &[u32]) -> Vec<u8> {
    debug_assert_eq!(indices.len(), bit_widths.len());
    let max_bits = bit_widths.iter().copied().max().unwrap_or(0);
    let mut key = Vec::with_capacity(bit_widths.iter().map(|width| *width as usize).sum());
    for bit in (0..max_bits).rev() {
        for (index, width) in indices.iter().zip(bit_widths) {
            if *width > bit {
                key.push(((index >> bit) & 1) as u8);
            }
        }
    }
    key
}

fn morton_order(shape: &[u64]) -> Result<Vec<MortonElement>, CodecError> {
    let num_elements = shape
        .iter()
        .try_fold(1u64, |product, dim| product.checked_mul(*dim))
        .ok_or_else(|| CodecError::Other("morton codec shape product overflow".to_string()))?;
    let num_elements_usize = usize::try_from(num_elements).map_err(|_| {
        CodecError::Other("morton codec shape product exceeds usize::MAX".to_string())
    })?;
    let bit_widths = morton_bit_widths(shape);
    let subset = ArraySubset::new_with_shape(shape.to_vec());
    let mut order = Vec::with_capacity(num_elements_usize);
    for (decoded_linear_index, decoded_indices) in
        subset.iter_indices().enumerate().map(|(linear, indices)| {
            (
                u64::try_from(linear).unwrap(),
                indices.into_iter().collect::<ArrayIndices>(),
            )
        })
    {
        order.push(MortonElement {
            morton_key: morton_key(&decoded_indices, &bit_widths),
            decoded_linear_index,
            decoded_indices,
        });
    }
    order.sort_by(|left, right| {
        left.morton_key
            .cmp(&right.morton_key)
            .then_with(|| left.decoded_indices.cmp(&right.decoded_indices))
    });
    Ok(order)
}

fn encoded_position_by_decoded_linear_index(shape: &[u64]) -> Result<Vec<u64>, CodecError> {
    let order = morton_order(shape)?;
    let mut positions = vec![0; order.len()];
    for (encoded_position, element) in order.iter().enumerate() {
        let decoded_linear_index = usize::try_from(element.decoded_linear_index).unwrap();
        positions[decoded_linear_index] = u64::try_from(encoded_position).unwrap();
    }
    Ok(positions)
}

fn one_dimensional_indices(indices: impl IntoIterator<Item = u64>) -> Vec<ArrayIndices> {
    indices.into_iter().map(|index| vec![index]).collect()
}

fn coalesced_one_dimensional_runs(positions: impl IntoIterator<Item = u64>) -> Vec<ArraySubset> {
    let mut positions = positions.into_iter();
    let Some(first) = positions.next() else {
        return vec![ArraySubset::new_empty(1)];
    };
    let mut runs = Vec::new();
    let mut start = first;
    let mut end = first + 1;
    for position in positions {
        if position == end {
            end += 1;
        } else {
            runs.push(
                ArraySubset::new_with_start_shape(vec![start], vec![end - start])
                    .expect("matching dimensionality"),
            );
            start = position;
            end = position + 1;
        }
    }
    runs.push(
        ArraySubset::new_with_start_shape(vec![start], vec![end - start])
            .expect("matching dimensionality"),
    );
    runs
}

fn encoded_runs_for_indexer(
    indexer: &dyn Indexer,
    decoded_shape: &[u64],
    encoded_position_by_decoded_linear_index: &[u64],
) -> Result<MortonPartialSelection, CodecError> {
    if indexer.dimensionality() != decoded_shape.len() {
        return Err(IndexerError::new_incompatible_dimensionality(
            indexer.dimensionality(),
            decoded_shape.len(),
        )
        .into());
    }

    let mut encoded_positions_with_output_index = indexer
        .iter_linearised_indices(decoded_shape)?
        .enumerate()
        .map(|(output_index, decoded_linear_index)| {
            let decoded_linear_index = usize::try_from(decoded_linear_index).unwrap();
            (
                encoded_position_by_decoded_linear_index[decoded_linear_index],
                u64::try_from(output_index).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    encoded_positions_with_output_index
        .sort_unstable_by_key(|(encoded_position, _)| *encoded_position);

    let encoded_positions_sorted = encoded_positions_with_output_index
        .iter()
        .map(|(encoded_position, _)| *encoded_position)
        .collect::<Vec<_>>();
    let decoded_output_indices_in_encoded_order = encoded_positions_with_output_index
        .iter()
        .map(|(_, output_index)| *output_index)
        .collect::<Vec<_>>();
    let runs = coalesced_one_dimensional_runs(encoded_positions_sorted.iter().copied());

    Ok((
        runs,
        encoded_positions_sorted,
        decoded_output_indices_in_encoded_order,
    ))
}

fn partial_positions_for_decoded_order(
    encoded_positions_sorted: &[u64],
    decoded_output_indices_in_encoded_order: &[u64],
) -> Vec<u64> {
    let mut partial_position_by_output_index = vec![0; encoded_positions_sorted.len()];
    for (partial_position, output_index) in decoded_output_indices_in_encoded_order
        .iter()
        .copied()
        .enumerate()
    {
        partial_position_by_output_index[usize::try_from(output_index).unwrap()] =
            u64::try_from(partial_position).unwrap();
    }
    partial_position_by_output_index
}

fn reorder_morton_to_decoded(
    bytes: &crate::array::ArrayBytes<'_>,
    decoded_shape: &[u64],
    data_type: &crate::array::DataType,
) -> Result<crate::array::ArrayBytes<'static>, CodecError> {
    let encoded_position_by_decoded_linear_index =
        encoded_position_by_decoded_linear_index(decoded_shape)?;
    let indices = one_dimensional_indices(encoded_position_by_decoded_linear_index);
    let encoded_shape = [u64::try_from(indices.len()).unwrap()];
    Ok(bytes
        .extract_array_subset(&indices, &encoded_shape, data_type)?
        .into_owned())
}

fn reorder_decoded_to_morton(
    bytes: &crate::array::ArrayBytes<'_>,
    decoded_shape: &[u64],
    data_type: &crate::array::DataType,
) -> Result<crate::array::ArrayBytes<'static>, CodecError> {
    let indices = morton_order(decoded_shape)?
        .into_iter()
        .map(|element| element.decoded_indices)
        .collect::<Vec<_>>();
    Ok(bytes
        .extract_array_subset(&indices, decoded_shape, data_type)?
        .into_owned())
}

#[cfg(test)]
mod tests;
