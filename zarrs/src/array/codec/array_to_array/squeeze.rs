//! The `squeeze` array to array codec (Experimental).
//!
//! Collapses dimensions with a size of 1.
//!
//! <div class="warning">
//! This codec is experimental and may be incompatible with other Zarr V3 implementations.
//! </div>
//!
//! ### Compatible Implementations
//! None
//!
//! ### Specification
//! - `https://codec.zarrs.dev/array_to_array/squeeze`
//!
//! ### Codec `name` Aliases (Zarr V3)
//! - `zarrs.squeeze`
//!
//! ### Codec `id` Aliases (Zarr V2)
//! None
//!
//! ### Codec `configuration` Example - [`SqueezeCodecConfiguration`]:
//! ```rust
//! # let JSON = r#"
//! {}
//! # "#;
//! # use zarrs::metadata_ext::codec::squeeze::SqueezeCodecConfiguration;
//! # let configuration: SqueezeCodecConfiguration = serde_json::from_str(JSON).unwrap();
//! ```

mod squeeze_codec;
mod squeeze_codec_partial;

use std::sync::Arc;

use itertools::{Itertools, izip};
pub use squeeze_codec::SqueezeCodec;
use zarrs_metadata::v3::MetadataV3;

use crate::array::{ArrayIndices, ArraySubset, ArraySubsetTraits, Indexer, IndexerError};
use zarrs_codec::{Codec, CodecError, CodecPluginV3, CodecTraitsV3};
pub use zarrs_metadata_ext::codec::squeeze::{
    SqueezeCodecConfiguration, SqueezeCodecConfigurationV0,
};

zarrs_plugin::impl_extension_aliases!(SqueezeCodec,
  v3: "zarrs.squeeze", []
);

// Register the V3 codec.
inventory::submit! {
    CodecPluginV3::new::<SqueezeCodec>()
}

impl CodecTraitsV3 for SqueezeCodec {
    fn create(metadata: &MetadataV3) -> Result<Codec, zarrs_codec::CodecCreateError> {
        crate::warn_experimental_extension(metadata.name(), "codec");
        let configuration: SqueezeCodecConfiguration = metadata.to_typed_configuration()?;
        let codec = Arc::new(SqueezeCodec::new_with_configuration(&configuration)?);
        Ok(Codec::ArrayToArray(codec))
    }
}

fn get_squeezed_array_subset(
    decoded_region: &dyn ArraySubsetTraits,
    shape: &[u64],
) -> Result<ArraySubset, CodecError> {
    if decoded_region.dimensionality() != shape.len() {
        return Err(IndexerError::new_incompatible_dimensionality(
            decoded_region.dimensionality(),
            shape.len(),
        )
        .into());
    }

    let decoded_region_start = decoded_region.start();
    let decoded_region_shape = decoded_region.shape();
    let ranges = izip!(
        decoded_region_start.iter(),
        decoded_region_shape.iter(),
        shape.iter()
    )
    .filter(|&(_, _, shape)| *shape > 1)
    .map(|(rstart, rshape, _)| *rstart..rstart + rshape);

    let decoded_region_squeeze = ArraySubset::from(ranges);
    Ok(decoded_region_squeeze)
}

fn get_squeezed_indexer(indexer: &dyn Indexer, shape: &[u64]) -> Result<impl Indexer, CodecError> {
    let indices = indexer
        .iter_indices()
        .map(|indices| {
            if indices.len() == shape.len() {
                Ok(indices
                    .into_iter()
                    .zip(shape)
                    .filter_map(|(indices, &shape)| if shape > 1 { Some(indices) } else { None })
                    .collect_vec())
            } else {
                Err(IndexerError::new_incompatible_dimensionality(
                    indices.len(),
                    shape.len(),
                ))
            }
        })
        .collect::<Result<Vec<ArrayIndices>, _>>()?;

    Ok(indices)
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;
    use std::sync::Arc;

    use super::*;
    use crate::array::chunk_grid::RegularChunkGrid;
    use crate::array::codec::BytesCodec;
    use crate::array::{ArrayBytes, ArraySubset, DataType, FillValue, data_type};
    use zarrs_chunk_grid::{ChunkGrid, ChunkShapeTraits};
    use zarrs_codec::{
        CodecOptions, SubchunkGrid, UnboundArrayToArrayCodecTraits, UnboundArrayToBytesCodecTraits,
    };
    use zarrs_metadata::{ArrayShape, ChunkShapeNonEmpty};

    const fn nz(value: u64) -> NonZeroU64 {
        NonZeroU64::new(value).unwrap()
    }

    fn codec_squeeze_round_trip_impl(
        json: &str,
        data_type: DataType,
        fill_value: impl Into<FillValue>,
    ) {
        let shape = vec![1, 2, 1, 2, 3];
        let fill_value = fill_value.into();
        let size = shape.num_elements_usize() * data_type.fixed_size().unwrap();
        let bytes: Vec<u8> = (0..size).map(|s| s as u8).collect();
        let bytes: ArrayBytes = bytes.into();

        let configuration: SqueezeCodecConfiguration = serde_json::from_str(json).unwrap();
        let codec = Arc::new(SqueezeCodec::new_with_configuration(&configuration).unwrap())
            .with_context(data_type.clone(), fill_value.clone())
            .unwrap();
        assert_eq!(codec.encoded_shape(&shape).unwrap(), vec![2, 2, 3,]);

        let encoded = codec
            .encode(bytes.clone(), &shape, &CodecOptions::default())
            .unwrap();
        let decoded = codec
            .decode(encoded, &shape, &CodecOptions::default())
            .unwrap();
        assert_eq!(bytes, decoded);
    }

    #[test]
    fn codec_squeeze_round_trip_array1() {
        const JSON: &str = r"{}";
        codec_squeeze_round_trip_impl(JSON, data_type::uint8(), 0u8);
    }

    fn test_squeeze_partial_decode_granularity(
        array_shape: ArrayShape,
        chunk_shape: ChunkShapeNonEmpty,
        inner_array_shape: ArrayShape,
        inner_subchunk_shape: ChunkShapeNonEmpty,
        expected_subchunk_grid_edge_lengths: Vec<Vec<NonZeroU64>>,
    ) {
        let codec = Arc::new(SqueezeCodec::new())
            .with_context(data_type::uint8(), FillValue::from(0u8))
            .unwrap();
        let chunk_grid =
            ChunkGrid::new(RegularChunkGrid::new(array_shape.clone(), chunk_shape).unwrap());
        let inner_subchunk_grid =
            ChunkGrid::new(RegularChunkGrid::new(inner_array_shape, inner_subchunk_shape).unwrap());
        let subchunk_grid = codec
            .decoded_subchunk_grid(&chunk_grid, &inner_subchunk_grid)
            .unwrap();
        let SubchunkGrid::Array(subchunk_grid) = subchunk_grid else {
            panic!("expected array subchunk grid");
        };
        for (axis, expected_subchunk_grid_edge_lengths_axis) in
            expected_subchunk_grid_edge_lengths.into_iter().enumerate()
        {
            assert_eq!(
                subchunk_grid.chunk_edge_lengths(axis).unwrap(),
                expected_subchunk_grid_edge_lengths_axis
            );
        }
    }

    #[test]
    fn codec_squeeze_partial_decode_granularity() {
        test_squeeze_partial_decode_granularity(
            vec![1, 10],
            vec![nz(1), nz(5)],
            vec![10],
            vec![nz(5)],
            vec![vec![nz(1)], vec![nz(5), nz(5)]],
        );

        test_squeeze_partial_decode_granularity(
            vec![4, 2, 20],
            vec![nz(2), nz(1), nz(10)],
            vec![4, 20],
            vec![nz(2), nz(10)],
            vec![vec![nz(2), nz(2)], vec![nz(1), nz(1)], vec![nz(10), nz(10)]],
        );
    }

    #[test]
    fn codec_squeeze_partial_decode() {
        let codec = Arc::new(SqueezeCodec::new());

        let elements: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let shape = vec![1, 4, 1, 4, 1];
        let data_type = data_type::float32();
        let fill_value = FillValue::from(0.0f32);
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes: ArrayBytes = bytes.into();

        let codec = codec
            .with_context(data_type.clone(), fill_value.clone())
            .unwrap();
        let encoded = codec
            .encode(bytes, &shape, &CodecOptions::default())
            .unwrap();
        let input_handle = Arc::new(encoded.into_fixed().unwrap());
        let bytes_codec = Arc::new(BytesCodec::default());
        let encoded_shape = codec.encoded_shape(&shape).unwrap();
        let encoded_data_type = codec.encoded_data_type().clone();
        let encoded_fill_value = codec.encoded_fill_value().clone();
        let bytes_codec = bytes_codec
            .with_context(encoded_data_type.clone(), encoded_fill_value.clone())
            .unwrap();
        let input_handle = bytes_codec
            .partial_decoder(input_handle, &encoded_shape, &CodecOptions::default())
            .unwrap();
        let partial_decoder = codec
            .partial_decoder(input_handle.clone(), &shape, &CodecOptions::default())
            .unwrap();
        assert_eq!(partial_decoder.size_held(), input_handle.size_held()); // squeeze partial decoder does not hold bytes

        let decoded_regions = [
            ArraySubset::new_with_ranges(&[0..1, 0..4, 0..1, 0..4, 0..1]),
            ArraySubset::new_with_ranges(&[0..1, 1..3, 0..1, 1..4, 0..1]),
            ArraySubset::new_with_ranges(&[0..1, 2..4, 0..1, 0..2, 0..1]),
        ];

        for (decoded_region, expected) in decoded_regions.into_iter().zip([
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0,
            ],
            vec![5.0, 6.0, 7.0, 9.0, 10.0, 11.0],
            vec![8.0, 9.0, 12.0, 13.0],
        ]) {
            let decoded_partial_chunk = partial_decoder
                .partial_decode(&decoded_region, &CodecOptions::default())
                .unwrap();
            let decoded_partial_chunk = crate::array::convert_from_bytes_slice::<f32>(
                &decoded_partial_chunk.into_fixed().unwrap(),
            );
            assert_eq!(decoded_partial_chunk, expected);
        }
    }
}
