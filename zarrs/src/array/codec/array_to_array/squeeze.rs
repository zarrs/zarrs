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
//! - `zarrs.squeeze`
//!
//! ### Codec `configuration` Example - [`SqueezeCodecConfiguration`]:
//! ```rust
//! # let JSON = r#"
//! {}
//! # "#;
//! # use zarrs_metadata_ext::codec::squeeze::SqueezeCodecConfiguration;
//! # let configuration: SqueezeCodecConfiguration = serde_json::from_str(JSON).unwrap();
//! ```

mod squeeze_codec;
mod squeeze_codec_partial;

use std::{num::NonZeroU64, sync::Arc};

use itertools::{izip, Itertools};
pub use squeeze_codec::SqueezeCodec;
pub use zarrs_metadata_ext::codec::squeeze::{
    SqueezeCodecConfiguration, SqueezeCodecConfigurationV0,
};
use zarrs_registry::codec::SQUEEZE;

use crate::{
    array::{
        codec::{Codec, CodecError, CodecPlugin},
        ArrayIndices,
    },
    array_subset::ArraySubset,
    indexer::{IncompatibleIndexerError, Indexer},
    metadata::v3::MetadataV3,
    plugin::{PluginCreateError, PluginMetadataInvalidError},
};

// Register the codec.
inventory::submit! {
    CodecPlugin::new(SQUEEZE, is_identifier_squeeze, create_codec_squeeze)
}

fn is_identifier_squeeze(identifier: &str) -> bool {
    identifier == SQUEEZE
}

pub(crate) fn create_codec_squeeze(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
    log::warn!("The `{}` codec is experimental and may be incompatible with other Zarr V3 implementations.", metadata.name());
    let configuration: SqueezeCodecConfiguration = metadata
        .to_configuration()
        .map_err(|_| PluginMetadataInvalidError::new(SQUEEZE, "codec", metadata.to_string()))?;
    let codec = Arc::new(SqueezeCodec::new_with_configuration(&configuration)?);
    Ok(Codec::ArrayToArray(codec))
}

fn get_squeezed_array_subset(
    decoded_region: &ArraySubset,
    shape: &[NonZeroU64],
) -> Result<ArraySubset, CodecError> {
    if decoded_region.dimensionality() != shape.len() {
        return Err(IncompatibleIndexerError::new_incompatible_dimensionality(
            decoded_region.dimensionality(),
            shape.len(),
        )
        .into());
    }

    let ranges = izip!(
        decoded_region.start().iter(),
        decoded_region.shape().iter(),
        shape.iter()
    )
    .filter(|(_, _, &shape)| shape.get() > 1)
    .map(|(rstart, rshape, _)| *rstart..rstart + rshape);

    let decoded_region_squeeze = ArraySubset::from(ranges);
    Ok(decoded_region_squeeze)
}

fn get_squeezed_indexer(
    indexer: &dyn Indexer,
    shape: &[NonZeroU64],
) -> Result<impl Indexer, CodecError> {
    let indices = indexer
        .iter_indices()
        .map(|indices| {
            if indices.len() == shape.len() {
                Ok(indices
                    .into_iter()
                    .zip(shape)
                    .filter_map(
                        |(indices, &shape)| if shape.get() > 1 { Some(indices) } else { None },
                    )
                    .collect_vec())
            } else {
                Err(IncompatibleIndexerError::new_incompatible_dimensionality(
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
    use std::{num::NonZeroU64, sync::Arc};

    use crate::{
        array::{
            codec::{ArrayToArrayCodecTraits, ArrayToBytesCodecTraits, BytesCodec, CodecOptions},
            ArrayBytes, ChunkRepresentation, DataType, FillValue,
        },
        array_subset::ArraySubset,
    };

    use super::*;

    fn codec_squeeze_round_trip_impl(
        json: &str,
        data_type: DataType,
        fill_value: impl Into<FillValue>,
    ) {
        let chunk_representation = ChunkRepresentation::new(
            vec![
                NonZeroU64::new(2).unwrap(),
                NonZeroU64::new(1).unwrap(),
                NonZeroU64::new(2).unwrap(),
                NonZeroU64::new(1).unwrap(),
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

        let configuration: SqueezeCodecConfiguration = serde_json::from_str(json).unwrap();
        let codec = SqueezeCodec::new_with_configuration(&configuration).unwrap();
        assert_eq!(
            codec.encoded_shape(chunk_representation.shape()).unwrap(),
            vec![
                NonZeroU64::new(2).unwrap(),
                NonZeroU64::new(2).unwrap(),
                NonZeroU64::new(3).unwrap(),
            ]
            .into()
        );

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
        // let array_representation_squeeze =
        //     ArrayRepresentation::new(vec![2, 3, 2], data_type.clone(), fill_value.clone()).unwrap();
        // let encoded_array = ndarray::ArrayViewD::from_shape(
        //     array_representation_squeeze.shape().to_vec(),
        //     &encoded,
        // )
        // .unwrap();
        // let decoded_array =
        //     ndarray::ArrayViewD::from_shape(array_representation.shape(), &decoded).unwrap();
    }

    #[test]
    fn codec_squeeze_round_trip_array1() {
        const JSON: &str = r#"{}"#;
        codec_squeeze_round_trip_impl(JSON, DataType::UInt8, 0u8);
    }

    #[test]
    fn codec_squeeze_partial_decode() {
        let codec = Arc::new(SqueezeCodec::new());

        let elements: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let chunk_representation = ChunkRepresentation::new(
            vec![
                NonZeroU64::new(1).unwrap(),
                NonZeroU64::new(4).unwrap(),
                NonZeroU64::new(1).unwrap(),
                NonZeroU64::new(4).unwrap(),
                NonZeroU64::new(1).unwrap(),
            ],
            DataType::Float32,
            0.0f32,
        )
        .unwrap();
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes: ArrayBytes = bytes.into();

        let encoded = codec
            .encode(bytes, &chunk_representation, &CodecOptions::default())
            .unwrap();
        let input_handle = Arc::new(encoded.into_fixed().unwrap());
        let bytes_codec = Arc::new(BytesCodec::default());
        let input_handle = bytes_codec
            .partial_decoder(
                input_handle,
                &codec.encoded_representation(&chunk_representation).unwrap(),
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
