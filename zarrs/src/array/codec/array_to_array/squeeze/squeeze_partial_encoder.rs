use std::{num::NonZeroU64, sync::Arc};

use itertools::{izip, Itertools};

use crate::{
    array::{
        codec::{
            ArrayBytes, ArrayPartialDecoderTraits, ArrayPartialEncoderTraits, ArraySubset,
            CodecError, CodecOptions,
        },
        ArrayIndices, ChunkRepresentation, DataType,
    },
    indexer::{IncompatibleIndexerError, Indexer},
};

// #[cfg(feature = "async")]
// use crate::array::codec::AsyncArrayPartialEncoderTraits;

/// Partial encoder for the `squeeze` codec.
pub(crate) struct SqueezePartialEncoder {
    input_output_handle: Arc<dyn ArrayPartialEncoderTraits>,
    decoded_representation: ChunkRepresentation,
}

impl SqueezePartialEncoder {
    /// Create a new partial encoder for the `squeeze` codec.
    pub(crate) fn new(
        input_output_handle: Arc<dyn ArrayPartialEncoderTraits>,
        decoded_representation: ChunkRepresentation,
    ) -> Self {
        Self {
            input_output_handle,
            decoded_representation,
        }
    }
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

// FIXME: Repeated from SqueezePartialDecoder
impl ArrayPartialDecoderTraits for SqueezePartialEncoder {
    fn data_type(&self) -> &DataType {
        self.decoded_representation.data_type()
    }

    fn size(&self) -> usize {
        self.input_output_handle.size()
    }

    fn partial_decode(
        &self,
        indexer: &dyn crate::indexer::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        if let Some(array_subset) = indexer.as_array_subset() {
            let array_subset_squeezed =
                get_squeezed_array_subset(array_subset, self.decoded_representation.shape())?;
            self.input_output_handle
                .partial_decode(&array_subset_squeezed, options)
        } else {
            let indexer_squeezed =
                get_squeezed_indexer(indexer, self.decoded_representation.shape())?;
            self.input_output_handle
                .partial_decode(&indexer_squeezed, options)
        }
    }
}

impl ArrayPartialEncoderTraits for SqueezePartialEncoder {
    fn into_dyn_decoder(self: Arc<Self>) -> Arc<dyn ArrayPartialDecoderTraits> {
        self.clone()
    }

    fn erase(&self) -> Result<(), CodecError> {
        self.input_output_handle.erase()
    }

    fn partial_encode(
        &self,
        indexer: &dyn crate::indexer::Indexer,
        bytes: &ArrayBytes<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        if let Some(array_subset) = indexer.as_array_subset() {
            let array_subset_squeezed =
                get_squeezed_array_subset(array_subset, self.decoded_representation.shape())?;
            self.input_output_handle
                .partial_encode(&array_subset_squeezed, bytes, options)
        } else {
            let indexer_squeezed =
                get_squeezed_indexer(indexer, self.decoded_representation.shape())?;
            self.input_output_handle
                .partial_encode(&indexer_squeezed, bytes, options)
        }
    }
}
