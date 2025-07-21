use std::{num::NonZeroU64, sync::Arc};

use itertools::{izip, Itertools};

use crate::{
    array::{
        codec::{ArrayBytes, ArrayPartialDecoderTraits, ArraySubset, CodecError, CodecOptions},
        ArrayIndices, ChunkRepresentation, DataType,
    },
    indexer::Indexer,
};

#[cfg(feature = "async")]
use crate::array::codec::AsyncArrayPartialDecoderTraits;

/// Partial decoder for the Squeeze codec.
pub(crate) struct SqueezePartialDecoder {
    input_handle: Arc<dyn ArrayPartialDecoderTraits>,
    decoded_representation: ChunkRepresentation,
}

impl SqueezePartialDecoder {
    /// Create a new partial decoder for the Squeeze codec.
    pub(crate) fn new(
        input_handle: Arc<dyn ArrayPartialDecoderTraits>,
        decoded_representation: ChunkRepresentation,
    ) -> Self {
        Self {
            input_handle,
            decoded_representation,
        }
    }
}

fn get_squeezed_array_subset(
    decoded_region: &ArraySubset,
    shape: &[NonZeroU64],
) -> Result<ArraySubset, CodecError> {
    if decoded_region.dimensionality() != shape.len() {
        return Err(CodecError::InvalidIndexerDimensionalityError(
            decoded_region.dimensionality(),
            shape.len(),
        ));
    }

    let ranges = izip!(
        decoded_region.start().iter(),
        decoded_region.shape().iter(),
        shape.iter()
    )
    .filter(|(_, _, &shape)| shape.get() > 1)
    .map(|(rstart, rshape, _)| (*rstart..rstart + rshape));

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
                Err(CodecError::InvalidIndexerDimensionalityError(
                    indices.len(),
                    shape.len(),
                ))
            }
        })
        .collect::<Result<Vec<ArrayIndices>, _>>()?;

    Ok(indices)
}

impl ArrayPartialDecoderTraits for SqueezePartialDecoder {
    fn data_type(&self) -> &DataType {
        self.decoded_representation.data_type()
    }

    fn size(&self) -> usize {
        self.input_handle.size()
    }

    fn partial_decode(
        &self,
        indexer: &dyn crate::indexer::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        if let Some(array_subset) = indexer.as_array_subset() {
            let array_subset_squeezed =
                get_squeezed_array_subset(array_subset, self.decoded_representation.shape())?;
            self.input_handle
                .partial_decode(&array_subset_squeezed, options)
        } else {
            let indexer_squeezed =
                get_squeezed_indexer(indexer, self.decoded_representation.shape())?;
            self.input_handle.partial_decode(&indexer_squeezed, options)
        }
    }
}

#[cfg(feature = "async")]
/// Asynchronous partial decoder for the Squeeze codec.
pub(crate) struct AsyncSqueezePartialDecoder {
    input_handle: Arc<dyn AsyncArrayPartialDecoderTraits>,
    decoded_representation: ChunkRepresentation,
}

#[cfg(feature = "async")]
impl AsyncSqueezePartialDecoder {
    /// Create a new partial decoder for the Squeeze codec.
    pub(crate) fn new(
        input_handle: Arc<dyn AsyncArrayPartialDecoderTraits>,
        decoded_representation: ChunkRepresentation,
    ) -> Self {
        Self {
            input_handle,
            decoded_representation,
        }
    }
}

#[cfg(feature = "async")]
#[async_trait::async_trait]
impl AsyncArrayPartialDecoderTraits for AsyncSqueezePartialDecoder {
    fn data_type(&self) -> &DataType {
        self.decoded_representation.data_type()
    }

    async fn partial_decode(
        &self,
        indexer: &dyn crate::indexer::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        if let Some(array_subset) = indexer.as_array_subset() {
            let array_subset_squeezed =
                get_squeezed_array_subset(array_subset, self.decoded_representation.shape())?;
            self.input_handle
                .partial_decode(&array_subset_squeezed, options)
                .await
        } else {
            let indexer_squeezed =
                get_squeezed_indexer(indexer, self.decoded_representation.shape())?;
            self.input_handle
                .partial_decode(&indexer_squeezed, options)
                .await
        }
    }
}
