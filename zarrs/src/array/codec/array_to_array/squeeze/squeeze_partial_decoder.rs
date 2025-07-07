use std::{num::NonZeroU64, sync::Arc};

use itertools::izip;

use crate::array::{
    codec::{ArrayBytes, ArrayPartialDecoderTraits, ArraySubset, CodecError, CodecOptions},
    ChunkRepresentation, DataType,
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

fn get_decoded_regions_squeezed(
    decoded_regions: &[ArraySubset],
    shape: &[NonZeroU64],
) -> Result<Vec<ArraySubset>, CodecError> {
    let mut decoded_regions_squeezed = Vec::with_capacity(decoded_regions.len());
    for decoded_region in decoded_regions {
        if decoded_region.dimensionality() != shape.len() {
            return Err(CodecError::InvalidArraySubsetDimensionalityError(
                decoded_region.clone(),
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
        decoded_regions_squeezed.push(decoded_region_squeeze);
    }
    Ok(decoded_regions_squeezed)
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
        decoded_regions: &[ArraySubset],
        options: &CodecOptions,
    ) -> Result<Vec<ArrayBytes<'_>>, CodecError> {
        let decoded_regions_squeezed =
            get_decoded_regions_squeezed(decoded_regions, self.decoded_representation.shape())?;
        self.input_handle
            .partial_decode(&decoded_regions_squeezed, options)
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
        decoded_regions: &[ArraySubset],
        options: &CodecOptions,
    ) -> Result<Vec<ArrayBytes<'_>>, CodecError> {
        let decoded_regions_squeezed =
            get_decoded_regions_squeezed(decoded_regions, self.decoded_representation.shape())?;
        self.input_handle
            .partial_decode(&decoded_regions_squeezed, options)
            .await
    }
}
