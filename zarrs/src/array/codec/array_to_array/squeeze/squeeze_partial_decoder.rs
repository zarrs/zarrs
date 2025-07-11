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
    decoded_region: &ArraySubset,
    shape: &[NonZeroU64],
) -> Result<ArraySubset, CodecError> {
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
    Ok(decoded_region_squeeze)
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
        indexer: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        // let Some(decoded_region) = indexer.as_array_subset() else {
        //     todo!("Generic indexer support")
        // };
        let decoded_region = indexer;

        let decoded_region_squeezed =
            get_decoded_regions_squeezed(decoded_region, self.decoded_representation.shape())?;
        self.input_handle
            .partial_decode(&decoded_region_squeezed, options)
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
        indexer: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        // let Some(decoded_region) = indexer.as_array_subset() else {
        //     todo!("Generic indexer support")
        // };
        let decoded_region = indexer;

        let decoded_region_squeezed =
            get_decoded_regions_squeezed(decoded_region, self.decoded_representation.shape())?;
        self.input_handle
            .partial_decode(&decoded_region_squeezed, options)
            .await
    }
}
