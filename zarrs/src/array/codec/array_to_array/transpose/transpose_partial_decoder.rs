use std::sync::Arc;

use super::{calculate_order_decode, permute, transpose_array, TransposeOrder};
use crate::array::{
    codec::{ArrayBytes, ArrayPartialDecoderTraits, ArraySubset, CodecError, CodecOptions},
    ChunkRepresentation, DataType,
};

#[cfg(feature = "async")]
use crate::array::codec::AsyncArrayPartialDecoderTraits;

/// Partial decoder for the Transpose codec.
pub(crate) struct TransposePartialDecoder {
    input_handle: Arc<dyn ArrayPartialDecoderTraits>,
    decoded_representation: ChunkRepresentation,
    order: TransposeOrder,
}

impl TransposePartialDecoder {
    /// Create a new partial decoder for the Transpose codec.
    pub(crate) fn new(
        input_handle: Arc<dyn ArrayPartialDecoderTraits>,
        decoded_representation: ChunkRepresentation,
        order: TransposeOrder,
    ) -> Self {
        Self {
            input_handle,
            decoded_representation,
            order,
        }
    }
}

fn validate_regions(
    decoded_regions: &[ArraySubset],
    dimensionality: usize,
) -> Result<(), CodecError> {
    for array_subset in decoded_regions {
        if array_subset.dimensionality() != dimensionality {
            return Err(CodecError::InvalidArraySubsetDimensionalityError(
                array_subset.clone(),
                dimensionality,
            ));
        }
    }
    Ok(())
}

fn get_decoded_regions_transposed(
    order: &TransposeOrder,
    decoded_regions: &[ArraySubset],
) -> Vec<ArraySubset> {
    let mut decoded_regions_transposed = Vec::with_capacity(decoded_regions.len());
    for decoded_region in decoded_regions {
        let start = permute(decoded_region.start(), &order.0);
        let size = permute(decoded_region.shape(), &order.0);
        let ranges = start.iter().zip(size).map(|(&st, si)| (st..(st + si)));
        let decoded_region_transpose = ArraySubset::from(ranges);
        decoded_regions_transposed.push(decoded_region_transpose);
    }
    decoded_regions_transposed
}

/// Reverse the transpose on each subset
fn do_transpose<'a>(
    encoded_value: Vec<ArrayBytes<'a>>,
    decoded_regions: &[ArraySubset],
    order: &TransposeOrder,
    decoded_representation: &ChunkRepresentation,
) -> Result<Vec<ArrayBytes<'a>>, CodecError> {
    let order_decode = calculate_order_decode(order, decoded_representation.shape().len());
    let data_type_size = decoded_representation.data_type().size();
    std::iter::zip(decoded_regions, encoded_value)
        .map(|(subset, bytes)| {
            bytes.validate(subset.num_elements(), data_type_size)?;
            match bytes {
                ArrayBytes::Variable(bytes, offsets) => {
                    let mut order_decode = vec![0; decoded_representation.shape().len()];
                    for (i, val) in order.0.iter().enumerate() {
                        order_decode[*val] = i;
                    }
                    Ok(super::transpose_vlen(
                        &bytes,
                        &offsets,
                        &subset.shape_usize(),
                        order_decode,
                    ))
                }
                ArrayBytes::Fixed(bytes) => {
                    let data_type_size = decoded_representation.data_type().fixed_size().unwrap();
                    let bytes = transpose_array(
                        &order_decode,
                        &permute(subset.shape(), &order.0),
                        data_type_size,
                        &bytes,
                    )
                    .map_err(|_| CodecError::Other("transpose_array error".to_string()))?;
                    Ok(ArrayBytes::from(bytes))
                }
            }
        })
        .collect::<Result<Vec<_>, CodecError>>()
}

impl ArrayPartialDecoderTraits for TransposePartialDecoder {
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
        validate_regions(
            decoded_regions,
            self.decoded_representation.dimensionality(),
        )?;
        let decoded_regions_transposed =
            get_decoded_regions_transposed(&self.order, decoded_regions);
        let encoded_value = self
            .input_handle
            .partial_decode(&decoded_regions_transposed, options)?;
        do_transpose(
            encoded_value,
            decoded_regions,
            &self.order,
            &self.decoded_representation,
        )
    }
}

#[cfg(feature = "async")]
/// Asynchronous partial decoder for the Transpose codec.
pub(crate) struct AsyncTransposePartialDecoder {
    input_handle: Arc<dyn AsyncArrayPartialDecoderTraits>,
    decoded_representation: ChunkRepresentation,
    order: TransposeOrder,
}

#[cfg(feature = "async")]
impl AsyncTransposePartialDecoder {
    /// Create a new partial decoder for the Transpose codec.
    pub(crate) fn new(
        input_handle: Arc<dyn AsyncArrayPartialDecoderTraits>,
        decoded_representation: ChunkRepresentation,
        order: TransposeOrder,
    ) -> Self {
        Self {
            input_handle,
            decoded_representation,
            order,
        }
    }
}

#[cfg(feature = "async")]
#[async_trait::async_trait]
impl AsyncArrayPartialDecoderTraits for AsyncTransposePartialDecoder {
    fn data_type(&self) -> &DataType {
        self.decoded_representation.data_type()
    }

    async fn partial_decode(
        &self,
        decoded_regions: &[ArraySubset],
        options: &CodecOptions,
    ) -> Result<Vec<ArrayBytes<'_>>, CodecError> {
        validate_regions(
            decoded_regions,
            self.decoded_representation.dimensionality(),
        )?;
        let decoded_regions_transposed =
            get_decoded_regions_transposed(&self.order, decoded_regions);
        let encoded_value = self
            .input_handle
            .partial_decode(&decoded_regions_transposed, options)
            .await?;
        do_transpose(
            encoded_value,
            decoded_regions,
            &self.order,
            &self.decoded_representation,
        )
    }
}
