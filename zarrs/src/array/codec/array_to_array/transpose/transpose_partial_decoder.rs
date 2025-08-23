use std::sync::Arc;

use super::{calculate_order_decode, permute, transpose_array, TransposeOrder};
use crate::{
    array::{
        codec::{ArrayBytes, ArrayPartialDecoderTraits, ArraySubset, CodecError, CodecOptions},
        ChunkRepresentation, DataType,
    },
    indexer::{IncompatibleIndexerError, Indexer},
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
    indexer: &dyn crate::indexer::Indexer,
    dimensionality: usize,
) -> Result<(), CodecError> {
    if indexer.dimensionality() == dimensionality {
        Ok(())
    } else {
        Err(IncompatibleIndexerError::new_incompatible_dimensionality(
            indexer.dimensionality(),
            dimensionality,
        )
        .into())
    }
}

fn get_transposed_array_subset(
    order: &TransposeOrder,
    decoded_region: &ArraySubset,
) -> ArraySubset {
    let start = permute(decoded_region.start(), &order.0);
    let size = permute(decoded_region.shape(), &order.0);
    let ranges = start.iter().zip(size).map(|(&st, si)| st..(st + si));
    ArraySubset::from(ranges)
}

fn get_transposed_indexer(order: &TransposeOrder, indexer: &dyn Indexer) -> impl Indexer {
    indexer
        .iter_indices()
        .map(|indices| permute(&indices, &order.0))
        .collect::<Vec<_>>()
}

/// Reverse the transpose on each subset
fn do_transpose<'a>(
    encoded_value: ArrayBytes<'a>,
    subset: &ArraySubset,
    order: &TransposeOrder,
    decoded_representation: &ChunkRepresentation,
) -> Result<ArrayBytes<'a>, CodecError> {
    let order_decode = calculate_order_decode(order, decoded_representation.shape().len());
    let data_type_size = decoded_representation.data_type().size();
    encoded_value.validate(subset.num_elements(), data_type_size)?;
    match encoded_value {
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
        indexer: &dyn crate::indexer::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        validate_regions(indexer, self.decoded_representation.dimensionality())?;

        if let Some(array_subset) = indexer.as_array_subset() {
            let array_subset_transposed = get_transposed_array_subset(&self.order, array_subset);
            let encoded_value = self
                .input_handle
                .partial_decode(&array_subset_transposed, options)?;
            do_transpose(
                encoded_value,
                array_subset,
                &self.order,
                &self.decoded_representation,
            )
        } else {
            let indexer_transposed = get_transposed_indexer(&self.order, indexer);
            self.input_handle
                .partial_decode(&indexer_transposed, options)
        }
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
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncArrayPartialDecoderTraits for AsyncTransposePartialDecoder {
    fn data_type(&self) -> &DataType {
        self.decoded_representation.data_type()
    }

    async fn partial_decode(
        &self,
        indexer: &dyn crate::indexer::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        validate_regions(indexer, self.decoded_representation.dimensionality())?;

        if let Some(array_subset) = indexer.as_array_subset() {
            let array_subset_transposed = get_transposed_array_subset(&self.order, array_subset);
            let encoded_value = self
                .input_handle
                .partial_decode(&array_subset_transposed, options)
                .await?;
            do_transpose(
                encoded_value,
                array_subset,
                &self.order,
                &self.decoded_representation,
            )
        } else {
            let indexer_transposed = get_transposed_indexer(&self.order, indexer);
            self.input_handle
                .partial_decode(&indexer_transposed, options)
                .await
        }
    }
}
