use std::sync::Arc;

use zarrs_registry::codec::BITROUND;

use crate::{
    array::{
        codec::{ArrayBytes, ArrayPartialDecoderTraits, CodecError, CodecOptions},
        DataType,
    },
    array_subset::ArraySubset,
};

#[cfg(feature = "async")]
use crate::array::codec::AsyncArrayPartialDecoderTraits;

use super::round_bytes;

/// Partial decoder for the `bitround` codec.
pub(crate) struct BitroundPartialDecoder {
    input_handle: Arc<dyn ArrayPartialDecoderTraits>,
    data_type: DataType,
    keepbits: u32,
}

impl BitroundPartialDecoder {
    /// Create a new partial decoder for the `bitround` codec.
    pub(crate) fn new(
        input_handle: Arc<dyn ArrayPartialDecoderTraits>,
        data_type: &DataType,
        keepbits: u32,
    ) -> Result<Self, CodecError> {
        match data_type {
            super::supported_dtypes!() => Ok(Self {
                input_handle,
                data_type: data_type.clone(),
                keepbits,
            }),
            super::unsupported_dtypes!() => Err(CodecError::UnsupportedDataType(
                data_type.clone(),
                BITROUND.to_string(),
            )),
        }
    }
}

impl ArrayPartialDecoderTraits for BitroundPartialDecoder {
    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn size(&self) -> usize {
        self.input_handle.size()
    }

    fn partial_decode(
        &self,
        indexer: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        let bytes = self.input_handle.partial_decode(indexer, options)?;

        let mut bytes = bytes.into_fixed()?;
        round_bytes(bytes.to_mut(), &self.data_type, self.keepbits)?;
        Ok(bytes.into())
    }
}

#[cfg(feature = "async")]
/// Asynchronous partial decoder for the `bitround` codec.
pub(crate) struct AsyncBitroundPartialDecoder {
    input_handle: Arc<dyn AsyncArrayPartialDecoderTraits>,
    data_type: DataType,
    keepbits: u32,
}

#[cfg(feature = "async")]
impl AsyncBitroundPartialDecoder {
    /// Create a new partial decoder for the `bitround` codec.
    pub(crate) fn new(
        input_handle: Arc<dyn AsyncArrayPartialDecoderTraits>,
        data_type: &DataType,
        keepbits: u32,
    ) -> Result<Self, CodecError> {
        match data_type {
            super::supported_dtypes!() => Ok(Self {
                input_handle,
                data_type: data_type.clone(),
                keepbits,
            }),
            super::unsupported_dtypes!() => Err(CodecError::UnsupportedDataType(
                data_type.clone(),
                BITROUND.to_string(),
            )),
        }
    }
}

#[cfg(feature = "async")]
#[async_trait::async_trait]
impl AsyncArrayPartialDecoderTraits for AsyncBitroundPartialDecoder {
    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    async fn partial_decode(
        &self,
        indexer: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        let bytes = self.input_handle.partial_decode(indexer, options).await?;

        let mut bytes = bytes.into_fixed()?;
        round_bytes(bytes.to_mut(), &self.data_type, self.keepbits)?;
        Ok(bytes.into())
    }
}
