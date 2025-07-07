use std::sync::Arc;

use zarrs_registry::codec::ZFP;

use crate::{
    array::{
        codec::{
            ArrayBytes, ArrayPartialDecoderTraits, BytesPartialDecoderTraits, CodecError,
            CodecOptions,
        },
        ArraySize, ChunkRepresentation, DataType,
    },
    array_subset::ArraySubset,
    byte_range::extract_byte_ranges_concat,
};

#[cfg(feature = "async")]
use crate::array::codec::{AsyncArrayPartialDecoderTraits, AsyncBytesPartialDecoderTraits};

use super::{zarr_to_zfp_data_type, zfp_decode, ZfpMode};

/// Partial decoder for the `zfp` codec.
pub(crate) struct ZfpPartialDecoder {
    input_handle: Arc<dyn BytesPartialDecoderTraits>,
    decoded_representation: ChunkRepresentation,
    mode: ZfpMode,
    write_header: bool,
}

impl ZfpPartialDecoder {
    /// Create a new partial decoder for the `zfp` codec.
    pub(crate) fn new(
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        decoded_representation: &ChunkRepresentation,
        mode: ZfpMode,
        write_header: bool,
    ) -> Result<Self, CodecError> {
        if zarr_to_zfp_data_type(decoded_representation.data_type()).is_some() {
            Ok(Self {
                input_handle,
                decoded_representation: decoded_representation.clone(),
                mode,
                write_header,
            })
        } else {
            Err(CodecError::from(
                "data type {} is unsupported for zfp codec",
            ))
        }
    }
}

impl ArrayPartialDecoderTraits for ZfpPartialDecoder {
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
        let data_type_size = self.data_type().fixed_size().ok_or_else(|| {
            CodecError::UnsupportedDataType(self.data_type().clone(), ZFP.to_string())
        })?;
        for array_subset in decoded_regions {
            if array_subset.dimensionality() != self.decoded_representation.dimensionality() {
                return Err(CodecError::InvalidArraySubsetDimensionalityError(
                    array_subset.clone(),
                    self.decoded_representation.dimensionality(),
                ));
            }
        }

        let encoded_value = self.input_handle.decode(options)?;
        let mut out = Vec::with_capacity(decoded_regions.len());
        let chunk_shape = self.decoded_representation.shape_u64();
        match encoded_value {
            Some(mut encoded_value) => {
                let decoded_value = zfp_decode(
                    &self.mode,
                    self.write_header,
                    encoded_value.to_mut(), // FIXME: Does zfp **really** need the encoded value as mutable?
                    &self.decoded_representation,
                    false, // FIXME
                )?;
                for array_subset in decoded_regions {
                    let byte_ranges = array_subset.byte_ranges(&chunk_shape, data_type_size)?;
                    out.push(ArrayBytes::from(extract_byte_ranges_concat(
                        &decoded_value,
                        &byte_ranges,
                    )?));
                }
            }
            None => {
                for decoded_region in decoded_regions {
                    let array_size = ArraySize::new(
                        self.decoded_representation.data_type().size(),
                        decoded_region.num_elements(),
                    );
                    let fill_value = ArrayBytes::new_fill_value(
                        array_size,
                        self.decoded_representation.fill_value(),
                    );
                    out.push(fill_value);
                }
            }
        }
        Ok(out)
    }
}

#[cfg(feature = "async")]
/// Asynchronous partial decoder for the `zfp` codec.
pub(crate) struct AsyncZfpPartialDecoder {
    input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
    decoded_representation: ChunkRepresentation,
    mode: ZfpMode,
    write_header: bool,
}

#[cfg(feature = "async")]
impl AsyncZfpPartialDecoder {
    /// Create a new partial decoder for the `zfp` codec.
    pub(crate) fn new(
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        decoded_representation: &ChunkRepresentation,
        mode: ZfpMode,
        write_header: bool,
    ) -> Result<Self, CodecError> {
        if zarr_to_zfp_data_type(decoded_representation.data_type()).is_some() {
            Ok(Self {
                input_handle,
                decoded_representation: decoded_representation.clone(),
                mode,
                write_header,
            })
        } else {
            Err(CodecError::from(
                "data type {} is unsupported for zfp codec",
            ))
        }
    }
}

#[cfg(feature = "async")]
#[async_trait::async_trait]
impl AsyncArrayPartialDecoderTraits for AsyncZfpPartialDecoder {
    fn data_type(&self) -> &DataType {
        self.decoded_representation.data_type()
    }

    async fn partial_decode(
        &self,
        decoded_regions: &[ArraySubset],
        options: &CodecOptions,
    ) -> Result<Vec<ArrayBytes<'_>>, CodecError> {
        let data_type_size = self.data_type().fixed_size().ok_or_else(|| {
            CodecError::UnsupportedDataType(self.data_type().clone(), ZFP.to_string())
        })?;
        for array_subset in decoded_regions {
            if array_subset.dimensionality() != self.decoded_representation.dimensionality() {
                return Err(CodecError::InvalidArraySubsetDimensionalityError(
                    array_subset.clone(),
                    self.decoded_representation.dimensionality(),
                ));
            }
        }

        let encoded_value = self.input_handle.decode(options).await?;
        let chunk_shape = self.decoded_representation.shape_u64();
        let mut out = Vec::with_capacity(decoded_regions.len());
        match encoded_value {
            Some(mut encoded_value) => {
                let decoded_value = zfp_decode(
                    &self.mode,
                    self.write_header,
                    encoded_value.to_mut(), // FIXME: Does zfp **really** need the encoded value as mutable?
                    &self.decoded_representation,
                    false, // FIXME
                )?;
                for array_subset in decoded_regions {
                    let byte_ranges = array_subset.byte_ranges(&chunk_shape, data_type_size)?;
                    out.push(ArrayBytes::from(extract_byte_ranges_concat(
                        &decoded_value,
                        &byte_ranges,
                    )?));
                }
            }
            None => {
                for decoded_region in decoded_regions {
                    let array_size = ArraySize::new(
                        self.decoded_representation.data_type().size(),
                        decoded_region.num_elements(),
                    );
                    let fill_value = ArrayBytes::new_fill_value(
                        array_size,
                        self.decoded_representation.fill_value(),
                    );
                    out.push(fill_value);
                }
            }
        }
        Ok(out)
    }
}
