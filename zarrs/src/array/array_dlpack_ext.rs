use std::{ffi::c_void, sync::Arc};

use crate::array::DataType;
use derive_more::Display;
use dlpark::{ffi::Device, ShapeAndStrides, ToTensor};
use thiserror::Error;

use super::{ChunkRepresentation, RawBytes};

mod array_dlpack_ext_async;
mod array_dlpack_ext_sync;

pub use array_dlpack_ext_async::AsyncArrayDlPackExt;
pub use array_dlpack_ext_sync::ArrayDlPackExt;

/// [`RawBytes`] for use in a [`dlpark::ManagerCtx`].
pub struct RawBytesDlPack {
    bytes: Arc<RawBytes<'static>>,
    dtype: dlpark::ffi::DataType,
    shape: Vec<i64>,
}

/// Errors related to [`[Async]ArrayDlPackExt`](ArrayDlPackExt) methods.
#[derive(Clone, Debug, Display, Error)]
#[non_exhaustive]
pub enum ArrayDlPackExtError {
    /// The Zarr data type is not supported by `DLPack`.
    UnsupportedDataType,
}

macro_rules! unsupported_dtypes {
    // TODO: Add support for extensions?
    () => {
        DataType::Int2
            | DataType::Int4
            | DataType::UInt2
            | DataType::UInt4
            | DataType::Float4E2M1FN
            | DataType::Float6E2M3FN
            | DataType::Float6E3M2FN
            | DataType::Float8E3M4
            | DataType::Float8E4M3
            | DataType::Float8E4M3B11FNUZ
            | DataType::Float8E4M3FNUZ
            | DataType::Float8E5M2
            | DataType::Float8E5M2FNUZ
            | DataType::Float8E8M0FNU
            | DataType::ComplexBFloat16
            | DataType::ComplexFloat16
            | DataType::ComplexFloat32
            | DataType::ComplexFloat64
            | DataType::ComplexFloat4E2M1FN
            | DataType::ComplexFloat6E2M3FN
            | DataType::ComplexFloat6E3M2FN
            | DataType::ComplexFloat8E3M4
            | DataType::ComplexFloat8E4M3
            | DataType::ComplexFloat8E4M3B11FNUZ
            | DataType::ComplexFloat8E4M3FNUZ
            | DataType::ComplexFloat8E5M2
            | DataType::ComplexFloat8E5M2FNUZ
            | DataType::ComplexFloat8E8M0FNU
            | DataType::Complex64
            | DataType::Complex128
            | DataType::RawBits(_)
            | DataType::String
            | DataType::Bytes
            | DataType::NumpyDateTime64 {
                unit: _,
                scale_factor: _,
            }
            | DataType::NumpyTimeDelta64 {
                unit: _,
                scale_factor: _,
            }
            | DataType::Extension(_)
    };
}

impl RawBytesDlPack {
    /// Create a new [`RawBytesDlPack`].
    ///
    /// # Errors
    /// Returns [`ArrayDlPackExtError::UnsupportedDataType`] if the data type is not supported.
    ///
    /// # Panics
    /// Panics if an element in the shape cannot be encoded in a `i64`.
    pub fn new(
        bytes: Arc<RawBytes<'static>>,
        representation: &ChunkRepresentation,
    ) -> Result<Self, ArrayDlPackExtError> {
        let dtype = match representation.data_type() {
            DataType::Bool => dlpark::ffi::DataType::BOOL,
            DataType::Int8 => dlpark::ffi::DataType::I8,
            DataType::Int16 => dlpark::ffi::DataType::I16,
            DataType::Int32 => dlpark::ffi::DataType::I32,
            DataType::Int64 => dlpark::ffi::DataType::I64,
            DataType::UInt8 => dlpark::ffi::DataType::U8,
            DataType::UInt16 => dlpark::ffi::DataType::U16,
            DataType::UInt32 => dlpark::ffi::DataType::U32,
            DataType::UInt64 => dlpark::ffi::DataType::U64,
            DataType::Float16 => dlpark::ffi::DataType::F16,
            DataType::Float32 => dlpark::ffi::DataType::F32,
            DataType::Float64 => dlpark::ffi::DataType::F64,
            DataType::BFloat16 => dlpark::ffi::DataType::BF16,
            unsupported_dtypes!() => Err(ArrayDlPackExtError::UnsupportedDataType)?,
        };
        let shape = representation
            .shape()
            .iter()
            .map(|s| i64::try_from(s.get()).unwrap())
            .collect();
        Ok(Self {
            bytes,
            dtype,
            shape,
        })
    }
}

impl ToTensor for RawBytesDlPack {
    fn data_ptr(&self) -> *mut c_void {
        self.bytes.as_ptr().cast::<c_void>().cast_mut()
    }

    fn byte_offset(&self) -> u64 {
        0
    }

    fn device(&self) -> Device {
        Device::CPU
    }

    fn dtype(&self) -> dlpark::ffi::DataType {
        self.dtype
    }

    fn shape_and_strides(&self) -> ShapeAndStrides {
        ShapeAndStrides::new_contiguous(&self.shape)
    }
}

#[cfg(test)]
mod tests {
    use dlpark::{IntoDLPack, ManagedTensor};
    use zarrs_storage::store::MemoryStore;

    use crate::{
        array::{codec::CodecOptions, ArrayBuilder, ArrayDlPackExt, DataType},
        array_subset::ArraySubset,
    };

    #[test]
    fn array_dlpack_ext_sync() {
        let store = MemoryStore::new();
        let array = ArrayBuilder::new(vec![4, 4], vec![2, 2], DataType::Float32, -1.0f32)
            .build(store.into(), "/")
            .unwrap();
        array
            .store_chunk_elements::<f32>(&[0, 0], &[0.0, 1.0, 2.0, 3.0])
            .unwrap();
        let tensor = array
            .retrieve_chunks_dlpack(
                &ArraySubset::new_with_shape(vec![1, 2]),
                &CodecOptions::default(),
            )
            .unwrap();

        assert_eq!(
            ManagedTensor::new(tensor.into_dlpack()).as_slice::<f32>(),
            &[0.0, 1.0, -1.0, -1.0, 2.0, 3.0, -1.0, -1.0]
        );
    }
}
