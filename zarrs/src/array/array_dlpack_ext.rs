use std::ffi::c_void;

use dlpark::ffi::Device;
use dlpark::traits::{RowMajorCompactLayout, TensorLike};

use super::{DataType, Tensor, TensorError};

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
            | DataType::Optional(_)
            | DataType::Extension(_)
    };
}

/// Convert a zarrs [`DataType`] to a [`dlpark::ffi::DataType`].
///
/// # Errors
/// Returns [`TensorError::UnsupportedDataType`] if the data type is not supported.
fn data_type_to_dlpack(data_type: &DataType) -> Result<dlpark::ffi::DataType, TensorError> {
    match data_type {
        DataType::Bool => Ok(dlpark::ffi::DataType::BOOL),
        DataType::Int8 => Ok(dlpark::ffi::DataType::I8),
        DataType::Int16 => Ok(dlpark::ffi::DataType::I16),
        DataType::Int32 => Ok(dlpark::ffi::DataType::I32),
        DataType::Int64 => Ok(dlpark::ffi::DataType::I64),
        DataType::UInt8 => Ok(dlpark::ffi::DataType::U8),
        DataType::UInt16 => Ok(dlpark::ffi::DataType::U16),
        DataType::UInt32 => Ok(dlpark::ffi::DataType::U32),
        DataType::UInt64 => Ok(dlpark::ffi::DataType::U64),
        DataType::Float16 => Ok(dlpark::ffi::DataType::F16),
        DataType::Float32 => Ok(dlpark::ffi::DataType::F32),
        DataType::Float64 => Ok(dlpark::ffi::DataType::F64),
        DataType::BFloat16 => Ok(dlpark::ffi::DataType::BF16),
        unsupported_dtypes!() => Err(TensorError::UnsupportedDataType(data_type.clone())),
    }
}

impl TensorLike<RowMajorCompactLayout> for Tensor {
    type Error = TensorError;

    fn data_ptr(&self) -> *mut c_void {
        self.bytes().as_ptr().cast::<c_void>().cast_mut()
    }

    fn memory_layout(&self) -> RowMajorCompactLayout {
        let shape: Vec<i64> = self
            .shape()
            .iter()
            .map(|s| i64::try_from(*s).unwrap())
            .collect();
        RowMajorCompactLayout::new(shape)
    }

    fn byte_offset(&self) -> u64 {
        0
    }

    fn device(&self) -> Result<Device, Self::Error> {
        Ok(Device::CPU)
    }

    fn data_type(&self) -> Result<dlpark::ffi::DataType, Self::Error> {
        data_type_to_dlpack(self.data_type())
    }
}

#[cfg(test)]
mod tests {
    // use dlpark::{IntoDLPack, ManagedTensor};

    use crate::array::{Tensor, transmute_to_bytes};
    use crate::storage::store::MemoryStore;
    use crate::{
        array::{ArrayBuilder, DataType, codec::CodecOptions},
        array_subset::ArraySubset,
    };

    #[test]
    fn array_dlpack_ext_sync() {
        let store = MemoryStore::new();
        let array = ArrayBuilder::new(vec![4, 4], vec![2, 2], DataType::Float32, -1.0f32)
            .build(store.into(), "/")
            .unwrap();
        array
            .store_chunk(&[0, 0], &[0.0f32, 1.0, 2.0, 3.0])
            .unwrap();
        let tensor: Tensor = array
            .retrieve_chunks_opt(
                &ArraySubset::new_with_shape(vec![1, 2]),
                &CodecOptions::default(),
            )
            .unwrap();

        let managed_tensor = dlpark::versioned::SafeManagedTensorVersioned::new(tensor).unwrap();
        let bytes: &[u8] = &managed_tensor;

        assert_eq!(
            bytes,
            transmute_to_bytes(&[0.0f32, 1.0, -1.0, -1.0, 2.0, 3.0, -1.0, -1.0])
        );
    }
}
