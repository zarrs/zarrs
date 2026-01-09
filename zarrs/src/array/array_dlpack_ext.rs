use std::ffi::c_void;

use dlpark::ffi::Device;
use dlpark::traits::{RowMajorCompactLayout, TensorLike};
use zarrs_plugin::ExtensionIdentifier;

use super::{DataType, Tensor, TensorError};
use crate::array::data_type as dt;

/// Convert a zarrs [`DataType`] to a [`dlpark::ffi::DataType`].
///
/// # Errors
/// Returns [`TensorError::UnsupportedDataType`] if the data type is not supported.
fn data_type_to_dlpack(data_type: &DataType) -> Result<dlpark::ffi::DataType, TensorError> {
    match data_type.identifier() {
        dt::BoolDataType::IDENTIFIER => Ok(dlpark::ffi::DataType::BOOL),
        dt::Int8DataType::IDENTIFIER => Ok(dlpark::ffi::DataType::I8),
        dt::Int16DataType::IDENTIFIER => Ok(dlpark::ffi::DataType::I16),
        dt::Int32DataType::IDENTIFIER => Ok(dlpark::ffi::DataType::I32),
        dt::Int64DataType::IDENTIFIER => Ok(dlpark::ffi::DataType::I64),
        dt::UInt8DataType::IDENTIFIER => Ok(dlpark::ffi::DataType::U8),
        dt::UInt16DataType::IDENTIFIER => Ok(dlpark::ffi::DataType::U16),
        dt::UInt32DataType::IDENTIFIER => Ok(dlpark::ffi::DataType::U32),
        dt::UInt64DataType::IDENTIFIER => Ok(dlpark::ffi::DataType::U64),
        dt::Float16DataType::IDENTIFIER => Ok(dlpark::ffi::DataType::F16),
        dt::Float32DataType::IDENTIFIER => Ok(dlpark::ffi::DataType::F32),
        dt::Float64DataType::IDENTIFIER => Ok(dlpark::ffi::DataType::F64),
        dt::BFloat16DataType::IDENTIFIER => Ok(dlpark::ffi::DataType::BF16),
        _ => Err(TensorError::UnsupportedDataType(data_type.clone())),
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

    use crate::array::codec::CodecOptions;
    use crate::array::{ArrayBuilder, ArraySubset, Tensor, data_type, transmute_to_bytes};
    use crate::storage::store::MemoryStore;

    #[test]
    fn array_dlpack_ext_sync() {
        let store = MemoryStore::new();
        let array = ArrayBuilder::new(vec![4, 4], vec![2, 2], data_type::float32(), -1.0f32)
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
