use std::ffi::c_void;

use dlpark::ffi::Device;
use dlpark::traits::{RowMajorCompactLayout, TensorLike};
use std::any::TypeId;

use super::{DataType, Tensor, TensorError};
use crate::array::data_type as dt;

/// Convert a zarrs [`DataType`] to a [`dlpark::ffi::DataType`].
///
/// # Errors
/// Returns [`TensorError::UnsupportedDataType`] if the data type is not supported.
fn data_type_to_dlpack(data_type: &DataType) -> Result<dlpark::ffi::DataType, TensorError> {
    let type_id = data_type.as_any().type_id();
    // https://github.com/rust-lang/rust/issues/70861 for match?
    if type_id == TypeId::of::<dt::BoolDataType>() {
        Ok(dlpark::ffi::DataType::BOOL)
    } else if type_id == TypeId::of::<dt::Int8DataType>() {
        Ok(dlpark::ffi::DataType::I8)
    } else if type_id == TypeId::of::<dt::Int16DataType>() {
        Ok(dlpark::ffi::DataType::I16)
    } else if type_id == TypeId::of::<dt::Int32DataType>() {
        Ok(dlpark::ffi::DataType::I32)
    } else if type_id == TypeId::of::<dt::Int64DataType>() {
        Ok(dlpark::ffi::DataType::I64)
    } else if type_id == TypeId::of::<dt::UInt8DataType>() {
        Ok(dlpark::ffi::DataType::U8)
    } else if type_id == TypeId::of::<dt::UInt16DataType>() {
        Ok(dlpark::ffi::DataType::U16)
    } else if type_id == TypeId::of::<dt::UInt32DataType>() {
        Ok(dlpark::ffi::DataType::U32)
    } else if type_id == TypeId::of::<dt::UInt64DataType>() {
        Ok(dlpark::ffi::DataType::U64)
    } else if type_id == TypeId::of::<dt::Float16DataType>() {
        Ok(dlpark::ffi::DataType::F16)
    } else if type_id == TypeId::of::<dt::Float32DataType>() {
        Ok(dlpark::ffi::DataType::F32)
    } else if type_id == TypeId::of::<dt::Float64DataType>() {
        Ok(dlpark::ffi::DataType::F64)
    } else if type_id == TypeId::of::<dt::BFloat16DataType>() {
        Ok(dlpark::ffi::DataType::BF16)
    } else {
        Err(TensorError::UnsupportedDataType(data_type.clone()))
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
