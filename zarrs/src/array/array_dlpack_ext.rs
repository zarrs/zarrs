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

impl Tensor {
    fn validate_dlpack_shape(&self) -> Result<(), TensorError> {
        let unsupported = || TensorError::UnsupportedShape(self.shape().to_vec());
        i32::try_from(self.shape().len()).map_err(|_| unsupported())?;
        if self.shape().iter().any(|&dim| i64::try_from(dim).is_err()) {
            return Err(unsupported());
        }
        Ok(())
    }

    fn validate_dlpack_bytes(&self) -> Result<(), TensorError> {
        self.validate_dlpack_shape()?;
        let element_size = data_type_to_dlpack(self.data_type())?.size();
        if self.data_type().fixed_size() != Some(element_size) {
            return Err(TensorError::UnsupportedDataType(self.data_type().clone()));
        }
        // A zero-length axis makes the tensor empty, regardless of the other axes.
        let num_elements = if self.shape().contains(&0) {
            Some(0)
        } else {
            self.shape()
                .iter()
                .try_fold(1usize, |n, &dim| n.checked_mul(usize::try_from(dim).ok()?))
        };
        let expected = num_elements.and_then(|n| n.checked_mul(element_size));
        let actual = self.bytes().len();
        if expected.is_none_or(|expected| expected > actual) {
            return Err(TensorError::InsufficientBytes {
                expected: expected.unwrap_or(usize::MAX),
                actual,
            });
        }
        Ok(())
    }
}

impl TensorLike<RowMajorCompactLayout> for Tensor {
    type Error = TensorError;

    fn data_ptr(&self) -> *mut c_void {
        self.bytes().as_ptr().cast::<c_void>().cast_mut()
    }

    fn memory_layout(&self) -> RowMajorCompactLayout {
        // dlpark 0.6 constructs the layout before calling the fallible device/data-type
        // hooks. Keep this infallible hook panic-free; either hook rejects an invalid
        // shape before a managed tensor can be returned.
        let shape = if self.validate_dlpack_shape().is_ok() {
            self.shape().iter().map(|&dim| dim as i64).collect()
        } else {
            vec![]
        };
        RowMajorCompactLayout::new(shape)
    }

    fn byte_offset(&self) -> u64 {
        0
    }

    fn device(&self) -> Result<Device, Self::Error> {
        self.validate_dlpack_bytes()?;
        Ok(Device::CPU)
    }

    fn data_type(&self) -> Result<dlpark::ffi::DataType, Self::Error> {
        self.validate_dlpack_bytes()?;
        data_type_to_dlpack(self.data_type())
    }
}

#[cfg(test)]
mod tests {
    use crate::array::{
        ArrayBuilder, ArraySubset, Tensor, TensorError, data_type, transmute_to_bytes,
    };
    use zarrs_codec::CodecOptions;
    use zarrs_storage::store::MemoryStore;

    #[test]
    fn dlpack_rejects_invalid_buffer_and_shape() {
        use dlpark::versioned::SafeManagedTensorVersioned;
        assert!(matches!(
            SafeManagedTensorVersioned::new(Tensor::new(vec![0u8], data_type::float32(), vec![2])),
            Err(TensorError::InsufficientBytes {
                expected: 8,
                actual: 1
            })
        ));
        assert!(matches!(
            SafeManagedTensorVersioned::new(Tensor::new(
                vec![],
                data_type::uint8(),
                vec![u64::MAX]
            )),
            Err(TensorError::UnsupportedShape(_))
        ));
        assert!(matches!(
            SafeManagedTensorVersioned::new(Tensor::new(
                vec![],
                data_type::float64(),
                vec![i64::MAX as u64, 4]
            )),
            Err(TensorError::InsufficientBytes { .. })
        ));
        assert!(
            SafeManagedTensorVersioned::new(Tensor::new(vec![], data_type::uint8(), vec![]))
                .is_err()
        );
        assert!(
            SafeManagedTensorVersioned::new(Tensor::new(vec![], data_type::string(), vec![0]))
                .is_err()
        );
    }

    #[test]
    fn dlpack_accepts_empty_scalar_and_extra_bytes() {
        use dlpark::versioned::SafeManagedTensorVersioned;
        for (bytes, shape) in [
            (vec![], vec![0]),
            (vec![1], vec![]),
            (vec![1, 2], vec![1]),
            (vec![], vec![i64::MAX as u64, 4, 0]),
        ] {
            assert!(
                SafeManagedTensorVersioned::new(Tensor::new(bytes, data_type::uint8(), shape))
                    .is_ok()
            );
        }
    }

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
