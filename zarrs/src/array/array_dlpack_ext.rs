use std::any::TypeId;
use std::ffi::c_void;

use dlpark::ffi::{DLDataType, DLDataTypeCode, DLDevice};
use dlpark::metadata::CopiedSlice;
use dlpark::tensor::compact_strides;
use dlpark::{Builder, DlpackFlags};

use super::{DataType, Tensor, TensorError};
use crate::array::data_type as dt;

/// A [`dlpark::Builder`] that exports a [`Tensor`] as a `DLPack` managed tensor.
///
/// Create one with `Builder::try_from(Box::new(tensor))`, then finish it with
/// [`try_build`](dlpark::Builder::try_build) to get a
/// [`versioned::Dlpack`](dlpark::versioned::Dlpack) or a
/// [`legacy::Dlpack`](dlpark::legacy::Dlpack).
///
/// The [`Tensor`] is moved into the builder and keeps its data alive for as long as the managed
/// tensor exists. No element data is copied; only the shape and strides are copied into the
/// managed tensor allocation.
///
/// # Sub-byte data types
/// `zarrs` stores sub-byte data types (`int2`, `int4`, `uint2`, `uint4`, `float4_e2m1fn`,
/// `float6_e2m3fn`, and `float6_e3m2fn`) padded to one byte per element, whereas `DLPack` assumes
/// that sub-byte elements are packed. The builder sets
/// [`DlpackFlags::IS_SUBBYTE_TYPE_PADDED`] to signal this, but the legacy `DLManagedTensor` ABI has
/// no flags field. **Sub-byte tensors must therefore be built as a
/// [`versioned::Dlpack`](dlpark::versioned::Dlpack)**; building one as a
/// [`legacy::Dlpack`](dlpark::legacy::Dlpack) drops the flag and consumers will misinterpret the
/// data. Note that [`num_bytes`](dlpark::ManagedBox::num_bytes) reports the *packed* size for these
/// data types, which is smaller than the exported buffer.
///
/// # Examples
/// ```rust
/// # use zarrs::array::{ArrayBuilder, ArraySubset, Tensor, data_type};
/// # use zarrs_storage::store::MemoryStore;
/// # let store = MemoryStore::new();
/// # let array = ArrayBuilder::new(vec![4, 4], vec![2, 2], data_type::float32(), -1.0f32)
/// #     .build(store.into(), "/")?;
/// # array.store_chunk(&[0, 0], &[0.0f32, 1.0, 2.0, 3.0])?;
/// use dlpark::{Builder, versioned};
///
/// let tensor: Tensor = array.retrieve_chunks(&ArraySubset::new_with_shape(vec![1, 2]))?;
/// let dlpack: versioned::Dlpack = Builder::try_from(Box::new(tensor))?.try_build()?;
///
/// assert_eq!(dlpack.shape()?, &[2, 4]);
/// assert_eq!(
///     dlpack.cpu_data_slice::<f32>()?,
///     &[0.0, 1.0, -1.0, -1.0, 2.0, 3.0, -1.0, -1.0]
/// );
/// # Ok::<_, Box<dyn std::error::Error>>(())
/// ```
pub type TensorDlpackBuilder = Builder<Box<Tensor>, CopiedSlice<Vec<i64>, Vec<i64>>>;

/// Convert a zarrs [`DataType`] to a [`DLDataType`].
///
/// Data types with fewer than 8 bits are described as `DLPack` sub-byte types. `zarrs` stores
/// these padded to a whole byte, whereas `DLPack` assumes they are packed unless the exporter sets
/// [`DlpackFlags::IS_SUBBYTE_TYPE_PADDED`], see [`data_type_dlpack_flags`].
///
/// # Errors
/// Returns [`TensorError::UnsupportedDataType`] if the data type is not supported.
fn data_type_to_dlpack(data_type: &DataType) -> Result<DLDataType, TensorError> {
    let type_id = data_type.as_any().type_id();
    let unsupported = || TensorError::UnsupportedDataType(data_type.clone());
    // https://github.com/rust-lang/rust/issues/70861 for match?
    let dtype = if type_id == TypeId::of::<dt::BoolDataType>() {
        // By array library convention, the underlying storage size of a bool is 8 bits
        DLDataType::scalar(DLDataTypeCode::BOOL, 8)
    } else if type_id == TypeId::of::<dt::Int2DataType>() {
        DLDataType::scalar(DLDataTypeCode::INT, 2)
    } else if type_id == TypeId::of::<dt::Int4DataType>() {
        DLDataType::scalar(DLDataTypeCode::INT, 4)
    } else if type_id == TypeId::of::<dt::Int8DataType>() {
        DLDataType::scalar(DLDataTypeCode::INT, 8)
    } else if type_id == TypeId::of::<dt::Int16DataType>() {
        DLDataType::scalar(DLDataTypeCode::INT, 16)
    } else if type_id == TypeId::of::<dt::Int32DataType>() {
        DLDataType::scalar(DLDataTypeCode::INT, 32)
    } else if type_id == TypeId::of::<dt::Int64DataType>() {
        DLDataType::scalar(DLDataTypeCode::INT, 64)
    } else if type_id == TypeId::of::<dt::UInt2DataType>() {
        DLDataType::scalar(DLDataTypeCode::UINT, 2)
    } else if type_id == TypeId::of::<dt::UInt4DataType>() {
        DLDataType::scalar(DLDataTypeCode::UINT, 4)
    } else if type_id == TypeId::of::<dt::UInt8DataType>() {
        DLDataType::scalar(DLDataTypeCode::UINT, 8)
    } else if type_id == TypeId::of::<dt::UInt16DataType>() {
        DLDataType::scalar(DLDataTypeCode::UINT, 16)
    } else if type_id == TypeId::of::<dt::UInt32DataType>() {
        DLDataType::scalar(DLDataTypeCode::UINT, 32)
    } else if type_id == TypeId::of::<dt::UInt64DataType>() {
        DLDataType::scalar(DLDataTypeCode::UINT, 64)
    } else if type_id == TypeId::of::<dt::Float16DataType>() {
        DLDataType::scalar(DLDataTypeCode::FLOAT, 16)
    } else if type_id == TypeId::of::<dt::Float32DataType>() {
        DLDataType::scalar(DLDataTypeCode::FLOAT, 32)
    } else if type_id == TypeId::of::<dt::Float64DataType>() {
        DLDataType::scalar(DLDataTypeCode::FLOAT, 64)
    } else if type_id == TypeId::of::<dt::BFloat16DataType>() {
        DLDataType::scalar(DLDataTypeCode::BFLOAT, 16)
    } else if type_id == TypeId::of::<dt::Float8E3M4DataType>() {
        DLDataType::scalar(DLDataTypeCode::FLOAT8_E3M4, 8)
    } else if type_id == TypeId::of::<dt::Float8E4M3DataType>() {
        DLDataType::scalar(DLDataTypeCode::FLOAT8_E4M3, 8)
    } else if type_id == TypeId::of::<dt::Float8E4M3B11FNUZDataType>() {
        DLDataType::scalar(DLDataTypeCode::FLOAT8_E4M3B11FNUZ, 8)
    } else if type_id == TypeId::of::<dt::Float8E4M3FNUZDataType>() {
        DLDataType::scalar(DLDataTypeCode::FLOAT8_E4M3FNUZ, 8)
    } else if type_id == TypeId::of::<dt::Float8E5M2DataType>() {
        DLDataType::scalar(DLDataTypeCode::FLOAT8_E5M2, 8)
    } else if type_id == TypeId::of::<dt::Float8E5M2FNUZDataType>() {
        DLDataType::scalar(DLDataTypeCode::FLOAT8_E5M2FNUZ, 8)
    } else if type_id == TypeId::of::<dt::Float8E8M0FNUDataType>() {
        DLDataType::scalar(DLDataTypeCode::FLOAT8_E8M0FNU, 8)
    } else if type_id == TypeId::of::<dt::Float6E2M3FNDataType>() {
        DLDataType::scalar(DLDataTypeCode::FLOAT6_E2M3FN, 6)
    } else if type_id == TypeId::of::<dt::Float6E3M2FNDataType>() {
        DLDataType::scalar(DLDataTypeCode::FLOAT6_E3M2FN, 6)
    } else if type_id == TypeId::of::<dt::Float4E2M1FNDataType>() {
        DLDataType::scalar(DLDataTypeCode::FLOAT4_E2M1FN, 4)
    } else if type_id == TypeId::of::<dt::ComplexFloat16DataType>() {
        // `DLPack` complex `bits` is the width of the whole complex number, not one component
        DLDataType::scalar(DLDataTypeCode::COMPLEX, 32)
    } else if type_id == TypeId::of::<dt::Complex64DataType>()
        || type_id == TypeId::of::<dt::ComplexFloat32DataType>()
    {
        DLDataType::scalar(DLDataTypeCode::COMPLEX, 64)
    } else if type_id == TypeId::of::<dt::Complex128DataType>()
        || type_id == TypeId::of::<dt::ComplexFloat64DataType>()
    {
        DLDataType::scalar(DLDataTypeCode::COMPLEX, 128)
    } else {
        // Unsupported data types include:
        // - `complex_bfloat16`, which needs a data type code that postdates DLPack 1.3
        // - the complex subfloats, which have no `DLPack` data type code
        // - the variable-sized, time, and raw data types
        return Err(unsupported());
    };

    // Guard against a data type whose in-memory element size disagrees with its `DLPack`
    // descriptor, which would misrepresent the tensor bytes.
    if data_type.fixed_size() == Some(dtype.element_size()) {
        Ok(dtype)
    } else {
        Err(unsupported())
    }
}

/// Get the [`DlpackFlags`] implied by an exported [`DLDataType`].
fn data_type_dlpack_flags(dtype: DLDataType) -> DlpackFlags {
    if dtype.bits.is_multiple_of(8) {
        DlpackFlags::empty()
    } else {
        // `zarrs` pads sub-byte elements to a whole byte, but `DLPack` assumes they are packed
        DlpackFlags::IS_SUBBYTE_TYPE_PADDED
    }
}

/// Convert a tensor shape into `DLPack` shape and row-major compact strides (in elements).
///
/// # Errors
/// Returns [`TensorError::UnsupportedShape`] if a dimension or a stride is not representable as an
/// [`i64`].
fn shape_and_strides(shape: &[u64]) -> Result<(Vec<i64>, Vec<i64>), TensorError> {
    let unsupported = || TensorError::UnsupportedShape(shape.to_vec());
    let shape = shape
        .iter()
        .map(|s| i64::try_from(*s).map_err(|_| unsupported()))
        .collect::<Result<Vec<i64>, TensorError>>()?;
    let strides = compact_strides(&shape).map_err(|_| unsupported())?;
    Ok((shape, strides))
}

/// Converts a boxed [`Tensor`] into a configurable `DLPack` builder.
///
/// The tensor bytes are not copied. [`IS_COPIED`](DlpackFlags::IS_COPIED) is not set: a [`Tensor`]
/// may reference data that it does not own, so exclusive ownership cannot be asserted here.
/// Callers that know their tensor owns its bytes can add it with
/// [`insert_flags_unchecked`](dlpark::Builder::insert_flags_unchecked).
impl TryFrom<Box<Tensor>> for TensorDlpackBuilder {
    type Error = TensorError;

    fn try_from(tensor: Box<Tensor>) -> Result<Self, Self::Error> {
        let dtype = data_type_to_dlpack(tensor.data_type())?;
        let (shape, strides) = shape_and_strides(tensor.shape())?;
        let data = if tensor.bytes().is_empty() {
            std::ptr::null_mut()
        } else {
            tensor.bytes().as_ptr().cast::<c_void>().cast_mut()
        };
        let builder = Builder::new(tensor, CopiedSlice::new(shape, strides));
        // SAFETY: the boxed tensor context owns the initialized byte allocation addressed by
        // `data` for the lifetime of the managed tensor, and `data`/`dtype`/shape/strides describe
        // it as a compact row-major tensor.
        let builder = unsafe { builder.data(data) }
            .dtype(dtype)
            .device(DLDevice::CPU);
        Ok(builder
            .insert_flags(data_type_dlpack_flags(dtype))
            .expect("the flags do not include IS_COPIED"))
    }
}

#[cfg(test)]
mod tests {
    use dlpark::ffi::{DLDataType, DLDataTypeCode};
    use dlpark::{Builder, DlpackFlags, legacy, versioned};
    use zarrs_storage::store::MemoryStore;

    use crate::array::{ArrayBuilder, ArraySubset, Tensor, data_type};

    fn test_tensor() -> Tensor {
        let store = MemoryStore::new();
        let array = ArrayBuilder::new(vec![4, 4], vec![2, 2], data_type::float32(), -1.0f32)
            .build(store.into(), "/")
            .unwrap();
        array
            .store_chunk(&[0, 0], &[0.0f32, 1.0, 2.0, 3.0])
            .unwrap();
        array
            .retrieve_chunks(&ArraySubset::new_with_shape(vec![1, 2]))
            .unwrap()
    }

    #[test]
    fn array_dlpack_ext_versioned() {
        let dlpack: versioned::Dlpack = Builder::try_from(Box::new(test_tensor()))
            .unwrap()
            .try_build()
            .unwrap();

        assert_eq!(dlpack.shape().unwrap(), &[2, 4]);
        assert_eq!(dlpack.strides().unwrap().unwrap(), &[4, 1]);
        assert_eq!(dlpack.num_bytes().unwrap(), 8 * size_of::<f32>());
        assert_eq!(
            dlpack.cpu_data_slice::<f32>().unwrap(),
            &[0.0f32, 1.0, -1.0, -1.0, 2.0, 3.0, -1.0, -1.0]
        );
    }

    #[test]
    fn array_dlpack_ext_legacy() {
        let dlpack: legacy::Dlpack = Builder::try_from(Box::new(test_tensor()))
            .unwrap()
            .try_build()
            .unwrap();

        assert_eq!(dlpack.shape().unwrap(), &[2, 4]);
        assert_eq!(
            dlpack.cpu_data_slice::<f32>().unwrap(),
            &[0.0f32, 1.0, -1.0, -1.0, 2.0, 3.0, -1.0, -1.0]
        );
    }

    #[test]
    fn array_dlpack_ext_data_types() {
        // (data type, expected code, expected bits)
        let data_types = [
            (data_type::bool(), DLDataTypeCode::BOOL, 8),
            (data_type::int2(), DLDataTypeCode::INT, 2),
            (data_type::int4(), DLDataTypeCode::INT, 4),
            (data_type::int8(), DLDataTypeCode::INT, 8),
            (data_type::int16(), DLDataTypeCode::INT, 16),
            (data_type::int32(), DLDataTypeCode::INT, 32),
            (data_type::int64(), DLDataTypeCode::INT, 64),
            (data_type::uint2(), DLDataTypeCode::UINT, 2),
            (data_type::uint4(), DLDataTypeCode::UINT, 4),
            (data_type::uint8(), DLDataTypeCode::UINT, 8),
            (data_type::uint16(), DLDataTypeCode::UINT, 16),
            (data_type::uint32(), DLDataTypeCode::UINT, 32),
            (data_type::uint64(), DLDataTypeCode::UINT, 64),
            (data_type::float16(), DLDataTypeCode::FLOAT, 16),
            (data_type::float32(), DLDataTypeCode::FLOAT, 32),
            (data_type::float64(), DLDataTypeCode::FLOAT, 64),
            (data_type::bfloat16(), DLDataTypeCode::BFLOAT, 16),
            (data_type::float8_e3m4(), DLDataTypeCode::FLOAT8_E3M4, 8),
            (data_type::float8_e4m3(), DLDataTypeCode::FLOAT8_E4M3, 8),
            (
                data_type::float8_e4m3b11fnuz(),
                DLDataTypeCode::FLOAT8_E4M3B11FNUZ,
                8,
            ),
            (
                data_type::float8_e4m3fnuz(),
                DLDataTypeCode::FLOAT8_E4M3FNUZ,
                8,
            ),
            (data_type::float8_e5m2(), DLDataTypeCode::FLOAT8_E5M2, 8),
            (
                data_type::float8_e5m2fnuz(),
                DLDataTypeCode::FLOAT8_E5M2FNUZ,
                8,
            ),
            (
                data_type::float8_e8m0fnu(),
                DLDataTypeCode::FLOAT8_E8M0FNU,
                8,
            ),
            (data_type::float6_e2m3fn(), DLDataTypeCode::FLOAT6_E2M3FN, 6),
            (data_type::float6_e3m2fn(), DLDataTypeCode::FLOAT6_E3M2FN, 6),
            (data_type::float4_e2m1fn(), DLDataTypeCode::FLOAT4_E2M1FN, 4),
            (data_type::complex_float16(), DLDataTypeCode::COMPLEX, 32),
            (data_type::complex64(), DLDataTypeCode::COMPLEX, 64),
            (data_type::complex_float32(), DLDataTypeCode::COMPLEX, 64),
            (data_type::complex128(), DLDataTypeCode::COMPLEX, 128),
            (data_type::complex_float64(), DLDataTypeCode::COMPLEX, 128),
        ];

        for (data_type, code, bits) in data_types {
            let dtype = super::data_type_to_dlpack(&data_type)
                .unwrap_or_else(|err| panic!("{data_type} should be supported: {err}"));
            assert!(
                dtype.matches(DLDataType::scalar(code, bits)),
                "{data_type} mapped to {dtype:?}"
            );
            // The DLPack element size must match the zarrs in-memory element size
            assert_eq!(data_type.fixed_size(), Some(dtype.element_size()));
        }
    }

    #[test]
    fn array_dlpack_ext_subbyte_flag() {
        // `zarrs` pads sub-byte elements to a byte, so the padded flag must be set
        let tensor = Tensor::new(vec![0u8; 4], data_type::float4_e2m1fn(), vec![4]);
        let dlpack: versioned::Dlpack = Builder::try_from(Box::new(tensor))
            .unwrap()
            .try_build()
            .unwrap();
        assert_eq!(dlpack.flags(), DlpackFlags::IS_SUBBYTE_TYPE_PADDED);

        // Byte-sized data types are neither packed nor padded
        let tensor = Tensor::new(vec![0u8; 4], data_type::float8_e4m3(), vec![4]);
        let dlpack: versioned::Dlpack = Builder::try_from(Box::new(tensor))
            .unwrap()
            .try_build()
            .unwrap();
        assert_eq!(dlpack.flags(), DlpackFlags::empty());
    }

    #[test]
    fn array_dlpack_ext_unsupported_data_type() {
        // `complex_bfloat16` needs a data type code that postdates DLPack 1.3
        let tensor = Tensor::new(vec![0u8; 4], data_type::complex_bfloat16(), vec![1]);
        assert!(Builder::try_from(Box::new(tensor)).is_err());

        // Complex subfloats have no DLPack data type code
        let tensor = Tensor::new(vec![0u8; 2], data_type::complex_float8_e4m3(), vec![1]);
        assert!(Builder::try_from(Box::new(tensor)).is_err());

        // Variable-sized data types cannot be represented
        let tensor = Tensor::new(vec![0u8; 8], data_type::string(), vec![1]);
        assert!(Builder::try_from(Box::new(tensor)).is_err());
    }

    #[test]
    fn array_dlpack_ext_unsupported_shape() {
        // A dimension that does not fit in an i64
        let tensor = Tensor::new(vec![], data_type::uint8(), vec![u64::MAX]);
        assert!(Builder::try_from(Box::new(tensor)).is_err());

        // A shape whose compact strides overflow an i64
        let tensor = Tensor::new(vec![], data_type::uint8(), vec![2, i64::MAX as u64]);
        assert!(Builder::try_from(Box::new(tensor)).is_err());
    }
}
