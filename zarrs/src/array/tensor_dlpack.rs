use std::any::TypeId;
use std::ffi::c_void;

use dlpark::ffi::{DLDataType, DLDataTypeCode, DLDevice};
use dlpark::metadata::CopiedSlice;
use dlpark::tensor::compact_strides;
use dlpark::{Builder, DlpackFlags, versioned};

use super::element_layout::{ElementLayout, ElementPacking};
use super::{DataType, Tensor, TensorError};
use crate::array::data_type as dt;

/// Convert a zarrs [`DataType`] and [`ElementLayout`] to a [`DLDataType`] and the [`DlpackFlags`]
/// that the layout implies.
///
/// Data types with fewer than 8 bits are described as `DLPack` sub-byte types. `DLPack` assumes
/// these are packed, so a [`ElementPacking::Padded`] layout is declared with
/// [`DlpackFlags::IS_SUBBYTE_TYPE_PADDED`].
///
/// # Errors
/// Returns [`TensorError::UnsupportedDataType`] if the data type or the layout is not supported.
fn data_type_to_dlpack(
    data_type: &DataType,
    layout: ElementLayout,
) -> Result<(DLDataType, DlpackFlags), TensorError> {
    // TODO: return `TensorError::UnsupportedLayout(layout)` for the layout failures once
    // `ElementLayout` is public
    let unsupported = || TensorError::UnsupportedDataType(data_type.clone());

    // `DLPack` assumes the native endianness
    if !layout.endianness.is_native() {
        return Err(unsupported());
    }

    let type_id = data_type.as_any().type_id();
    // https://github.com/rust-lang/rust/issues/70861 for match?
    let dtype = if type_id == TypeId::of::<dt::BoolDataType>() {
        // `DLPack` fixes the storage size of a bool at 8 bits, by array library convention, so a
        // bool packed at 1 bit per element cannot be described
        if layout.packing == ElementPacking::PackedLsb0 {
            return Err(unsupported());
        }
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
    if data_type.fixed_size() != Some(dtype.element_size()) {
        return Err(unsupported());
    }

    let flags = match layout.packing {
        // `DLPack` assumes sub-byte elements are packed, so padding must be declared
        ElementPacking::Padded if !dtype.bits.is_multiple_of(8) => {
            DlpackFlags::IS_SUBBYTE_TYPE_PADDED
        }
        // Whole-byte elements are neither padded nor packed, and packed sub-byte elements match
        // what `DLPack` assumes, so neither needs a flag
        ElementPacking::Padded | ElementPacking::PackedLsb0 => DlpackFlags::empty(),
    };
    Ok((dtype, flags))
}

/// The number of bytes an exported tensor occupies, including [`ElementLayout::byte_offset`].
///
/// Returns [`None`] if the number of bytes is not representable as a [`usize`].
fn expected_num_bytes(
    num_elements: u64,
    dtype: DLDataType,
    layout: ElementLayout,
) -> Option<usize> {
    let bits = u64::from(dtype.bits);
    let elements = match layout.packing {
        ElementPacking::Padded => num_elements.checked_mul(bits.div_ceil(8))?,
        ElementPacking::PackedLsb0 => num_elements.checked_mul(bits)?.div_ceil(8),
    };
    usize::try_from(layout.byte_offset.checked_add(elements)?).ok()
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

impl Tensor<'static> {
    /// Convert this tensor into a versioned `DLPack` managed tensor.
    ///
    /// The tensor is moved into the managed tensor and keeps its data alive. Only the shape and
    /// strides are copied, not the elements.
    ///
    /// A consumer may hold the data for as long as it likes, so the tensor must be `'static`. Use
    /// [`into_static`](Tensor::into_static) to export borrowed bytes, which copies them.
    ///
    /// # Errors
    ///
    /// Returns an error if the data type, element layout, or shape cannot be represented by
    /// `DLPack`, or if the tensor does not contain enough bytes for its shape and data type.
    ///
    /// # Examples
    /// ```rust
    /// # use zarrs::array::{ArrayBuilder, ArraySubset, Tensor, data_type};
    /// # use zarrs_storage::store::MemoryStore;
    /// # let store = MemoryStore::new();
    /// # let array = ArrayBuilder::new(vec![4, 4], vec![2, 2], data_type::float32(), -1.0f32)
    /// #     .build(store.into(), "/")?;
    /// # array.store_chunk(&[0, 0], &[0.0f32, 1.0, 2.0, 3.0])?;
    /// let subset = ArraySubset::new_with_shape(vec![1, 2]);
    /// let tensor: Tensor<'static> = array.retrieve_chunks(&subset)?;
    /// let dlpack = tensor.into_dlpack()?;
    ///
    /// assert_eq!(dlpack.shape()?, &[2, 4]);
    /// assert_eq!(
    ///     dlpack.cpu_data_slice::<f32>()?,
    ///     &[0.0, 1.0, -1.0, -1.0, 2.0, 3.0, -1.0, -1.0]
    /// );
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// Exporting borrowed bytes without [`into_static`](Tensor::into_static) does not compile:
    /// ```rust,compile_fail
    /// # use zarrs::array::{Tensor, data_type};
    /// let bytes = vec![0u8; 16];
    /// let tensor = Tensor::new(&bytes[..], data_type::float32(), vec![2, 2]);
    /// let dlpack = tensor.into_dlpack()?; // `bytes` does not live for `'static`
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    pub fn into_dlpack(self) -> Result<versioned::Dlpack, TensorError> {
        let unsupported_shape = self.shape().to_vec();
        let tensor = Box::new(self);
        let layout = tensor.layout();
        let (dtype, flags) = data_type_to_dlpack(tensor.data_type(), layout)?;
        let (shape, strides) = shape_and_strides(tensor.shape())?;

        // The exported tensor must not describe bytes beyond the end of the buffer
        let expected = expected_num_bytes(tensor.num_elements(), dtype, layout);
        let actual = tensor.bytes().len();
        if expected.is_none_or(|expected| expected > actual) {
            return Err(TensorError::InsufficientBytes {
                expected: expected.unwrap_or(usize::MAX),
                actual,
            });
        }

        let data = if tensor.bytes().is_empty() {
            std::ptr::null_mut()
        } else {
            tensor.bytes().as_ptr().cast::<c_void>().cast_mut()
        };
        let builder = Builder::new(tensor, CopiedSlice::new(shape, strides));
        // SAFETY: the boxed tensor owns the bytes at `data` and outlives the managed tensor, which
        // needs the `'static` tensor. `dtype`/shape/strides/`byte_offset` describe those bytes as a
        // compact row-major tensor, and there are enough of them for the elements described.
        let builder = unsafe { builder.data(data) }
            .dtype(dtype)
            .device(DLDevice::CPU)
            .byte_offset(layout.byte_offset);
        // SAFETY: `data_type_to_dlpack` only returns layout flags and never `IS_COPIED`.
        unsafe { builder.flags_unchecked(flags) }
            .try_build()
            .map_err(|_| TensorError::UnsupportedShape(unsupported_shape))
    }
}

#[cfg(test)]
mod tests {
    use dlpark::DlpackFlags;
    use zarrs_metadata::Endianness;

    use super::{ElementLayout, ElementPacking};
    use crate::array::data_type;

    #[test]
    fn tensor_dlpack_layout_flags() {
        let packed = ElementLayout {
            packing: ElementPacking::PackedLsb0,
            ..ElementLayout::default()
        };
        let flags = |data_type: &_, layout| {
            super::data_type_to_dlpack(data_type, layout).map(|(_dtype, flags)| flags)
        };

        // A padded sub-byte element does not match what DLPack assumes, so it must be declared
        assert_eq!(
            flags(&data_type::int4(), ElementLayout::default()).unwrap(),
            DlpackFlags::IS_SUBBYTE_TYPE_PADDED
        );
        // A whole-byte element is neither padded nor packed
        assert_eq!(
            flags(&data_type::uint16(), ElementLayout::default()).unwrap(),
            DlpackFlags::empty()
        );
        // A packed sub-byte element matches what DLPack assumes, so it needs no flag
        assert_eq!(
            flags(&data_type::int4(), packed).unwrap(),
            DlpackFlags::empty()
        );
        // DLPack fixes the storage size of a bool at 8 bits, but packbits packs it at 1 bit
        assert_eq!(
            flags(&data_type::bool(), ElementLayout::default()).unwrap(),
            DlpackFlags::empty()
        );
        assert!(flags(&data_type::bool(), packed).is_err());

        // DLPack assumes the native endianness
        let non_native = ElementLayout {
            endianness: match Endianness::native() {
                Endianness::Big => Endianness::Little,
                Endianness::Little => Endianness::Big,
            },
            ..ElementLayout::default()
        };
        assert!(flags(&data_type::uint16(), non_native).is_err());
    }

    #[test]
    fn tensor_dlpack_expected_num_bytes() {
        let (dtype, _flags) =
            super::data_type_to_dlpack(&data_type::int4(), ElementLayout::default()).unwrap();
        let packed = ElementLayout {
            packing: ElementPacking::PackedLsb0,
            ..ElementLayout::default()
        };

        // Eight int4 elements occupy eight padded bytes, or four packed bytes
        assert_eq!(
            super::expected_num_bytes(8, dtype, ElementLayout::default()),
            Some(8)
        );
        assert_eq!(super::expected_num_bytes(8, dtype, packed), Some(4));
        // A partial trailing byte is included
        assert_eq!(super::expected_num_bytes(7, dtype, packed), Some(4));
        // The byte offset precedes the elements
        let offset = ElementLayout {
            byte_offset: 1,
            ..packed
        };
        assert_eq!(super::expected_num_bytes(8, dtype, offset), Some(5));
        // A zero element tensor needs no bytes
        assert_eq!(
            super::expected_num_bytes(0, dtype, ElementLayout::default()),
            Some(0)
        );
        // An unrepresentable number of bytes is rejected
        assert_eq!(super::expected_num_bytes(u64::MAX, dtype, packed), None);
    }
}
