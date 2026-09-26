use zfp_rs::{ZfpField, ZfpFieldMut, ZfpScalar, ZfpScalarType};

use super::ZfpEncoding;
use crate::array::transmute_to_bytes_vec;

/// Create a zfp field from C-order `shape` (zfp dimensions are ordered fastest varying first).
fn new_field<'a, T: ZfpScalar>(data: &'a [T], shape: &[usize]) -> Option<ZfpField<'a>> {
    match *shape {
        [nx] => Some(ZfpField::new(data, [nx])),
        [ny, nx] => Some(ZfpField::new(data, [nx, ny])),
        [nz, ny, nx] => Some(ZfpField::new(data, [nx, ny, nz])),
        [nw, nz, ny, nx] => Some(ZfpField::new(data, [nx, ny, nz, nw])),
        _ => None,
    }
}

/// Create a mutable zfp field from C-order `shape` (zfp dimensions are ordered fastest varying first).
fn new_field_mut<'a, T: ZfpScalar>(data: &'a mut [T], shape: &[usize]) -> Option<ZfpFieldMut<'a>> {
    match *shape {
        [nx] => Some(ZfpFieldMut::new(data, [nx])),
        [ny, nx] => Some(ZfpFieldMut::new(data, [nx, ny])),
        [nz, ny, nx] => Some(ZfpFieldMut::new(data, [nx, ny, nz])),
        [nw, nz, ny, nx] => Some(ZfpFieldMut::new(data, [nx, ny, nz, nw])),
        _ => None,
    }
}

/// A zfp array holding decoded data along with the original encoding.
///
/// This pairs the zfp native array data with the source encoding,
/// enabling type-safe demotion back to the original data type.
#[derive(Debug)]
pub(super) enum ZfpArray {
    /// i8 data stored as promoted i32 values
    Int8(Vec<i32>),
    /// i16 data stored as promoted i32 values
    Int16(Vec<i32>),
    /// Native i32 data
    Int32(Vec<i32>),
    /// Native i64 data
    Int64(Vec<i64>),
    /// u8 data stored as promoted i32 values
    UInt8(Vec<i32>),
    /// u16 data stored as promoted i32 values
    UInt16(Vec<i32>),
    /// u32 data stored as clamped i32 values
    UInt32(Vec<i32>),
    /// u64 data stored as clamped i64 values
    UInt64(Vec<i64>),
    /// Native f32 data
    Float32(Vec<f32>),
    /// Native f64 data
    Float64(Vec<f64>),
}

impl ZfpArray {
    /// Returns the number of elements in the array.
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Int8(v)
            | Self::Int16(v)
            | Self::Int32(v)
            | Self::UInt8(v)
            | Self::UInt16(v)
            | Self::UInt32(v) => v.len(),
            Self::Int64(v) | Self::UInt64(v) => v.len(),
            Self::Float32(v) => v.len(),
            Self::Float64(v) => v.len(),
        }
    }

    /// Creates a new zeroed array for the given encoding and number of elements.
    pub(super) fn new_zeroed(encoding: ZfpEncoding, num_elements: usize) -> Self {
        match encoding {
            ZfpEncoding::Int8 => Self::Int8(vec![0; num_elements]),
            ZfpEncoding::Int16 => Self::Int16(vec![0; num_elements]),
            ZfpEncoding::Int32 => Self::Int32(vec![0; num_elements]),
            ZfpEncoding::Int64 => Self::Int64(vec![0; num_elements]),
            ZfpEncoding::UInt8 => Self::UInt8(vec![0; num_elements]),
            ZfpEncoding::UInt16 => Self::UInt16(vec![0; num_elements]),
            ZfpEncoding::UInt32 => Self::UInt32(vec![0; num_elements]),
            ZfpEncoding::UInt64 => Self::UInt64(vec![0; num_elements]),
            ZfpEncoding::Float32 => Self::Float32(vec![0.0; num_elements]),
            ZfpEncoding::Float64 => Self::Float64(vec![0.0; num_elements]),
        }
    }

    /// Returns the zfp scalar type of the array.
    pub(super) fn scalar_type(&self) -> ZfpScalarType {
        match self {
            Self::Int8(_)
            | Self::Int16(_)
            | Self::Int32(_)
            | Self::UInt8(_)
            | Self::UInt16(_)
            | Self::UInt32(_) => ZfpScalarType::Int32,
            Self::Int64(_) | Self::UInt64(_) => ZfpScalarType::Int64,
            Self::Float32(_) => ZfpScalarType::Float,
            Self::Float64(_) => ZfpScalarType::Double,
        }
    }

    /// Returns a zfp field over the array with the given C-order `shape`.
    ///
    /// Returns [`None`] if the shape is not 1-4 dimensional or does not match the array length.
    pub(super) fn field(&self, shape: &[usize]) -> Option<ZfpField<'_>> {
        if shape.iter().product::<usize>() != self.len() {
            return None;
        }
        match self {
            Self::Int8(v)
            | Self::Int16(v)
            | Self::Int32(v)
            | Self::UInt8(v)
            | Self::UInt16(v)
            | Self::UInt32(v) => new_field(v, shape),
            Self::Int64(v) | Self::UInt64(v) => new_field(v, shape),
            Self::Float32(v) => new_field(v, shape),
            Self::Float64(v) => new_field(v, shape),
        }
    }

    /// Returns a mutable zfp field over the array with the given C-order `shape`.
    ///
    /// Returns [`None`] if the shape is not 1-4 dimensional or does not match the array length.
    pub(super) fn field_mut(&mut self, shape: &[usize]) -> Option<ZfpFieldMut<'_>> {
        if shape.iter().product::<usize>() != self.len() {
            return None;
        }
        match self {
            Self::Int8(v)
            | Self::Int16(v)
            | Self::Int32(v)
            | Self::UInt8(v)
            | Self::UInt16(v)
            | Self::UInt32(v) => new_field_mut(v, shape),
            Self::Int64(v) | Self::UInt64(v) => new_field_mut(v, shape),
            Self::Float32(v) => new_field_mut(v, shape),
            Self::Float64(v) => new_field_mut(v, shape),
        }
    }

    /// Demotes the zfp array back to its original byte representation.
    #[allow(clippy::cast_sign_loss)]
    pub(super) fn into_bytes(self) -> Vec<u8> {
        match self {
            Self::Int32(v) => transmute_to_bytes_vec(v),
            Self::Int64(v) => transmute_to_bytes_vec(v),
            Self::Float32(v) => transmute_to_bytes_vec(v),
            Self::Float64(v) => transmute_to_bytes_vec(v),
            Self::Int8(v) => transmute_to_bytes_vec(
                v.into_iter()
                    .map(|i| i8::try_from((i >> 23).clamp(-0x80, 0x7f)).unwrap())
                    .collect::<Vec<_>>(),
            ),
            Self::UInt8(v) => transmute_to_bytes_vec(
                v.into_iter()
                    .map(|i| u8::try_from(((i >> 23) + 0x80).clamp(0x00, 0xff)).unwrap())
                    .collect::<Vec<_>>(),
            ),
            Self::Int16(v) => transmute_to_bytes_vec(
                v.into_iter()
                    .map(|i| i16::try_from((i >> 15).clamp(-0x8000, 0x7fff)).unwrap())
                    .collect::<Vec<_>>(),
            ),
            Self::UInt16(v) => transmute_to_bytes_vec(
                v.into_iter()
                    .map(|i| u16::try_from(((i >> 15) + 0x8000).clamp(0x0000, 0xffff)).unwrap())
                    .collect::<Vec<_>>(),
            ),
            Self::UInt32(v) => transmute_to_bytes_vec(
                v.into_iter()
                    .map(|i| core::cmp::max(i, 0) as u32)
                    .collect::<Vec<_>>(),
            ),
            Self::UInt64(v) => transmute_to_bytes_vec(
                v.into_iter()
                    .map(|i| core::cmp::max(i, 0) as u64)
                    .collect::<Vec<_>>(),
            ),
        }
    }
}
