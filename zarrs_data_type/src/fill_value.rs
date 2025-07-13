//! Zarr fill values.
//!
//! See <https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#fill-value>.

use thiserror::Error;
use zarrs_metadata::v3::FillValueMetadataV3;

/// A data type and fill value metadata incompatibility error.
#[derive(Clone, Debug, Error)]
#[error("incompatible fill value {} for data type {}", _1.to_string(), _0.to_string())]
pub struct DataTypeFillValueMetadataError(String, FillValueMetadataV3);

impl DataTypeFillValueMetadataError {
    /// Create a new [`DataTypeFillValueMetadataError`].
    #[must_use]
    pub fn new(data_type: String, fill_value_metadata: FillValueMetadataV3) -> Self {
        Self(data_type, fill_value_metadata)
    }
}

/// A data type and fill value incompatibility error.
#[derive(Clone, Debug, Error)]
#[error("incompatible fill value {1} for data type {0}")]
pub struct DataTypeFillValueError(String, FillValue);

impl DataTypeFillValueError {
    /// Create a new incompatible fill value error.
    #[must_use]
    pub const fn new(data_type_name: String, fill_value: FillValue) -> Self {
        Self(data_type_name, fill_value)
    }
}

/// A fill value.
///
/// Provides an element value to use for uninitialised portions of the Zarr array.
#[derive(Clone, Eq, PartialEq, Debug)]
pub struct FillValue(Vec<u8>);

impl core::fmt::Display for FillValue {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl From<&[u8]> for FillValue {
    fn from(value: &[u8]) -> Self {
        Self(value.to_vec())
    }
}

impl<const N: usize> From<[u8; N]> for FillValue {
    fn from(value: [u8; N]) -> Self {
        Self(value.to_vec())
    }
}

impl<const N: usize> From<&[u8; N]> for FillValue {
    fn from(value: &[u8; N]) -> Self {
        Self(value.to_vec())
    }
}

impl From<Vec<u8>> for FillValue {
    fn from(value: Vec<u8>) -> Self {
        Self(value)
    }
}

impl From<bool> for FillValue {
    fn from(value: bool) -> Self {
        Self(vec![u8::from(value)])
    }
}

impl From<u8> for FillValue {
    fn from(value: u8) -> Self {
        Self(value.to_ne_bytes().to_vec())
    }
}

impl From<u16> for FillValue {
    fn from(value: u16) -> Self {
        Self(value.to_ne_bytes().to_vec())
    }
}

impl From<u32> for FillValue {
    fn from(value: u32) -> Self {
        Self(value.to_ne_bytes().to_vec())
    }
}

impl From<u64> for FillValue {
    fn from(value: u64) -> Self {
        Self(value.to_ne_bytes().to_vec())
    }
}

impl From<i8> for FillValue {
    fn from(value: i8) -> Self {
        Self(value.to_ne_bytes().to_vec())
    }
}

impl From<i16> for FillValue {
    fn from(value: i16) -> Self {
        Self(value.to_ne_bytes().to_vec())
    }
}

impl From<i32> for FillValue {
    fn from(value: i32) -> Self {
        Self(value.to_ne_bytes().to_vec())
    }
}

impl From<i64> for FillValue {
    fn from(value: i64) -> Self {
        Self(value.to_ne_bytes().to_vec())
    }
}

impl From<half::f16> for FillValue {
    fn from(value: half::f16) -> Self {
        Self(value.to_ne_bytes().to_vec())
    }
}

impl From<half::bf16> for FillValue {
    fn from(value: half::bf16) -> Self {
        Self(value.to_ne_bytes().to_vec())
    }
}

impl From<f32> for FillValue {
    fn from(value: f32) -> Self {
        Self(value.to_ne_bytes().to_vec())
    }
}

impl From<f64> for FillValue {
    fn from(value: f64) -> Self {
        Self(value.to_ne_bytes().to_vec())
    }
}

impl<T> From<num::complex::Complex<T>> for FillValue
where
    FillValue: From<T>,
{
    fn from(value: num::complex::Complex<T>) -> Self {
        let re = FillValue::from(value.re);
        let im = FillValue::from(value.im);
        let mut bytes = Vec::with_capacity(size_of::<num::complex::Complex<T>>());
        bytes.extend(re.as_ne_bytes());
        bytes.extend(im.as_ne_bytes());
        Self(bytes)
    }
}

impl From<String> for FillValue {
    fn from(value: String) -> Self {
        Self(value.into_bytes())
    }
}

impl From<&str> for FillValue {
    fn from(value: &str) -> Self {
        Self(value.as_bytes().to_vec())
    }
}

impl FillValue {
    /// Create a new fill value composed of `bytes`.
    #[must_use]
    pub fn new(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }

    /// Returns the size in bytes of the fill value.
    #[must_use]
    pub fn size(&self) -> usize {
        self.0.len()
    }

    /// Return the byte representation of the fill value.
    #[must_use]
    pub fn as_ne_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Check if the bytes are equal to a sequence of the fill value.
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn equals_all(&self, bytes: &[u8]) -> bool {
        // Special cases for variable length data
        if !num::Integer::is_multiple_of(&bytes.len(), &self.0.len()) || bytes.len() < self.0.len()
        {
            return false;
        }

        match self.0.len() {
            0 => bytes.is_empty(),
            1 => {
                let fill_value = self.0[0];
                let fill_value_128 = u128::from_ne_bytes([self.0[0]; 16]);
                let (prefix, aligned, suffix) = unsafe { bytes.align_to::<u128>() };
                prefix.iter().all(|x| x == &fill_value)
                    && suffix.iter().all(|x| x == &fill_value)
                    && aligned.iter().all(|x| x == &fill_value_128)
            }
            2 => {
                let (prefix, aligned, suffix) = unsafe { bytes.align_to::<u128>() };
                if prefix.is_empty() && suffix.is_empty() {
                    let fill_value_128 =
                        u128::from_ne_bytes(self.0[..2].repeat(8).try_into().unwrap());
                    aligned.iter().all(|x| x == &fill_value_128)
                } else {
                    bytes.chunks_exact(2).all(|x| x == self.0)
                }
            }
            4 => {
                let (prefix, aligned, suffix) = unsafe { bytes.align_to::<u128>() };
                if prefix.is_empty() && suffix.is_empty() {
                    let fill_value_128 =
                        u128::from_ne_bytes(self.0[..4].repeat(4).try_into().unwrap());
                    aligned.iter().all(|x| x == &fill_value_128)
                } else {
                    bytes.chunks_exact(4).all(|x| x == self.0)
                }
            }
            8 => {
                let (prefix, aligned, suffix) = unsafe { bytes.align_to::<u128>() };
                if prefix.is_empty() && suffix.is_empty() {
                    let fill_value_128 =
                        u128::from_ne_bytes(self.0[..8].repeat(2).try_into().unwrap());
                    aligned.iter().all(|x| x == &fill_value_128)
                } else {
                    bytes.chunks_exact(8).all(|x| x == self.0)
                }
            }
            16 => {
                let (prefix, aligned, suffix) = unsafe { bytes.align_to::<u128>() };
                if prefix.is_empty() && suffix.is_empty() {
                    let fill_value_128 = u128::from_ne_bytes(self.0[..16].try_into().unwrap());
                    aligned.iter().all(|x| x == &fill_value_128)
                } else {
                    bytes.chunks_exact(16).all(|x| x == self.0)
                }
            }
            _ => bytes
                .chunks_exact(self.0.len())
                .all(|element| element == self.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Convert from `&[T]` to `Vec<u8>`.
    #[must_use]
    fn convert_to_bytes_vec<T: bytemuck::NoUninit>(from: &[T]) -> Vec<u8> {
        bytemuck::allocation::pod_collect_to_vec(from)
    }

    /// Transmute from `Vec<T>` to `Vec<u8>`.
    #[must_use]
    fn transmute_to_bytes_vec<T: bytemuck::NoUninit>(from: Vec<T>) -> Vec<u8> {
        bytemuck::allocation::try_cast_vec(from)
            .unwrap_or_else(|(_err, from)| convert_to_bytes_vec(&from))
    }

    #[test]
    fn fill_value() {
        assert_eq!(
            FillValue::from([0u8, 1u8, 2u8].as_slice()).as_ne_bytes(),
            &[0u8, 1u8, 2u8]
        );
        assert_eq!(
            FillValue::from(vec![0u8, 1u8, 2u8]).as_ne_bytes(),
            &[0u8, 1u8, 2u8]
        );
        assert_eq!(
            FillValue::new(vec![0u8, 1u8, 2u8]).as_ne_bytes(),
            &[0u8, 1u8, 2u8]
        );
        assert_eq!(FillValue::from(false).as_ne_bytes(), &[0u8]);
        assert_eq!(FillValue::from(true).as_ne_bytes(), &[1u8]);
        assert_eq!(FillValue::from(1u8).as_ne_bytes(), 1u8.to_ne_bytes());
        assert_eq!(FillValue::from(1u16).as_ne_bytes(), 1u16.to_ne_bytes());
        assert_eq!(FillValue::from(1u32).as_ne_bytes(), 1u32.to_ne_bytes());
        assert_eq!(FillValue::from(1u64).as_ne_bytes(), 1u64.to_ne_bytes());
        assert_eq!(FillValue::from(1i8).as_ne_bytes(), 1i8.to_ne_bytes());
        assert_eq!(FillValue::from(1i16).as_ne_bytes(), 1i16.to_ne_bytes());
        assert_eq!(FillValue::from(1i32).as_ne_bytes(), 1i32.to_ne_bytes());
        assert_eq!(FillValue::from(1i64).as_ne_bytes(), 1i64.to_ne_bytes());
        assert_eq!(
            FillValue::from(half::f16::from_f32_const(1.0)).as_ne_bytes(),
            half::f16::from_f32_const(1.0).to_ne_bytes()
        );
        assert_eq!(
            FillValue::from(half::bf16::from_f32_const(1.0)).as_ne_bytes(),
            half::bf16::from_f32_const(1.0).to_ne_bytes()
        );
        assert_eq!(
            FillValue::from(1.0_f32).as_ne_bytes(),
            1.0_f32.to_ne_bytes()
        );
        assert_eq!(
            FillValue::from(1.0_f64).as_ne_bytes(),
            1.0_f64.to_ne_bytes()
        );
        assert_eq!(
            FillValue::from(num::complex::Complex32::new(1.0, 2.0)).as_ne_bytes(),
            [1.0_f32.to_ne_bytes(), 2.0_f32.to_ne_bytes()].concat()
        );
        assert_eq!(
            FillValue::from(num::complex::Complex64::new(1.0, 2.0)).as_ne_bytes(),
            [1.0_f64.to_ne_bytes(), 2.0_f64.to_ne_bytes()].concat()
        );
        assert_eq!(FillValue::from("test").as_ne_bytes(), "test".as_bytes());
        assert_eq!(
            FillValue::from("test".to_string()).as_ne_bytes(),
            "test".as_bytes()
        );
    }

    #[test]
    fn fill_value_equals_u8() {
        assert!(FillValue::from(vec![1u8; 32]).equals_all(&vec![1u8; 32 * 5]));
    }

    #[test]
    fn fill_value_equals_u16() {
        assert!(FillValue::from(1u16).equals_all(&transmute_to_bytes_vec(vec![1u16; 5])));
        assert!(!FillValue::from(1u16).equals_all(&transmute_to_bytes_vec(vec![0u16; 5])));
    }

    #[test]
    fn fill_value_equals_u32() {
        assert!(FillValue::from(1u32).equals_all(&transmute_to_bytes_vec(vec![1u32; 5])));
        assert!(!FillValue::from(1u32).equals_all(&transmute_to_bytes_vec(vec![0u32; 5])));
    }

    #[test]
    fn fill_value_equals_u64() {
        assert!(FillValue::from(1u64).equals_all(&transmute_to_bytes_vec(vec![1u64; 5])));
        assert!(!FillValue::from(1u64).equals_all(&transmute_to_bytes_vec(vec![0u64; 5])));
    }

    #[test]
    fn fill_value_equals_u128() {
        assert!(FillValue::from(vec![1u8; 16]).equals_all(&transmute_to_bytes_vec(vec![1u8; 16])));
    }

    #[test]
    fn fill_value_equals_complex32() {
        assert!(
            FillValue::from(num::complex::Complex32::new(1.0, 2.0)).equals_all(
                &transmute_to_bytes_vec(
                    FillValue::from(num::complex::Complex32::new(1.0, 2.0))
                        .as_ne_bytes()
                        .repeat(5)
                )
            )
        );
    }

    #[test]
    fn fill_value_equals_complex64() {
        assert!(
            FillValue::from(num::complex::Complex64::new(1.0, 2.0)).equals_all(
                &transmute_to_bytes_vec(
                    FillValue::from(num::complex::Complex64::new(1.0, 2.0))
                        .as_ne_bytes()
                        .repeat(5)
                )
            )
        );
    }
}
