//! Codec data type traits.
//!
//! These traits define how data types interact with various codecs in `zarrs`.
//! Custom data types should implement these traits where applicable.
//!
//! # Core Traits
//!
//! - [`BytesDataTypeTraits`] - For the `bytes` codec (endianness handling)
//! - [`PackBitsDataTypeTraits`] - For the `packbits` codec (bit packing)
//! - [`BitroundDataTypeTraits`] - For the `bitround` codec (mantissa rounding)
//! - [`FixedScaleOffsetDataTypeTraits`] - For the `fixedscaleoffset` codec
//! - [`PcodecDataTypeTraits`] - For the `pcodec` codec
//! - [`ZfpDataTypeTraits`] - For the `zfp` codec
//!
//! # Convenience Macros
//!
//! - [`impl_bytes_data_type_traits`] - Implement `BytesDataTypeTraits`
//! - [`impl_pack_bits_data_type_traits`] - Implement `PackBitsDataTypeTraits`
//! - [`impl_fixed_scale_offset_data_type_traits`] - Implement `FixedScaleOffsetDataTypeTraits`
//! - [`impl_pcodec_data_type_traits`] - Implement `PcodecDataTypeTraits`
//! - [`impl_zfp_data_type_traits`] - Implement `ZfpDataTypeTraits`

pub mod bitround;
pub mod bytes;
pub mod fixedscaleoffset;
pub mod packbits;
pub mod pcodec;
pub mod zfp;

pub use bitround::*;
pub use bytes::*;
pub use fixedscaleoffset::*;
pub use packbits::*;
pub use pcodec::*;
pub use zfp::*;
