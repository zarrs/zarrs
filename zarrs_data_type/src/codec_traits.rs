//! Codec data type traits.
//!
//! These traits define how data types interact with various codecs in `zarrs`.
//! Custom data types should implement these traits where applicable.
//!
//! # Convenience Macros
//!
//! The following macros are provided to simplify the implementation of codec traits for data types.
//! Note that some data types may require custom implementations beyond what the macros provide.
//!
//! - [`impl_bitround_codec`] - Implement [`bitround::BitroundDataTypeTraits`]
//! - [`impl_bytes_data_type_traits`] - Implement [`bytes::BytesDataTypeTraits`]
//! - [`impl_fixed_scale_offset_data_type_traits`] - Implement [`fixedscaleoffset::FixedScaleOffsetDataTypeTraits`]
//! - [`impl_pack_bits_data_type_traits`] - Implement [`packbits::PackBitsDataTypeTraits`]
//! - [`impl_pcodec_data_type_traits`] - Implement [`pcodec::PcodecDataTypeTraits`]
//! - [`impl_zfp_data_type_traits`] - Implement [`zfp::ZfpDataTypeTraits`]

pub mod bitround;
pub mod bytes;
pub mod fixedscaleoffset;
pub mod packbits;
pub mod pcodec;
pub mod zfp;

pub use bitround::impl_bitround_codec;
pub use bytes::impl_bytes_data_type_traits;
pub use fixedscaleoffset::impl_fixed_scale_offset_data_type_traits;
pub use packbits::impl_pack_bits_data_type_traits;
pub use pcodec::impl_pcodec_data_type_traits;
pub use zfp::impl_zfp_data_type_traits;
