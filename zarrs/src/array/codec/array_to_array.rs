//! Array to array codecs.

#[cfg(feature = "bitround")]
pub mod bitround;
pub mod cast_value;
pub mod fixedscaleoffset;
pub mod reshape;
pub mod scale_offset;
pub mod squeeze;
#[cfg(feature = "transpose")]
pub mod transpose;
