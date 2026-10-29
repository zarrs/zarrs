//! Array to array codecs.

#[cfg(feature = "bitround")]
pub mod bitround;
pub mod cast_value;
pub mod fixedscaleoffset;
pub mod numcodecs_packbits;
pub mod reshape;
pub mod squeeze;
#[cfg(feature = "transpose")]
pub mod transpose;
