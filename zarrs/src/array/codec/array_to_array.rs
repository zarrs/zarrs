//! Array to array codecs.

#[cfg(feature = "bitround")]
pub mod bitround;
pub mod fixedscaleoffset;
pub mod reshape;
pub mod squeeze;
#[cfg(feature = "transpose")]
pub mod transpose;
