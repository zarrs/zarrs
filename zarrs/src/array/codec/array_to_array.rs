//! Array to array codecs.

mod subchunk_forwarding;

#[cfg(feature = "bitround")]
pub mod bitround;
pub mod cast_value;
pub mod fixedscaleoffset;
pub mod reshape;
pub mod squeeze;
#[cfg(feature = "transpose")]
pub mod transpose;
