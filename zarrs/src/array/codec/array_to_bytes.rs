//! Array to bytes codecs.

pub mod bytes;
pub mod codec_chain;
pub mod optional;
pub mod packbits;
pub mod vlen;
pub mod vlen_array;
pub mod vlen_bytes;
pub mod vlen_utf8;
pub mod vlen_v2;

#[cfg(feature = "pcodec")]
pub mod pcodec;
#[cfg(feature = "sharding")]
pub mod sharding;
#[cfg(feature = "zfp")]
pub mod zfp;
#[cfg(feature = "zfp")]
pub mod zfpy;
