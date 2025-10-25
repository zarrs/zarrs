//! Zarr array codec metadata.
//!
//! See <https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#array-metadata-codecs>
//!
//! This module defines versioned structures that map to the `configuration` field of codecs.

/// Registered codec metadata.
/// See <https://github.com/zarr-developers/zarr-extensions>.
mod registered {
    /// `blosc` codec metadata (registered, core).
    pub mod blosc;
    /// `bytes` codec metadata (registered, core).
    pub mod bytes;
    /// `crc32c` codec metadata (registered, ZEP0002).
    pub mod crc32c;
    /// `gzip` codec metadata (registered, core).
    pub mod gzip;
    /// `optional` codec metadata (registered).
    pub mod optional;
    /// `packbits` codec metadata (registered).
    pub mod packbits;
    /// `reshape` codec metadata (draft).
    pub mod reshape;
    /// `sharding` codec metadata (registered, ZEP0002).
    pub mod sharding;
    /// `transpose` codec metadata (registered, core).
    pub mod transpose;
    /// `vlen-bytes` codec metadata (registered).
    pub mod vlen_bytes;
    /// `vlen-utf8` codec metadata (registered).
    pub mod vlen_utf8;
    /// `zfp` codec metadata (registered).
    pub mod zfp;
    /// `zstd` codec metadata (registered).
    pub mod zstd;
}
pub use registered::*;

/// `zarrs` codec metadata.
mod zarrs {
    /// `gdeflate` codec metadata (`zarrs` experimental).
    pub mod gdeflate;
    /// `squeeze` codec metadata (`zarrs` experimental).
    pub mod squeeze;
    /// `vlen` codec metadata (`zarrs` experimental).
    pub mod vlen;
    /// `vlen_v2` codec metadata (`zarrs` experimental).
    pub mod vlen_v2;
}
pub use zarrs::*;

/// `numcodecs` codec metadata.
mod numcodecs {
    /// `adler32` codec metadata (`numcodecs`).
    pub mod adler32;
    /// `bitround` codec metadata (`numcodecs`).
    pub mod bitround;
    /// `bz2` codec metadata (`numcodecs`).
    pub mod bz2;
    /// `fixedscaleoffset` codec metadata (`numcodecs`).
    pub mod fixedscaleoffset;
    /// `fletcher32` codec metadata (`numcodecs`).
    pub mod fletcher32;
    /// `pcodec` codec metadata (`numcodecs`).
    pub mod pcodec;
    /// `shuffle` codec metadata (`numcodecs`).
    pub mod shuffle;
    /// `vlen-array` codec metadata (`numcodecs`).
    pub mod vlen_array;
    /// `zfpy` codec metadata (`numcodecs`).
    pub mod zfpy;
    /// `zlib` codec metadata (`numcodecs`).
    pub mod zlib;
}
pub use numcodecs::*;
