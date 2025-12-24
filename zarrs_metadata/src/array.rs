//! Zarr common array metadata.
//!
//! See <https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#array-metadata>.

mod chunk_key_separator;
pub use chunk_key_separator::ChunkKeySeparator;

/// A dimension name.
pub type DimensionName = Option<String>;

mod endianness;
pub use endianness::Endianness;

/// A trait for types convertible to a [`DimensionName`].
pub trait IntoDimensionName {
    /// Convert into a [`DimensionName`].
    fn into_dimension_name(self) -> DimensionName;
}

impl IntoDimensionName for &str {
    fn into_dimension_name(self) -> DimensionName {
        Some(self.to_string())
    }
}

impl IntoDimensionName for Option<&str> {
    fn into_dimension_name(self) -> DimensionName {
        self.map(ToString::to_string)
    }
}

impl IntoDimensionName for String {
    fn into_dimension_name(self) -> DimensionName {
        Some(self)
    }
}

impl IntoDimensionName for Option<String> {
    fn into_dimension_name(self) -> DimensionName {
        self
    }
}
