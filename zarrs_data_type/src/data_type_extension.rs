use std::{any::Any, borrow::Cow, fmt::Debug, sync::Arc};

use zarrs_metadata::{v3::FillValueMetadataV3, Configuration, DataTypeSize};
use zarrs_plugin::{MaybeSend, MaybeSync, ZarrVersions};

use crate::{DataTypeFillValueError, DataTypeFillValueMetadataError, FillValue};

/// A data type.
///
/// This is a type alias for `Arc<dyn DataTypeExtension>`, providing a unified
/// interface for all data types (both built-in and custom extensions).
pub type DataType = Arc<dyn DataTypeExtension>;

/// Traits for a data type extension.
///
/// The in-memory size of a data type can differ between its associated Rust structure and the *serialised* [`ArrayBytes`](https://docs.rs/zarrs/latest/zarrs/array/enum.ArrayBytes.html) passed into the codec pipeline.
/// For example, a Rust struct that has padding bytes can be converted to tightly packed bytes before it is passed into the codec pipeline for encoding, and vice versa for decoding.
///
/// It is recommended to define a concrete structure representing a single element of a custom data type that implements [`Element`](https://docs.rs/zarrs/latest/zarrs/array/trait.Element.html) and [`ElementOwned`](https://docs.rs/zarrs/latest/zarrs/array/trait.ElementOwned.html).
/// These traits have `into_array_bytes` and `from_array_bytes` methods for this purpose that enable custom data types to be used with the [`Array::{store,retrieve}_*_elements`](https://docs.rs/zarrs/latest/zarrs/array/struct.Array.html) variants.
/// These methods should encode data to and from native endianness if endianness is applicable, unless the endianness should be explicitly fixed.
/// Note that codecs that act on numerical data typically expect the data to be in native endianness.
///
/// A custom data type must also directly handle conversion of fill value metadata to fill value bytes, and vice versa.
pub trait DataTypeExtension: Debug + MaybeSend + MaybeSync {
    /// The identifier of the data type.
    fn identifier(&self) -> &'static str;

    /// The name to use when creating metadata for this data type.
    ///
    /// This is used when creating metadata. Most data types return their identifier,
    /// but some (like `RawBitsDataType`) return a version-specific name like `r{bits}`.
    #[allow(unused_variables)]
    fn default_name(&self, zarr_version: ZarrVersions) -> Option<Cow<'static, str>> {
        None
    }

    /// The configuration of the data type.
    fn configuration(&self) -> Configuration;

    /// The size of the data type.
    ///
    /// This size may differ from the size in memory of the data type.
    /// It represents the size of elements passing through array to array and array to bytes codecs in the codec pipeline (i.e., after conversion to [`ArrayBytes`](https://docs.rs/zarrs/latest/zarrs/array/enum.ArrayBytes.html)).
    fn size(&self) -> DataTypeSize;

    /// Create a fill value from metadata.
    ///
    /// # Errors
    /// Returns [`DataTypeFillValueMetadataError`] if the fill value is incompatible with the data type.
    fn fill_value(
        &self,
        fill_value_metadata: &FillValueMetadataV3,
    ) -> Result<FillValue, DataTypeFillValueMetadataError>;

    /// Create fill value metadata.
    ///
    /// # Errors
    /// Returns an [`DataTypeFillValueError`] if the metadata cannot be created from the fill value.
    fn metadata_fill_value(
        &self,
        fill_value: &FillValue,
    ) -> Result<FillValueMetadataV3, DataTypeFillValueError>;

    /// Compare this data type with another for equality.
    ///
    /// The default implementation compares identifier and configuration.
    /// Custom data types may override this for more efficient comparison.
    fn data_type_eq(&self, other: &dyn DataTypeExtension) -> bool {
        self.identifier() == other.identifier() && self.configuration() == other.configuration()
    }

    /// Returns self as `Any` for downcasting.
    ///
    /// This enables accessing concrete type-specific methods (like `OptionalDataType::data_type()`).
    fn as_any(&self) -> &dyn Any;
}

/// A data type extension error.
#[derive(Debug, Clone, thiserror::Error, derive_more::Display)]
#[non_exhaustive]
pub enum DataTypeExtensionError {
    /// Codec not supported
    #[display("The {codec} codec is not supported by the {data_type} extension data type")]
    CodecUnsupported {
        /// The data type name.
        data_type: String,
        /// The codec name.
        codec: String,
    },
}
