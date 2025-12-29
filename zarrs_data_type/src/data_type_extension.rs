use std::{any::Any, borrow::Cow, fmt::Debug, sync::Arc};

use zarrs_metadata::{v3::FillValueMetadataV3, Configuration, DataTypeSize};
use zarrs_plugin::{MaybeSend, MaybeSync, ZarrVersions};

use crate::{
    data_type_extension_bitround_codec::DataTypeExtensionBitroundCodec,
    data_type_extension_fixedscaleoffset_codec::DataTypeExtensionFixedScaleOffsetCodec,
    data_type_extension_packbits_codec::DataTypeExtensionPackBitsCodec,
    data_type_extension_pcodec_codec::DataTypeExtensionPcodecCodec,
    data_type_extension_zfp_codec::DataTypeExtensionZfpCodec, DataTypeExtensionBytesCodec,
    DataTypeExtensionBytesCodecError, DataTypeFillValueError, DataTypeFillValueMetadataError,
    FillValue,
};

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
/// The [`DataTypeExtensionBytesCodec`] traits methods allow a fixed-size custom data type to be encoded with the `bytes` codec with a requested endianness.
/// These methods are not invoked for variable-size data types, and can be pass-through for a fixed-size data types that use an explicitly fixed endianness or where endianness is not applicable.
///
/// A custom data type must also directly handle conversion of fill value metadata to fill value bytes, and vice versa.
pub trait DataTypeExtension: Debug + MaybeSend + MaybeSync {
    /// The identifier of the data type.
    fn identifier(&self) -> &'static str;

    /// The name to use when creating metadata for this data type.
    ///
    /// This is used when creating metadata. Most data types return their identifier,
    /// but some (like `RawBitsDataType`) return a version-specific name like `r{bits}`.
    ///
    /// The default implementation returns the identifier.
    ///
    /// Note: This is distinct from `ExtensionIdentifier::default_name()` which operates
    /// on the type level. This method operates on the instance level, allowing for
    /// data types that have instance-specific names (like `r16`, `r32`, etc.).
    fn metadata_name(&self, _zarr_version: ZarrVersions) -> Cow<'static, str> {
        Cow::Borrowed(self.identifier())
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

    /// Return [`DataTypeExtensionBytesCodec`] if the data type supports the `bytes` codec.
    ///
    /// Fixed-size data types are expected to support the `bytes` codec, even if bytes pass through it unmodified.
    ///
    /// The default implementation returns `None`.
    fn codec_bytes(&self) -> Option<&dyn DataTypeExtensionBytesCodec> {
        None
    }

    /// Return [`DataTypeExtensionPackBitsCodec`] if the data type supports the `packbits` codec.
    ///
    /// Types that can be encoded smaller in less than a byte should support the `packbits` codec.
    ///
    /// The default implementation returns `None`.
    fn codec_packbits(&self) -> Option<&dyn DataTypeExtensionPackBitsCodec> {
        None
    }

    /// Return [`DataTypeExtensionBitroundCodec`] if the data type supports the `bitround` codec.
    ///
    /// Integer and floating-point types should support the `bitround` codec.
    ///
    /// The default implementation returns `None`.
    fn codec_bitround(&self) -> Option<&dyn DataTypeExtensionBitroundCodec> {
        None
    }

    /// Return [`DataTypeExtensionPcodecCodec`] if the data type supports the `pcodec` codec.
    ///
    /// 16-bit and larger numeric types should support the `pcodec` codec.
    ///
    /// The default implementation returns `None`.
    fn codec_pcodec(&self) -> Option<&dyn DataTypeExtensionPcodecCodec> {
        None
    }

    /// Return [`DataTypeExtensionFixedScaleOffsetCodec`] if the data type supports the `fixedscaleoffset` codec.
    ///
    /// Numeric types should support the `fixedscaleoffset` codec.
    ///
    /// The default implementation returns `None`.
    fn codec_fixedscaleoffset(&self) -> Option<&dyn DataTypeExtensionFixedScaleOffsetCodec> {
        None
    }

    /// Return [`DataTypeExtensionZfpCodec`] if the data type supports the `zfp` codec.
    ///
    /// 32-bit and 64-bit integer and floating-point types should support the `zfp` codec.
    /// 8-bit and 16-bit types are supported through promotion.
    ///
    /// The default implementation returns `None`.
    fn codec_zfp(&self) -> Option<&dyn DataTypeExtensionZfpCodec> {
        None
    }

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
    /// A `bytes` codec error.
    BytesCodec(#[from] DataTypeExtensionBytesCodecError),
}
