//! The `bytes` codec data type traits.

use std::borrow::Cow;

use zarrs_metadata::Endianness;

/// Error indicating the bytes codec requires endianness to be specified.
#[derive(Debug, Clone, Copy, thiserror::Error)]
#[error("endianness must be specified for multi-byte data types")]
pub struct BytesCodecEndiannessMissingError;

/// Traits for a data type supporting the `bytes` codec.
pub trait BytesDataTypeTraits {
    /// Encode the bytes of a fixed-size data type to a specified endianness for the `bytes` codec.
    ///
    /// Returns the input bytes unmodified for fixed-size data where endianness is not applicable
    /// (i.e. the bytes are serialised directly from the in-memory representation).
    ///
    /// # Errors
    /// Returns a [`BytesCodecEndiannessMissingError`] if `endianness` is [`None`] but must be specified.
    #[allow(unused_variables)]
    fn encode<'a>(
        &self,
        bytes: Cow<'a, [u8]>,
        endianness: Option<Endianness>,
    ) -> Result<Cow<'a, [u8]>, BytesCodecEndiannessMissingError>;

    /// Decode the bytes of a fixed-size data type from a specified endianness for the `bytes` codec.
    ///
    /// This performs the inverse operation of [`encode`](BytesDataTypeTraits::encode).
    ///
    /// # Errors
    /// Returns a [`BytesCodecEndiannessMissingError`] if `endianness` is [`None`] but must be specified.
    #[allow(unused_variables)]
    fn decode<'a>(
        &self,
        bytes: Cow<'a, [u8]>,
        endianness: Option<Endianness>,
    ) -> Result<Cow<'a, [u8]>, BytesCodecEndiannessMissingError>;
}

// Generate the codec support infrastructure using the generic macro
crate::define_data_type_support!(Bytes);

/// Macro to implement `BytesDataTypeTraits` for data types and register support.
///
/// The second parameter is the component size in bytes. Use `1` for single-byte types
/// (passthrough, no endianness conversion) or a larger value for multi-byte types
/// (endianness handling via byte reversal).
///
/// # Usage
/// ```ignore
/// // Single-byte types (passthrough)
/// zarrs_data_type::impl_bytes_data_type_traits!(BoolDataType, 1);
/// zarrs_data_type::impl_bytes_data_type_traits!(UInt4DataType, 1);
///
/// // Multi-byte types (endianness handling)
/// zarrs_data_type::impl_bytes_data_type_traits!(NumpyDateTime64DataType, 8);
///
/// // Const expressions also work
/// zarrs_data_type::impl_bytes_data_type_traits!(ComplexFloat32DataType, { 8 / 2 });
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! _impl_bytes_data_type_traits {
    ($marker:ty, 1) => {
        // Passthrough for single-byte components (no endianness conversion needed)
        impl $crate::codec_traits::bytes::BytesDataTypeTraits for $marker {
            fn encode<'a>(
                &self,
                bytes: ::std::borrow::Cow<'a, [u8]>,
                _endianness: Option<::zarrs_metadata::Endianness>,
            ) -> Result<
                ::std::borrow::Cow<'a, [u8]>,
                $crate::codec_traits::bytes::BytesCodecEndiannessMissingError,
            > {
                Ok(bytes)
            }

            fn decode<'a>(
                &self,
                bytes: ::std::borrow::Cow<'a, [u8]>,
                _endianness: Option<::zarrs_metadata::Endianness>,
            ) -> Result<
                ::std::borrow::Cow<'a, [u8]>,
                $crate::codec_traits::bytes::BytesCodecEndiannessMissingError,
            > {
                Ok(bytes)
            }
        }
        $crate::register_data_type_extension_codec!(
            $marker,
            $crate::codec_traits::bytes::BytesDataTypePlugin,
            $crate::codec_traits::bytes::BytesDataTypeTraits
        );
    };
    ($marker:ty, $component_size:tt) => {
        // Multi-byte components need endianness handling
        impl $crate::codec_traits::bytes::BytesDataTypeTraits for $marker {
            fn encode<'a>(
                &self,
                bytes: ::std::borrow::Cow<'a, [u8]>,
                endianness: Option<::zarrs_metadata::Endianness>,
            ) -> Result<
                ::std::borrow::Cow<'a, [u8]>,
                $crate::codec_traits::bytes::BytesCodecEndiannessMissingError,
            > {
                const COMPONENT_SIZE: usize = $component_size;
                let endianness = endianness
                    .ok_or($crate::codec_traits::bytes::BytesCodecEndiannessMissingError)?;
                if endianness == ::zarrs_metadata::Endianness::native() {
                    Ok(bytes)
                } else {
                    let mut result = bytes.into_owned();
                    for chunk in result.as_chunks_mut::<COMPONENT_SIZE>().0 {
                        chunk.reverse();
                    }
                    Ok(::std::borrow::Cow::Owned(result))
                }
            }

            fn decode<'a>(
                &self,
                bytes: ::std::borrow::Cow<'a, [u8]>,
                endianness: Option<::zarrs_metadata::Endianness>,
            ) -> Result<
                ::std::borrow::Cow<'a, [u8]>,
                $crate::codec_traits::bytes::BytesCodecEndiannessMissingError,
            > {
                self.encode(bytes, endianness)
            }
        }
        $crate::register_data_type_extension_codec!(
            $marker,
            $crate::codec_traits::bytes::BytesDataTypePlugin,
            $crate::codec_traits::bytes::BytesDataTypeTraits
        );
    };
}

#[doc(inline)]
pub use _impl_bytes_data_type_traits as impl_bytes_data_type_traits;
