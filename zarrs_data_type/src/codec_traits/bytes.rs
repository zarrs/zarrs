//! The `bytes` codec data type traits.

use cowbytes::CowBytes;
use zarrs_metadata::Endianness;

/// Error indicating the bytes codec requires endianness to be specified.
#[derive(Debug, Clone, Copy, thiserror::Error)]
#[error("endianness must be specified for multi-byte data types")]
pub struct BytesCodecEndiannessMissingError;

/// An error decoding the bytes of a fixed-size data type in place for the `bytes` codec.
#[derive(Debug, Clone, Copy, thiserror::Error)]
pub enum BytesCodecDecodeInPlaceError {
    /// Endianness must be specified for multi-byte data types.
    #[error(transparent)]
    EndiannessMissing(#[from] BytesCodecEndiannessMissingError),
    /// Decoding changed the length of the bytes, which in-place decoding cannot represent.
    #[error("decoding {len} bytes in place produced {decoded_len} bytes")]
    LengthChanged {
        /// The length of the bytes before decoding.
        len: usize,
        /// The length of the bytes after decoding.
        decoded_len: usize,
    },
}

/// Traits for a data type supporting the `bytes` codec.
pub trait BytesDataTypeTraits {
    /// Returns whether decoding with `endianness` is the identity on the encoded bytes.
    ///
    /// An implementation returning `true` guarantees that, for every valid input, the decoded
    /// bytes are the encoded bytes unchanged (i.e. already in native in-memory byte order), with
    /// no validation or transformation of values.
    /// The `bytes` codec may then skip [`decode`](Self::decode) and use its input directly.
    ///
    /// The default implementation is conservative and returns `false`.
    #[allow(unused_variables)]
    fn is_decode_passthrough(&self, endianness: Option<Endianness>) -> bool {
        false
    }

    /// Encode the bytes of a fixed-size data type to a specified endianness for the `bytes` codec.
    ///
    /// Returns the input bytes unmodified for fixed-size data where endianness is not applicable
    /// (i.e. the bytes are serialised directly from the in-memory representation).
    ///
    /// # Errors
    /// Returns a [`BytesCodecEndiannessMissingError`] if `endianness` is [`None`] but must be specified.
    fn encode<'a>(
        &self,
        bytes: CowBytes<'a>,
        endianness: Option<Endianness>,
    ) -> Result<CowBytes<'a>, BytesCodecEndiannessMissingError>;

    /// Decode the bytes of a fixed-size data type from a specified endianness for the `bytes` codec.
    ///
    /// This performs the inverse operation of [`encode`](BytesDataTypeTraits::encode).
    ///
    /// # Errors
    /// Returns a [`BytesCodecEndiannessMissingError`] if `endianness` is [`None`] but must be specified.
    fn decode<'a>(
        &self,
        bytes: CowBytes<'a>,
        endianness: Option<Endianness>,
    ) -> Result<CowBytes<'a>, BytesCodecEndiannessMissingError>;

    /// Returns whether [`decode_in_place`](Self::decode_in_place) transforms bytes without allocating.
    ///
    /// The `bytes` codec uses this to decode in place only where it is cheaper than [`decode`](Self::decode).
    /// The default implementation of [`decode_in_place`](Self::decode_in_place) is correct but allocates, so the default is `false`.
    fn is_decode_in_place_efficient(&self) -> bool {
        false
    }

    /// Decode the bytes of a fixed-size data type from a specified endianness in place.
    ///
    /// This is equivalent to [`decode`](BytesDataTypeTraits::decode) but transforms `bytes` in place.
    /// The decoded bytes must be the same length as `bytes`.
    ///
    /// The default implementation decodes a copy of `bytes`, so it allocates.
    /// Override this and [`is_decode_in_place_efficient`](Self::is_decode_in_place_efficient) to decode without allocating.
    ///
    /// An override must check `endianness` before it inspects `bytes`, so that it returns the same error for empty `bytes` as for any other.
    /// The `bytes` codec decodes empty bytes before a decoder writes to its output so that an invalid `endianness` is an error that leaves the output unwritten.
    ///
    /// # Errors
    /// Returns a [`BytesCodecDecodeInPlaceError`] if `endianness` is [`None`] but must be specified, or (default implementation) if [`decode`](BytesDataTypeTraits::decode) does not preserve the length of `bytes`.
    fn decode_in_place(
        &self,
        bytes: &mut [u8],
        endianness: Option<Endianness>,
    ) -> Result<(), BytesCodecDecodeInPlaceError> {
        let decoded = self.decode(CowBytes::from(bytes.to_vec()), endianness)?;
        if decoded.len() != bytes.len() {
            return Err(BytesCodecDecodeInPlaceError::LengthChanged {
                len: bytes.len(),
                decoded_len: decoded.len(),
            });
        }
        bytes.copy_from_slice(&decoded);
        Ok(())
    }
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
            fn is_decode_passthrough(
                &self,
                _endianness: Option<::zarrs_metadata::Endianness>,
            ) -> bool {
                true
            }

            fn encode<'a>(
                &self,
                bytes: $crate::CowBytes<'a>,
                _endianness: Option<::zarrs_metadata::Endianness>,
            ) -> Result<
                $crate::CowBytes<'a>,
                $crate::codec_traits::bytes::BytesCodecEndiannessMissingError,
            > {
                Ok(bytes)
            }

            fn decode<'a>(
                &self,
                bytes: $crate::CowBytes<'a>,
                _endianness: Option<::zarrs_metadata::Endianness>,
            ) -> Result<
                $crate::CowBytes<'a>,
                $crate::codec_traits::bytes::BytesCodecEndiannessMissingError,
            > {
                Ok(bytes)
            }

            fn is_decode_in_place_efficient(&self) -> bool {
                true
            }

            fn decode_in_place(
                &self,
                _bytes: &mut [u8],
                _endianness: Option<::zarrs_metadata::Endianness>,
            ) -> Result<(), $crate::codec_traits::bytes::BytesCodecDecodeInPlaceError> {
                Ok(())
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
            fn is_decode_passthrough(
                &self,
                endianness: Option<::zarrs_metadata::Endianness>,
            ) -> bool {
                endianness.is_some_and(::zarrs_metadata::Endianness::is_native)
            }

            fn encode<'a>(
                &self,
                bytes: $crate::CowBytes<'a>,
                endianness: Option<::zarrs_metadata::Endianness>,
            ) -> Result<
                $crate::CowBytes<'a>,
                $crate::codec_traits::bytes::BytesCodecEndiannessMissingError,
            > {
                const COMPONENT_SIZE: usize = $component_size;
                let endianness = endianness
                    .ok_or($crate::codec_traits::bytes::BytesCodecEndiannessMissingError)?;
                if endianness == ::zarrs_metadata::Endianness::native() {
                    Ok(bytes)
                } else {
                    let mut result = bytes.into_vec();
                    for chunk in result.as_chunks_mut::<COMPONENT_SIZE>().0 {
                        chunk.reverse();
                    }
                    Ok($crate::CowBytes::from(result))
                }
            }

            fn decode<'a>(
                &self,
                bytes: $crate::CowBytes<'a>,
                endianness: Option<::zarrs_metadata::Endianness>,
            ) -> Result<
                $crate::CowBytes<'a>,
                $crate::codec_traits::bytes::BytesCodecEndiannessMissingError,
            > {
                self.encode(bytes, endianness)
            }

            fn is_decode_in_place_efficient(&self) -> bool {
                true
            }

            fn decode_in_place(
                &self,
                bytes: &mut [u8],
                endianness: Option<::zarrs_metadata::Endianness>,
            ) -> Result<(), $crate::codec_traits::bytes::BytesCodecDecodeInPlaceError> {
                const COMPONENT_SIZE: usize = $component_size;
                let endianness = endianness
                    .ok_or($crate::codec_traits::bytes::BytesCodecEndiannessMissingError)?;
                if endianness != ::zarrs_metadata::Endianness::native() {
                    for chunk in bytes.as_chunks_mut::<COMPONENT_SIZE>().0 {
                        chunk.reverse();
                    }
                }
                Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A data type with a `decode` that drops the last byte.
    struct LengthChanging;

    impl BytesDataTypeTraits for LengthChanging {
        fn encode<'a>(
            &self,
            bytes: CowBytes<'a>,
            _endianness: Option<Endianness>,
        ) -> Result<CowBytes<'a>, BytesCodecEndiannessMissingError> {
            Ok(bytes)
        }

        fn decode<'a>(
            &self,
            bytes: CowBytes<'a>,
            _endianness: Option<Endianness>,
        ) -> Result<CowBytes<'a>, BytesCodecEndiannessMissingError> {
            Ok(CowBytes::from(bytes[..bytes.len() - 1].to_vec()))
        }
    }

    #[test]
    fn default_decode_in_place_length_changed() {
        let mut bytes = [1, 2, 3, 4];
        assert!(!LengthChanging.is_decode_in_place_efficient());
        assert!(matches!(
            LengthChanging.decode_in_place(&mut bytes, None),
            Err(BytesCodecDecodeInPlaceError::LengthChanged {
                len: 4,
                decoded_len: 3
            })
        ));
        assert_eq!(bytes, [1, 2, 3, 4]);
    }
}
