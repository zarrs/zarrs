use std::borrow::Cow;

#[cfg(doc)]
use super::ArrayBytes;

/// Array element bytes.
///
/// These can represent:
/// - [`ArrayBytes::Fixed`]: fixed length elements of an array in C-contiguous order,
/// - [`ArrayBytes::Variable`]: variable length elements of an array in C-contiguous order with padding permitted,
/// - Encoded array bytes after an array to bytes or bytes to bytes codecs.
pub type RawBytes<'a> = Cow<'a, [u8]>;
