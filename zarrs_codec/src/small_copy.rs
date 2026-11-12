//! Helpers for copying many small byte runs.

/// Evaluate `$body` with `$len` bound to `$length`, specialised for common small lengths.
///
/// Small lengths are bound as constants, so copies of `$len` bytes in `$body` compile to single moves rather than calls to `memcpy`.
macro_rules! with_small_len {
    ($length:expr, |$len:ident| $body:expr) => {
        match $length {
            1 => {
                let $len: usize = 1;
                $body
            }
            2 => {
                let $len: usize = 2;
                $body
            }
            4 => {
                let $len: usize = 4;
                $body
            }
            8 => {
                let $len: usize = 8;
                $body
            }
            16 => {
                let $len: usize = 16;
                $body
            }
            $len => $body,
        }
    };
}
pub(crate) use with_small_len;
