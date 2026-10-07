//! Helpers for testing the `decode_into` of array to bytes codecs.

use std::num::NonZeroU64;
use std::sync::Arc;

use unsafe_cell_slice::UnsafeCellSlice;
use zarrs_codec::{ArrayToBytesCodecTraits, CodecError, CodecOptions};

use crate::array::{ArrayBytesFixedDisjointView, ArraySubset, CowBytes};

/// Decode `encoded` for an array with the shape `chunk_shape` with `decode_into` into a view.
///
/// The view has the shape `view_shape` and is at the origin of an array of 4 rows and `array_columns` columns.
/// The array starts as `0xFF` so that bytes that are not written are detected.
/// The array is `offset` bytes into its allocation, which determines its alignment.
///
/// Returns the bytes of the array, which are written to partially if decoding fails, and the result of decoding.
pub(crate) fn decode_into_view_array(
    codec: &Arc<dyn ArrayToBytesCodecTraits>,
    encoded: &CowBytes<'_>,
    chunk_shape: &[NonZeroU64],
    view_shape: [u64; 2],
    array_columns: usize,
    offset: usize,
    element_size: usize,
) -> (Vec<u8>, Result<(), CodecError>) {
    let array_shape = [4, u64::try_from(array_columns).unwrap()];
    let mut allocation = vec![0xFFu8; offset + 4 * array_columns * element_size];
    let result = {
        let mut view = unsafe {
            ArrayBytesFixedDisjointView::new(
                UnsafeCellSlice::new(&mut allocation[offset..]),
                element_size,
                &array_shape,
                ArraySubset::new_with_shape(view_shape.to_vec()),
            )
        }
        .unwrap();
        codec.decode_into(
            encoded.clone().into(),
            chunk_shape,
            (&mut view).into(),
            &CodecOptions::default(),
        )
    };
    (allocation.split_off(offset), result)
}

/// Decode `encoded` with `decode_into` into a view, as [`decode_into_view_array`].
///
/// Returns the bytes of the array.
pub(crate) fn decode_into_view(
    codec: &Arc<dyn ArrayToBytesCodecTraits>,
    encoded: &CowBytes<'_>,
    chunk_shape: &[NonZeroU64],
    view_shape: [u64; 2],
    array_columns: usize,
    offset: usize,
    element_size: usize,
) -> Result<Vec<u8>, CodecError> {
    let (array, result) = decode_into_view_array(
        codec,
        encoded,
        chunk_shape,
        view_shape,
        array_columns,
        offset,
        element_size,
    );
    result.map(|()| array)
}
