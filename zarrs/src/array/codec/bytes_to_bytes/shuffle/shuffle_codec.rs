use std::mem::MaybeUninit;
use std::sync::Arc;

use zarrs_plugin::{PluginCreateError, ZarrVersion};

use super::{ShuffleCodecConfiguration, ShuffleCodecConfigurationV1};
use crate::array::{BytesRepresentation, CowBytes};
use zarrs_codec::{
    BytesToBytesCodecDecodeInto, BytesToBytesCodecTraits, CodecError, CodecMetadataOptions,
    CodecOptions, CodecTraits, ComputeResources, InvalidBytesLengthError, PartialDecoderCapability,
    PartialEncoderCapability, RecommendedConcurrency,
};
use zarrs_metadata::Configuration;

/// A `shuffle` codec implementation.
#[derive(Clone, Debug, Default)]
pub struct ShuffleCodec {
    elementsize: usize,
}

impl ShuffleCodec {
    /// Create a new `shuffle` codec.
    #[must_use]
    pub fn new(elementsize: usize) -> Self {
        Self { elementsize }
    }

    /// Create a new `shuffle` codec.
    ///
    /// # Errors
    /// Returns an error if the configuration is not supported.
    pub fn new_with_configuration(
        configuration: &ShuffleCodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        match configuration {
            ShuffleCodecConfiguration::V1(configuration) if configuration.elementsize == 0 => {
                Err(PluginCreateError::Other(
                    "the shuffle codec elementsize must be non-zero".to_string(),
                ))
            }
            ShuffleCodecConfiguration::V1(configuration) => Ok(Self {
                elementsize: configuration.elementsize,
            }),
            _ => Err(PluginCreateError::Other(
                "this shuffle codec configuration variant is unsupported".to_string(),
            )),
        }
    }

    /// Return the number of elements in `len` bytes.
    ///
    /// # Errors
    /// Returns an error if the elementsize is zero or `len` is not an integer multiple of it.
    fn element_count(&self, len: usize) -> Result<usize, CodecError> {
        if self.elementsize == 0 {
            return Err(CodecError::Other(
                "the shuffle codec elementsize must be non-zero".to_string(),
            ));
        }
        if !len.is_multiple_of(self.elementsize) {
            return Err(CodecError::Other("the shuffle codec expects the input byte length to be an integer multiple of the elementsize".to_string()));
        }
        Ok(len / self.elementsize)
    }

    /// Unshuffle `encoded_value` into `decoded_value`, initialising every byte of `decoded_value` on success.
    ///
    /// # Errors
    /// Returns an error if the lengths differ, the elementsize is zero, or the length is not an integer multiple of the elementsize.
    fn unshuffle(
        &self,
        encoded_value: &[u8],
        decoded_value: &mut [MaybeUninit<u8>],
    ) -> Result<(), CodecError> {
        // Shuffling preserves the length, so the decoded length is the encoded length
        if encoded_value.len() != decoded_value.len() {
            return Err(
                InvalidBytesLengthError::new(encoded_value.len(), decoded_value.len()).into(),
            );
        }

        let count = self.element_count(encoded_value.len())?;
        unshuffle(self.elementsize, count, encoded_value, decoded_value);
        Ok(())
    }
}

impl CodecTraits for ShuffleCodec {
    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        let configuration = ShuffleCodecConfiguration::V1(ShuffleCodecConfigurationV1 {
            elementsize: self.elementsize,
        });
        Some(configuration.into())
    }

    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        PartialDecoderCapability {
            partial_read: false,
            partial_decode: false,
        }
    }

    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        PartialEncoderCapability {
            partial_encode: false,
        }
    }
}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl BytesToBytesCodecTraits for ShuffleCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn BytesToBytesCodecTraits> {
        self as Arc<dyn BytesToBytesCodecTraits>
    }

    fn recommended_concurrency(
        &self,
        _decoded_representation: &BytesRepresentation,
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new_maximum(1))
    }

    fn encode<'a>(
        &self,
        decoded_value: CowBytes<'a>,
        _options: &CodecOptions,
        _resources: &ComputeResources,
    ) -> Result<CowBytes<'a>, CodecError> {
        let count = self.element_count(decoded_value.len())?;
        let len = decoded_value.len();
        let mut encoded_value = Vec::with_capacity(len);
        shuffle(
            self.elementsize,
            count,
            &decoded_value,
            &mut encoded_value.spare_capacity_mut()[..len],
        );
        // SAFETY: `shuffle` initialised the first `len <= capacity` elements.
        unsafe { encoded_value.set_len(len) };
        Ok(CowBytes::from(encoded_value))
    }

    fn decode<'a>(
        &self,
        encoded_value: CowBytes<'a>,
        _decoded_representation: &BytesRepresentation,
        _options: &CodecOptions,
        _resources: &ComputeResources,
    ) -> Result<CowBytes<'a>, CodecError> {
        let len = encoded_value.len();
        let mut decoded_value = Vec::with_capacity(len);
        self.unshuffle(
            &encoded_value,
            &mut decoded_value.spare_capacity_mut()[..len],
        )?;
        // SAFETY: `unshuffle` succeeded, so it initialised the first `len <= capacity` elements.
        unsafe { decoded_value.set_len(len) };
        Ok(CowBytes::from(decoded_value))
    }

    fn as_decode_into(&self) -> Option<&dyn BytesToBytesCodecDecodeInto> {
        Some(self)
    }

    fn encoded_representation(
        &self,
        decoded_representation: &BytesRepresentation,
    ) -> BytesRepresentation {
        *decoded_representation
    }
}

// SAFETY: `unshuffle` initialises every byte of `output` from the equal-length encoded bytes.
unsafe impl BytesToBytesCodecDecodeInto for ShuffleCodec {
    fn decode_into(
        &self,
        encoded_value: CowBytes<'_>,
        _decoded_representation: &BytesRepresentation,
        output: &mut [MaybeUninit<u8>],
        _options: &CodecOptions,
        _resources: &ComputeResources,
    ) -> Result<(), CodecError> {
        self.unshuffle(&encoded_value, output)
    }
}

/// Shuffle `count` elements of `elementsize` bytes: byte `i` of element `k` of `decoded` goes to `encoded[i * count + k]`.
///
/// `decoded` and `encoded` must have `count * elementsize` bytes, and every byte of `encoded` is initialised.
fn shuffle(elementsize: usize, count: usize, decoded: &[u8], encoded: &mut [MaybeUninit<u8>]) {
    match elementsize {
        1 => {
            crate::write_copy_of_slice(encoded, decoded);
        }
        2 => shuffle_swar::<2>(count, decoded, encoded),
        4 => shuffle_swar::<4>(count, decoded, encoded),
        8 => shuffle_swar::<8>(count, decoded, encoded),
        16 => shuffle_swar::<16>(count, decoded, encoded),
        _ => {
            for (k, element) in decoded.chunks_exact(elementsize).enumerate() {
                for (i, byte) in element.iter().enumerate() {
                    encoded[i * count + k].write(*byte);
                }
            }
        }
    }
}

/// The inverse of [`shuffle`], which initialises every byte of `decoded`.
fn unshuffle(elementsize: usize, count: usize, encoded: &[u8], decoded: &mut [MaybeUninit<u8>]) {
    match elementsize {
        1 => {
            crate::write_copy_of_slice(decoded, encoded);
        }
        2 => unshuffle_rows::<2>(count, encoded, decoded),
        4 => unshuffle_swar::<4>(count, encoded, decoded),
        8 => unshuffle_swar::<8>(count, encoded, decoded),
        16 => unshuffle_swar::<16>(count, encoded, decoded),
        _ => {
            for (i, lane) in encoded.chunks_exact(count.max(1)).enumerate() {
                for (k, byte) in lane.iter().enumerate() {
                    decoded[k * elementsize + i].write(*byte);
                }
            }
        }
    }
}

/// Transpose an 8x8 matrix of bytes, where row `j` is the little-endian bytes of `rows[j]`.
fn transpose_8x8(rows: &mut [u64; 8]) {
    /// Swap the high `shift` bits of each `2 * shift` bits of row `a` with the low bits of row `b`.
    fn swap(rows: &mut [u64; 8], a: usize, b: usize, shift: u32, mask: u64) {
        let t = ((rows[a] >> shift) ^ rows[b]) & mask;
        rows[b] ^= t;
        rows[a] ^= t << shift;
    }
    // Swap the off-diagonal 1x1, then 2x2, then 4x4 blocks. Unrolled by hand: as a loop it was not, and ran ~8x slower
    const M1: u64 = 0x00FF_00FF_00FF_00FF;
    const M2: u64 = 0x0000_FFFF_0000_FFFF;
    const M4: u64 = 0x0000_0000_FFFF_FFFF;
    swap(rows, 0, 1, 8, M1);
    swap(rows, 2, 3, 8, M1);
    swap(rows, 4, 5, 8, M1);
    swap(rows, 6, 7, 8, M1);
    swap(rows, 0, 2, 16, M2);
    swap(rows, 1, 3, 16, M2);
    swap(rows, 4, 6, 16, M2);
    swap(rows, 5, 7, 16, M2);
    swap(rows, 0, 4, 32, M4);
    swap(rows, 1, 5, 32, M4);
    swap(rows, 2, 6, 32, M4);
    swap(rows, 3, 7, 32, M4);
}

/// [`shuffle`] 8 elements at a time with 8x8 byte transposes, in 8-byte groups of each element.
///
/// Each group is transposed for all elements before the next, so a pass writes 8 lanes.
fn shuffle_swar<const N: usize>(count: usize, decoded: &[u8], encoded: &mut [MaybeUninit<u8>]) {
    const { assert!(N <= 8 || N.is_multiple_of(8)) };
    // A constant group width keeps the row copies inline
    let width = if N <= 8 { N } else { 8 };
    let elements = &decoded.as_chunks::<N>().0[..count];
    let (blocks, remainder) = elements.as_chunks::<8>();
    for group in 0..N.div_ceil(8) {
        for (b, block) in blocks.iter().enumerate() {
            let mut rows: [u64; 8] = std::array::from_fn(|k| {
                let mut row = [0u8; 8];
                row[..width].copy_from_slice(&block[k][8 * group..][..width]);
                u64::from_le_bytes(row)
            });
            transpose_8x8(&mut rows);
            for (i, row) in rows.iter().enumerate().take(width) {
                crate::write_copy_of_slice(
                    &mut encoded[(8 * group + i) * count + 8 * b..][..8],
                    &row.to_le_bytes(),
                );
            }
        }
    }
    let start = blocks.len() * 8;
    for (k, element) in remainder.iter().enumerate() {
        for (i, byte) in element.iter().enumerate() {
            encoded[i * count + start + k].write(*byte);
        }
    }
}

/// [`unshuffle`] 8 elements at a time with 8x8 byte transposes, in 8-byte groups of each element.
fn unshuffle_swar<const N: usize>(count: usize, encoded: &[u8], decoded: &mut [MaybeUninit<u8>]) {
    const { assert!(N <= 8 || N.is_multiple_of(8)) };
    let width = if N <= 8 { N } else { 8 };
    let elements = &mut decoded.as_chunks_mut::<N>().0[..count];
    let (blocks, remainder) = elements.as_chunks_mut::<8>();
    for (b, block) in blocks.iter_mut().enumerate() {
        for group in 0..N.div_ceil(8) {
            let mut rows: [u64; 8] = std::array::from_fn(|i| {
                if i < width {
                    let lane = &encoded[(8 * group + i) * count + 8 * b..][..8];
                    u64::from_le_bytes(lane.try_into().unwrap())
                } else {
                    0
                }
            });
            transpose_8x8(&mut rows);
            for (element, row) in block.iter_mut().zip(rows) {
                crate::write_copy_of_slice(
                    &mut element[8 * group..][..width],
                    &row.to_le_bytes()[..width],
                );
            }
        }
    }
    let start = blocks.len() * 8;
    for (k, element) in remainder.iter_mut().enumerate() {
        for (i, byte) in element.iter_mut().enumerate() {
            byte.write(encoded[i * count + start + k]);
        }
    }
}

/// [`unshuffle`] one element at a time, building each element from its lanes.
///
/// Faster than [`unshuffle_swar`] for 2-byte elements, where the compiler vectorises it.
fn unshuffle_rows<const N: usize>(count: usize, encoded: &[u8], decoded: &mut [MaybeUninit<u8>]) {
    let elements = &mut decoded.as_chunks_mut::<N>().0[..count];
    let lanes: [&[u8]; N] = std::array::from_fn(|i| &encoded[i * count..][..count]);
    for (k, element) in elements.iter_mut().enumerate() {
        *element = std::array::from_fn(|i| MaybeUninit::new(lanes[i][k]));
    }
}
