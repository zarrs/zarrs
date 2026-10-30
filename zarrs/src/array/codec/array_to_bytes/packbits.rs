//! The `packbits` array to bytes codec.
//!
//! Packs together values with non-byte-aligned sizes.
//!
//! The bits of each component are packed from the least significant bit, and each encoded byte is filled from its least significant bit.
//! That is, bit `j` of the packed bit sequence is bit `j % 8` of byte `j / 8`.
//! Whole-byte components are therefore packed as in the little-endian `bytes` codec.
//!
//! `numcodecs.packbits` (`packbits` in Zarr V2) fills each byte from its most significant bit instead, and is implemented by [`NumcodecsPackBitsCodec`](crate::array::codec::NumcodecsPackBitsCodec).
//! See <https://github.com/zarr-developers/zarr-extensions/issues/85>.
//!
//! ### Specification
//! - <https://github.com/zarr-developers/zarr-extensions/blob/8870ab31408e8b64072614b765f5e7cb265c5ec0/codecs/packbits/README.md>
//!
//! ### Codec `name` Aliases (Zarr V3)
//! - `packbits`
//!
//! ### Codec `id` Aliases (Zarr V2)
//! None
//!
//! ### Codec `configuration` Example - [`PackBitsCodecConfiguration`]:
//! ```rust
//! # let JSON = r#"
//! {
//!     "padding_encoding": "first_byte",
//!     "first_bit": null,
//!     "last_bit": null
//! }
//! # "#;
//! # use zarrs::metadata_ext::codec::packbits::PackBitsCodecConfiguration;
//! # serde_json::from_str::<PackBitsCodecConfiguration>(JSON).unwrap();
//! ```

mod data_type_extension_packbits_codec;
mod packbits_codec;
mod packbits_partial_decoder;

use std::mem::MaybeUninit;
use std::sync::Arc;

pub use packbits_codec::PackBitsCodec;
pub(crate) use packbits_codec::PackBitsCodecBound;
use zarrs_metadata::Endianness;
use zarrs_metadata::v3::MetadataV3;

use crate::array::DataType;
use zarrs_codec::{Codec, CodecPluginV3, CodecTraitsV3};
pub use zarrs_metadata_ext::codec::packbits::{
    PackBitsCodecConfiguration, PackBitsCodecConfigurationV1,
};

zarrs_plugin::impl_extension_aliases!(PackBitsCodec, v3: "packbits");

inventory::submit! {
    CodecPluginV3::new::<PackBitsCodec>()
}

impl CodecTraitsV3 for PackBitsCodec {
    fn create(metadata: &MetadataV3) -> Result<Codec, zarrs_codec::CodecCreateError> {
        let configuration: PackBitsCodecConfiguration = metadata.to_typed_configuration()?;
        let codec = Arc::new(PackBitsCodec::new_with_configuration(&configuration)?);
        Ok(Codec::ArrayToBytes(codec))
    }
}

// Re-export extension trait from zarrs_data_type
pub use zarrs_data_type::codec_traits::packbits::{
    PackBitsDataTypeExt, PackBitsDataTypePlugin, PackBitsDataTypeTraits,
    impl_pack_bits_data_type_traits,
};

#[derive(Debug, Clone, Copy)]
pub(super) struct PackBitsCodecComponents {
    pub component_size_bits: u64,
    pub num_components: u64,
    pub sign_extension: bool,
    /// The byte order that components are read and written in.
    pub endianness: Endianness,
    /// Whether decoded bytes are converted to and from `endianness` with the `bytes` codec.
    pub convert_bytes: bool,
}

fn pack_bits_components(
    data_type: &DataType,
) -> Result<PackBitsCodecComponents, zarrs_data_type::DataTypeCodecError> {
    let packbits = data_type.codec_packbits()?;
    // Components are packed from their little-endian `bytes` codec representation.
    // Decoded bytes that are already a `bytes` codec representation are read directly in its byte order.
    let (endianness, convert_bytes) = if packbits.is_decode_passthrough(Some(Endianness::Little)) {
        (Endianness::Little, false)
    } else if packbits.is_decode_passthrough(Some(Endianness::Big)) {
        (Endianness::Big, false)
    } else {
        (Endianness::Little, true)
    };
    Ok(PackBitsCodecComponents {
        component_size_bits: packbits.component_size_bits(),
        num_components: packbits.num_components(),
        sign_extension: packbits.sign_extension(),
        endianness,
        convert_bytes,
    })
}

/// The order that the packed bit sequence fills each byte in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PackBitsBitOrder {
    /// Bit `j` of the sequence is bit `j % 8` of byte `j / 8`, numbered from the least significant bit (`packbits`).
    LsbFirst,
    /// Bit `j` of the sequence is bit `7 - j % 8` of byte `j / 8`, numbered from the least significant bit (`numcodecs.packbits`).
    MsbFirst,
}

/// How each decoded component maps to the packed bit sequence.
#[derive(Debug, Clone, Copy)]
pub(super) struct PackBitsLayout {
    /// The size of a decoded component in bytes, at most 8.
    component_size_bytes: usize,
    /// The first packed bit of a component, numbered from its least significant bit.
    first_bit: u32,
    /// The last packed bit of a component, numbered from its least significant bit.
    last_bit: u32,
    /// Whether `last_bit` is sign extended to the higher bits when decoding.
    sign_extension: bool,
    /// The byte order of a component.
    endianness: Endianness,
    /// The order that the packed bit sequence fills each byte in.
    bit_order: PackBitsBitOrder,
}

impl PackBitsLayout {
    /// Create the layout of packed `components`.
    ///
    /// `first_bit <= last_bit < component_size_bits <= 64` must hold.
    #[expect(clippy::cast_possible_truncation, reason = "the bits are at most 64")]
    fn new(
        components: PackBitsCodecComponents,
        first_bit: u64,
        last_bit: u64,
        bit_order: PackBitsBitOrder,
    ) -> Self {
        debug_assert!(first_bit <= last_bit);
        debug_assert!(last_bit < components.component_size_bits);
        debug_assert!(components.component_size_bits <= 64);
        Self {
            component_size_bytes: components.component_size_bits.div_ceil(8) as usize,
            first_bit: first_bit as u32,
            last_bit: last_bit as u32,
            sign_extension: components.sign_extension,
            endianness: components.endianness,
            bit_order,
        }
    }

    /// The number of packed bits of a component.
    fn packed_bits(self) -> u32 {
        self.last_bit - self.first_bit + 1
    }

    /// Whether all bits of whole-byte components are packed (e.g. `float32`), so there are no padding bits.
    fn packs_whole_bytes(self) -> bool {
        self.packed_bits() as usize == self.component_size_bytes * 8
    }

    /// A mask of the low [`packed_bits`](Self::packed_bits) bits.
    fn mask(self) -> u64 {
        u64::MAX >> (64 - self.packed_bits())
    }

    /// Read a decoded component.
    fn read_component<const N: usize>(self, component: &[u8; N]) -> u64 {
        let mut bytes = [0u8; 8];
        match self.endianness {
            Endianness::Little => {
                bytes[..N].copy_from_slice(component);
                u64::from_le_bytes(bytes)
            }
            Endianness::Big => {
                bytes[8 - N..].copy_from_slice(component);
                u64::from_be_bytes(bytes)
            }
        }
    }

    /// Write a decoded component.
    fn write_component<const N: usize>(self, value: u64, component: &mut [MaybeUninit<u8>; N]) {
        let (bytes, start) = match self.endianness {
            Endianness::Little => (value.to_le_bytes(), 0),
            Endianness::Big => (value.to_be_bytes(), 8 - N),
        };
        *component = std::array::from_fn(|i| MaybeUninit::new(bytes[start + i]));
    }
}

/// Call `$f::<N, MSB_FIRST>($args)`, where `N` is the component size in bytes of `$layout`
/// and `MSB_FIRST` is whether its bit order is [`PackBitsBitOrder::MsbFirst`].
macro_rules! with_component_size {
    ($layout:expr, $f:ident($($args:expr),*)) => {
        match $layout.bit_order {
            PackBitsBitOrder::LsbFirst => with_component_size!(@size $layout, $f::<false>($($args),*)),
            PackBitsBitOrder::MsbFirst => with_component_size!(@size $layout, $f::<true>($($args),*)),
        }
    };
    (@size $layout:expr, $f:ident::<$msb_first:literal>($($args:expr),*)) => {
        match $layout.component_size_bytes {
            1 => $f::<1, $msb_first>($($args),*),
            2 => $f::<2, $msb_first>($($args),*),
            3 => $f::<3, $msb_first>($($args),*),
            4 => $f::<4, $msb_first>($($args),*),
            5 => $f::<5, $msb_first>($($args),*),
            6 => $f::<6, $msb_first>($($args),*),
            7 => $f::<7, $msb_first>($($args),*),
            8 => $f::<8, $msb_first>($($args),*),
            _ => unreachable!("packbits components are at most 64 bits"),
        }
    };
}

/// Pack the components of `decoded` into `packed`.
///
/// Components are packed from their least significant bit, and each byte of `packed` is filled in the bit order of `layout`.
/// `packed` must have the length of the packed bits rounded up to whole bytes, and is initialised.
fn pack(layout: PackBitsLayout, decoded: &[u8], packed: &mut [MaybeUninit<u8>]) {
    with_component_size!(layout, pack_components(layout, decoded, packed));
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the pending bits are written a word at a time"
)]
fn pack_components<const N: usize, const MSB_FIRST: bool>(
    layout: PackBitsLayout,
    decoded: &[u8],
    mut packed: &mut [MaybeUninit<u8>],
) {
    // Store a word of the bit sequence, with its first bit in the least significant bit.
    // Reversing the bits of the word reverses the bits of each byte and the byte order.
    let word_bytes = |word: u64| {
        if MSB_FIRST {
            word.reverse_bits().to_be_bytes()
        } else {
            word.to_le_bytes()
        }
    };
    let packed_bits = layout.packed_bits();
    let mask = layout.mask();
    // The pending bits, with the next bit in the least significant bit
    let mut acc: u128 = 0;
    let mut nbits: u32 = 0;
    for component in decoded.as_chunks::<N>().0 {
        let value = (layout.read_component(component) >> layout.first_bit) & mask;
        acc |= u128::from(value) << nbits;
        nbits += packed_bits;
        if nbits >= 64 {
            let (word, rest) = std::mem::take(&mut packed)
                .split_first_chunk_mut::<8>()
                .expect("packed holds all bits");
            *word = word_bytes(acc as u64).map(MaybeUninit::new);
            packed = rest;
            acc >>= 64;
            nbits -= 64;
        }
    }
    // Initialise all of the remaining bytes, which hold the pending bits
    let word = word_bytes(acc as u64);
    let word = &word[..packed.len()];
    for (byte, word_byte) in packed.iter_mut().zip(word) {
        byte.write(*word_byte);
    }
}

/// Unpack the components of `decoded` from `packed`, skipping its first `bit_offset` bits.
///
/// This is the inverse of [`pack`], and also sign extends components if `layout.sign_extension`.
/// `packed` must hold the bits of every component of `decoded`, which is initialised.
fn unpack(layout: PackBitsLayout, packed: &[u8], bit_offset: u8, decoded: &mut [MaybeUninit<u8>]) {
    with_component_size!(
        layout,
        unpack_components(layout, packed, bit_offset, decoded)
    );
}

/// Unpack `len` decoded bytes from `packed` into a new vector, as in [`unpack`].
fn unpack_to_vec(layout: PackBitsLayout, packed: &[u8], bit_offset: u8, len: usize) -> Vec<u8> {
    let mut decoded = Vec::with_capacity(len);
    unpack(
        layout,
        packed,
        bit_offset,
        &mut decoded.spare_capacity_mut()[..len],
    );
    // SAFETY: `unpack` initialised the first `len <= capacity` elements.
    unsafe { decoded.set_len(len) };
    decoded
}

fn unpack_components<const N: usize, const MSB_FIRST: bool>(
    layout: PackBitsLayout,
    packed: &[u8],
    bit_offset: u8,
    decoded: &mut [MaybeUninit<u8>],
) {
    /// The number of packed bytes that are copied (and bit reversed if `MSB_FIRST`) at a time.
    const BLOCK_BYTES: usize = 1024;
    let packed_bits = layout.packed_bits() as usize;
    let mask = layout.mask();
    // Bits past the end of `packed` would be unpacked as zero
    assert!(
        packed.len() * 8 >= usize::from(bit_offset) + decoded.len() / N * packed_bits,
        "packed holds all bits"
    );
    // Every byte of `decoded` is initialised, so it must hold whole components
    let (components, []) = decoded.as_chunks_mut::<N>() else {
        panic!("decoded holds whole components");
    };
    let mut components = components.iter_mut();
    // A block of packed bytes with the bits in sequence from the least significant bit.
    // It is uninitialised, as zeroing it all would dominate unpacking short ranges.
    let mut block = [MaybeUninit::<u8>::uninit(); BLOCK_BYTES];
    // The next bit to unpack, relative to the start of `packed` and then the block
    let mut bit = usize::from(bit_offset);
    loop {
        let block_start = bit / 8;
        bit %= 8;
        let block_packed = packed.get(block_start..).unwrap_or_default();
        let block_len = block_packed.len().min(BLOCK_BYTES);
        for (byte, packed_byte) in block.iter_mut().zip(&block_packed[..block_len]) {
            byte.write(if MSB_FIRST {
                packed_byte.reverse_bits()
            } else {
                *packed_byte
            });
        }
        // Zero the bytes after `packed` that the 9 byte window of a component starting in `packed` can reach,
        // and one more so that a window always fits
        let init_len = (block_len + 9).min(BLOCK_BYTES);
        block[block_len..init_len].fill(MaybeUninit::new(0));
        // SAFETY: `block[..block_len]` was written from `block_packed[..block_len]` and `block[block_len..init_len]` was zeroed,
        // so the first `init_len <= BLOCK_BYTES` elements of `block` are initialised.
        // `MaybeUninit<u8>` has the same size and alignment as `u8`, and `block` is not written while `block_init` is borrowed.
        let block_init: &[u8] =
            unsafe { std::slice::from_raw_parts(block.as_ptr().cast::<u8>(), init_len) };

        // Unpack components while the 9 bytes holding their (at most 64) bits are initialised
        while let Some(bytes) = block_init.get(bit / 8..bit / 8 + 9) {
            let Some(component) = components.next() else {
                return;
            };
            let shift = bit % 8;
            let lo = u64::from_le_bytes(bytes[..8].try_into().unwrap());
            let hi = u64::from(bytes[8]);
            // Shift `hi` in two steps, as it is shifted out entirely if `shift` is zero
            let value = ((lo >> shift) | (hi << 1 << (63 - shift))) & mask;
            let value = if layout.sign_extension {
                // Move the sign bit to the most significant bit, then shift it back down with sign extension
                ((value << (64 - packed_bits)).cast_signed() >> (63 - layout.last_bit))
                    .cast_unsigned()
            } else {
                value << layout.first_bit
            };
            layout.write_component(value, component);
            bit += packed_bits;
        }
        bit += block_start * 8;
    }
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
mod tests {
    use std::mem::MaybeUninit;
    use std::num::NonZeroU64;
    use std::sync::Arc;

    use num::Integer;
    use zarrs_data_type::FillValue;
    use zarrs_metadata::Endianness;
    use zarrs_metadata::v3::MetadataV3;
    use zarrs_plugin::ExtensionName;

    use crate::array::codec::BytesCodec;
    use crate::array::element::{Element, ElementOwned};
    use crate::array::{ArrayBytes, ArraySubset, CowBytes, DataType, data_type};
    use zarrs_codec::{
        BytesPartialDecoderTraits, Codec, CodecMetadataOptions, CodecOptions, CodecSpecificOptions,
        UnboundArrayToBytesCodecTraits,
    };
    use zarrs_metadata_ext::codec::packbits::PackBitsPaddingEncoding;

    use super::{
        PackBitsBitOrder, PackBitsCodec, PackBitsCodecConfiguration, PackBitsLayout, pack, unpack,
    };

    /// A xorshift pseudorandom number generator.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        /// A random number in `range`.
        fn range(&mut self, range: std::ops::Range<u64>) -> u64 {
            range.start + self.next() % (range.end - range.start)
        }

        fn bytes(&mut self, len: usize) -> Vec<u8> {
            (0..len).map(|_| self.next().to_le_bytes()[0]).collect()
        }
    }

    /// Split the padding encoding byte, if any, from the packed elements of `encoded`.
    /// Returns `len` bytes initialised to `0xFF` and then written by `f`, so that bytes it does not write are detected.
    fn with_output(len: usize, f: impl FnOnce(&mut [MaybeUninit<u8>])) -> Vec<u8> {
        let mut output = vec![MaybeUninit::new(0xFF); len];
        f(&mut output);
        // SAFETY: every byte was initialised, and `pack` and `unpack` only write initialised bytes.
        output
            .into_iter()
            .map(|byte| unsafe { byte.assume_init() })
            .collect()
    }

    fn split_padding(encoding: PackBitsPaddingEncoding, encoded: &[u8]) -> (Option<u8>, &[u8]) {
        match encoding {
            PackBitsPaddingEncoding::None => (None, encoded),
            PackBitsPaddingEncoding::FirstByte => {
                let (padding, elements) = encoded.split_first().unwrap();
                (Some(*padding), elements)
            }
            PackBitsPaddingEncoding::LastByte => {
                let (padding, elements) = encoded.split_last().unwrap();
                (Some(*padding), elements)
            }
        }
    }

    /// The decoded byte and bit of bit `bit` of a component, numbered from its least significant bit.
    fn reference_component_bit(layout: PackBitsLayout, bit: u32) -> (usize, u32) {
        let byte = (bit / 8) as usize;
        let byte = match layout.endianness {
            Endianness::Little => byte,
            Endianness::Big => layout.component_size_bytes - 1 - byte,
        };
        (byte, bit % 8)
    }

    /// The packed byte and bit of bit `bit` of the packed bit sequence, numbered from its least significant bit.
    fn reference_packed_bit(layout: PackBitsLayout, bit: usize) -> (usize, u32) {
        #[expect(clippy::cast_possible_truncation)]
        let byte_bit = (bit % 8) as u32;
        let byte_bit = match layout.bit_order {
            PackBitsBitOrder::LsbFirst => byte_bit,
            PackBitsBitOrder::MsbFirst => 7 - byte_bit,
        };
        (bit / 8, byte_bit)
    }

    /// Pack a bit at a time.
    #[expect(clippy::cast_possible_truncation)]
    fn reference_pack(layout: PackBitsLayout, decoded: &[u8]) -> Vec<u8> {
        let packed_bits = layout.packed_bits() as usize;
        let num_components = decoded.len() / layout.component_size_bytes;
        let mut packed = vec![0u8; (num_components * packed_bits).div_ceil(8)];
        for (i, component) in decoded
            .chunks_exact(layout.component_size_bytes)
            .enumerate()
        {
            for bit in 0..packed_bits {
                let (byte, byte_bit) =
                    reference_component_bit(layout, layout.first_bit + bit as u32);
                let (packed_byte, packed_byte_bit) =
                    reference_packed_bit(layout, i * packed_bits + bit);
                packed[packed_byte] |= ((component[byte] >> byte_bit) & 1) << packed_byte_bit;
            }
        }
        packed
    }

    /// Unpack a bit at a time, starting at bit `bit_offset` of `packed`.
    #[expect(clippy::cast_possible_truncation)]
    fn reference_unpack(
        layout: PackBitsLayout,
        packed: &[u8],
        bit_offset: usize,
        num_components: usize,
    ) -> Vec<u8> {
        let packed_bits = layout.packed_bits() as usize;
        let mut decoded = vec![0u8; num_components * layout.component_size_bytes];
        for (i, component) in decoded
            .chunks_exact_mut(layout.component_size_bytes)
            .enumerate()
        {
            let mut sign = 0;
            for bit in 0..packed_bits {
                let (packed_byte, packed_byte_bit) =
                    reference_packed_bit(layout, bit_offset + i * packed_bits + bit);
                sign = (packed[packed_byte] >> packed_byte_bit) & 1;
                let (byte, byte_bit) =
                    reference_component_bit(layout, layout.first_bit + bit as u32);
                component[byte] |= sign << byte_bit;
            }
            if layout.sign_extension && sign == 1 {
                for bit in layout.last_bit + 1..layout.component_size_bytes as u32 * 8 {
                    let (byte, byte_bit) = reference_component_bit(layout, bit);
                    component[byte] |= 1 << byte_bit;
                }
            }
        }
        decoded
    }

    #[test]
    fn pack_unpack_endianness() {
        // Bits 2..=11 of 0x025C and 0x0FFC, from the least significant bit
        for ((bit_order, packed), (endianness, decoded)) in itertools::iproduct!(
            [
                (
                    PackBitsBitOrder::LsbFirst,
                    [0b1001_0111, 0b1111_1100, 0b0000_1111]
                ),
                (
                    PackBitsBitOrder::MsbFirst,
                    [0b1110_1001, 0b0011_1111, 0b1111_0000]
                ),
            ],
            [
                (Endianness::Little, [0x5C, 0x02, 0xFC, 0x0F]),
                (Endianness::Big, [0x02, 0x5C, 0x0F, 0xFC]),
            ]
        ) {
            let layout = |sign_extension| PackBitsLayout {
                component_size_bytes: 2,
                first_bit: 2,
                last_bit: 11,
                sign_extension,
                endianness,
                bit_order,
            };
            let packed_out = with_output(3, |out| pack(layout(false), &decoded, out));
            assert_eq!(packed_out, packed);

            let decoded_out = with_output(4, |out| unpack(layout(false), &packed, 0, out));
            assert_eq!(decoded_out, decoded);

            // 0x0FFC is negative in 12 bits, so its high bits are set
            let decoded_out = with_output(4, |out| unpack(layout(true), &packed, 0, out));
            let negative = match endianness {
                Endianness::Little => [0xFC, 0xFF],
                Endianness::Big => [0xFF, 0xFC],
            };
            assert_eq!(decoded_out[..2], decoded[..2]);
            assert_eq!(decoded_out[2..], negative);
        }
    }

    #[test]
    fn unpack_sign_extends_sub_byte_components() {
        for (bit_order, packed) in [
            (PackBitsBitOrder::LsbFirst, 0b0011_1001),
            (PackBitsBitOrder::MsbFirst, 0b1001_1100),
        ] {
            let layout = PackBitsLayout {
                component_size_bytes: 1,
                first_bit: 0,
                last_bit: 1,
                sign_extension: true,
                endianness: Endianness::native(),
                bit_order,
            };
            let decoded = with_output(4, |out| unpack(layout, &[packed], 0, out));
            assert_eq!(decoded, [0x01, 0xFE, 0xFF, 0x00]);
        }
    }

    /// `pack` and `unpack` match packing a bit at a time, for both byte orders and bit orders.
    #[test]
    #[expect(clippy::cast_possible_truncation)]
    fn pack_unpack_reference() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for (endianness, bit_order) in itertools::iproduct!(
            [Endianness::Little, Endianness::Big],
            [PackBitsBitOrder::LsbFirst, PackBitsBitOrder::MsbFirst]
        ) {
            for component_size_bits in [1u32, 2, 4, 6, 8, 12, 16, 32, 64] {
                for _ in 0..50 {
                    let first_bit = rng.range(0..u64::from(component_size_bits)) as u32;
                    let last_bit =
                        rng.range(u64::from(first_bit)..u64::from(component_size_bits)) as u32;
                    let layout = PackBitsLayout {
                        component_size_bytes: component_size_bits.div_ceil(8) as usize,
                        first_bit,
                        last_bit,
                        sign_extension: rng.next().is_multiple_of(2),
                        endianness,
                        bit_order,
                    };
                    // Span several blocks of `unpack`
                    let max_components = 3 * 8 * 1024 / u64::from(layout.packed_bits());
                    let num_components = rng.range(0..max_components) as usize;
                    let decoded = rng.bytes(num_components * layout.component_size_bytes);

                    let packed = reference_pack(layout, &decoded);
                    let packed_out = with_output(packed.len(), |out| pack(layout, &decoded, out));
                    assert_eq!(packed_out, packed, "{layout:?}");

                    let decoded_out =
                        with_output(decoded.len(), |out| unpack(layout, &packed, 0, out));
                    assert_eq!(
                        decoded_out,
                        reference_unpack(layout, &packed, 0, num_components),
                        "{layout:?}"
                    );

                    // Unpack a subset from the bytes holding it
                    if num_components > 0 {
                        let packed_bits = layout.packed_bits() as usize;
                        let start = rng.range(0..num_components as u64) as usize;
                        let end = rng.range(start as u64 + 1..num_components as u64 + 1) as usize;
                        let bit_start = start * packed_bits;
                        let bit_end = end * packed_bits;
                        let packed_subset = &packed[bit_start / 8..bit_end.div_ceil(8)];
                        let decoded_out =
                            with_output((end - start) * layout.component_size_bytes, |out| {
                                unpack(layout, packed_subset, (bit_start % 8) as u8, out);
                            });
                        assert_eq!(
                            decoded_out,
                            reference_unpack(layout, &packed, bit_start, end - start),
                            "{layout:?}"
                        );
                    }
                }
            }
        }
    }

    /// `unpack` reads components that straddle the blocks it reverses, from any bit offset.
    #[test]
    fn unpack_reference_wide_components() {
        let mut rng = Rng(0xD1B5_4A32_D192_ED03);
        for (endianness, bit_order) in itertools::iproduct!(
            [Endianness::Little, Endianness::Big],
            [PackBitsBitOrder::LsbFirst, PackBitsBitOrder::MsbFirst]
        ) {
            for first_bit in [0, 1, 7] {
                let layout = PackBitsLayout {
                    component_size_bytes: 8,
                    first_bit,
                    last_bit: 63,
                    sign_extension: false,
                    endianness,
                    bit_order,
                };
                let packed_bits = layout.packed_bits() as usize;
                let num_components = 3 * 8 * 1024 / packed_bits;
                let packed = reference_pack(layout, &rng.bytes(num_components * 8));
                for start in 0..16 {
                    let bit_start = start * packed_bits;
                    let decoded = with_output((num_components - start) * 8, |out| {
                        unpack(layout, &packed[bit_start / 8..], (bit_start % 8) as u8, out);
                    });
                    assert_eq!(
                        decoded,
                        reference_unpack(layout, &packed, bit_start, num_components - start),
                        "{layout:?} {start}"
                    );
                }
            }
        }
    }

    /// The codec matches packing a bit at a time, including partial decoding.
    #[test]
    #[expect(clippy::cast_possible_truncation)]
    fn codec_packbits_reference() -> Result<(), Box<dyn std::error::Error>> {
        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        let options = CodecOptions::default();
        let data_types = [
            data_type::bool(),
            data_type::uint2(),
            data_type::int2(),
            data_type::int4(),
            data_type::float6_e2m3fn(),
            data_type::complex_float4_e2m1fn(),
            data_type::int8(),
            data_type::complex_float8_e4m3(),
            data_type::uint16(),
            data_type::int16(),
            data_type::int32(),
            data_type::complex64(),
            data_type::int64(),
            data_type::complex128(),
        ];
        for data_type in data_types {
            let components = super::pack_bits_components(&data_type)?;
            let data_type_size = data_type.fixed_size().unwrap();
            for _ in 0..20 {
                let first_bit = rng.range(0..components.component_size_bits);
                let last_bit = rng.range(first_bit..components.component_size_bits);
                let padding_encoding = match rng.range(0..3) {
                    0 => PackBitsPaddingEncoding::None,
                    1 => PackBitsPaddingEncoding::FirstByte,
                    _ => PackBitsPaddingEncoding::LastByte,
                };
                let codec = Arc::new(super::PackBitsCodec::new(
                    padding_encoding,
                    Some(first_bit),
                    Some(last_bit),
                )?)
                .with_context(
                    data_type.clone(),
                    FillValue::new(vec![0; data_type_size]),
                    &CodecSpecificOptions::default(),
                )?;
                let layout = super::PackBitsLayout::new(
                    components,
                    first_bit,
                    last_bit,
                    PackBitsBitOrder::LsbFirst,
                );

                let shape = [rng.range(1..20), rng.range(1..20)];
                let num_elements = (shape[0] * shape[1]) as usize;
                let chunk_shape = shape.map(|s| NonZeroU64::new(s).unwrap());
                let bytes = rng.bytes(num_elements * data_type_size);

                // Encoding
                let encoded =
                    codec.encode(ArrayBytes::new_flen(bytes.clone()), &chunk_shape, &options)?;
                let packed = reference_pack(layout, &bytes);
                let (_, elements) = split_padding(padding_encoding, &encoded);
                assert_eq!(elements, packed, "{data_type:?} {layout:?}");

                // Decoding
                let num_components = num_elements * components.num_components as usize;
                let expected =
                    ArrayBytes::new_flen(reference_unpack(layout, &packed, 0, num_components));
                let decoded = codec.decode(encoded.clone(), &chunk_shape, &options)?;
                assert_eq!(decoded, expected, "{data_type:?} {layout:?}");

                // Partial decoding
                let partial_decoder =
                    codec.partial_decoder(Arc::new(encoded), &chunk_shape, &options)?;
                for _ in 0..4 {
                    let start = shape.map(|s| rng.range(0..s));
                    let end = [
                        rng.range(start[0] + 1..shape[0] + 1),
                        rng.range(start[1] + 1..shape[1] + 1),
                    ];
                    let subset =
                        ArraySubset::new_with_ranges(&[start[0]..end[0], start[1]..end[1]]);
                    assert_eq!(
                        partial_decoder.partial_decode(&subset, &options)?,
                        expected.extract_array_subset(&subset, &shape, &data_type)?,
                        "{data_type:?} {layout:?} {subset:?}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn codec_packbits_v3() -> Result<(), Box<dyn std::error::Error>> {
        let Codec::ArrayToBytes(codec) = Codec::from_metadata(&MetadataV3::new("packbits"))? else {
            panic!("expected an array to bytes codec");
        };
        let configuration: PackBitsCodecConfiguration =
            serde_json::from_str(r#"{"padding_encoding":"first_byte"}"#)?;
        for codec in [
            codec,
            Arc::new(PackBitsCodec::default()),
            Arc::new(PackBitsCodec::new(
                PackBitsPaddingEncoding::FirstByte,
                None,
                None,
            )?),
            Arc::new(PackBitsCodec::new_with_configuration(&configuration)?),
        ] {
            let options = CodecMetadataOptions::default();
            assert_eq!(codec.name_v3().as_deref(), Some("packbits"));
            assert!(codec.configuration_v3(&options).is_some());
            assert!(codec.name_v2().is_none());
            assert!(codec.configuration_v2(&options).is_none());
            codec.with_context(
                data_type::float32(),
                FillValue::from(0.0f32),
                &CodecSpecificOptions::default(),
            )?;
        }
        Ok(())
    }

    #[test]
    fn codec_packbits_bool() -> Result<(), Box<dyn std::error::Error>> {
        for encoding in [
            PackBitsPaddingEncoding::None,
            PackBitsPaddingEncoding::FirstByte,
            PackBitsPaddingEncoding::LastByte,
        ] {
            let chunk_shape = vec![NonZeroU64::new(8).unwrap(), NonZeroU64::new(5).unwrap()];
            let data_type = data_type::bool();
            let fill_value = FillValue::from(false);
            let codec = Arc::new(super::PackBitsCodec::new(encoding, None, None).unwrap())
                .with_context(
                    data_type.clone(),
                    fill_value.clone(),
                    &CodecSpecificOptions::default(),
                )?;

            let elements: Vec<bool> = (0..40).map(|i| i % 3 == 0).collect();
            let bytes = bool::into_array_bytes(&data_type, elements)?.into_owned();
            // T F F T F
            // F T F F T
            // F F T F F
            // T F F T F
            // ...

            // Encoding
            let encoded = codec.encode(bytes.clone(), &chunk_shape, &CodecOptions::default())?;
            assert!((encoded.len() as u64) <= 40.div_ceil(&8) + 1);

            // Decoding
            let decoded = codec
                .decode(encoded.clone(), &chunk_shape, &CodecOptions::default())
                .unwrap();
            assert_eq!(bytes, decoded);

            // Partial decoding
            let decoded_region = ArraySubset::new_with_ranges(&[1..4, 1..4]);
            let input_handle = Arc::new(encoded);
            let partial_decoder = codec
                .partial_decoder(input_handle.clone(), &chunk_shape, &CodecOptions::default())
                .unwrap();
            assert_eq!(partial_decoder.size_held(), input_handle.size_held()); // packbits partial decoder does not hold bytes
            let decoded_partial_chunk = partial_decoder
                .partial_decode(&decoded_region, &CodecOptions::default())
                .unwrap();
            let decoded_partial_chunk =
                bool::from_array_bytes(&data_type, decoded_partial_chunk).unwrap();
            let answer: Vec<bool> =
                vec![true, false, false, false, true, false, false, false, true];
            assert_eq!(answer, decoded_partial_chunk);
        }
        Ok(())
    }

    /// Bytes are filled from the least significant bit, as in zarrs 0.20-0.23.
    #[test]
    fn codec_packbits_bit_order() -> Result<(), Box<dyn std::error::Error>> {
        let encode = |data_type: zarrs_data_type::DataType,
                      fill_value: FillValue,
                      bytes: Vec<u8>|
         -> Result<Vec<u8>, Box<dyn std::error::Error>> {
            let shape = [NonZeroU64::new(bytes.len() as u64).unwrap()];
            let codec = Arc::new(super::PackBitsCodec::new(
                PackBitsPaddingEncoding::FirstByte,
                None,
                None,
            )?)
            .with_context(data_type, fill_value, &CodecSpecificOptions::default())?;
            let encoded = codec.encode(
                ArrayBytes::new_flen(bytes.clone()),
                &shape,
                &CodecOptions::default(),
            )?;
            assert_eq!(
                codec.decode(encoded.clone(), &shape, &CodecOptions::default())?,
                ArrayBytes::new_flen(bytes)
            );
            Ok(encoded.to_vec())
        };

        assert_eq!(
            encode(
                data_type::bool(),
                FillValue::from(false),
                vec![1, 0, 0, 0, 0, 0, 0, 0]
            )?,
            [0, 0b0000_0001]
        );
        assert_eq!(
            encode(data_type::bool(), FillValue::from(false), vec![1, 0, 0, 1])?,
            [4, 0b0000_1001]
        );
        // Component bits are in the sequence from the least significant bit
        assert_eq!(
            encode(data_type::uint4(), FillValue::from(0u8), vec![1, 2, 7])?,
            [4, 0b0010_0001, 0b0000_0111]
        );
        // Whole-byte components are packed as in the `bytes` codec
        assert_eq!(
            encode(data_type::uint8(), FillValue::from(0u8), vec![1, 0x0F])?,
            [0, 1, 0x0F]
        );
        Ok(())
    }

    #[test]
    fn codec_packbits_rejects_out_of_range_last_bit_when_bound() {
        let data_type = data_type::uint2();
        let fill_value = FillValue::from(0u8);
        let codec = Arc::new(
            super::PackBitsCodec::new(PackBitsPaddingEncoding::None, None, Some(2)).unwrap(),
        );

        assert!(
            codec
                .with_context(data_type, fill_value, &CodecSpecificOptions::default())
                .is_err()
        );
    }

    /// Shapes with sizes that overflow are errors rather than panics.
    #[test]
    fn codec_packbits_rejects_overflowing_shape() -> Result<(), Box<dyn std::error::Error>> {
        let codec = Arc::new(super::PackBitsCodec::default()).with_context(
            data_type::uint4(),
            FillValue::from(0u8),
            &CodecSpecificOptions::default(),
        )?;
        let options = CodecOptions::default();
        let max = NonZeroU64::new(u64::MAX).unwrap();
        let shapes = [
            // The number of elements overflows
            vec![max, NonZeroU64::new(2).unwrap()],
            // The number of bits overflows
            vec![NonZeroU64::new(u64::MAX / 2).unwrap()],
        ];
        for shape in shapes {
            assert!(codec.encoded_representation(&shape).is_err());
            assert!(
                codec
                    .encode(ArrayBytes::new_flen(vec![0u8]), &shape, &options)
                    .is_err()
            );
            assert!(
                codec
                    .decode(CowBytes::from(vec![0u8]), &shape, &options)
                    .is_err()
            );
        }
        Ok(())
    }

    #[test]
    fn codec_packbits_float32() -> Result<(), Box<dyn std::error::Error>> {
        for encoding in [
            PackBitsPaddingEncoding::None,
            PackBitsPaddingEncoding::FirstByte,
            PackBitsPaddingEncoding::LastByte,
        ] {
            let chunk_shape = vec![NonZeroU64::new(8).unwrap(), NonZeroU64::new(5).unwrap()];
            let data_type = data_type::float32();
            let fill_value = FillValue::from(0.0f32);
            let codec = Arc::new(super::PackBitsCodec::new(encoding, None, None).unwrap())
                .with_context(
                    data_type.clone(),
                    fill_value.clone(),
                    &CodecSpecificOptions::default(),
                )?;

            let elements: Vec<f32> = (0..40).map(|i| i as f32).collect();
            let bytes = f32::to_array_bytes(&data_type, &elements)?.into_owned();

            // Encoding
            let encoded = codec.encode(bytes.clone(), &chunk_shape, &CodecOptions::default())?;
            // A padding encoding byte holds the number of padding bits, which is zero
            let (padding, elements) = split_padding(encoding, &encoded);
            assert!(padding.is_none_or(|padding| padding == 0));
            assert_eq!(elements.len(), 40 * 4);
            assert_eq!(
                codec.encoded_representation(&chunk_shape)?,
                crate::array::BytesRepresentation::FixedSize(encoded.len() as u64)
            );

            // Decoding
            let decoded = codec
                .decode(encoded.clone(), &chunk_shape, &CodecOptions::default())
                .unwrap();
            assert_eq!(bytes, decoded);

            // Check the elements match the little endian `bytes` codec
            let decoded = Arc::new(BytesCodec::little())
                .with_context(
                    data_type.clone(),
                    fill_value.clone(),
                    &CodecSpecificOptions::default(),
                )?
                .decode(
                    CowBytes::from(elements.to_vec()),
                    &chunk_shape,
                    &CodecOptions::default(),
                )
                .unwrap();
            assert_eq!(bytes, decoded);

            // Partial decoding must skip the padding byte, including across separate rows.
            let partial_decoder =
                codec.partial_decoder(Arc::new(encoded), &chunk_shape, &CodecOptions::default())?;
            let decoded_partial_chunk = partial_decoder.partial_decode(
                &ArraySubset::new_with_ranges(&[2..5, 1..4]),
                &CodecOptions::default(),
            )?;
            assert_eq!(
                vec![11.0, 12.0, 13.0, 16.0, 17.0, 18.0, 21.0, 22.0, 23.0],
                f32::from_array_bytes(&data_type, decoded_partial_chunk)?
            );
        }
        Ok(())
    }

    /// Returns each requested byte range without its last byte, like a store that clamps ranges.
    struct TruncatingPartialDecoder(CowBytes<'static>);

    impl BytesPartialDecoderTraits for TruncatingPartialDecoder {
        fn exists(&self) -> Result<bool, zarrs_storage::StorageError> {
            Ok(true)
        }

        fn size_held(&self) -> usize {
            0
        }

        fn partial_decode_many(
            &self,
            decoded_regions: zarrs_storage::byte_range::ByteRangeIterator,
            options: &CodecOptions,
        ) -> Result<Option<Vec<CowBytes<'_>>>, zarrs_codec::CodecError> {
            Ok(self
                .0
                .partial_decode_many(decoded_regions, options)?
                .map(|bytes| {
                    bytes
                        .into_iter()
                        .map(|bytes| bytes.slice(..bytes.len().saturating_sub(1)))
                        .collect()
                }))
        }

        fn supports_partial_decode(&self) -> bool {
            true
        }
    }

    /// The decoded bytes of built-in data types are read directly, without converting them with the `bytes` codec.
    #[test]
    fn codec_packbits_reads_built_in_data_types_directly() -> Result<(), Box<dyn std::error::Error>>
    {
        for data_type in [
            data_type::bool(),
            data_type::uint4(),
            data_type::int8(),
            data_type::int16(),
            data_type::uint32(),
            data_type::int64(),
            data_type::bfloat16(),
            data_type::float32(),
            data_type::float64(),
            data_type::complex64(),
            data_type::complex_float64(),
        ] {
            let fill_value = FillValue::new(vec![0; data_type.fixed_size().unwrap()]);
            let codec = PackBitsCodec::default().bind(
                data_type.clone(),
                fill_value,
                PackBitsBitOrder::LsbFirst,
            )?;
            assert!(!codec.convert_bytes, "{data_type}");
            if codec.layout.component_size_bytes > 1 {
                assert_eq!(codec.layout.endianness, Endianness::native(), "{data_type}");
            }
        }
        Ok(())
    }

    /// Decoded bytes that are converted with the `bytes` codec pack as if they were read directly.
    #[test]
    fn codec_packbits_converts_bytes() -> Result<(), Box<dyn std::error::Error>> {
        use zarrs_codec::ArrayToBytesCodecTraits;

        let options = CodecOptions::default();
        let chunk_shape = [NonZeroU64::new(5).unwrap(), NonZeroU64::new(7).unwrap()];
        let subset = ArraySubset::new_with_ranges(&[1..4, 2..6]);
        let data_type = data_type::int16();
        let elements: Vec<i16> = (-17..18).map(|i| i * 100).collect();
        let bytes = i16::to_array_bytes(&data_type, &elements)?.into_owned();
        for (first_bit, last_bit) in [(None, None), (Some(2), Some(12))] {
            let codec =
                PackBitsCodec::new(PackBitsPaddingEncoding::FirstByte, first_bit, last_bit)?.bind(
                    data_type.clone(),
                    FillValue::from(0i16),
                    PackBitsBitOrder::LsbFirst,
                )?;
            // Convert to the other byte order, so that bytes are swapped on any host
            let mut converted = codec.clone();
            converted.convert_bytes = true;
            converted.layout.endianness = match Endianness::native() {
                Endianness::Little => Endianness::Big,
                Endianness::Big => Endianness::Little,
            };
            let (codec, converted) = (Arc::new(codec), Arc::new(converted));

            let encoded = codec.encode(bytes.clone(), &chunk_shape, &options)?;
            assert_eq!(
                converted.encode(bytes.clone(), &chunk_shape, &options)?,
                encoded
            );
            assert_eq!(
                converted.decode(encoded.clone(), &chunk_shape, &options)?,
                codec.decode(encoded.clone(), &chunk_shape, &options)?
            );
            assert_eq!(
                converted
                    .clone()
                    .partial_decoder(Arc::new(encoded.clone()), &chunk_shape, &options)?
                    .partial_decode(&subset, &options)?,
                codec
                    .clone()
                    .partial_decoder(Arc::new(encoded.clone()), &chunk_shape, &options)?
                    .partial_decode(&subset, &options)?
            );
        }
        Ok(())
    }

    /// Encoded bytes with an incorrect length or padding encoding byte are rejected by decoding.
    ///
    /// Partial decoding only checks the byte ranges it reads.
    /// Chunks with the length of those written by zarrs 0.20-0.23 are tested by [`codec_packbits_reads_legacy_chunks`].
    #[test]
    fn codec_packbits_rejects_invalid_encoded_bytes() -> Result<(), Box<dyn std::error::Error>> {
        let options = CodecOptions::default();
        let chunk_shape = [NonZeroU64::new(4).unwrap(), NonZeroU64::new(3).unwrap()];
        let subset = ArraySubset::new_with_ranges(&[1..3, 1..2]);
        for (data_type, fill_value) in [
            (data_type::float32(), FillValue::from(0.0f32)),
            (data_type::uint4(), FillValue::from(0u8)),
        ] {
            let elements: Vec<u8> = (0..12 * data_type.fixed_size().unwrap())
                .map(|i| u8::try_from(i % 7).unwrap())
                .collect();
            for encoding in [
                PackBitsPaddingEncoding::None,
                PackBitsPaddingEncoding::FirstByte,
                PackBitsPaddingEncoding::LastByte,
            ] {
                let codec = Arc::new(PackBitsCodec::new(encoding, None, None)?).with_context(
                    data_type.clone(),
                    fill_value.clone(),
                    &CodecSpecificOptions::default(),
                )?;
                let encoded = codec
                    .encode(
                        ArrayBytes::new_flen(elements.clone()),
                        &chunk_shape,
                        &options,
                    )?
                    .into_static();
                let partial_decode = |encoded: &CowBytes<'static>| {
                    codec
                        .clone()
                        .partial_decoder(Arc::new(encoded.clone()), &chunk_shape, &options)?
                        .partial_decode(&subset, &options)
                        .map(ArrayBytes::into_owned)
                };
                codec.decode(encoded.clone(), &chunk_shape, &options)?;
                let expected = partial_decode(&encoded)?;

                // Invalid encoded bytes, and whether partial decoding can read them as the valid bytes
                let mut invalid = vec![([encoded.as_slice(), &[0]].concat(), true)];
                // One byte shorter is the length of a legacy chunk with whole-byte components (e.g. `float32`)
                let legacy =
                    encoding != PackBitsPaddingEncoding::None && data_type == data_type::float32();
                if !legacy {
                    invalid.push((encoded[..encoded.len() - 1].to_vec(), true));
                }
                if encoding != PackBitsPaddingEncoding::None {
                    // Omitting the padding encoding byte offsets the packed elements with `first_byte`
                    if !legacy {
                        let (_, packed) = split_padding(encoding, &encoded);
                        invalid.push((
                            packed.to_vec(),
                            encoding == PackBitsPaddingEncoding::LastByte,
                        ));
                    }
                    let mut incorrect_padding = encoded.to_vec();
                    let index = if encoding == PackBitsPaddingEncoding::FirstByte {
                        0
                    } else {
                        encoded.len() - 1
                    };
                    incorrect_padding[index] ^= 1;
                    invalid.push((incorrect_padding, true));
                }
                for (invalid, partial_decode_valid) in invalid {
                    let invalid = CowBytes::from(invalid);
                    assert!(
                        codec
                            .decode(invalid.clone(), &chunk_shape, &options)
                            .is_err(),
                        "{data_type} {encoding:?} {invalid:?}"
                    );
                    // Partial decoding only checks what it needs to, so may succeed
                    if partial_decode_valid && let Ok(decoded) = partial_decode(&invalid) {
                        assert_eq!(decoded, expected, "{data_type} {encoding:?} {invalid:?}");
                    }
                }

                // Short byte ranges are not unpacked as zeros
                let partial_decoder = codec.clone().partial_decoder(
                    Arc::new(TruncatingPartialDecoder(encoded.clone())),
                    &chunk_shape,
                    &options,
                )?;
                assert!(partial_decoder.partial_decode(&subset, &options).is_err());
            }
        }
        Ok(())
    }

    /// A legacy chunk with its padding encoding, data type, and decoded elements.
    type LegacyChunk = (
        PackBitsPaddingEncoding,
        DataType,
        CowBytes<'static>,
        ArrayBytes<'static>,
    );

    /// The legacy chunks of zarrs 0.20-0.23 and their elements for [`codec_packbits_reads_legacy_chunks`].
    ///
    /// These omitted the padding encoding byte if all bits of whole-byte components are packed.
    fn legacy_chunks() -> Result<Vec<LegacyChunk>, Box<dyn std::error::Error>> {
        let options = CodecOptions::default();
        let mut chunks = Vec::new();
        for (data_type, fill_value) in [
            (data_type::float32(), FillValue::from(0.0f32)),
            (data_type::int8(), FillValue::from(0i8)),
        ] {
            let elements = ArrayBytes::new_flen(
                (0..12 * data_type.fixed_size().unwrap())
                    .map(|i| u8::try_from(i * 37 % 251).unwrap())
                    .collect::<Vec<u8>>(),
            );
            for encoding in [
                PackBitsPaddingEncoding::FirstByte,
                PackBitsPaddingEncoding::LastByte,
            ] {
                let codec = Arc::new(PackBitsCodec::new(encoding, None, None)?).with_context(
                    data_type.clone(),
                    fill_value.clone(),
                    &CodecSpecificOptions::default(),
                )?;
                let encoded = codec.encode(elements.clone(), &LEGACY_CHUNK_SHAPE, &options)?;
                let (padding, packed) = split_padding(encoding, &encoded);
                assert_eq!(padding, Some(0));
                chunks.push((
                    encoding,
                    data_type.clone(),
                    CowBytes::from(packed.to_vec()),
                    elements.clone(),
                ));
            }
        }
        Ok(chunks)
    }

    const LEGACY_CHUNK_SHAPE: [NonZeroU64; 2] =
        [NonZeroU64::new(4).unwrap(), NonZeroU64::new(3).unwrap()];

    /// Subsets at the start, in the middle, touching the last element, and of all elements of a legacy chunk.
    fn legacy_chunk_subsets() -> [ArraySubset; 4] {
        [
            ArraySubset::new_with_ranges(&[0..1, 0..2]),
            ArraySubset::new_with_ranges(&[1..3, 1..2]),
            ArraySubset::new_with_ranges(&[2..4, 1..3]),
            ArraySubset::new_with_ranges(&[0..4, 0..3]),
        ]
    }

    /// The chunks of zarrs 0.20-0.23 that omitted the padding encoding byte are identified by their length.
    #[test]
    fn codec_packbits_reads_legacy_chunks() -> Result<(), Box<dyn std::error::Error>> {
        let options = CodecOptions::default();
        let shape = [4, 3];
        for (encoding, data_type, legacy, elements) in legacy_chunks()? {
            let codec = Arc::new(PackBitsCodec::new(encoding, None, None)?).with_context(
                data_type.clone(),
                FillValue::new(vec![0; data_type.fixed_size().unwrap()]),
                &CodecSpecificOptions::default(),
            )?;
            assert_eq!(
                codec.decode(legacy.clone(), &LEGACY_CHUNK_SHAPE, &options)?,
                elements,
                "{data_type} {encoding:?}"
            );
            for input in [
                Arc::new(legacy.clone()) as Arc<dyn BytesPartialDecoderTraits>,
                // A store that clamps byte ranges to the end of the chunk instead of erroring
                Arc::new(ClampingPartialDecoder(legacy.clone())),
            ] {
                let partial_decoder =
                    codec
                        .clone()
                        .partial_decoder(input, &LEGACY_CHUNK_SHAPE, &options)?;
                for subset in legacy_chunk_subsets() {
                    assert_eq!(
                        partial_decoder.partial_decode(&subset, &options)?,
                        elements.extract_array_subset(&subset, &shape, &data_type)?,
                        "{data_type} {encoding:?} {subset:?}"
                    );
                }
            }

            // Chunks of other lengths are still rejected
            let short = CowBytes::from(legacy[..legacy.len() - 1].to_vec());
            assert!(
                codec
                    .decode(short.clone(), &LEGACY_CHUNK_SHAPE, &options)
                    .is_err()
            );
            if encoding == PackBitsPaddingEncoding::FirstByte {
                let partial_decoder = codec.clone().partial_decoder(
                    Arc::new(short),
                    &LEGACY_CHUNK_SHAPE,
                    &options,
                )?;
                for subset in legacy_chunk_subsets() {
                    assert!(partial_decoder.partial_decode(&subset, &options).is_err());
                }
            }
        }
        Ok(())
    }

    /// Legacy chunks are read from a store, which rejects byte ranges beyond the end of a chunk.
    #[test]
    fn codec_packbits_reads_legacy_chunks_from_store() -> Result<(), Box<dyn std::error::Error>> {
        use crate::array::{Array, ArrayMetadata};
        use zarrs_storage::store::MemoryStore;
        use zarrs_storage::{StoreKey, WritableStorageTraits};

        for (encoding, data_type, legacy, elements) in legacy_chunks()? {
            let store = Arc::new(MemoryStore::new());
            let array = Array::new_with_metadata(
                store.clone(),
                "/",
                ArrayMetadata::V3(legacy_array_metadata(encoding, &data_type)?),
            )?;
            store.set(&StoreKey::new("c/0/0")?, legacy)?;
            assert_eq!(array.retrieve_chunk::<ArrayBytes>(&[0, 0])?, elements);
            for subset in legacy_chunk_subsets() {
                assert_eq!(
                    array.retrieve_array_subset::<ArrayBytes>(&subset)?,
                    elements.extract_array_subset(&subset, &[4, 3], &data_type)?,
                    "{data_type} {encoding:?} {subset:?}"
                );
            }
        }
        Ok(())
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn codec_packbits_async_reads_legacy_chunks() -> Result<(), Box<dyn std::error::Error>> {
        use crate::array::{Array, ArrayMetadata};
        use zarrs_storage::store::AsyncMemoryStore;
        use zarrs_storage::{AsyncWritableStorageTraits, StoreKey};

        let options = CodecOptions::default();
        for (encoding, data_type, legacy, elements) in legacy_chunks()? {
            let codec = Arc::new(PackBitsCodec::new(encoding, None, None)?).with_context(
                data_type.clone(),
                FillValue::new(vec![0; data_type.fixed_size().unwrap()]),
                &CodecSpecificOptions::default(),
            )?;
            let partial_decoder = codec
                .async_partial_decoder(Arc::new(legacy.clone()), &LEGACY_CHUNK_SHAPE, &options)
                .await?;
            for subset in legacy_chunk_subsets() {
                assert_eq!(
                    partial_decoder.partial_decode(&subset, &options).await?,
                    elements.extract_array_subset(&subset, &[4, 3], &data_type)?,
                    "{data_type} {encoding:?} {subset:?}"
                );
            }

            let store = Arc::new(AsyncMemoryStore::new());
            let array = Array::new_with_metadata(
                store.clone(),
                "/",
                ArrayMetadata::V3(legacy_array_metadata(encoding, &data_type)?),
            )?;
            store.set(&StoreKey::new("c/0/0")?, legacy).await?;
            assert_eq!(
                array.async_retrieve_chunk::<ArrayBytes>(&[0, 0]).await?,
                elements
            );
            for subset in legacy_chunk_subsets() {
                assert_eq!(
                    array
                        .async_retrieve_array_subset::<ArrayBytes>(&subset)
                        .await?,
                    elements.extract_array_subset(&subset, &[4, 3], &data_type)?,
                    "{data_type} {encoding:?} {subset:?}"
                );
            }
        }
        Ok(())
    }

    fn legacy_array_metadata(
        encoding: PackBitsPaddingEncoding,
        data_type: &DataType,
    ) -> Result<zarrs_metadata::v3::ArrayMetadataV3, serde_json::Error> {
        serde_json::from_value(serde_json::json!({
            "zarr_format": 3,
            "node_type": "array",
            "shape": [4, 3],
            "data_type": data_type.name_v3().unwrap(),
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [4, 3]}},
            "chunk_key_encoding": {"name": "default"},
            "fill_value": 0,
            "codecs": [{"name": "packbits", "configuration": {"padding_encoding": encoding}}],
        }))
    }

    /// Returns each requested byte range clamped to the end of the bytes, like a store that does not reject them.
    struct ClampingPartialDecoder(CowBytes<'static>);

    impl BytesPartialDecoderTraits for ClampingPartialDecoder {
        fn exists(&self) -> Result<bool, zarrs_storage::StorageError> {
            Ok(true)
        }

        fn size_held(&self) -> usize {
            0
        }

        fn partial_decode_many(
            &self,
            decoded_regions: zarrs_storage::byte_range::ByteRangeIterator,
            _options: &CodecOptions,
        ) -> Result<Option<Vec<CowBytes<'_>>>, zarrs_codec::CodecError> {
            let size = self.0.len() as u64;
            Ok(Some(
                decoded_regions
                    .map(|byte_range| {
                        let start = byte_range.start(size).min(size) as usize;
                        let end = byte_range.end(size).min(size) as usize;
                        self.0.slice(start..end)
                    })
                    .collect(),
            ))
        }

        fn supports_partial_decode(&self) -> bool {
            true
        }
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn codec_packbits_float32_async_partial_decode() -> Result<(), Box<dyn std::error::Error>>
    {
        let chunk_shape = [NonZeroU64::new(4).unwrap()];
        let data_type = data_type::float32();
        let elements = [1.0f32, 2.0, 3.0, 4.0];
        let options = CodecOptions::default();
        for encoding in [
            PackBitsPaddingEncoding::None,
            PackBitsPaddingEncoding::FirstByte,
            PackBitsPaddingEncoding::LastByte,
        ] {
            let codec = Arc::new(super::PackBitsCodec::new(encoding, None, None)?).with_context(
                data_type.clone(),
                FillValue::from(0.0f32),
                &CodecSpecificOptions::default(),
            )?;
            let encoded = codec.encode(
                f32::to_array_bytes(&data_type, &elements)?.into_owned(),
                &chunk_shape,
                &options,
            )?;
            let decoded = codec.decode(encoded.clone(), &chunk_shape, &options)?;
            let decoded = f32::from_array_bytes(&data_type, decoded)?;
            assert_eq!(decoded, elements);

            let partial_decoder = codec
                .async_partial_decoder(Arc::new(encoded), &chunk_shape, &options)
                .await?;
            let decoded_partial = partial_decoder
                .partial_decode(&ArraySubset::new_with_ranges(&[1..3]), &options)
                .await?;
            assert_eq!(
                f32::from_array_bytes(&data_type, decoded_partial)?,
                decoded[1..3]
            );
        }
        Ok(())
    }

    #[test]
    fn codec_packbits_int16() -> Result<(), Box<dyn std::error::Error>> {
        for last_bit in 11..15 {
            for first_bit in 0..4 {
                for encoding in [
                    PackBitsPaddingEncoding::None,
                    PackBitsPaddingEncoding::FirstByte,
                    PackBitsPaddingEncoding::LastByte,
                ] {
                    let chunk_shape =
                        vec![NonZeroU64::new(8).unwrap(), NonZeroU64::new(5).unwrap()];
                    let data_type = data_type::int16();
                    let fill_value = FillValue::from(0i16);
                    let codec = Arc::new(
                        super::PackBitsCodec::new(encoding, Some(first_bit), Some(last_bit))
                            .unwrap(),
                    )
                    .with_context(
                        data_type.clone(),
                        fill_value.clone(),
                        &CodecSpecificOptions::default(),
                    )?;
                    let elements: Vec<i16> = (-20..20).map(|i| (i as i16) << first_bit).collect();
                    let bytes = i16::to_array_bytes(&data_type, &elements)?.into_owned();

                    // Encoding
                    let encoded =
                        codec.encode(bytes.clone(), &chunk_shape, &CodecOptions::default())?;
                    assert!(
                        (encoded.len() as u64) <= (40 * (last_bit - first_bit + 1)).div_ceil(8) + 1
                    );

                    // Decoding
                    let decoded = codec
                        .decode(encoded.clone(), &chunk_shape, &CodecOptions::default())
                        .unwrap();
                    assert_eq!(elements, i16::from_array_bytes(&data_type, decoded)?);

                    // Partial decoding
                    let partial_decoder = codec.partial_decoder(
                        Arc::new(encoded),
                        &chunk_shape,
                        &CodecOptions::default(),
                    )?;
                    let decoded_partial_chunk = partial_decoder.partial_decode(
                        &ArraySubset::new_with_ranges(&[2..5, 1..4]),
                        &CodecOptions::default(),
                    )?;
                    let answer: Vec<i16> = (2..5)
                        .flat_map(|i| (1..4).map(move |j| i * 5 + j))
                        .map(|idx| elements[idx])
                        .collect();
                    assert_eq!(
                        answer,
                        i16::from_array_bytes(&data_type, decoded_partial_chunk)?
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn codec_packbits_int8_first_bit() -> Result<(), Box<dyn std::error::Error>> {
        let chunk_shape = vec![NonZeroU64::new(16).unwrap()];
        let data_type = data_type::int8();
        let fill_value = FillValue::from(0i8);
        let codec = Arc::new(
            super::PackBitsCodec::new(PackBitsPaddingEncoding::None, Some(2), Some(5)).unwrap(),
        )
        .with_context(
            data_type.clone(),
            fill_value.clone(),
            &CodecSpecificOptions::default(),
        )?;
        let elements: Vec<i8> = (-8..8).map(|i| i << 2).collect();
        let bytes = i8::to_array_bytes(&data_type, &elements)?.into_owned();
        let encoded = codec.encode(bytes, &chunk_shape, &CodecOptions::default())?;

        // Decoding
        let decoded = codec.decode(encoded.clone(), &chunk_shape, &CodecOptions::default())?;
        assert_eq!(elements, i8::from_array_bytes(&data_type, decoded)?);

        // Partial decoding
        let partial_decoder =
            codec.partial_decoder(Arc::new(encoded), &chunk_shape, &CodecOptions::default())?;
        let decoded_partial_chunk = partial_decoder.partial_decode(
            &ArraySubset::new_with_ranges(&[3..13]),
            &CodecOptions::default(),
        )?;
        assert_eq!(
            &elements[3..13],
            i8::from_array_bytes(&data_type, decoded_partial_chunk)?
        );
        Ok(())
    }

    #[test]
    fn codec_packbits_int16_sign_extension() -> Result<(), Box<dyn std::error::Error>> {
        // The sign bit is in the low byte, so sign extension must extend through the high byte
        for last_bit in 0..8 {
            let chunk_shape = vec![NonZeroU64::new(8).unwrap(), NonZeroU64::new(4).unwrap()];
            let data_type = data_type::int16();
            let fill_value = FillValue::from(0i16);
            let codec = Arc::new(
                super::PackBitsCodec::new(PackBitsPaddingEncoding::None, None, Some(last_bit))
                    .unwrap(),
            )
            .with_context(
                data_type.clone(),
                fill_value.clone(),
                &CodecSpecificOptions::default(),
            )?;
            let max = 1i16 << last_bit;
            let elements: Vec<i16> = (0..32).map(|i| -max + i % (2 * max)).collect();
            let bytes = i16::to_array_bytes(&data_type, &elements)?.into_owned();
            let encoded = codec.encode(bytes, &chunk_shape, &CodecOptions::default())?;

            // Decoding
            let decoded = codec.decode(encoded.clone(), &chunk_shape, &CodecOptions::default())?;
            assert_eq!(elements, i16::from_array_bytes(&data_type, decoded)?);

            // Partial decoding
            let partial_decoder =
                codec.partial_decoder(Arc::new(encoded), &chunk_shape, &CodecOptions::default())?;
            let decoded_partial_chunk = partial_decoder.partial_decode(
                &ArraySubset::new_with_ranges(&[4..8, 0..4]),
                &CodecOptions::default(),
            )?;
            assert_eq!(
                &elements[16..],
                i16::from_array_bytes(&data_type, decoded_partial_chunk)?
            );
        }
        Ok(())
    }

    #[test]
    fn codec_packbits_uint2() -> Result<(), Box<dyn std::error::Error>> {
        for encoding in [
            PackBitsPaddingEncoding::None,
            PackBitsPaddingEncoding::FirstByte,
            PackBitsPaddingEncoding::LastByte,
        ] {
            let chunk_shape = vec![NonZeroU64::new(4).unwrap(), NonZeroU64::new(1).unwrap()];
            let data_type = data_type::uint2();
            let fill_value = FillValue::from(0u8);
            let codec = Arc::new(super::PackBitsCodec::new(encoding, None, None).unwrap())
                .with_context(
                    data_type.clone(),
                    fill_value.clone(),
                    &CodecSpecificOptions::default(),
                )?;

            let elements: Vec<u8> = (0..4).map(|i| i as u8).collect();
            let bytes = u8::to_array_bytes(&data_type, &elements)?.into_owned();

            // Encoding
            let encoded = codec.encode(bytes.clone(), &chunk_shape, &CodecOptions::default())?;
            assert!((encoded.len() as u64) <= (4 * 4).div_ceil(&8) + 1);

            // Decoding
            let decoded = codec
                .decode(encoded.clone(), &chunk_shape, &CodecOptions::default())
                .unwrap();
            assert_eq!(elements, u8::from_array_bytes(&data_type, decoded)?);
        }
        Ok(())
    }

    #[test]
    fn codec_packbits_uint4() -> Result<(), Box<dyn std::error::Error>> {
        for encoding in [
            PackBitsPaddingEncoding::None,
            PackBitsPaddingEncoding::FirstByte,
            PackBitsPaddingEncoding::LastByte,
        ] {
            let chunk_shape = vec![NonZeroU64::new(16).unwrap(), NonZeroU64::new(1).unwrap()];
            let data_type = data_type::uint4();
            let fill_value = FillValue::from(0u8);
            let codec = Arc::new(super::PackBitsCodec::new(encoding, None, None).unwrap())
                .with_context(
                    data_type.clone(),
                    fill_value.clone(),
                    &CodecSpecificOptions::default(),
                )?;

            let elements: Vec<u8> = (0..16).map(|i| i as u8).collect();
            let bytes = u8::to_array_bytes(&data_type, &elements)?.into_owned();

            // Encoding
            let encoded = codec.encode(bytes.clone(), &chunk_shape, &CodecOptions::default())?;
            assert!((encoded.len() as u64) <= (4 * 16).div_ceil(&8) + 1);

            // Decoding
            let decoded = codec
                .decode(encoded.clone(), &chunk_shape, &CodecOptions::default())
                .unwrap();
            assert_eq!(elements, u8::from_array_bytes(&data_type, decoded)?);
        }
        Ok(())
    }

    #[test]
    fn codec_packbits_int2() -> Result<(), Box<dyn std::error::Error>> {
        for encoding in [
            PackBitsPaddingEncoding::None,
            PackBitsPaddingEncoding::FirstByte,
            PackBitsPaddingEncoding::LastByte,
        ] {
            let chunk_shape = vec![NonZeroU64::new(4).unwrap(), NonZeroU64::new(1).unwrap()];
            let data_type = data_type::int2();
            let fill_value = FillValue::from(0i8);
            let codec = Arc::new(super::PackBitsCodec::new(encoding, None, None).unwrap())
                .with_context(
                    data_type.clone(),
                    fill_value.clone(),
                    &CodecSpecificOptions::default(),
                )?;

            let elements: Vec<i8> = (-2..2).map(|i| i as i8).collect();
            let bytes = i8::to_array_bytes(&data_type, &elements)?.into_owned();

            // Encoding
            let encoded = codec.encode(bytes.clone(), &chunk_shape, &CodecOptions::default())?;
            assert!((encoded.len() as u64) <= (4 * 4).div_ceil(&8) + 1);

            // Decoding
            let decoded = codec
                .decode(encoded.clone(), &chunk_shape, &CodecOptions::default())
                .unwrap();
            assert_eq!(elements, i8::from_array_bytes(&data_type, decoded)?);

            // Partial decoding
            let partial_decoder =
                codec.partial_decoder(Arc::new(encoded), &chunk_shape, &CodecOptions::default())?;
            let decoded_partial_chunk = partial_decoder.partial_decode(
                &ArraySubset::new_with_ranges(&[0..2, 0..1]),
                &CodecOptions::default(),
            )?;
            assert_eq!(
                &elements[..2],
                i8::from_array_bytes(&data_type, decoded_partial_chunk)?
            );
        }
        Ok(())
    }

    #[test]
    fn codec_packbits_int4() -> Result<(), Box<dyn std::error::Error>> {
        for encoding in [
            PackBitsPaddingEncoding::None,
            PackBitsPaddingEncoding::FirstByte,
            PackBitsPaddingEncoding::LastByte,
        ] {
            let chunk_shape = vec![NonZeroU64::new(16).unwrap(), NonZeroU64::new(1).unwrap()];
            let data_type = data_type::int4();
            let fill_value = FillValue::from(0i8);
            let codec = Arc::new(super::PackBitsCodec::new(encoding, None, None).unwrap())
                .with_context(
                    data_type.clone(),
                    fill_value.clone(),
                    &CodecSpecificOptions::default(),
                )?;

            let elements: Vec<i8> = (-8..8).map(|i| i as i8).collect();
            let bytes = i8::to_array_bytes(&data_type, &elements)?.into_owned();

            // Encoding
            let encoded = codec.encode(bytes.clone(), &chunk_shape, &CodecOptions::default())?;
            assert!((encoded.len() as u64) <= (4 * 16).div_ceil(&8) + 1);

            // Decoding
            let decoded = codec
                .decode(encoded.clone(), &chunk_shape, &CodecOptions::default())
                .unwrap();
            assert_eq!(elements, i8::from_array_bytes(&data_type, decoded)?);

            // Partial decoding
            let partial_decoder =
                codec.partial_decoder(Arc::new(encoded), &chunk_shape, &CodecOptions::default())?;
            let decoded_partial_chunk = partial_decoder.partial_decode(
                &ArraySubset::new_with_ranges(&[0..2, 0..1]),
                &CodecOptions::default(),
            )?;
            assert_eq!(
                &elements[..2],
                i8::from_array_bytes(&data_type, decoded_partial_chunk)?
            );
        }
        Ok(())
    }

    #[test]
    fn codec_packbits_float4_e2m1fn() -> Result<(), Box<dyn std::error::Error>> {
        for encoding in [
            PackBitsPaddingEncoding::None,
            PackBitsPaddingEncoding::FirstByte,
            PackBitsPaddingEncoding::LastByte,
        ] {
            let chunk_shape = vec![NonZeroU64::new(16).unwrap(), NonZeroU64::new(1).unwrap()];
            let data_type = data_type::float4_e2m1fn();
            let fill_value = FillValue::from(0u8);
            let codec = Arc::new(super::PackBitsCodec::new(encoding, None, None).unwrap())
                .with_context(
                    data_type.clone(),
                    fill_value.clone(),
                    &CodecSpecificOptions::default(),
                )?;

            let bytes = ArrayBytes::new_flen((0..16).map(|i| i as u8).collect::<Vec<u8>>());

            // Encoding
            let encoded = codec.encode(bytes.clone(), &chunk_shape, &CodecOptions::default())?;
            assert!((encoded.len() as u64) <= (4 * 16).div_ceil(&8) + 1);

            // Decoding
            let decoded = codec
                .decode(encoded.clone(), &chunk_shape, &CodecOptions::default())
                .unwrap();
            assert_eq!(bytes, decoded);
        }
        Ok(())
    }

    #[test]
    fn codec_packbits_float6_e2m3fn() -> Result<(), Box<dyn std::error::Error>> {
        for encoding in [
            PackBitsPaddingEncoding::None,
            PackBitsPaddingEncoding::FirstByte,
            PackBitsPaddingEncoding::LastByte,
        ] {
            let chunk_shape = vec![NonZeroU64::new(64).unwrap(), NonZeroU64::new(1).unwrap()];
            let data_type = data_type::float6_e2m3fn();
            let fill_value = FillValue::from(0u8);
            let codec = Arc::new(super::PackBitsCodec::new(encoding, None, None).unwrap())
                .with_context(
                    data_type.clone(),
                    fill_value.clone(),
                    &CodecSpecificOptions::default(),
                )?;

            let bytes = ArrayBytes::new_flen((0..64).map(|i| i as u8).collect::<Vec<u8>>());

            // Encoding
            let encoded = codec.encode(bytes.clone(), &chunk_shape, &CodecOptions::default())?;
            assert!((encoded.len() as u64) <= (6 * 64).div_ceil(&8) + 1);

            // Decoding
            let decoded = codec
                .decode(encoded.clone(), &chunk_shape, &CodecOptions::default())
                .unwrap();
            assert_eq!(bytes, decoded);
        }
        Ok(())
    }

    #[test]
    fn codec_packbits_float6_e3m2fn() -> Result<(), Box<dyn std::error::Error>> {
        for encoding in [
            PackBitsPaddingEncoding::None,
            PackBitsPaddingEncoding::FirstByte,
            PackBitsPaddingEncoding::LastByte,
        ] {
            let chunk_shape = vec![NonZeroU64::new(64).unwrap(), NonZeroU64::new(1).unwrap()];
            let data_type = data_type::float6_e3m2fn();
            let fill_value = FillValue::from(0u8);
            let codec = Arc::new(super::PackBitsCodec::new(encoding, None, None).unwrap())
                .with_context(
                    data_type.clone(),
                    fill_value.clone(),
                    &CodecSpecificOptions::default(),
                )?;

            let bytes = ArrayBytes::new_flen((0..64).map(|i| i as u8).collect::<Vec<u8>>());

            // Encoding
            let encoded = codec.encode(bytes.clone(), &chunk_shape, &CodecOptions::default())?;
            assert!((encoded.len() as u64) <= (6 * 64).div_ceil(&8) + 1);

            // Decoding
            let decoded = codec
                .decode(encoded.clone(), &chunk_shape, &CodecOptions::default())
                .unwrap();
            assert_eq!(bytes, decoded);
        }
        Ok(())
    }
}
