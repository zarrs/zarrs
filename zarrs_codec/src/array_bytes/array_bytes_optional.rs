use std::ops::Range;

use crate::{ArrayBytes, ArrayBytesOffsetsElement, ArrayBytesOffsetsSlice, CowBytes};

/// Optional array bytes composed of data and a validity mask.
///
/// The mask is 1 byte per element where 0 = null/missing and 1 = present.
/// Other mask values are invalid.
///
/// The mask is not checked at construction.
/// [`ArrayBytes::validate`] checks that every mask has one byte per element and contains only 0 or 1.
///
/// # Equality Semantics
///
/// Equality is semantic: two `ArrayBytesOptional` values are equal if they have the same
/// layout (nesting depth, element widths) and the same value at every element:
/// - Masks are compared at visible positions, i.e. where all ancestors are present.
/// - Payloads are compared only where all ancestors and the element itself are present.
///
/// Bytes beneath a null element are **not** compared, since they may differ after
/// round-trip encoding/decoding (e.g., fill value vs original bytes).
/// A null outer element is never equal to a present outer element holding a null (`None` ≠ `Some(None)`).
/// Variable-length elements are compared by their bytes, regardless of offset width or unreferenced bytes.
///
/// Values with a malformed layout (e.g. a mask length that differs from the element count) are
/// compared structurally instead. Validation is stricter than equality: [`ArrayBytes::validate`]
/// enforces the storage contract, whereas equality only defines when two values are semantically the same.
#[derive(Clone, Debug)]
pub struct ArrayBytesOptional<'a> {
    data: Box<ArrayBytes<'a>>,
    mask: CowBytes<'a>,
}

impl<'a> ArrayBytesOptional<'a> {
    /// Create a new `ArrayBytesOptional`.
    ///
    /// The mask is not validated, see [`ArrayBytes::validate`].
    pub fn new(data: impl Into<Box<ArrayBytes<'a>>>, mask: impl Into<CowBytes<'a>>) -> Self {
        let data = data.into();
        let mask = mask.into();

        Self { data, mask }
    }

    /// Get the underlying data.
    #[must_use]
    pub fn data(&self) -> &ArrayBytes<'a> {
        &self.data
    }

    /// Get the validity mask.
    #[must_use]
    pub fn mask(&self) -> &CowBytes<'a> {
        &self.mask
    }

    /// Get the number of elements.
    #[must_use]
    pub fn num_elements(&self) -> usize {
        self.mask.len()
    }

    /// Consume self and return the data and mask.
    #[must_use]
    pub fn into_parts(self) -> (Box<ArrayBytes<'a>>, CowBytes<'a>) {
        (self.data, self.mask)
    }

    /// Convert into owned [`ArrayBytesOptional<'static>`].
    #[must_use]
    pub fn into_owned(self) -> ArrayBytesOptional<'static> {
        ArrayBytesOptional {
            data: Box::new((*self.data).into_owned()),
            mask: self.mask.into_static(),
        }
    }
}

impl ArrayBytesOptional<'_> {
    /// Returns true if the mask contains only 0 (null) or 1 (present).
    pub(crate) fn is_mask_binary(&self) -> bool {
        self.mask.iter().fold(0u8, |acc, &b| acc | b) <= 1
    }

    /// Returns true if every layer has `num_elements` elements and a mask of only 0 or 1.
    fn is_well_formed(&self, num_elements: usize) -> bool {
        self.mask.len() == num_elements
            && self.is_mask_binary()
            && match self.data.as_ref() {
                // `is_multiple_of(0)` is only true for 0, so empty arrays must have empty bytes
                ArrayBytes::Fixed(bytes) => bytes.len().is_multiple_of(num_elements),
                ArrayBytes::Variable(bytes) => bytes.offsets().num_elements() == num_elements,
                ArrayBytes::Optional(inner) => inner.is_well_formed(num_elements),
            }
    }
}

/// Returns true if `a` and `b` have identical masks and bytes at every layer, without semantic comparison.
fn structural_eq(a: &ArrayBytesOptional<'_>, b: &ArrayBytesOptional<'_>) -> bool {
    a.mask == b.mask
        && match (a.data.as_ref(), b.data.as_ref()) {
            (ArrayBytes::Optional(a), ArrayBytes::Optional(b)) => structural_eq(a, b),
            (a, b) => a == b,
        }
}

/// Returns true if `a` and `b` have the same variant at each layer and the same fixed-length size.
fn same_layout(a: &ArrayBytes<'_>, b: &ArrayBytes<'_>) -> bool {
    match (a, b) {
        (ArrayBytes::Fixed(a), ArrayBytes::Fixed(b)) => a.len() == b.len(),
        (ArrayBytes::Variable(_), ArrayBytes::Variable(_)) => true,
        (ArrayBytes::Optional(a), ArrayBytes::Optional(b)) => same_layout(&a.data, &b.data),
        _ => false,
    }
}

/// The number of elements compared per block.
///
/// Blocks bound the visible mask of nested layers to a stack buffer, and allow an early exit.
const EQ_BLOCK: usize = 4096;

/// Returns true if well-formed `a` and `b` with the same layout are equal at every visible element.
///
/// Each block of elements is compared one layer at a time, outermost first, narrowing a mask of
/// visible elements (where all ancestors are present).
fn visible_eq(a: &ArrayBytesOptional<'_>, b: &ArrayBytesOptional<'_>) -> bool {
    let num_elements = a.num_elements();
    let mut buffer = [0u8; EQ_BLOCK];
    (0..num_elements).step_by(EQ_BLOCK).all(|start| {
        let range = start..num_elements.min(start + EQ_BLOCK);
        visible_eq_block(a, b, num_elements, range, &mut buffer)
    })
}

/// [`visible_eq`] for the elements in `range`, using `buffer` for the visible mask of nested layers.
fn visible_eq_block(
    a: &ArrayBytesOptional<'_>,
    b: &ArrayBytesOptional<'_>,
    num_elements: usize,
    range: Range<usize>,
    buffer: &mut [u8; EQ_BLOCK],
) -> bool {
    // Every element of the outermost layer is visible
    let outer_mask = &a.mask[range.clone()];
    if *outer_mask != b.mask[range.clone()] {
        return false;
    }
    let buffer = &mut buffer[..range.len()];
    let mut narrowed = false;
    let (mut a, mut b) = (a.data.as_ref(), b.data.as_ref());
    loop {
        let visible: &[u8] = if narrowed { buffer } else { outer_mask };
        match (a, b) {
            (ArrayBytes::Optional(a_inner), ArrayBytes::Optional(b_inner)) => {
                let a_mask = &a_inner.mask[range.clone()];
                if !masks_eq_where(a_mask, &b_inner.mask[range.clone()], visible) {
                    return false;
                }
                if narrowed {
                    for (visible, &mask) in buffer.iter_mut().zip(a_mask) {
                        *visible &= mask;
                    }
                } else {
                    for ((visible, &outer), &mask) in buffer.iter_mut().zip(outer_mask).zip(a_mask)
                    {
                        *visible = outer & mask;
                    }
                    narrowed = true;
                }
                (a, b) = (&a_inner.data, &b_inner.data);
            }
            (ArrayBytes::Fixed(a), ArrayBytes::Fixed(b)) => {
                let width = a.len() / num_elements;
                let bytes = range.start * width..range.end * width;
                return fixed_eq_where(&a[bytes.clone()], &b[bytes], width, visible);
            }
            (ArrayBytes::Variable(a), ArrayBytes::Variable(b)) => {
                let offsets = range.start..=range.end;
                return match (a.offsets().as_slice(), b.offsets().as_slice()) {
                    (
                        ArrayBytesOffsetsSlice::U32(a_offsets),
                        ArrayBytesOffsetsSlice::U32(b_offsets),
                    ) => variable_eq_where(
                        a.bytes(),
                        &a_offsets[offsets.clone()],
                        b.bytes(),
                        &b_offsets[offsets],
                        visible,
                    ),
                    (
                        ArrayBytesOffsetsSlice::U32(a_offsets),
                        ArrayBytesOffsetsSlice::U64(b_offsets),
                    ) => variable_eq_where(
                        a.bytes(),
                        &a_offsets[offsets.clone()],
                        b.bytes(),
                        &b_offsets[offsets],
                        visible,
                    ),
                    (
                        ArrayBytesOffsetsSlice::U64(a_offsets),
                        ArrayBytesOffsetsSlice::U32(b_offsets),
                    ) => variable_eq_where(
                        a.bytes(),
                        &a_offsets[offsets.clone()],
                        b.bytes(),
                        &b_offsets[offsets],
                        visible,
                    ),
                    (
                        ArrayBytesOffsetsSlice::U64(a_offsets),
                        ArrayBytesOffsetsSlice::U64(b_offsets),
                    ) => variable_eq_where(
                        a.bytes(),
                        &a_offsets[offsets.clone()],
                        b.bytes(),
                        &b_offsets[offsets],
                        visible,
                    ),
                };
            }
            _ => return false,
        }
    }
}

/// Returns true if binary masks `a` and `b` are equal where binary mask `visible` is set.
fn masks_eq_where(a: &[u8], b: &[u8], visible: &[u8]) -> bool {
    // Branchless, so it vectorises
    a.iter()
        .zip(b)
        .zip(visible)
        .fold(0u8, |acc, ((&a, &b), &visible)| acc | ((a ^ b) & visible))
        == 0
}

/// Returns true if elements of `width` bytes in `a` and `b` are equal where binary mask `visible` is set.
fn fixed_eq_where(a: &[u8], b: &[u8], width: usize, visible: &[u8]) -> bool {
    match width {
        0 => true,
        1 => fixed_eq_where_n::<1>(a, b, visible),
        2 => fixed_eq_where_n::<2>(a, b, visible),
        4 => fixed_eq_where_n::<4>(a, b, visible),
        8 => fixed_eq_where_n::<8>(a, b, visible),
        16 => fixed_eq_where_n::<16>(a, b, visible),
        width => fixed_eq_where_runs(a, b, width, visible),
    }
}

/// [`fixed_eq_where`] for elements of `N` bytes, compared without branching on `visible` so it vectorises.
fn fixed_eq_where_n<const N: usize>(a: &[u8], b: &[u8], visible: &[u8]) -> bool {
    let (a, b) = (a.as_chunks::<N>().0, b.as_chunks::<N>().0);
    a.iter()
        .zip(b)
        .zip(visible)
        .fold(0u8, |acc, ((a, b), &visible)| {
            acc | (u8::from(a != b) & visible)
        })
        == 0
}

/// [`fixed_eq_where`] for elements of `width` bytes, comparing runs of visible elements.
fn fixed_eq_where_runs(a: &[u8], b: &[u8], width: usize, visible: &[u8]) -> bool {
    let mut start = 0;
    while let Some(offset) = visible[start..].iter().position(|&v| v != 0) {
        start += offset;
        let end = visible[start..]
            .iter()
            .position(|&v| v == 0)
            .map_or(visible.len(), |len| start + len);
        let range = start * width..end * width;
        if a[range.clone()] != b[range] {
            return false;
        }
        start = end;
    }
    true
}

/// Returns true if variable-length elements of `a` and `b` are equal where binary mask `visible` is set.
fn variable_eq_where<A: ArrayBytesOffsetsElement, B: ArrayBytesOffsetsElement>(
    a: &[u8],
    a_offsets: &[A],
    b: &[u8],
    b_offsets: &[B],
    visible: &[u8],
) -> bool {
    fn element(bytes: &[u8], start: u64, end: u64) -> Option<&[u8]> {
        bytes.get(usize::try_from(start).ok()?..usize::try_from(end).ok()?)
    }
    a_offsets
        .windows(2)
        .zip(b_offsets.windows(2))
        .zip(visible)
        .all(|((a_range, b_range), &visible)| {
            visible == 0
                || match (
                    element(a, a_range[0].into(), a_range[1].into()),
                    element(b, b_range[0].into(), b_range[1].into()),
                ) {
                    (Some(a), Some(b)) => a == b,
                    _ => false,
                }
        })
}

impl PartialEq for ArrayBytesOptional<'_> {
    fn eq(&self, other: &Self) -> bool {
        // Structural equality implies semantic equality.
        if structural_eq(self, other) {
            return true;
        }
        // Otherwise, only well-formed values are compared semantically (ignoring bytes beneath a null).
        // Structural equality preserves well-formedness, so a malformed value never equals a well-formed one,
        // and `Eq` remains reflexive, symmetric and transitive.
        let num_elements = self.num_elements();
        self.is_well_formed(num_elements)
            && other.is_well_formed(num_elements)
            && same_layout(&self.data, &other.data)
            && visible_eq(self, other)
    }
}

impl Eq for ArrayBytesOptional<'_> {}

#[cfg(test)]
mod tests {
    use crate::{ArrayBytes, ArrayBytesOffsets};

    fn fixed(bytes: &[u8]) -> ArrayBytes<'static> {
        ArrayBytes::new_flen(bytes.to_vec())
    }

    fn vlen<T: crate::ArrayBytesOffsetsElement>(
        bytes: &[u8],
        offsets: Vec<T>,
    ) -> ArrayBytes<'static> {
        ArrayBytes::new_vlen(bytes.to_vec(), ArrayBytesOffsets::new(offsets).unwrap()).unwrap()
    }

    /// Wrap `data` in optional layers, innermost mask first.
    fn masked(data: ArrayBytes<'static>, masks: &[&[u8]]) -> ArrayBytes<'static> {
        masks
            .iter()
            .fold(data, |data, mask| data.with_optional_mask(mask.to_vec()))
    }

    #[test]
    fn optional_eq_ignores_hidden_elements() {
        // Depth 2: element 1 is null at the outer layer
        let a = masked(fixed(&[5, 6]), &[&[1, 0], &[1, 0]]);
        let b = masked(fixed(&[5, 9]), &[&[1, 1], &[1, 0]]);
        assert_eq!(a, b);
        // The same differences are visible if element 1 is present
        let a = masked(fixed(&[5, 6]), &[&[1, 0], &[1, 1]]);
        let b = masked(fixed(&[5, 9]), &[&[1, 1], &[1, 1]]);
        assert_ne!(a, b);
        let c = masked(fixed(&[5, 9]), &[&[1, 0], &[1, 1]]);
        assert_ne!(c, b);

        // Depth 3: element 0 is null at the middle layer, element 1 at the outer layer
        let a = masked(fixed(&[1, 2, 3]), &[&[1, 1, 1], &[0, 1, 1], &[1, 0, 1]]);
        let b = masked(fixed(&[9, 8, 3]), &[&[0, 0, 1], &[0, 0, 1], &[1, 0, 1]]);
        assert_eq!(a, b);
        for visible in [
            masked(fixed(&[9, 8, 3]), &[&[0, 0, 1], &[1, 0, 1], &[1, 0, 1]]),
            masked(fixed(&[9, 8, 3]), &[&[0, 0, 0], &[0, 0, 1], &[1, 0, 1]]),
            masked(fixed(&[9, 8, 4]), &[&[0, 0, 1], &[0, 0, 1], &[1, 0, 1]]),
            masked(fixed(&[9, 8, 3]), &[&[0, 0, 1], &[0, 0, 1], &[1, 1, 1]]),
        ] {
            assert_ne!(a, visible);
            assert_ne!(visible, a);
        }
    }

    #[test]
    fn optional_eq_widths_and_blocks() {
        // Specialised widths, runs of visible elements for other widths, and multiple blocks
        for width in [1, 2, 3, 4, 8, 12, 16, 17] {
            for num_elements in [1, 5, 3 * super::EQ_BLOCK + 7] {
                // Every third element is null, except the last
                let hidden = |i: usize| i % 3 == 1 && i + 1 != num_elements;
                let mask: Vec<u8> = (0..num_elements).map(|i| u8::from(!hidden(i))).collect();
                let a: Vec<u8> = (0..num_elements * width)
                    .map(|i| u8::try_from(i % 251).unwrap())
                    .collect();
                let mut b = a.clone();
                // Differences beneath nulls are ignored
                for i in (0..num_elements).filter(|&i| hidden(i)) {
                    b[i * width + width - 1] ^= 0xff;
                }
                let wrap = |data: Vec<u8>, mask: Vec<u8>| masked(fixed(&data), &[&mask]);
                assert_eq!(wrap(a.clone(), mask.clone()), wrap(b.clone(), mask.clone()));
                // A difference in any byte of the last visible element is not
                let last = (num_elements - 1) * width;
                for byte in [last, last + width - 1] {
                    let mut c = b.clone();
                    c[byte] ^= 1;
                    assert_ne!(wrap(a.clone(), mask.clone()), wrap(c, mask.clone()));
                }
                // Nor is a mask difference
                let mut mask_b = mask.clone();
                mask_b[num_elements - 1] ^= 1;
                assert_ne!(wrap(a.clone(), mask.clone()), wrap(b.clone(), mask_b));
            }
        }
    }

    #[test]
    fn optional_eq_none_is_not_some_none() {
        let none = masked(fixed(&[0]), &[&[0], &[0]]);
        let some_none = masked(fixed(&[0]), &[&[0], &[1]]);
        assert_ne!(none, some_none);
        assert_ne!(some_none, none);
    }

    #[test]
    fn optional_eq_layouts() {
        // Differing fixed widths
        for mask in [[1, 1], [0, 0]] {
            let wide = masked(fixed(&[1, 0, 2, 0]), &[&mask]);
            let narrow = masked(fixed(&[1, 2]), &[&mask]);
            assert_ne!(wide, narrow);
            assert_ne!(narrow, wide);
        }
        // Differing depths and variants
        let depth1 = masked(fixed(&[1, 2]), &[&[0, 0]]);
        let depth2 = masked(fixed(&[1, 2]), &[&[0, 0], &[0, 0]]);
        let variable = masked(vlen(b"ab", vec![0u32, 1, 2]), &[&[0, 0]]);
        assert_ne!(depth1, depth2);
        assert_ne!(depth1, variable);
        assert_ne!(depth2, variable);

        // Empty arrays and zero-width elements
        assert_eq!(masked(fixed(&[]), &[&[]]), masked(fixed(&[]), &[&[]]));
        assert_eq!(
            masked(fixed(&[]), &[&[1, 0, 1]]),
            masked(fixed(&[]), &[&[1, 0, 1]])
        );
        assert_ne!(
            masked(fixed(&[]), &[&[1, 0, 1]]),
            masked(fixed(&[]), &[&[1, 1, 1]])
        );
    }

    #[test]
    fn optional_eq_variable_offsets() {
        let a = masked(vlen(b"abcd", vec![0u32, 2, 2, 4]), &[&[1, 0, 1]]);
        let b = masked(vlen(b"abzzcd", vec![0u64, 2, 4, 6]), &[&[1, 0, 1]]);
        let c = masked(vlen(b"abzzce", vec![0u64, 2, 4, 6]), &[&[1, 0, 1]]);
        assert_eq!(a, b);
        assert_eq!(b, a);
        assert_ne!(a, c);
    }

    #[test]
    fn optional_eq_malformed() {
        let malformed = [
            // Short mask, with a fixed length not divisible by the element count
            masked(fixed(&[1, 2, 3]), &[&[1, 1]]),
            // Nonbinary mask
            masked(fixed(&[1, 2]), &[&[1, 2]]),
            // Elements without a mask
            masked(fixed(&[1, 2]), &[&[]]),
            // Variable length with fewer elements than the mask
            masked(vlen(b"ab", vec![0u32, 2]), &[&[1, 1]]),
            // Nested mask lengths differ
            masked(fixed(&[1, 2]), &[&[1], &[1, 1]]),
            masked(fixed(&[1, 2]), &[&[1, 1, 1], &[0, 0]]),
        ];
        for value in &malformed {
            assert_eq!(value, value);
            assert_eq!(value, &value.clone());
        }
        // A malformed value never equals a well-formed value
        assert_ne!(malformed[1], masked(fixed(&[1, 2]), &[&[1, 1]]));
        assert_ne!(masked(fixed(&[1, 2]), &[&[1, 1]]), malformed[1]);
    }

    #[test]
    fn optional_eq_is_equivalence() {
        let values = [
            masked(fixed(&[5, 6]), &[&[1, 0], &[1, 0]]),
            masked(fixed(&[5, 9]), &[&[1, 1], &[1, 0]]),
            masked(fixed(&[5, 7]), &[&[1, 1], &[1, 0]]),
            masked(fixed(&[5, 6]), &[&[1, 0], &[1, 1]]),
            masked(fixed(&[0, 0]), &[&[0, 0], &[0, 0]]),
            masked(fixed(&[1, 1]), &[&[1, 1], &[0, 0]]),
            masked(fixed(&[5, 6]), &[&[1, 0]]),
            masked(fixed(&[5, 6, 0, 0]), &[&[1, 0]]),
            masked(fixed(&[5, 7]), &[&[1, 0]]),
            masked(vlen(b"abcd", vec![0u32, 2, 2, 4]), &[&[1, 0, 1]]),
            masked(vlen(b"abzzcd", vec![0u64, 2, 4, 6]), &[&[1, 0, 1]]),
            masked(fixed(&[1, 2, 3]), &[&[1, 1]]),
            masked(fixed(&[5, 6]), &[&[1, 2]]),
            masked(fixed(&[5, 6]), &[&[1, 0], &[1, 1, 0]]),
            masked(fixed(&[]), &[&[]]),
        ];
        for a in &values {
            assert_eq!(a, a);
            for b in &values {
                assert_eq!(a == b, b == a, "{a:?} {b:?}");
                for c in &values {
                    if a == b && b == c {
                        assert_eq!(a, c);
                    }
                }
            }
        }
    }
}
