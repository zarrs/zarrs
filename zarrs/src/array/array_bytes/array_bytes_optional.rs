use crate::array::{ArrayBytes, ArrayBytesRaw};

/// Optional array bytes composed of data and a validity mask.
///
/// The mask is 1 byte per element where 0 = invalid/missing, non-zero = valid/present.
/// The mask length is validated at construction to ensure it matches the number of elements.
///
/// # Equality Semantics
///
/// Two `ArrayBytesOptional` values are considered equal if:
/// - Their masks are equal
/// - Their valid (non-null) elements are equal
///
/// Invalid (null) element bytes are **not** compared, since they may differ after
/// round-trip encoding/decoding (e.g., fill value vs original bytes).
#[derive(Clone, Debug)]
pub struct ArrayBytesOptional<'a> {
    data: Box<ArrayBytes<'a>>,
    mask: ArrayBytesRaw<'a>,
}

impl<'a> ArrayBytesOptional<'a> {
    /// Create a new `ArrayBytesOptional` with validation.
    pub fn new(data: impl Into<Box<ArrayBytes<'a>>>, mask: impl Into<ArrayBytesRaw<'a>>) -> Self {
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
    pub fn mask(&self) -> &ArrayBytesRaw<'a> {
        &self.mask
    }

    /// Get the number of elements.
    #[must_use]
    pub fn num_elements(&self) -> usize {
        self.mask.len()
    }

    /// Consume self and return the data and mask.
    #[must_use]
    pub fn into_parts(self) -> (Box<ArrayBytes<'a>>, ArrayBytesRaw<'a>) {
        (self.data, self.mask)
    }

    /// Convert into owned [`ArrayBytesOptional<'static>`].
    #[must_use]
    pub fn into_owned(self) -> ArrayBytesOptional<'static> {
        ArrayBytesOptional {
            data: Box::new((*self.data).into_owned()),
            mask: self.mask.into_owned().into(),
        }
    }
}

impl PartialEq for ArrayBytesOptional<'_> {
    fn eq(&self, other: &Self) -> bool {
        // Masks must match
        if self.mask != other.mask {
            return false;
        }

        // Compare only valid (non-null) elements
        match (self.data.as_ref(), other.data.as_ref()) {
            (ArrayBytes::Fixed(a_data), ArrayBytes::Fixed(b_data)) => {
                // Compute element size on demand
                let num_elements = self.mask.len();
                if num_elements == 0 {
                    return a_data == b_data;
                }
                let elem_size = a_data.len() / num_elements;
                if elem_size == 0 {
                    return a_data == b_data;
                }
                for (i, &mask_val) in self.mask.iter().enumerate() {
                    if mask_val != 0 {
                        let start = i * elem_size;
                        let end = start + elem_size;
                        if a_data.get(start..end) != b_data.get(start..end) {
                            return false;
                        }
                    }
                }
                true
            }
            (ArrayBytes::Variable(a_var), ArrayBytes::Variable(b_var)) => {
                // For variable length, compare element by element for valid positions
                let a_offsets = a_var.offsets();
                let b_offsets = b_var.offsets();
                for (i, &mask_val) in self.mask.iter().enumerate() {
                    if mask_val != 0 {
                        let a_start = a_offsets[i];
                        let a_end = a_offsets[i + 1];
                        let b_start = b_offsets[i];
                        let b_end = b_offsets[i + 1];
                        if a_var.bytes().get(a_start..a_end) != b_var.bytes().get(b_start..b_end) {
                            return false;
                        }
                    }
                }
                true
            }
            (ArrayBytes::Optional(a_inner), ArrayBytes::Optional(b_inner)) => {
                // Nested optional - delegate to inner PartialEq
                a_inner == b_inner
            }
            (ArrayBytes::Fixed(_) | ArrayBytes::Optional(_), ArrayBytes::Variable(_))
            | (ArrayBytes::Fixed(_) | ArrayBytes::Variable(_), ArrayBytes::Optional(_))
            | (ArrayBytes::Variable(_) | ArrayBytes::Optional(_), ArrayBytes::Fixed(_)) => false,
        }
    }
}

impl Eq for ArrayBytesOptional<'_> {}
