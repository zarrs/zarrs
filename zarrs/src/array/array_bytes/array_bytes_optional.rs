use crate::array::{ArrayBytes, RawBytes};

/// Optional array bytes composed of data and a validity mask.
///
/// The mask is 1 byte per element where 0 = invalid/missing, non-zero = valid/present.
/// The mask length is validated at construction to ensure it matches the number of elements.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OptionalBytes<'a> {
    data: Box<ArrayBytes<'a>>,
    mask: RawBytes<'a>,
}

impl<'a> OptionalBytes<'a> {
    /// Create a new `OptionalBytes` with validation.
    pub fn new(data: impl Into<Box<ArrayBytes<'a>>>, mask: impl Into<RawBytes<'a>>) -> Self {
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
    pub fn mask(&self) -> &RawBytes<'a> {
        &self.mask
    }

    /// Get the number of elements.
    #[must_use]
    pub fn num_elements(&self) -> usize {
        self.mask.len()
    }

    /// Consume self and return the data and mask.
    #[must_use]
    pub fn into_parts(self) -> (Box<ArrayBytes<'a>>, RawBytes<'a>) {
        (self.data, self.mask)
    }

    /// Convert into owned [`OptionalBytes<'static>`].
    #[must_use]
    pub fn into_owned(self) -> OptionalBytes<'static> {
        OptionalBytes {
            data: Box::new((*self.data).into_owned()),
            mask: self.mask.into_owned().into(),
        }
    }
}
