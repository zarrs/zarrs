use derive_more::Display;

/// The decoded representation of `bytes`.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Display)]
pub enum BytesRepresentation {
    /// The output size is fixed.
    #[display("fixed size: {_0}")]
    FixedSize(u64),
    /// The output size is bounded.
    #[display("bounded size: {_0}")]
    BoundedSize(u64),
    /// The output size is unbounded/indeterminate.
    #[display("unbounded size")]
    UnboundedSize,
}

impl BytesRepresentation {
    /// Return the fixed or bounded size of the bytes representations, or [`None`] if the size is unbounded.
    #[must_use]
    pub const fn size(&self) -> Option<u64> {
        match self {
            Self::FixedSize(size) | Self::BoundedSize(size) => Some(*size),
            Self::UnboundedSize => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_representation() {
        let bytes_representation_fixed = BytesRepresentation::FixedSize(10);
        assert_eq!(bytes_representation_fixed.size(), Some(10));
        assert_eq!(
            bytes_representation_fixed,
            bytes_representation_fixed.clone()
        );
        let bytes_representation_bounded = BytesRepresentation::BoundedSize(10);
        assert_eq!(bytes_representation_bounded.size(), Some(10));
        assert_ne!(bytes_representation_fixed, bytes_representation_bounded);
        let bytes_representation_unbounded = BytesRepresentation::UnboundedSize;
        assert_eq!(bytes_representation_unbounded.size(), None);
    }
}
