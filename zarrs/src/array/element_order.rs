//! The [`ElementOrder`] of array elements in memory.

use super::ArrayIndices;

/// The order of the elements of an array in memory.
///
/// Array operations decode into and encode from elements in this order (see [`ArrayOps::element_order`](super::ArrayOps::element_order)).
/// It does not affect how chunks are stored: the order of the elements in an encoded chunk is set by its codecs (e.g. `transpose`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ElementOrder {
    /// Row-major order, where the last dimension varies fastest.
    #[default]
    C,
    /// Column-major (Fortran) order, where the first dimension varies fastest.
    F,
}

impl ElementOrder {
    /// The strides (in elements) of an array with `shape` in this order.
    pub(crate) fn strides(self, shape: &[u64]) -> ArrayIndices {
        let mut strides: ArrayIndices = shape.iter().map(|_| 0).collect();
        let mut stride = 1u64;
        let mut set = |(s, &size): (&mut u64, &u64)| {
            *s = stride;
            stride = stride.wrapping_mul(size);
        };
        match self {
            Self::C => std::iter::zip(strides.iter_mut(), shape)
                .rev()
                .for_each(&mut set),
            Self::F => std::iter::zip(strides.iter_mut(), shape).for_each(&mut set),
        }
        strides
    }

    /// Return true if the elements of an array with `shape` differ in position between this order and C order.
    pub(crate) fn is_reordered(self, shape: &[u64]) -> bool {
        self == Self::F && shape.iter().filter(|&&size| size > 1).count() > 1
    }

    /// Return the order of C-ordered elements of an array with `shape`, preferring this order if both apply.
    ///
    /// This is this order if it places the elements of `shape` as C order does, and C order otherwise.
    pub(crate) fn for_c_order_elements(self, shape: &[u64]) -> Self {
        if self.is_reordered(shape) {
            Self::C
        } else {
            self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_order_strides() {
        assert_eq!(ElementOrder::C.strides(&[2, 3, 4]).as_slice(), [12, 4, 1]);
        assert_eq!(ElementOrder::F.strides(&[2, 3, 4]).as_slice(), [1, 2, 6]);
        assert!(ElementOrder::C.strides(&[]).is_empty());
        assert!(!ElementOrder::C.is_reordered(&[2, 3]));
        assert!(ElementOrder::F.is_reordered(&[2, 3]));
        assert!(!ElementOrder::F.is_reordered(&[1, 3, 1]));
        assert!(!ElementOrder::F.is_reordered(&[5]));
        assert_eq!(
            ElementOrder::F.for_c_order_elements(&[2, 3]),
            ElementOrder::C
        );
        assert_eq!(
            ElementOrder::F.for_c_order_elements(&[1, 3]),
            ElementOrder::F
        );
        assert_eq!(
            ElementOrder::C.for_c_order_elements(&[2, 3]),
            ElementOrder::C
        );
    }
}
