use std::any::Any;
use std::num::NonZeroU64;

use zarrs_data_type::{DataType, FillValue};
use zarrs_plugin::{MaybeSend, MaybeSync};

use crate::{CodecError, RecommendedConcurrency};

/// Traits shared by context-bound array codecs.
pub trait ArrayCodecTraits: MaybeSend + MaybeSync {
    /// Returns this codec as [`Any`].
    fn as_any(&self) -> &dyn Any;

    /// Return the decoded data type bound to this codec.
    fn data_type(&self) -> &DataType;

    /// Return the decoded fill value bound to this codec.
    fn fill_value(&self) -> &FillValue;

    /// Return the recommended concurrency for the requested decoded shape.
    ///
    /// # Errors
    /// Returns [`CodecError`] if the decoded `shape` is not valid for the codec.
    fn recommended_concurrency(
        &self,
        shape: &[NonZeroU64],
    ) -> Result<RecommendedConcurrency, CodecError>;
}
