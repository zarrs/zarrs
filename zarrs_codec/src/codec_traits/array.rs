use std::any::Any;

use zarrs_data_type::{DataType, FillValue};
use zarrs_plugin::{MaybeSend, MaybeSync};

/// Traits shared by context-bound array codecs.
pub trait ArrayCodecTraits: MaybeSend + MaybeSync {
    /// Returns this codec as [`Any`].
    fn as_any(&self) -> &dyn Any;

    /// Return the decoded data type bound to this codec.
    fn data_type(&self) -> &DataType;

    /// Return the decoded fill value bound to this codec.
    fn fill_value(&self) -> &FillValue;
}
