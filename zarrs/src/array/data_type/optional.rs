use std::ops::Deref;

use zarrs_data_type::FillValue;
use zarrs_metadata::Configuration;
use zarrs_metadata_ext::data_type::optional::OptionalDataTypeConfigurationV1;

use crate::array::NamedDataType;

/// The `optional` data type.
///
/// This wraps the inner [`NamedDataType`] and provides methods specific to optional types,
/// such as checking if a fill value represents null and extracting inner fill value bytes.
///
/// The newtype implements [`Deref`] to the inner [`NamedDataType`], so methods on the inner
/// type can be called directly.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OptionalDataType(Box<NamedDataType>);
zarrs_plugin::impl_extension_aliases!(OptionalDataType, "optional",
    v3: "zarrs.optional", []
);

impl OptionalDataType {
    /// Create a new optional data type wrapper.
    #[must_use]
    pub fn new(inner: NamedDataType) -> Self {
        Self(Box::new(inner))
    }

    /// Check if the fill value represents null (last byte is `0x00`).
    #[must_use]
    pub fn is_fill_value_null(&self, fill_value: &FillValue) -> bool {
        fill_value.as_ne_bytes().last() == Some(&0)
    }

    /// Get the inner fill value bytes (without optional suffix).
    ///
    /// For optional data types, returns all bytes except the last suffix byte.
    #[must_use]
    pub fn fill_value_inner_bytes<'a>(&self, fill_value: &'a FillValue) -> &'a [u8] {
        let bytes = fill_value.as_ne_bytes();
        if bytes.is_empty() {
            &[]
        } else {
            &bytes[..bytes.len() - 1]
        }
    }

    /// Returns the configuration for this optional data type.
    #[must_use]
    pub fn configuration(&self) -> Configuration {
        Configuration::from(OptionalDataTypeConfigurationV1 {
            name: self.0.name().to_string(),
            configuration: self.0.configuration(),
        })
    }
}

impl Deref for OptionalDataType {
    type Target = NamedDataType;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
