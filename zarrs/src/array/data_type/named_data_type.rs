use std::ops::Deref;

use zarrs_data_type::{DataTypeFillValueMetadataError, DataTypePlugin, FillValue};
use zarrs_metadata::v3::{FillValueMetadataV3, MetadataV3};
use zarrs_plugin::{PluginCreateError, ZarrVersions};

use crate::array::{DataType, data_type};

/// A named data type.
#[derive(Debug, Clone)]
pub struct NamedDataType {
    name: String,
    data_type: DataType,
}

impl PartialEq for NamedDataType {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.data_type.eq(other.data_type.as_ref())
    }
}

impl Eq for NamedDataType {}

impl NamedDataType {
    /// Create a new [`NamedDataType`].
    #[must_use]
    pub fn new(name: String, data_type: DataType) -> Self {
        Self { name, data_type }
    }

    /// Create a new [`NamedDataType`] with the default name for the data type.
    ///
    /// Uses the instance `default_name` if it provides one, otherwise uses the type-level registered default name.
    #[must_use]
    pub fn new_default_name(data_type: DataType) -> Self {
        let name = data_type.default_name(ZarrVersions::V3);
        if let Some(name) = name {
            Self::new(name.into_owned(), data_type)
        } else {
            for plugin in inventory::iter::<DataTypePlugin> {
                if plugin.identifier() == data_type.identifier() {
                    let default_name = plugin.default_name(ZarrVersions::V3);
                    return Self::new(default_name.into_owned(), data_type);
                }
            }
            Self::new(data_type.identifier().to_string(), data_type)
        }
    }

    /// The name of the data type.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The underlying data type extension.
    #[must_use]
    pub const fn data_type(&self) -> &DataType {
        &self.data_type
    }

    /// Wrap this data type in an optional type.
    ///
    /// Can be chained to create nested optional types.
    ///
    /// # Examples
    /// ```
    /// # use zarrs::array::{data_type, DataTypeExt};
    /// // Single level optional
    /// let opt_u8 = data_type::uint8().to_named().into_optional();
    /// # assert_eq!(opt_u8.identifier(), "optional");
    ///
    /// // Nested optional
    /// let opt_opt_u8 = opt_u8.into_optional();
    /// ```
    #[must_use]
    pub fn into_optional(self) -> Self {
        let data_type = data_type::optional(self);
        Self::new_default_name(data_type)
    }

    /// Create the data type metadata.
    #[must_use]
    pub fn metadata(&self) -> MetadataV3 {
        let configuration = self.data_type.configuration();
        if configuration.is_empty() {
            MetadataV3::new(self.name.clone())
        } else {
            MetadataV3::new_with_configuration(self.name.clone(), configuration)
        }
    }

    /// Create a fill value from metadata.
    ///
    /// # Errors
    /// Returns [`DataTypeFillValueMetadataError`] if the fill value is incompatible with the data type.
    pub fn fill_value_from_metadata(
        &self,
        fill_value: &FillValueMetadataV3,
    ) -> Result<FillValue, DataTypeFillValueMetadataError> {
        // Delegate to the trait method - each data type implementation handles its own fill value parsing
        self.data_type.fill_value(fill_value)
    }
}

impl Deref for NamedDataType {
    type Target = DataType;

    fn deref(&self) -> &Self::Target {
        &self.data_type
    }
}

impl From<NamedDataType> for DataType {
    fn from(value: NamedDataType) -> Self {
        value.data_type
    }
}

impl TryFrom<&MetadataV3> for NamedDataType {
    type Error = PluginCreateError;

    /// Create a [`NamedDataType`] from metadata.
    ///
    /// # Errors
    ///
    /// Returns [`PluginCreateError`] if the metadata is invalid or not associated with a registered data type plugin.
    #[allow(clippy::too_many_lines)]
    fn try_from(metadata: &MetadataV3) -> Result<Self, Self::Error> {
        let data_type = DataType::from_metadata(metadata)?;
        Ok(Self::new(metadata.name().to_string(), data_type))
    }
}
