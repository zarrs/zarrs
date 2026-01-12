use std::any::Any;
use std::borrow::Cow;
use std::fmt::Debug;
use std::sync::Arc;

use derive_more::{Deref, From};
use zarrs_metadata::v2::DataTypeMetadataV2;
use zarrs_metadata::v3::MetadataV3;
use zarrs_metadata::{Configuration, DataTypeSize, FillValueMetadata};
use zarrs_plugin::{
    ExtensionName, MaybeSend, MaybeSync, PluginCreateError, PluginUnsupportedError, ZarrVersions,
};

use crate::{
    DataTypeFillValueError, DataTypeFillValueMetadataError, DataTypePluginV2, DataTypePluginV3,
    FillValue, DATA_TYPE_RUNTIME_REGISTRY_V2, DATA_TYPE_RUNTIME_REGISTRY_V3,
};

/// A data type implementing [`DataTypeTraits`].
#[derive(Debug, Clone, Deref, From)]
pub struct DataType(Arc<dyn DataTypeTraits>);

impl<T: DataTypeTraits + 'static> From<Arc<T>> for DataType {
    fn from(data_type: Arc<T>) -> Self {
        Self(data_type)
    }
}

impl ExtensionName for DataType {
    fn name(&self, version: ZarrVersions) -> Option<Cow<'static, str>> {
        self.0.name(version)
    }
}

/// Data type metadata for different Zarr versions.
#[derive(Debug, Clone, Copy, derive_more::From)]
pub enum DataTypeMetadata<'a> {
    /// Zarr V3 metadata.
    V3(&'a MetadataV3),
    /// Zarr V2 metadata.
    V2(&'a DataTypeMetadataV2),
}

impl DataType {
    /// Create a data type.
    pub fn new<T: DataTypeTraits + 'static>(data_type: T) -> Self {
        let data_type: Arc<dyn DataTypeTraits> = Arc::new(data_type);
        data_type.into()
    }

    /// Create a data type from metadata.
    ///
    /// # Errors
    /// Returns [`PluginCreateError`] if the metadata is invalid or not associated with a registered codec plugin.
    pub fn from_metadata<'a>(
        metadata: impl Into<DataTypeMetadata<'a>>,
    ) -> Result<Self, PluginCreateError> {
        match metadata.into() {
            DataTypeMetadata::V2(metadata) => Self::from_metadata_v2(metadata),
            DataTypeMetadata::V3(metadata) => Self::from_metadata_v3(metadata),
        }
    }

    /// Create a data type from V3 metadata.
    ///
    /// # Errors
    ///
    /// Returns a [`PluginCreateError`] if the metadata is invalid or not associated with a registered data type plugin.
    fn from_metadata_v3(metadata: &MetadataV3) -> Result<Self, PluginCreateError> {
        // Validate must_understand for V3
        if !metadata.must_understand() {
            return Err(PluginCreateError::Other(
                r#"data type must not have `"must_understand": false`"#.to_string(),
            ));
        }

        let name = metadata.name();

        // Check runtime registry first (higher priority)
        {
            let result = DATA_TYPE_RUNTIME_REGISTRY_V3.with_plugins(|plugins| {
                for plugin in plugins {
                    if plugin.match_name(name) {
                        return Some(plugin.create(metadata));
                    }
                }
                None
            });
            if let Some(result) = result {
                return result;
            }
        }

        // Fall back to compile-time registered plugins
        for plugin in inventory::iter::<DataTypePluginV3> {
            if plugin.match_name(name) {
                return plugin.create(metadata);
            }
        }
        Err(PluginUnsupportedError::new(name.to_string(), "data type".to_string()).into())
    }

    /// Create a data type from V2 metadata.
    ///
    /// # Errors
    ///
    /// Returns a [`PluginCreateError`] if the metadata is invalid or not associated with a registered data type plugin.
    fn from_metadata_v2(metadata: &DataTypeMetadataV2) -> Result<Self, PluginCreateError> {
        let name = match metadata {
            DataTypeMetadataV2::Simple(s) => s.as_str(),
            DataTypeMetadataV2::Structured(_) => "structured_v2", // special case name for V2 structured types
        };

        // Check runtime registry first (higher priority)
        {
            let result = DATA_TYPE_RUNTIME_REGISTRY_V2.with_plugins(|plugins| {
                for plugin in plugins {
                    if plugin.match_name(name) {
                        return Some(plugin.create(metadata));
                    }
                }
                None
            });
            if let Some(result) = result {
                return result;
            }
        }

        // Fall back to compile-time registered plugins
        for plugin in inventory::iter::<DataTypePluginV2> {
            if plugin.match_name(name) {
                return plugin.create(metadata);
            }
        }
        Err(PluginUnsupportedError::new(name.to_string(), "data type".to_string()).into())
    }
}

/// Traits for a data type extension.
///
/// The in-memory size of a data type can differ between its associated Rust structure and the *serialised* [`ArrayBytes`](https://docs.rs/zarrs/latest/zarrs/array/enum.ArrayBytes.html) passed into the codec pipeline.
/// For example, a Rust struct that has padding bytes can be converted to tightly packed bytes before it is passed into the codec pipeline for encoding, and vice versa for decoding.
///
/// It is recommended to define a concrete structure representing a single element of a custom data type that implements [`Element`](https://docs.rs/zarrs/latest/zarrs/array/trait.Element.html) and [`ElementOwned`](https://docs.rs/zarrs/latest/zarrs/array/trait.ElementOwned.html).
/// These traits have `into_array_bytes` and `from_array_bytes` methods for this purpose that enable custom data types to be used with the [`Array::{store,retrieve}_*_elements`](https://docs.rs/zarrs/latest/zarrs/array/struct.Array.html) variants.
/// These methods should encode data to and from native endianness if endianness is applicable, unless the endianness should be explicitly fixed.
/// Note that codecs that act on numerical data typically expect the data to be in native endianness.
///
/// A custom data type must also directly handle conversion of fill value metadata to fill value bytes, and vice versa.
pub trait DataTypeTraits: ExtensionName + Debug + MaybeSend + MaybeSync {
    /// The configuration of the data type.
    fn configuration(&self, version: ZarrVersions) -> Configuration;

    /// The Zarr V3 configuration of the data type.
    fn configuration_v3(&self) -> Configuration {
        self.configuration(ZarrVersions::V3)
    }

    /// The Zarr V2 configuration of the data type.
    fn configuration_v2(&self) -> Configuration {
        self.configuration(ZarrVersions::V2)
    }

    /// The size of the data type.
    ///
    /// This size may differ from the size in memory of the data type.
    /// It represents the size of elements passing through array to array and array to bytes codecs in the codec pipeline (i.e., after conversion to [`ArrayBytes`](https://docs.rs/zarrs/latest/zarrs/array/enum.ArrayBytes.html)).
    fn size(&self) -> DataTypeSize;

    /// Create a fill value from metadata.
    ///
    /// # Errors
    /// Returns [`DataTypeFillValueMetadataError`] if the fill value is incompatible with the data type.
    fn fill_value(
        &self,
        fill_value_metadata: &FillValueMetadata,
        version: ZarrVersions,
    ) -> Result<FillValue, DataTypeFillValueMetadataError>;

    /// Create a fill value from Zarr V2 metadata.
    ///
    /// # Errors
    /// Returns [`DataTypeFillValueMetadataError`] if the fill value is incompatible with the data type.
    fn fill_value_v2(
        &self,
        fill_value_metadata: &FillValueMetadata,
    ) -> Result<FillValue, DataTypeFillValueMetadataError> {
        self.fill_value(fill_value_metadata, ZarrVersions::V2)
    }

    /// Create a fill value from Zarr V3 metadata.
    ///
    /// # Errors
    /// Returns [`DataTypeFillValueMetadataError`] if the fill value is incompatible with the data type.
    fn fill_value_v3(
        &self,
        fill_value_metadata: &FillValueMetadata,
    ) -> Result<FillValue, DataTypeFillValueMetadataError> {
        self.fill_value(fill_value_metadata, ZarrVersions::V3)
    }

    /// Create fill value metadata.
    ///
    /// # Errors
    /// Returns an [`DataTypeFillValueError`] if the metadata cannot be created from the fill value.
    fn metadata_fill_value(
        &self,
        fill_value: &FillValue,
    ) -> Result<FillValueMetadata, DataTypeFillValueError>;

    /// Compare this data type with another for equality.
    ///
    /// The default implementation compares type via [`TypeId`](std::any::TypeId) and configuration.
    /// Custom data types may override this for more efficient comparison.
    fn eq(&self, other: &dyn DataTypeTraits) -> bool {
        self.as_any().type_id() == other.as_any().type_id()
            && self.configuration(ZarrVersions::V3) == other.configuration(ZarrVersions::V3)
    }

    /// Returns self as `Any` for downcasting.
    ///
    /// This enables accessing concrete type-specific methods (like `OptionalDataType::data_type()`).
    fn as_any(&self) -> &dyn Any;
}
