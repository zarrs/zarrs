use std::sync::LazyLock;

use zarrs_metadata::v2::DataTypeMetadataV2;
use zarrs_metadata::v3::MetadataV3;
use zarrs_plugin::{
    ExtensionAliases, Plugin, PluginCreateError, RuntimePlugin, RuntimeRegistry, ZarrVersion2,
    ZarrVersion3,
};

use crate::DataType;

/// A Zarr V3 data type plugin.
#[derive(derive_more::Deref)]
pub struct DataTypePluginV3(Plugin<DataType, MetadataV3>);

inventory::collect!(DataTypePluginV3);

impl DataTypePluginV3 {
    /// Create a new [`DataTypePluginV3`] for a type implementing [`ExtensionAliases<ZarrVersion3>`].
    ///
    /// The `match_name_fn` is automatically derived from `T::matches_name`.
    pub const fn new<T: ExtensionAliases<ZarrVersion3>>(
        create_fn: fn(metadata: &MetadataV3) -> Result<DataType, PluginCreateError>,
    ) -> Self {
        Self(Plugin::new(|name| T::matches_name(name), create_fn))
    }
}

/// A Zarr V2 data type plugin.
#[derive(derive_more::Deref)]
pub struct DataTypePluginV2(Plugin<DataType, DataTypeMetadataV2>);

inventory::collect!(DataTypePluginV2);

impl DataTypePluginV2 {
    /// Create a new [`DataTypePluginV2`] for a type implementing [`ExtensionAliases<ZarrVersion2>`].
    ///
    /// The `match_name_fn` is automatically derived from `T::matches_name`.
    pub const fn new<T: ExtensionAliases<ZarrVersion2>>(
        create_fn: fn(metadata: &DataTypeMetadataV2) -> Result<DataType, PluginCreateError>,
    ) -> Self {
        Self(Plugin::new(|name| T::matches_name(name), create_fn))
    }
}

/// A runtime V3 data type plugin for dynamic registration.
pub type DataTypeRuntimePluginV3 = RuntimePlugin<DataType, MetadataV3>;

/// A runtime V2 data type plugin for dynamic registration.
pub type DataTypeRuntimePluginV2 = RuntimePlugin<DataType, DataTypeMetadataV2>;

/// Global runtime registry for V3 data type plugins.
pub static DATA_TYPE_RUNTIME_REGISTRY_V3: LazyLock<RuntimeRegistry<DataTypeRuntimePluginV3>> =
    LazyLock::new(RuntimeRegistry::new);

/// Global runtime registry for V2 data type plugins.
pub static DATA_TYPE_RUNTIME_REGISTRY_V2: LazyLock<RuntimeRegistry<DataTypeRuntimePluginV2>> =
    LazyLock::new(RuntimeRegistry::new);

/// A handle to a registered V3 data type plugin.
pub type DataTypeRuntimeRegistryHandleV3 = std::sync::Arc<DataTypeRuntimePluginV3>;

/// A handle to a registered V2 data type plugin.
pub type DataTypeRuntimeRegistryHandleV2 = std::sync::Arc<DataTypeRuntimePluginV2>;

/// Register a V3 data type plugin at runtime.
///
/// Runtime-registered plugins take precedence over compile-time registered plugins.
///
/// # Returns
///
/// A handle that can be used to unregister the plugin later.
///
/// # Example
///
/// ```ignore
/// use zarrs_data_type::{register_data_type_v3, DataTypeRuntimePluginV3};
///
/// let handle = register_data_type_v3(DataTypeRuntimePluginV3::new(
///     |name| name == "my.custom.dtype",
///     |metadata| Ok(Arc::new(MyCustomDataType::from_metadata(metadata)?)),
/// ));
/// ```
pub fn register_data_type_v3(plugin: DataTypeRuntimePluginV3) -> DataTypeRuntimeRegistryHandleV3 {
    DATA_TYPE_RUNTIME_REGISTRY_V3.register(plugin)
}

/// Register a V2 data type plugin at runtime.
///
/// Runtime-registered plugins take precedence over compile-time registered plugins.
///
/// # Returns
///
/// A handle that can be used to unregister the plugin later.
pub fn register_data_type_v2(plugin: DataTypeRuntimePluginV2) -> DataTypeRuntimeRegistryHandleV2 {
    DATA_TYPE_RUNTIME_REGISTRY_V2.register(plugin)
}

/// Unregister a runtime V3 data type plugin.
///
/// # Returns
/// `true` if the plugin was found and removed, `false` otherwise.
pub fn unregister_data_type_v3(handle: &DataTypeRuntimeRegistryHandleV3) -> bool {
    DATA_TYPE_RUNTIME_REGISTRY_V3.unregister(handle)
}

/// Unregister a runtime V2 data type plugin.
///
/// # Returns
/// `true` if the plugin was found and removed, `false` otherwise.
pub fn unregister_data_type_v2(handle: &DataTypeRuntimeRegistryHandleV2) -> bool {
    DATA_TYPE_RUNTIME_REGISTRY_V2.unregister(handle)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use zarrs_metadata::v3::MetadataV3;
    use zarrs_metadata::{Configuration, DataTypeSize, FillValueMetadata};
    use zarrs_plugin::ZarrVersion;

    use super::*;
    use crate::{
        DataTypeFillValueError, DataTypeFillValueMetadataError, DataTypeTraits, FillValue,
    };

    zarrs_plugin::impl_extension_aliases!(TestVoidDataType, v3: "zarrs.test_void");

    inventory::submit! {
        DataTypePluginV3::new::<TestVoidDataType>(create_test_void)
    }

    fn create_test_void(_metadata: &MetadataV3) -> Result<DataType, PluginCreateError> {
        Ok(Arc::new(TestVoidDataType).into())
    }

    #[derive(Debug)]
    struct TestVoidDataType;

    impl DataTypeTraits for TestVoidDataType {
        fn size(&self) -> DataTypeSize {
            DataTypeSize::Fixed(0)
        }

        fn configuration(&self, _version: ZarrVersion) -> Configuration {
            Configuration::default()
        }

        fn fill_value(
            &self,
            _fill_value_metadata: &FillValueMetadata,
            _version: ZarrVersion,
        ) -> Result<FillValue, DataTypeFillValueMetadataError> {
            Ok(FillValue::new(vec![]))
        }

        fn metadata_fill_value(
            &self,
            _fill_value: &FillValue,
        ) -> Result<FillValueMetadata, DataTypeFillValueError> {
            Ok(FillValueMetadata::Null)
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }
    #[test]
    fn data_type_plugin() {
        let mut found = false;
        for plugin in inventory::iter::<DataTypePluginV3> {
            if plugin.match_name("zarrs.test_void") {
                found = true;
                let metadata = MetadataV3::new("zarrs.test_void");
                let data_type = plugin.create(&metadata).unwrap();
                assert!(data_type.as_any().is::<TestVoidDataType>());
                assert_eq!(data_type.size(), DataTypeSize::Fixed(0));
                assert!(data_type.configuration_v3().is_empty());
                assert!(data_type.fill_value_v3(&FillValueMetadata::Null).is_ok());
                assert_eq!(
                    data_type
                        .metadata_fill_value(&FillValue::new(vec![]))
                        .unwrap(),
                    FillValueMetadata::Null
                );
            }
        }
        assert!(found);
    }
}
