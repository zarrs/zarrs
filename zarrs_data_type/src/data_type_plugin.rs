use std::sync::LazyLock;

use zarrs_metadata::v3::MetadataV3;
use zarrs_plugin::{Plugin, PluginCreateError, RuntimePlugin, RuntimeRegistry, ZarrVersions};

use crate::DataType;

/// A data type plugin.
#[derive(derive_more::Deref)]
pub struct DataTypePlugin(Plugin<DataType, MetadataV3>);

inventory::collect!(DataTypePlugin);

impl DataTypePlugin {
    /// Create a new [`DataTypePlugin`].
    pub const fn new(
        identifier: &'static str,
        match_name_fn: fn(name: &str, version: ZarrVersions) -> bool,
        default_name_fn: fn(ZarrVersions) -> std::borrow::Cow<'static, str>,
        create_fn: fn(metadata: &MetadataV3) -> Result<DataType, PluginCreateError>,
    ) -> Self {
        Self(Plugin::new(
            identifier,
            match_name_fn,
            default_name_fn,
            create_fn,
        ))
    }
}

/// A runtime data type plugin for dynamic registration.
pub type DataTypeRuntimePlugin = RuntimePlugin<DataType, MetadataV3>;

/// Global runtime registry for data type plugins.
pub static DATA_TYPE_RUNTIME_REGISTRY: LazyLock<RuntimeRegistry<DataTypeRuntimePlugin>> =
    LazyLock::new(RuntimeRegistry::new);

/// A handle to a registered data type plugin.
pub type DataTypeRuntimeRegistryHandle = std::sync::Arc<DataTypeRuntimePlugin>;

/// Register a data type plugin at runtime.
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
/// use zarrs_data_type::{register_data_type, DataTypeRuntimePlugin};
///
/// let handle = register_data_type(DataTypeRuntimePlugin::new(
///     "my.custom.dtype",
///     |name, zarr_version| name == "my.custom.dtype",
///     |zarr_version| "my.custom.dtype".into(),
///     |metadata| Ok(Arc::new(MyCustomDataType::from_metadata(metadata)?)),
/// ));
/// ```
pub fn register_data_type(plugin: DataTypeRuntimePlugin) -> DataTypeRuntimeRegistryHandle {
    DATA_TYPE_RUNTIME_REGISTRY.register(plugin)
}

/// Unregister a runtime data type plugin.
///
/// # Returns
/// `true` if the plugin was found and removed, `false` otherwise.
pub fn unregister_data_type(handle: &DataTypeRuntimeRegistryHandle) -> bool {
    DATA_TYPE_RUNTIME_REGISTRY.unregister(handle)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use zarrs_metadata::{v3::FillValueMetadataV3, Configuration, DataTypeSize};
    use zarrs_plugin::{PluginCreateError, ZarrVersions};

    use super::*;
    use crate::{
        DataTypeExtension, DataTypeFillValueError, DataTypeFillValueMetadataError, FillValue,
    };

    inventory::submit! {
        DataTypePlugin::new("zarrs.test_void", matches_name_test_void, default_name_test_void, create_test_void)
    }

    #[derive(Debug)]
    struct TestVoidDataType;

    impl DataTypeExtension for TestVoidDataType {
        fn identifier(&self) -> &'static str {
            "zarrs.test_void"
        }

        fn size(&self) -> DataTypeSize {
            DataTypeSize::Fixed(0)
        }

        fn configuration(&self) -> Configuration {
            Configuration::default()
        }

        fn fill_value(
            &self,
            _fill_value_metadata: &FillValueMetadataV3,
        ) -> Result<FillValue, DataTypeFillValueMetadataError> {
            Ok(FillValue::new(vec![]))
        }

        fn metadata_fill_value(
            &self,
            _fill_value: &FillValue,
        ) -> Result<FillValueMetadataV3, DataTypeFillValueError> {
            Ok(FillValueMetadataV3::Null)
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    fn matches_name_test_void(name: &str, _version: ZarrVersions) -> bool {
        name == "zarrs.test_void"
    }

    fn default_name_test_void(_version: ZarrVersions) -> std::borrow::Cow<'static, str> {
        "zarrs.test_void".into()
    }

    fn create_test_void(_metadata: &MetadataV3) -> Result<DataType, PluginCreateError> {
        Ok(Arc::new(TestVoidDataType))
    }

    #[test]
    fn data_type_plugin() {
        let mut found = false;
        for plugin in inventory::iter::<DataTypePlugin> {
            if plugin.match_name("zarrs.test_void", ZarrVersions::V3) {
                found = true;
                let data_type = plugin.create(&MetadataV3::new("zarrs.test_void")).unwrap();
                assert_eq!(data_type.identifier(), "zarrs.test_void");
                assert_eq!(data_type.size(), DataTypeSize::Fixed(0));
                assert!(data_type.configuration().is_empty());
                assert!(data_type.fill_value(&FillValueMetadataV3::Null).is_ok());
                assert_eq!(
                    data_type
                        .metadata_fill_value(&FillValue::new(vec![]))
                        .unwrap(),
                    FillValueMetadataV3::Null
                );
            }
        }
        assert!(found);
    }
}
