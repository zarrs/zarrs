#![allow(missing_docs)]

use zarrs_plugin::{Plugin, PluginCreateError, ZarrVersions};

struct TestPlugin;

// plugin can be an arbitraty input, usually zarrs_metadata::MetadataV3.
enum Input {
    Accept,
    Reject,
}

fn matches_name_test(name: &str, _version: ZarrVersions) -> bool {
    name == "test"
}

fn default_name_test(_version: ZarrVersions) -> std::borrow::Cow<'static, str> {
    "test".into()
}

fn create_test(input: &Input) -> Result<TestPlugin, PluginCreateError> {
    match input {
        Input::Accept => Ok(TestPlugin),
        Input::Reject => Err(PluginCreateError::from("rejected".to_string())),
    }
}

#[test]
fn plugin() {
    let plugin = Plugin::new("test", matches_name_test, default_name_test, create_test);
    assert!(!plugin.match_name("fail", ZarrVersions::V3));
    assert!(plugin.match_name("test", ZarrVersions::V3));
    assert_eq!(plugin.identifier(), "test");
    assert!(plugin.create(&Input::Accept).is_ok());
    assert!(plugin.create(&Input::Reject).is_err());
}
