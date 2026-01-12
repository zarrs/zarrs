#![allow(missing_docs)]

use zarrs_plugin::{Plugin, PluginCreateError};

struct TestPlugin;

// plugin can be an arbitrary input, usually zarrs_metadata::MetadataV3.
enum Input {
    Accept,
    Reject,
}

fn matches_name_test(name: &str) -> bool {
    name == "test"
}

fn create_test(input: &Input) -> Result<TestPlugin, PluginCreateError> {
    match input {
        Input::Accept => Ok(TestPlugin),
        Input::Reject => Err(PluginCreateError::from("rejected".to_string())),
    }
}

#[test]
fn plugin() {
    let plugin = Plugin::new(matches_name_test, create_test);
    assert!(!plugin.match_name("fail"));
    assert!(plugin.match_name("test"));
    assert!(plugin.create(&Input::Accept).is_ok());
    assert!(plugin.create(&Input::Reject).is_err());
}
