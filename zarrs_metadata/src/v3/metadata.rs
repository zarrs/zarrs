use std::fmt::Debug;

use serde::de::DeserializeOwned;
use serde::ser::SerializeMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{Configuration, ConfigurationError};

/// Zarr V3 generic metadata with a `name`, optional `configuration`, and optional `must_understand`.
///
/// Represents most fields in Zarr V3 array metadata (see [`ArrayMetadataV3`](crate::v3::ArrayMetadataV3)) which is either:
/// - a string name, or
/// - a JSON object with a required `name` field and optional `configuration` and `must_understand` fields.
///
/// `must_understand` is implicitly set to [`true`] if omitted.
/// See [ZEP0009](https://zarr.dev/zeps/draft/ZEP0009.html) for more information on this field and Zarr V3 extensions.
///
/// Metadata with a `configuration` (even if empty) is serialised as an object with the `configuration`, and metadata without a `configuration` is serialised as a string (a short-hand name) unless `must_understand` is [`false`].
///
/// Codec metadata needs a `configuration` to be serialised as an object such as `{"name":"...","configuration":{}}`, even though it *could* be simplified to a short-hand name.
/// `zarrs` creates codec metadata with a `configuration`, and adds an empty `configuration` to codec metadata without one when storing array metadata (see `Array::metadata_opt`).
/// This is for compatibility:
/// - Zarr 3.0 specified that array `codecs` metadata must be a list of JSON objects.
///   Codec metadata can be short-hand names since Zarr 3.1, but they are not supported by Zarr 3.0 implementations (e.g. `zarr-python` 3.4.1 and `tensorstore` 0.1.85).
/// - NON-CONFORMANT: `zarr-python` (as of 3.4.1) requires a `configuration` for `numcodecs.*` codecs (e.g. `{"name":"numcodecs.fletcher32"}` is rejected), even though it is optional.
///
/// ### Example Metadata
/// ```json
/// "bytes"
/// ```
///
/// ```json
/// {
///     "name": "bytes",
/// }
/// ```
///
/// ```json
/// {
///     "name": "bytes",
///     "configuration": {
///       "endian": "little"
///     }
/// }
/// ```
///
/// ```json
/// {
///     "name": "bytes",
///     "configuration": {
///       "endian": "little"
///     },
///     "must_understand": False
/// }
/// ```
#[derive(Clone, Eq, PartialEq, Debug)]
pub struct MetadataV3 {
    name: String,
    configuration: Option<Configuration>,
    must_understand: bool,
}

impl From<MetadataV3> for Option<Configuration> {
    fn from(metadata: MetadataV3) -> Self {
        metadata.configuration
    }
}

impl std::str::FromStr for MetadataV3 {
    type Err = serde_json::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        serde_json::from_str(s)
    }
}

impl TryFrom<&str> for MetadataV3 {
    type Error = serde_json::Error;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        s.parse()
    }
}

impl core::fmt::Display for MetadataV3 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if let Some(configuration) = &self.configuration {
            write!(
                f,
                "{} {}",
                self.name,
                serde_json::to_string(configuration).unwrap_or_default()
            )
        } else {
            write!(f, "{}", self.name)
        }
    }
}

impl serde::Serialize for MetadataV3 {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        if self.configuration.is_none() && self.must_understand {
            return s.serialize_str(self.name.as_str());
        }
        let len =
            1 + usize::from(self.configuration.is_some()) + usize::from(!self.must_understand);
        let mut s = s.serialize_map(Some(len))?;
        s.serialize_entry("name", &self.name)?;
        // An empty configuration is serialised for compatibility (see the type documentation)
        if let Some(configuration) = &self.configuration {
            s.serialize_entry("configuration", configuration)?;
        }
        if !self.must_understand {
            s.serialize_entry("must_understand", &false)?;
        }
        s.end()
    }
}

fn default_must_understand() -> bool {
    true
}

impl<'de> serde::Deserialize<'de> for MetadataV3 {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct MetadataNameConfiguration {
            name: String,
            #[serde(default)]
            configuration: Option<Configuration>,
            #[serde(default = "default_must_understand")]
            must_understand: bool,
        }

        #[derive(Deserialize)]
        #[serde(untagged)]
        enum MetadataIntermediate {
            Name(String),
            NameConfiguration(MetadataNameConfiguration),
        }

        let metadata = MetadataIntermediate::deserialize(d).map_err(|_| {
            serde::de::Error::custom(r#"Expected metadata "<name>" or {"name":"<name>"} or {"name":"<name>","configuration":{}}"#)
        })?;
        match metadata {
            MetadataIntermediate::Name(name) => Ok(Self {
                name,
                configuration: None,
                must_understand: true,
            }),
            MetadataIntermediate::NameConfiguration(metadata) => Ok(Self {
                name: metadata.name,
                configuration: metadata.configuration,
                must_understand: metadata.must_understand,
            }),
        }
    }
}

impl MetadataV3 {
    /// Create metadata from `name`.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            configuration: None,
            must_understand: true,
        }
    }

    /// Create metadata from `name` and `configuration`.
    #[must_use]
    pub fn new_with_configuration(
        name: impl Into<String>,
        configuration: impl Into<Configuration>,
    ) -> Self {
        Self {
            name: name.into(),
            configuration: Some(configuration.into()),
            must_understand: true,
        }
    }

    /// Set the value of the `must_understand` field.
    #[must_use]
    pub fn with_must_understand(mut self, must_understand: bool) -> Self {
        self.must_understand = must_understand;
        self
    }

    /// Convert a serializable configuration to [`MetadataV3`].
    ///
    /// # Errors
    /// Returns [`serde_json::Error`] if `configuration` cannot be converted to [`MetadataV3`].
    pub fn new_with_serializable_configuration<TConfiguration: serde::Serialize>(
        name: String,
        configuration: &TConfiguration,
    ) -> Result<Self, serde_json::Error> {
        let configuration = serde_json::to_value(configuration)?;
        if let Value::Object(configuration) = configuration {
            Ok(Self::new_with_configuration(name, configuration))
        } else {
            Err(serde::ser::Error::custom(
                "the configuration cannot be serialized to a JSON struct",
            ))
        }
    }

    /// Try and convert [`Configuration`] to a specific serializable configuration.
    ///
    /// # Errors
    /// Returns a [`serde_json`] error if the metadata cannot be converted.
    pub fn to_typed_configuration<TConfiguration: DeserializeOwned>(
        &self,
    ) -> Result<TConfiguration, std::sync::Arc<serde_json::Error>> {
        if let Some(configuration) = &self.configuration {
            configuration.to_typed()
        } else {
            Configuration::default().to_typed()
        }
    }

    /// Try and convert [`MetadataV3`] to a serializable configuration.
    ///
    /// # Errors
    /// Returns a [`serde_json`] error if the metadata cannot be converted.
    // TODO: #[deprecated(since = "0.8.0", note = "Use .to_typed() instead")]
    pub fn to_configuration<TConfiguration: DeserializeOwned>(
        &self,
    ) -> Result<TConfiguration, ConfigurationError> {
        let err = |_| ConfigurationError::new(self.name.clone(), self.configuration.clone());
        let configuration = self.configuration.clone().unwrap_or_default();
        let value = serde_json::to_value(configuration).map_err(err)?;
        serde_json::from_value(value).map_err(err)
    }

    /// Returns the metadata `name`.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Mutate the metadata `name`.
    pub fn set_name(&mut self, name: String) -> &mut Self {
        self.name = name;
        self
    }

    /// Returns the metadata configuration.
    #[must_use]
    pub const fn configuration(&self) -> Option<&Configuration> {
        self.configuration.as_ref()
    }

    /// Return whether the metadata must be understood as indicated by the `must_understand` field.
    ///
    /// The `must_understand` field is implicitly `true` if omitted.
    #[must_use]
    pub fn must_understand(&self) -> bool {
        self.must_understand
    }

    /// Returns true if the configuration is none or an empty map.
    #[must_use]
    pub fn configuration_is_none_or_empty(&self) -> bool {
        self.configuration
            .as_ref()
            .map_or(true, |configuration| configuration.is_empty())
    }
}

/// A Zarr V3 additional field in array or group metadata.
///
/// A field that is not recognised / supported by `zarrs` will be considered an additional field.
/// Additional fields can be any JSON type.
/// An array / group cannot be created with an additional field, unless the additional field is an object with a `"must_understand": false` field.
///
/// ### Example additional field JSON
/// ```json
///  "unknown_field": {
///      "key": "value",
///      "must_understand": false
///  }
/// ```
/// ```json
///  "unsupported_field_1": {
///      "key": "value",
///      "must_understand": true
///  }
/// ```
/// ```json
///  "unsupported_field_2": {
///      "key": "value"
///  }
/// ```
/// ```json
///  "unsupported_field_3": []
/// ```
/// ```json
///  "unsupported_field_4": "test"
/// ```
#[derive(Clone, Eq, PartialEq, Debug, Default)]
pub struct AdditionalFieldV3 {
    field: Value,
    must_understand: bool,
}

impl AdditionalFieldV3 {
    /// Create a new additional field.
    #[must_use]
    pub fn new(field: impl Into<Value>, must_understand: bool) -> Self {
        Self {
            field: field.into(),
            must_understand,
        }
    }

    /// Return the underlying value.
    #[must_use]
    pub const fn as_value(&self) -> &Value {
        &self.field
    }

    /// Return the `must_understand` component of the additional field.
    #[must_use]
    pub const fn must_understand(&self) -> bool {
        self.must_understand
    }
}

impl Serialize for AdditionalFieldV3 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match &self.field {
            Value::Object(object) => {
                let mut map = serializer.serialize_map(Some(object.len() + 1))?;
                map.serialize_entry("must_understand", &Value::Bool(self.must_understand))?;
                for (k, v) in object {
                    map.serialize_entry(k, v)?;
                }
                map.end()
            }
            _ => self.field.serialize(serializer),
        }
    }
}

impl<'de> serde::Deserialize<'de> for AdditionalFieldV3 {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(d)?;
        Ok(value.into())
    }
}

impl<T> From<T> for AdditionalFieldV3
where
    T: Into<Value>,
{
    fn from(field: T) -> Self {
        let mut value: Value = field.into();
        let must_understand = if let Some(object) = value.as_object_mut() {
            if let Some(Value::Bool(must_understand)) = object.remove("must_understand") {
                must_understand
            } else {
                true
            }
        } else {
            true
        };
        Self {
            must_understand,
            field: value,
        }
    }
}

/// Zarr V3 additional fields in array or group metadata.
// NOTE: It would be nice if this was just a serde_json::Map, but it only has implementations for `<String, Value>`.
pub type AdditionalFieldsV3 = std::collections::BTreeMap<String, AdditionalFieldV3>;

#[cfg(test)]
mod tests {
    use super::MetadataV3;

    #[test]
    fn metadata_must_understand_implicit_string() {
        let metadata = r#""test""#;
        let metadata: MetadataV3 = serde_json::from_str(metadata).unwrap();
        assert!(metadata.name() == "test");
        assert!(metadata.must_understand());
    }

    #[test]
    fn metadata_must_understand_implicit() {
        let metadata = r#"{
    "name": "test"
}"#;
        let metadata: MetadataV3 = serde_json::from_str(metadata).unwrap();
        assert!(metadata.name() == "test");
        assert!(metadata.must_understand());
    }

    #[test]
    fn metadata_must_understand_true() {
        let metadata = r#"{
    "name": "test",
    "must_understand": true
}"#;
        let metadata: MetadataV3 = serde_json::from_str(metadata).unwrap();
        assert!(metadata.name() == "test");
        assert!(metadata.must_understand());
    }

    #[test]
    fn metadata_must_understand_false() {
        let metadata = r#"{
    "name": "test",
    "must_understand": false
}"#;
        let metadata: MetadataV3 = serde_json::from_str(metadata).unwrap();
        assert!(metadata.name() == "test");
        assert!(!metadata.must_understand());
        assert_ne!(metadata, MetadataV3::new("test".to_string()));
        assert_eq!(
            metadata,
            MetadataV3::new("test".to_string()).with_must_understand(false)
        );
    }

    #[test]
    fn metadata_serialise_forms() {
        let roundtrip = |json: &str| {
            let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
            serde_json::to_string(&metadata).unwrap()
        };
        assert_eq!(roundtrip(r#""crc32c""#), r#""crc32c""#);
        let metadata: MetadataV3 = serde_json::from_str(r#"{"name":"crc32c"}"#).unwrap();
        assert!(metadata.configuration().is_none());
        assert_eq!(metadata, MetadataV3::new("crc32c"));
        assert_eq!(roundtrip(r#"{"name":"crc32c"}"#), r#""crc32c""#);
        assert_eq!(
            roundtrip(r#"{"name":"crc32c","configuration":{}}"#),
            r#"{"name":"crc32c","configuration":{}}"#
        );
        assert_eq!(
            roundtrip(r#"{"name":"test","must_understand":false}"#),
            r#"{"name":"test","must_understand":false}"#
        );
        assert_eq!(
            roundtrip(r#"{"name":"test","configuration":{"a":1},"must_understand":false}"#),
            r#"{"name":"test","configuration":{"a":1},"must_understand":false}"#
        );
        assert_eq!(
            serde_json::to_string(&MetadataV3::new("test").with_must_understand(false)).unwrap(),
            r#"{"name":"test","must_understand":false}"#
        );
        assert_eq!(
            serde_json::to_string(&MetadataV3::new_with_configuration(
                "crc32c",
                serde_json::Map::new()
            ))
            .unwrap(),
            r#"{"name":"crc32c","configuration":{}}"#
        );
    }
}
