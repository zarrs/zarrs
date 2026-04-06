---
name: zarrs-new-codec
description: Creates a new codec in the zarrs crate from a codec specification URL.
argument-hint: <spec-url>
---

# zarrs New Codec Skill

This skill implements a new codec in the zarrs crate from a codec specification URL.

## Arguments

- `<spec-url>` (required): URL pointing to the codec specification (e.g., a zarr-specs or zarr-extensions page)

## Instructions

When invoked with `/zarrs-new-codec <spec-url>`:

### Step 1: Read the Specification

Fetch the specification URL and extract:
- The codec **name/identifier** (the string used in Zarr metadata, e.g. `"gzip"`)
- Any **alias identifiers** (e.g. numcodecs ID, V2 names)
- The codec **type**: Array-to-Array (A→A), Array-to-Bytes (A→B), or Bytes-to-Bytes (B→B)
- The **configuration parameters** (fields, types, valid ranges, defaults)
- Whether the configuration is versioned (V1, V2, etc.)

### Step 2: Critique the Specification

Before writing any code, critically review the specification and raise concerns with the user. Look for:

- **Ambiguities**: Parameters or behaviours that are underspecified (e.g. "what happens if `level` is omitted — is there a default, or is it required?")
- **Inconsistencies**: Contradictions between different parts of the spec, or between the spec and the codec's stated behaviour
- **Missing information**: Absence of error handling semantics, edge cases (empty input, zero-length chunks), or behaviour for invalid configurations
- **Type/range gaps**: Numeric parameters with no stated valid range or default value
- **Versioning**: Whether the spec defines a configuration version and how future versions should be handled
- **V2 compatibility**: Whether a Zarr V2 `id` alias is defined and whether V2 configuration differs from V3
- **Data type constraints**: Whether the codec is restricted to certain data types (e.g. floats only) and whether this is clearly stated

Present findings as a concise list of questions or concerns. For example:

> - The spec says `level` controls compression but gives no valid range. What values are valid?
> - The spec does not state what happens on decoding if the input is truncated. Should this be an error?
> - No V2 `id` is mentioned — is V2 support out of scope?

Wait for the user to resolve any blocking ambiguities before proceeding. Minor uncertainties (e.g. an undocumented upper bound that can be inferred from the reference implementation) can be flagged and proceeded with a stated assumption.

### Step 3: Ask About Partial Encoding/Decoding

Before proceeding, ask the user:

> Do you want to support partial decoding and/or partial encoding for this codec?
>
> - **No** (recommended for initial development): Use `partial_read: false, partial_decode: false` and `partial_encode: false`. Simpler to implement; partial operations fall back to full encode/decode.
> - **Yes**: Requires implementing separate `*PartialDecoder` types. More complex but more efficient for large chunks where only a subset is needed. Some codecs cannot support partial decoding, and most cannot support partial encoding.

Recommend **No** unless the codec is inherently seekable (e.g. a checksum/filter that preserves byte offsets).

### Step 4: Determine Codec Category and Crate Placement

Classify the codec for placement in `zarrs_metadata_ext/src/codec.rs`:
- **`registered/`**: Core Zarr V3 spec codecs or registered at [zarr-developers/zarr-extensions](https://github.com/zarr-developers/zarr-extensions/)
- **`numcodecs/`**: Codecs originating from the numcodecs project
- **`zarrs/`**: Experimental `zarrs`-specific codecs not yet registered

And in `zarrs/src/array/codec/`:
- **`bytes_to_bytes/`**: B→B (compression, checksums, filters on raw bytes)
- **`array_to_bytes/`**: A→B (serialises the array to bytes, e.g. `bytes`, `zfp`, `pcodec`)
- **`array_to_array/`**: A→A (transforms array before bytes codec, e.g. `transpose`, `bitround`)

### Step 5: Add Metadata to `zarrs_metadata_ext`

Create `zarrs_metadata_ext/src/codec/<category>/<codec_name>.rs`:

```rust
use derive_more::{Display, From};
use serde::{Deserialize, Serialize};
use zarrs_metadata::ConfigurationSerialize;

/// A wrapper to handle various versions of `<codec_name>` codec configuration parameters.
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display, From)]
#[non_exhaustive]
#[serde(untagged)]
pub enum <CodecName>CodecConfiguration {
    /// Version 1.0.
    V1(<CodecName>CodecConfigurationV1),
}

impl ConfigurationSerialize for <CodecName>CodecConfiguration {}

/// `<codec_name>` codec configuration parameters (version 1.0).
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct <CodecName>CodecConfigurationV1 {
    // TODO: Add fields from the specification, e.g.:
    // pub level: <ParameterType>,
}
```

If a parameter has a constrained value range, use a validated newtype (see `GzipCompressionLevel` pattern):

```rust
/// A <parameter> value.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Display)]
pub struct <CodecName><ParameterName>(<BaseType>);

/// An invalid <parameter> value.
#[derive(Debug, thiserror::Error)]
#[error("Invalid <parameter> {0}, must be <range>")]
pub struct <CodecName><ParameterName>Error(<BaseType>);

impl TryFrom<<BaseType>> for <CodecName><ParameterName> {
    type Error = <CodecName><ParameterName>Error;
    fn try_from(value: <BaseType>) -> Result<Self, Self::Error> {
        if /* valid */ {
            Ok(Self(value))
        } else {
            Err(<CodecName><ParameterName>Error(value))
        }
    }
}
// Also implement custom Serialize/Deserialize for validation
```

Then wire it up in `zarrs_metadata_ext/src/codec.rs` — add to the appropriate `mod` block:

```rust
// In the `registered` / `zarrs` / `numcodecs` mod block:
/// `<codec_name>` codec metadata (<category>).
pub mod <codec_name>;
```

### Step 6: Check if a Data Type Trait is Needed

A data type trait in `zarrs_data_type` is **only needed** if:
- The codec must behave differently per data type (e.g. float-specific bit manipulation like `bitround` or `zfp`)
- The codec needs to query type size, endianness, or perform type-specific validation

**B→B codecs do NOT need a data type trait.** Skip this step for compression codecs (gzip, zstd, bz2, etc.) and simple checksum codecs.

If a data type trait IS needed:
1. Create `zarrs_data_type/src/codec_traits/<codec_name>.rs`
2. Define the trait (see `zarrs_data_type/src/codec_traits/bitround.rs` or `zfp.rs` for examples)
3. Add to `zarrs_data_type/src/codec_traits.rs`:
   ```rust
   pub mod <codec_name>;
   pub use <codec_name>::*;
   ```
4. Use the `define_data_type_support!` macro to register support for built-in data types

### Step 7: Implement the Codec in `zarrs`

#### 7a. Add a Cargo Feature (only if an external dependency is needed)

If the codec requires an external crate, add an optional feature to `zarrs/Cargo.toml`:

```toml
[features]
<codec_name> = ["dep:<dep_crate>"]

[dependencies]
<dep_crate> = { version = "<version>", optional = true }
```

And gate all codec module declarations and re-exports on the feature (steps 7e–7f below).

If the codec has **no** external dependency, skip this step. Include the module unconditionally, matching the pattern used by codecs like `reshape`, `squeeze`, and `packbits`.

#### 7b. Create the Codec Directory

Create:
- `zarrs/src/array/codec/<codec_type>/<codec_name>.rs` (module wrapper + registration)
- `zarrs/src/array/codec/<codec_type>/<codec_name>/<codec_name>_codec.rs` (implementation)

#### 7c. Write the Module Wrapper

`zarrs/src/array/codec/<codec_type>/<codec_name>.rs`:

```rust
//! The `<codec_name>` <codec_type> codec.
//!
//! <Brief description from specification>
//!
//! ### Specification
//! - <spec_url>
//!
//! ### Codec `name` Aliases (Zarr V3)
//! - `<identifier>`

mod <codec_name>_codec;

use std::sync::Arc;

pub use <codec_name>_codec::<CodecName>Codec;
use zarrs_metadata::v3::MetadataV3;
// Include V2 if needed:
// use zarrs_metadata::v2::MetadataV2;

use zarrs_codec::{Codec, CodecPluginV3, CodecTraitsV3};
// Include V2 if needed:
// use zarrs_codec::{CodecPluginV2, CodecTraitsV2};
pub use zarrs_metadata_ext::codec::<codec_name>::{
    <CodecName>CodecConfiguration,
    <CodecName>CodecConfigurationV1,
    // ... other public types
};
use zarrs_plugin::PluginCreateError;

zarrs_plugin::impl_extension_aliases!(<CodecName>Codec, v3: "<identifier>");
// With V2 alias:
// zarrs_plugin::impl_extension_aliases!(<CodecName>Codec, v3: "<identifier>", v2: "<v2_identifier>");

// Register the V3 codec.
inventory::submit! {
    CodecPluginV3::new::<CodecName>Codec>()
}

// Uncomment to also register as a V2 codec:
// inventory::submit! {
//     CodecPluginV2::new::<CodecName>Codec>()
// }

impl CodecTraitsV3 for <CodecName>Codec {
    fn create(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
        let configuration: <CodecName>CodecConfiguration = metadata.to_typed_configuration()?;
        let codec = Arc::new(<CodecName>Codec::new_with_configuration(&configuration)?);
        Ok(Codec::BytesToBytes(codec))  // or Codec::ArrayToBytes / Codec::ArrayToArray
    }
}
```

#### 7d. Write the Codec Implementation

`zarrs/src/array/codec/<codec_type>/<codec_name>/<codec_name>_codec.rs`:

**For Bytes-to-Bytes (B→B):**

```rust
use std::borrow::Cow;
use std::sync::Arc;

use super::<CodecName>CodecConfiguration; // and other config types

use crate::array::{ArrayBytesRaw, BytesRepresentation};
use zarrs_codec::{
    BytesToBytesCodecTraits, CodecError, CodecMetadataOptions, CodecOptions, CodecTraits,
    PartialDecoderCapability, PartialEncoderCapability, RecommendedConcurrency,
};
use zarrs_metadata::Configuration;
use zarrs_plugin::{PluginCreateError, ZarrVersion};

/// A `<codec_name>` codec implementation.
#[derive(Clone, Debug)]
pub struct <CodecName>Codec {
    // TODO: store configuration parameters here
}

impl <CodecName>Codec {
    /// Create a new `<codec_name>` codec.
    pub fn new(/* params */) -> Self {
        Self { /* ... */ }
    }

    /// Create a new `<codec_name>` codec from configuration.
    ///
    /// # Errors
    /// Returns an error if the configuration is not supported.
    pub fn new_with_configuration(
        configuration: &<CodecName>CodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        match configuration {
            <CodecName>CodecConfiguration::V1(cfg) => Ok(Self::new(/* cfg.fields */)),
            _ => Err(PluginCreateError::Other(
                "unsupported <codec_name> codec configuration variant".to_string(),
            )),
        }
    }
}

impl CodecTraits for <CodecName>Codec {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        let configuration = <CodecName>CodecConfiguration::V1(<CodecName>CodecConfigurationV1 {
            // TODO: populate from self fields
        });
        Some(configuration.into())
    }

    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        PartialDecoderCapability {
            partial_read: false,  // Set true only if partial reading is supported
            partial_decode: false,
        }
    }

    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        PartialEncoderCapability {
            partial_encode: false,
        }
    }
}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl BytesToBytesCodecTraits for <CodecName>Codec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn BytesToBytesCodecTraits> {
        self as Arc<dyn BytesToBytesCodecTraits>
    }

    fn recommended_concurrency(
        &self,
        _decoded_representation: &BytesRepresentation,
    ) -> Result<RecommendedConcurrency, CodecError> {
        // Use maximum(1) for single-threaded codecs to allow chunk-level parallelism
        Ok(RecommendedConcurrency::new_maximum(1))
    }

    fn encode<'a>(
        &self,
        decoded_value: ArrayBytesRaw<'a>,
        _options: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError> {
        // TODO: implement encoding
        todo!()
    }

    fn decode<'a>(
        &self,
        encoded_value: ArrayBytesRaw<'a>,
        _decoded_representation: &BytesRepresentation,
        _options: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError> {
        // TODO: implement decoding
        todo!()
    }

    fn encoded_representation(
        &self,
        decoded_representation: &BytesRepresentation,
    ) -> BytesRepresentation {
        // Return BoundedSize if you can compute a worst-case bound,
        // otherwise UnboundedSize.
        decoded_representation
            .size()
            .map_or(BytesRepresentation::UnboundedSize, |size| {
                // TODO: compute maximum encoded size bound
                BytesRepresentation::BoundedSize(size + /* overhead */)
            })
    }
}
```

**For Array-to-Array (A→A):**

Import `ArrayToArrayCodecTraits`, `ArrayCodecTraits`, `ArrayBytes`, `ChunkShape`, `DataType`, `FillValue` instead, and implement `encoded_shape`, `decoded_shape`, `encoded_data_type`, `encoded_fill_value`, `encode`, `decode`, and optionally `partial_decoder`.

**For Array-to-Bytes (A→B):**

Import `ArrayToBytesCodecTraits`, `ArrayCodecTraits`, and implement `encode`, `decode`, `encoded_representation` (returns `BytesRepresentation`), and optionally `partial_decoder`.

#### 7e. Expose in the Codec Module

In `zarrs/src/array/codec.rs`, add with appropriate `cfg` gate:

```rust
#[cfg(feature = "<codec_name>")]
pub use <codec_type>::<codec_name>::*;
```

#### 7f. Wire up the Module

In `zarrs/src/array/codec/<codec_type>.rs` (or the inline `mod` in `codec.rs`), add:

```rust
#[cfg(feature = "<codec_name>")]
pub mod <codec_name>;
```

### Step 8: Write Tests

In the module wrapper file, add a `#[cfg(test)]` section testing:

1. **Valid configuration parses correctly**
2. **Invalid configurations are rejected** (for each constraint)
3. **Round-trip encode/decode** produces identical bytes
4. If V2 is supported, test V2 configuration parsing

```rust
#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use super::*;
    use crate::array::BytesRepresentation;
    use zarrs_codec::{BytesToBytesCodecTraits, CodecOptions};

    const JSON_VALID: &str = r#"{ /* ... */ }"#;

    #[test]
    fn codec_<codec_name>_configuration_valid() {
        assert!(serde_json::from_str::<<CodecName>CodecConfiguration>(JSON_VALID).is_ok());
    }

    #[test]
    fn codec_<codec_name>_round_trip() {
        let elements: Vec<u16> = (0..32).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes_representation = BytesRepresentation::FixedSize(bytes.len() as u64);

        let configuration: <CodecName>CodecConfiguration =
            serde_json::from_str(JSON_VALID).unwrap();
        let codec = <CodecName>Codec::new_with_configuration(&configuration).unwrap();

        let encoded = codec
            .encode(Cow::Borrowed(&bytes), &CodecOptions::default())
            .unwrap();
        let decoded = codec
            .decode(encoded, &bytes_representation, &CodecOptions::default())
            .unwrap();
        assert_eq!(bytes, decoded.to_vec());
    }
}
```

### Step 9: Add to Snapshot Tests

The snapshot test suite in `zarrs/tests/codec_snapshot_tests.rs` runs the codec against all data types and verifies round-trip correctness, recording results as on-disk snapshots.

#### 9a. Enable in the feature gate

At the top of `codec_snapshot_tests.rs`, the file is conditionally compiled on all optional features. If the codec has a feature flag, add it to the `#[cfg(...)]` attribute:

```rust
#![cfg(all(
    // ... existing features ...
    feature = "<codec_name>",
))]
```

Skip this if the codec has no feature flag.

#### 9b. Add a `CodecDef` to `codec_registry()`

In the `codec_registry()` function, add an entry in the appropriate section (`Bytes-to-Bytes Codecs`, `Array-to-Array Codecs`, or `Array-to-Bytes Codecs`). Follow the pattern of an existing similar codec:

```rust
#[cfg(feature = "<codec_name>")]   // omit if no feature flag
codecs.push(CodecDef {
    name: <CodecName>Codec::aliases_v3().default_name.clone(),
    category: CodecCategory::BytesToBytes,  // or ArrayToArray / ArrayToBytes
    name_suffix: Some("<descriptive_config_suffix>"),  // e.g. "level5"; None if config-free
    factory: |_dt| CodecInstance::BytesToBytes(Arc::new(<CodecName>Codec::new(/* ... */))),
    lossy: false,        // true if encode/decode is not bit-exact
    non_deterministic: false,
    skip: None,          // Some(predicate_fn) to skip incompatible data types
});
```

For `name_suffix`, include a short human-readable description of the configuration used (e.g. `"level5"`, `"keepbits10"`). Use `None` only if the codec has no meaningful configuration.

#### 9c. Add to `registered_codecs()` in the compatibility matrix

In the `compatibility_matrix::registered_codecs()` function, add the codec to the appropriate category list:

```rust
// Array-to-Array / Array-to-Bytes / Bytes-to-Bytes section:
(
    codec::<CodecName>Codec::aliases_v3().default_name.clone(),
    "b2b",  // or "a2a" / "a2b"
),
```

#### 9d. Generate and commit the new snapshots

Run the test in add-mode to generate snapshot data for the new codec:

```bash
ADD_SNAPSHOTS=1 cargo test --all-features -p zarrs codec_snapshot_tests
```

This writes new snapshot directories (and `unsupported` markers for incompatible data types) under `zarrs/tests/data/snapshots/`. Review the output, then commit the new snapshot files.

### Step 10: Update `codecs.md`

Add a row for the new codec to `zarrs/doc/status/codecs.md`.

In the table, add to the appropriate codec type section:

```markdown
| <Codec Type>   | 🚧[`<identifier>`]                 | `<v2_id>` (or -)                    | <feature_flag> |
```

Use `🚧` prefix if the codec is experimental or not yet registered. Omit it for stable registered codecs.

Then add the link reference at the bottom of the file in the appropriate group:

```markdown
[`<identifier>`]: crate::array::codec::<codec_type>::<codec_name>
```

And if it has a spec link, add it to the spec links section:

```markdown
[<Spec Title>]: <spec_url>
```

### Step 11: Update Changelogs

Update the `## [Unreleased]` section in the `CHANGELOG.md` of each crate that was modified.

**`zarrs_metadata_ext/CHANGELOG.md`** — always updated (new codec metadata was added):
```markdown
## [Unreleased]

### Added
- Add `<CodecName>CodecConfiguration`, `<CodecName>CodecConfigurationV1` (and any other public types)
```

**`zarrs/CHANGELOG.md`** — always updated (new codec implementation was added):
```markdown
## [Unreleased]

### Added
- Add `<codec_name>` codec (feature: `<codec_name>`)
```

**`zarrs_data_type/CHANGELOG.md`** — only if a new data type trait was added (Step 5):
```markdown
## [Unreleased]

### Added
- Add `<CodecName>DataTypeTraits` trait for the `<codec_name>` codec
```

Follow the existing changelog style: use imperative mood, be concise, and group under `Added` / `Changed` / `Fixed` / `Removed` as appropriate.

### Step 12: Verify Build

Run:
```bash
cargo check -p zarrs --features <codec_name>
cargo test -p zarrs --features <codec_name> -- codec_<codec_name>
```

And if the metadata was added to `zarrs_metadata_ext`:
```bash
cargo test -p zarrs_metadata_ext -- codec_<codec_name>
```

## Quick Reference: Key Types

| Type | Import From | Used In |
|------|-------------|---------|
| `ArrayBytesRaw<'a>` | `crate::array` | B→B encode/decode input/output |
| `ArrayBytes<'a>` | `zarrs_codec` | A→A / A→B encode/decode |
| `BytesRepresentation` | `crate::array` | Describes byte size (Fixed/Bounded/Unbounded) |
| `CodecOptions` | `zarrs_codec` | Runtime options passed to encode/decode |
| `CodecMetadataOptions` | `zarrs_codec` | Options for `configuration()` method |
| `RecommendedConcurrency` | `zarrs_codec` | Hint for codec parallelism |
| `PartialDecoderCapability` | `zarrs_codec` | Declares partial decode support |
| `PartialEncoderCapability` | `zarrs_codec` | Declares partial encode support |
| `Configuration` | `zarrs_metadata` | Serialised codec configuration |
| `ZarrVersion` | `zarrs_plugin` | V2/V3 discriminant for `configuration()` |
| `PluginCreateError` | `zarrs_plugin` | Error for `new_with_configuration` |
| `ConfigurationSerialize` | `zarrs_metadata` | Trait for config enum serialisation |

## Key Macros

- `zarrs_plugin::impl_extension_aliases!(Codec, v3: "name")` — implements `ExtensionIdentifier`/`ExtensionAliases`
- `inventory::submit! { CodecPluginV3::new::<Codec>() }` — auto-registers the codec at startup
- `inventory::submit! { CodecPluginV2::new::<Codec>() }` — also register as V2 if applicable

## Implementation Checklist

- [ ] Specification URL fetched and understood
- [ ] Partial encode/decode decision made with user
- [ ] `zarrs_metadata_ext/src/codec/<category>/<codec_name>.rs` created
- [ ] Entry added to `zarrs_metadata_ext/src/codec.rs`
- [ ] Data type trait in `zarrs_data_type` (if needed)
- [ ] Feature flag added to `zarrs/Cargo.toml`
- [ ] Dependency added to `zarrs/Cargo.toml` (if needed)
- [ ] `zarrs/src/array/codec/<codec_type>/<codec_name>.rs` created (module + registration)
- [ ] `zarrs/src/array/codec/<codec_type>/<codec_name>/<codec_name>_codec.rs` created (implementation)
- [ ] Module added to `<codec_type>.rs` or `codec.rs`
- [ ] Re-export added to `zarrs/src/array/codec.rs`
- [ ] Tests written and passing
- [ ] Feature flag added to `#[cfg(...)]` in `codec_snapshot_tests.rs` (if feature-gated)
- [ ] `CodecDef` added to `codec_registry()` in `codec_snapshot_tests.rs`
- [ ] Codec added to `registered_codecs()` in the compatibility matrix
- [ ] Snapshots generated with `ADD_SNAPSHOTS=1 cargo test --all-features -p zarrs codec_snapshot_tests`
- [ ] `zarrs/doc/status/codecs.md` updated with new row and link reference
- [ ] `zarrs_metadata_ext/CHANGELOG.md` updated
- [ ] `zarrs/CHANGELOG.md` updated
- [ ] `zarrs_data_type/CHANGELOG.md` updated (if data type trait was added)
- [ ] `cargo check` and `cargo test` pass
