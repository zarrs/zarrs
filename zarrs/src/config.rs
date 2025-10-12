//! `zarrs` global configuration options.
//!
//! See [`Config`] for the list of options.

use std::sync::{LazyLock, RwLock, RwLockReadGuard, RwLockWriteGuard};

use zarrs_registry::{
    ExtensionAliasesCodecV2, ExtensionAliasesCodecV3, ExtensionAliasesDataTypeV2,
    ExtensionAliasesDataTypeV3,
};

#[cfg(doc)]
use crate::array::{codec::CodecOptions, ArrayMetadataOptions};

/// Global configuration options for the `zarrs` crate.
///
/// Retrieve the global [`Config`] with [`global_config`] and modify it with [`global_config_mut`].
///
/// ## Codec / Chunk Options
///
/// ### Validate Checksums
///  > default: [`true`]
///
/// [`CodecOptions::validate_checksums()`] defaults to [`Config::validate_checksums()`].
///
/// If validate checksums is enabled, checksum codecs (e.g. `crc32c`, `fletcher32`) will validate that encoded data matches stored checksums, otherwise validation is skipped.
/// Note that regardless of this configuration option, checksum codecs may skip validation when partial decoding.
///
/// ### Store Empty Chunks
///  > default: [`false`]
///
/// [`CodecOptions::store_empty_chunks()`] defaults to [`Config::store_empty_chunks()`].
///
/// If `false`, empty chunks (where all elements match the fill value) will not be stored.
/// This incurs a computational overhead as each element must be tested for equality to the fill value before a chunk is encoded.
/// If `true`, the aforementioned test is skipped and empty chunks will be stored.
/// Note that empty chunks must still be stored explicitly (e.g. with [`Array::store_chunk`](crate::array::Array::store_chunk)).
///
/// ### Codec Concurrent Target
/// > default: [`std::thread::available_parallelism`]`()`
///
/// [`CodecOptions::concurrent_target()`] defaults to [`Config::codec_concurrent_target()`].
///
/// The default number of concurrent operations to target for codec encoding and decoding.
/// Limiting concurrent operations is needed to reduce memory usage and improve performance.
/// Concurrency is unconstrained if the concurrent target if set to zero.
///
/// Note that the default codec concurrent target can be overridden for any encode/decode operation.
/// This is performed automatically for many array operations (see the [chunk concurrent minimum](#chunk-concurrent-minimum) option).
///
/// ### Chunk Concurrent Minimum
/// > default: `4`
///
/// Array operations involving multiple chunks can tune the chunk and codec concurrency to improve performance/reduce memory usage.
/// This option sets the preferred minimum chunk concurrency.
/// The concurrency of internal codecs is adjusted to accomodate for the chunk concurrency in accordance with the concurrent target set in the [`CodecOptions`] parameter of an encode or decode method.
///
/// ### Experimental Partial Encoding
/// > default: [`false`]
///
/// If `true`, [`Array::store_chunk_subset`](crate::array::Array::store_chunk_subset) and [`Array::store_array_subset`](crate::array::Array::store_array_subset) and variants can use partial encoding.
/// This is relevant when using the sharding codec, as it enables inner chunks to be written without reading and writing entire shards.
///
/// This is an experimental feature for now until it has more comprehensively tested and support is added in the async API.
///
/// ## Metadata Options
///
/// ### Codec Store Metadata If Encode Only
/// > default: [`true`]
///
/// Some codecs perform potentially irreversible transformations during encoding that decoders do not need to be aware of.
/// If this option is `false`, codecs with this behaviour will not write their metadata.
/// This enables arrays to be consumed by other Zarr V3 implementations that do not support the codec.
/// Currently, this options only affects the `bitround` codec.
///
/// ### Metadata Convert Version
/// > default: [`MetadataConvertVersion::Default`] (keep existing version)
///
/// Determines the Zarr version of metadata created with [`Array::metadata_opt`](crate::array::Array::metadata_opt) and [`Group::metadata_opt`](crate::group::Group::metadata_opt).
/// These methods are used internally by the `store_metadata` and `store_metadata_opt` methods of [`crate::array::Array`] and [`crate::group::Group`].
///
/// ### Metadata Erase Version
/// > default: [`MetadataEraseVersion::Default`] (erase existing version)
///
/// The default behaviour for the `erase_metadata` methods of [`crate::array::Array`] and [`crate::group::Group`].
/// Determines whether to erase metadata of a specific Zarr version, the same version as the array/group was created with, or all known versions.
///
/// ### Include `zarrs` Metadata
/// > default: [`true`]
///
/// [`ArrayMetadataOptions::include_zarrs_metadata`](crate::array::ArrayMetadataOptions::include_zarrs_metadata) defaults to [`Config::include_zarrs_metadata`].
///
/// If true, array metadata generated with [`Array::metadata_opt`](crate::array::Array::metadata_opt) (used internally by [`Array::store_metadata`](crate::array::Array::store_metadata)) includes the `zarrs` version and a link to its source code.
/// For example:
/// ```json
/// "_zarrs": {
///    "description": "This array was created with zarrs",
///    "repository": "https://github.com/zarrs/zarrs",
///    "version": "0.15.0"
///  }
/// ```
///
/// ### Codec Aliases
/// > default: see [`ExtensionAliasesCodecV3::default`] and [`ExtensionAliasesCodecV2::default`].
///
/// The default codec `name`s used when serialising codecs, and recognised codec `name` aliases when deserialising codecs.
/// Codec default `name`s and aliases can be modified at runtime.
///
/// Note that the [`NamedCodec`](crate::array::codec::NamedCodec) mechanism means that a serialised codec `name` can differ from the default `name`.
/// By default, updating and storing the metadata of an array will NOT convert aliased codec names to the default codec name.
/// This behaviour can be changed with the [convert aliased extension names](#convert-aliased-extension-names) configuration option.
///
/// The codec maps enable support for unstandardised codecs, such as:
/// - codecs registered in the official [`zarr-extensions`](https://github.com/zarr-developers/zarr-extensions) repository that are compatible with `zarrs`,
/// - `zarrs` experimental codecs with `name`s that have since changed, and
/// - user-defined custom codecs.
///
/// If a codec is not present in the codec maps, the `name` will be inferred as the unique codec identifier.
/// Codecs registered for that identifier work without any changes required for the codec maps.
///
/// ### Data Type Aliases
/// > default: see [`ExtensionAliasesDataTypeV3::default`] and [`ExtensionAliasesDataTypeV2::default`].
///
/// These operate similarly to codec maps, but for data types.
///
/// ### Convert Aliased Extension Names
/// > default: [`false`]
///
/// If true, then aliased extension names will be replaced by the standard name if metadata is resaved.
/// This sets the default for the association option of [`crate::array::ArrayMetadataOptions`].
#[derive(Debug)]
#[allow(clippy::struct_excessive_bools)]
pub struct Config {
    validate_checksums: bool,
    store_empty_chunks: bool,
    codec_concurrent_target: usize,
    chunk_concurrent_minimum: usize,
    codec_store_metadata_if_encode_only: bool,
    metadata_convert_version: MetadataConvertVersion,
    metadata_erase_version: MetadataEraseVersion,
    include_zarrs_metadata: bool,
    codec_aliases_v3: ExtensionAliasesCodecV3,
    codec_aliases_v2: ExtensionAliasesCodecV2,
    data_type_aliases_v3: ExtensionAliasesDataTypeV3,
    data_type_aliases_v2: ExtensionAliasesDataTypeV2,
    experimental_partial_encoding: bool,
    convert_aliased_extension_names: bool,
}

#[allow(clippy::derivable_impls)]
impl Default for Config {
    fn default() -> Self {
        Self {
            validate_checksums: true,
            store_empty_chunks: false,
            codec_concurrent_target: rayon::current_num_threads(),
            chunk_concurrent_minimum: 4,
            codec_store_metadata_if_encode_only: true,
            metadata_convert_version: MetadataConvertVersion::Default,
            metadata_erase_version: MetadataEraseVersion::Default,
            include_zarrs_metadata: true,
            codec_aliases_v3: ExtensionAliasesCodecV3::default(),
            codec_aliases_v2: ExtensionAliasesCodecV2::default(),
            data_type_aliases_v3: ExtensionAliasesDataTypeV3::default(),
            data_type_aliases_v2: ExtensionAliasesDataTypeV2::default(),
            experimental_partial_encoding: false,
            convert_aliased_extension_names: false,
        }
    }
}

impl Config {
    /// Get the [validate checksums](#validate-checksums) configuration.
    #[must_use]
    pub fn validate_checksums(&self) -> bool {
        self.validate_checksums
    }

    /// Set the [validate checksums](#validate-checksums) configuration.
    pub fn set_validate_checksums(&mut self, validate_checksums: bool) -> &mut Self {
        self.validate_checksums = validate_checksums;
        self
    }

    /// Get the [store empty chunks](#store-empty-chunks) configuration.
    #[must_use]
    pub fn store_empty_chunks(&self) -> bool {
        self.store_empty_chunks
    }

    /// Set the [store empty chunks](#store-empty-chunks) configuration.
    pub fn set_store_empty_chunks(&mut self, store_empty_chunks: bool) -> &mut Self {
        self.store_empty_chunks = store_empty_chunks;
        self
    }

    /// Get the [codec concurrent target](#codec-concurrent-target) configuration.
    #[must_use]
    pub fn codec_concurrent_target(&self) -> usize {
        self.codec_concurrent_target
    }

    /// Set the [codec concurrent target](#codec-concurrent-target) configuration.
    pub fn set_codec_concurrent_target(&mut self, concurrent_target: usize) -> &mut Self {
        self.codec_concurrent_target = concurrent_target;
        self
    }

    /// Get the [chunk concurrent minimum](#chunk-concurrent-minimum) configuration.
    #[must_use]
    pub fn chunk_concurrent_minimum(&self) -> usize {
        self.chunk_concurrent_minimum
    }

    /// Set the [chunk concurrent minimum](#chunk-concurrent-minimum) configuration.
    pub fn set_chunk_concurrent_minimum(&mut self, concurrent_minimum: usize) -> &mut Self {
        self.chunk_concurrent_minimum = concurrent_minimum;
        self
    }

    /// Get the [codec store metadata if encode only](#codec-store-metadata-if-encode-only) configuration.
    #[must_use]
    pub fn codec_store_metadata_if_encode_only(&self) -> bool {
        self.codec_store_metadata_if_encode_only
    }

    /// Set the [codec store metadata if encode only](#codec-store-metadata-if-encode-only) configuration.
    pub fn set_codec_store_metadata_if_encode_only(&mut self, enabled: bool) -> &mut Self {
        self.codec_store_metadata_if_encode_only = enabled;
        self
    }

    /// Get the [metadata convert version](#metadata-convert-version) configuration.
    #[must_use]
    pub fn metadata_convert_version(&self) -> MetadataConvertVersion {
        self.metadata_convert_version
    }

    /// Set the [metadata convert version](#metadata-convert-version) configuration.
    pub fn set_metadata_convert_version(&mut self, version: MetadataConvertVersion) -> &mut Self {
        self.metadata_convert_version = version;
        self
    }

    /// Get the [metadata erase version](#metadata-erase-version) configuration.
    #[must_use]
    pub fn metadata_erase_version(&self) -> MetadataEraseVersion {
        self.metadata_erase_version
    }

    /// Set the [metadata erase version](#metadata-erase-version) configuration.
    pub fn set_metadata_erase_version(&mut self, version: MetadataEraseVersion) -> &mut Self {
        self.metadata_erase_version = version;
        self
    }

    /// Get the [include zarrs metadata](#include-zarrs-metadata) configuration.
    #[must_use]
    pub fn include_zarrs_metadata(&self) -> bool {
        self.include_zarrs_metadata
    }

    /// Set the [include zarrs metadata](#include-zarrs-metadata) configuration.
    pub fn set_include_zarrs_metadata(&mut self, include_zarrs_metadata: bool) -> &mut Self {
        self.include_zarrs_metadata = include_zarrs_metadata;
        self
    }

    /// Get the Zarr V3 [codec aliases](#codec-aliases) configuration.
    #[must_use]
    pub fn codec_aliases_v3(&self) -> &ExtensionAliasesCodecV3 {
        &self.codec_aliases_v3
    }

    /// Get a mutable reference to the Zarr V3 [codec aliases](#codec-aliases) configuration.
    pub fn codec_aliases_v3_mut(&mut self) -> &mut ExtensionAliasesCodecV3 {
        &mut self.codec_aliases_v3
    }

    /// Get the Zarr V3 [data type aliases](#data-type-aliases) configuration.
    #[must_use]
    pub fn data_type_aliases_v3(&self) -> &ExtensionAliasesDataTypeV3 {
        &self.data_type_aliases_v3
    }

    /// Get a mutable reference to the Zarr V3 [data type aliases](#data-type-aliases) configuration.
    pub fn data_type_aliases_v3_mut(&mut self) -> &mut ExtensionAliasesDataTypeV3 {
        &mut self.data_type_aliases_v3
    }

    /// Get the Zarr V2 [codec aliases](#codec-aliases) configuration.
    #[must_use]
    pub fn codec_aliases_v2(&self) -> &ExtensionAliasesCodecV2 {
        &self.codec_aliases_v2
    }

    /// Get a mutable reference to the Zarr V2 [codec aliases](#codec-aliases) configuration.
    pub fn codec_aliases_v2_mut(&mut self) -> &mut ExtensionAliasesCodecV2 {
        &mut self.codec_aliases_v2
    }

    /// Get the Zarr V2 [data type aliases](#data-type-aliases) configuration.
    #[must_use]
    pub fn data_type_aliases_v2(&self) -> &ExtensionAliasesDataTypeV2 {
        &self.data_type_aliases_v2
    }

    /// Get a mutable reference to the Zarr V2 [data type aliases](#data-type-aliases) configuration.
    pub fn data_type_aliases_v2_mut(&mut self) -> &mut ExtensionAliasesDataTypeV2 {
        &mut self.data_type_aliases_v2
    }

    /// Get the [experimental partial encoding](#experimental-partial-encoding) configuration.
    #[must_use]
    pub fn experimental_partial_encoding(&self) -> bool {
        self.experimental_partial_encoding
    }

    /// Set the [experimental partial encoding](#experimental-partial-encoding) configuration.
    pub fn set_experimental_partial_encoding(
        &mut self,
        experimental_partial_encoding: bool,
    ) -> &mut Self {
        self.experimental_partial_encoding = experimental_partial_encoding;
        self
    }

    /// Set the [convert aliased extension names](#convert-aliased-extension-names) configuration.
    #[must_use]
    pub fn convert_aliased_extension_names(&self) -> bool {
        self.convert_aliased_extension_names
    }

    /// Set the [convert aliased extension names](#convert-aliased-extension-names) configuration.
    pub fn set_convert_aliased_extension_names(
        &mut self,
        convert_aliased_extension_names: bool,
    ) -> &mut Self {
        self.convert_aliased_extension_names = convert_aliased_extension_names;
        self
    }
}

static CONFIG: LazyLock<RwLock<Config>> = LazyLock::new(|| RwLock::new(Config::default()));

/// Returns a reference to the global `zarrs` configuration.
///
/// # Panics
/// This function panics if the underlying lock has been poisoned and might panic if the global config is already held by the current thread.
pub fn global_config() -> RwLockReadGuard<'static, Config> {
    CONFIG.read().unwrap()
}

/// Returns a mutable reference to the global `zarrs` configuration.
///
/// # Panics
/// This function panics if the underlying lock has been poisoned and might panic if the global config is already held by the current thread.
pub fn global_config_mut() -> RwLockWriteGuard<'static, Config> {
    CONFIG.write().unwrap()
}

/// The metadata version to retrieve.
///
/// Used with [`crate::array::Array::open_opt`], [`crate::group::Group::open_opt`].
pub enum MetadataRetrieveVersion {
    /// Either Zarr V3 or V2. V3 is prioritised over V2 if found.
    Default,
    /// Zarr V3.
    V3,
    /// Zarr V2.
    V2,
}

/// Version options for [`Array::store_metadata`](crate::array::Array::store_metadata) and [`Group::store_metadata`](crate::group::Group::store_metadata), and their async variants.
#[derive(Debug, Clone, Copy)]
pub enum MetadataConvertVersion {
    /// Write the same version as the input metadata.
    Default,
    /// Write Zarr V3 metadata. Zarr V2 metadata will not be automatically removed if it exists.
    V3,
}

/// Version options for [`Array::erase_metadata`](crate::array::Array::erase_metadata) and [`Group::erase_metadata`](crate::group::Group::erase_metadata), and their async variants.
#[derive(Debug, Clone, Copy)]
pub enum MetadataEraseVersion {
    /// Erase the same version as the input metadata.
    Default,
    /// Erase all metadata.
    All,
    /// Erase Zarr V3 metadata.
    V3,
    /// Erase Zarr V2 metadata.
    V2,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_validate_checksums() {
        assert!(global_config().validate_checksums());
        global_config_mut().set_validate_checksums(false);
        assert!(!global_config().validate_checksums());
        global_config_mut().set_validate_checksums(true);
    }
}
