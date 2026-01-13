use std::borrow::Cow;
use std::num::NonZeroU64;
use std::sync::Arc;

use derive_more::From;
use zarrs_plugin::ExtensionName;

use super::chunk_key_encoding::DefaultChunkKeyEncoding;
use super::{
    Array, ArrayCreateError, ArrayMetadata, ArrayMetadataV3, ArrayShape, ChunkShape, CodecChain,
    DimensionName, StorageTransformerChain,
};
use crate::array::array_builder::array_builder_fill_value::ArrayBuilderFillValueImpl;
use crate::array::{ArrayMetadataOptions, ChunkGrid};
use crate::config::global_config;
use crate::node::NodePath;
use zarrs_chunk_key_encoding::ChunkKeyEncoding;
use zarrs_codec::{
    ArrayToArrayCodecTraits, ArrayToBytesCodecTraits, BytesToBytesCodecTraits, CodecOptions,
};
use zarrs_metadata::v3::{AdditionalFieldsV3, MetadataV3};
use zarrs_metadata::{ChunkKeySeparator, IntoDimensionName};

mod array_builder_chunk_grid;
pub use array_builder_chunk_grid::ArrayBuilderChunkGrid;

mod array_builder_chunk_grid_metadata;
pub use array_builder_chunk_grid_metadata::ArrayBuilderChunkGridMetadata;

mod array_builder_data_type;
pub use array_builder_data_type::ArrayBuilderDataType;

mod array_builder_fill_value;
pub use array_builder_fill_value::ArrayBuilderFillValue;

/// An [`Array`] builder.
///
/// [`ArrayBuilder`] is initialised from an array shape, data type, chunk grid, and fill value.
///  - The default array-to-bytes codec is dependent on the data type:
///    - [`bytes`](crate::array::codec::array_to_bytes::bytes) for fixed-length data types,
///    - [`vlen-utf8`](crate::array::codec::array_to_bytes::vlen_utf8) for the [`StringDataType`](crate::array::data_type::StringDataType) variable-length data type,
///    - [`vlen-bytes`](crate::array::codec::array_to_bytes::vlen_bytes) for the [`BytesDataType`](crate::array::data_type::BytesDataType) variable-length data type, and
///    - [`vlen`](crate::array::codec::array_to_bytes::vlen) for any other variable-length data type.
///  - Array-to-array and bytes-to-bytes codecs are empty by default.
///  - The default chunk key encoding is [`default`](crate::array::chunk_key_encoding::default::DefaultChunkKeyEncoding) with the `/` chunk key separator.
///  - Attributes, storage transformers, and dimension names are empty.
///
/// Use the methods in the array builder to change the configuration away from these defaults, and then build the array at a path of some storage with [`ArrayBuilder::build`].
///
/// [`build`](ArrayBuilder::build) does not modify the store! Array metadata has to be explicitly written with [`Array::store_metadata`].
///
/// ### Simple Example
/// This array is uncompressed, and has no dimension names or attributes.
///
/// ```rust
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # use std::sync::Arc;
/// use zarrs::array::{ArrayBuilder, DataType, FillValue, ZARR_NAN_F32, data_type};
/// # let store = Arc::new(zarrs::storage::store::MemoryStore::new());
/// let mut array = ArrayBuilder::new(
///     vec![8, 8], // array shape
///     vec![4, 4], // regular chunk shape
///     data_type::float32(), // data type
///     f32::NAN, // fill value
/// )
/// .build(store.clone(), "/group/array")?;
/// array.store_metadata()?; // write metadata to the store
/// # Ok(())
/// # }
/// ```
///
/// ### Advanced Example
/// The array is compressed with the [`zstd`](crate::array::codec::bytes_to_bytes::zstd) codec, dimension names are set, and the experimental [`rectangular`](crate::array::chunk_grid::rectangular) chunk grid is used.
///
/// This example uses alternative types to specify the array shape, data type, chunk grid, and fill value.
/// In general you don't want to use strings, prefer concrete types (like [`RectangularChunkGridConfiguration`](crate::array::chunk_grid::RectangularChunkGridConfiguration), for example).
///
/// ```rust
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # use std::sync::Arc;
/// use zarrs::array::{ArrayBuilder, DataType, FillValue, ZARR_NAN_F32};
/// # let store = Arc::new(zarrs::storage::store::MemoryStore::new());
/// let mut array = ArrayBuilder::new(
///     [8, 8],
///     r#"{"name":"rectangular","configuration":{"chunk_shape": [[1, 2, 5], 4]}}"#,
///     "float32",
///     "NaN",
/// )
/// .bytes_to_bytes_codecs(vec![
///     #[cfg(feature = "zstd")]
///     Arc::new(zarrs::array::codec::ZstdCodec::new(5, false)),
/// ])
/// .dimension_names(Some(["y", "x"]))
/// .build(store.clone(), "/group/array")?;
/// array.store_metadata()?; // write metadata to the store
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct ArrayBuilder {
    /// Data type.
    data_type: ArrayBuilderDataType,
    /// Chunk grid.
    chunk_grid: ArrayBuilderChunkGridMaybe,
    /// Chunk key encoding.
    chunk_key_encoding: ChunkKeyEncoding,
    /// Fill value.
    fill_value: ArrayBuilderFillValue,
    /// The array-to-array codecs.
    array_to_array_codecs: Vec<Arc<dyn ArrayToArrayCodecTraits>>,
    /// The array-to-bytes codec.
    array_to_bytes_codec: Option<Arc<dyn ArrayToBytesCodecTraits>>,
    /// The bytes-to-bytes codecs. If [`None`], chooses a default based on the data type.
    bytes_to_bytes_codecs: Vec<Arc<dyn BytesToBytesCodecTraits>>,
    /// Storage transformer chain.
    storage_transformers: StorageTransformerChain,
    /// Attributes.
    attributes: serde_json::Map<String, serde_json::Value>,
    /// Dimension names.
    dimension_names: Option<Vec<DimensionName>>,
    /// Additional fields.
    additional_fields: AdditionalFieldsV3,
    /// Subchunk (inner chunk) shape for sharding.
    #[cfg(feature = "sharding")]
    subchunk_shape: Option<ArrayShape>,
    /// Codec options.
    codec_options: CodecOptions,
    /// Metadata options.
    metadata_options: ArrayMetadataOptions,
}

#[derive(Debug, From)]
enum ArrayBuilderChunkGridMaybe {
    ChunkGrid(ArrayBuilderChunkGrid),
    Metadata(ArrayShape, ArrayBuilderChunkGridMetadata),
}

impl ArrayBuilder {
    /// Create a new array builder for an array at `path` from an array shape and chunk grid metadata.
    ///
    /// The length of the array shape must match the dimensionality of the intended array.
    /// Some chunk grids (e.g. `regular`) support all zero shape, indicating the shape is unbounded.
    #[must_use]
    pub fn new(
        shape: impl Into<ArrayShape>,
        chunk_grid_metadata: impl Into<ArrayBuilderChunkGridMetadata>,
        data_type: impl Into<ArrayBuilderDataType>,
        fill_value: impl Into<ArrayBuilderFillValue>,
    ) -> Self {
        let shape = shape.into();
        let data_type = data_type.into();
        let chunk_grid_metadata: ArrayBuilderChunkGridMetadata = chunk_grid_metadata.into();
        let chunk_grid: ArrayBuilderChunkGridMaybe = (shape, chunk_grid_metadata).into();
        let fill_value = fill_value.into();

        let (codec_options, metadata_options) = {
            let config = global_config();
            (config.codec_options(), config.array_metadata_options())
        };

        Self {
            data_type,
            chunk_grid,
            chunk_key_encoding: DefaultChunkKeyEncoding::default().into(),
            fill_value,
            array_to_array_codecs: Vec::default(),
            array_to_bytes_codec: None,
            bytes_to_bytes_codecs: Vec::default(),
            attributes: serde_json::Map::default(),
            storage_transformers: StorageTransformerChain::default(),
            dimension_names: None,
            additional_fields: AdditionalFieldsV3::default(),
            #[cfg(feature = "sharding")]
            subchunk_shape: None,
            codec_options,
            metadata_options,
        }
    }

    /// Create a new array builder with a concrete chunk grid (with an associated array shape).
    pub fn new_with_chunk_grid(
        chunk_grid: impl Into<ArrayBuilderChunkGrid>,
        data_type: impl Into<ArrayBuilderDataType>,
        fill_value: impl Into<ArrayBuilderFillValue>,
    ) -> Self {
        let data_type = data_type.into();
        let chunk_grid: ArrayBuilderChunkGrid = chunk_grid.into();
        let chunk_grid: ArrayBuilderChunkGridMaybe = chunk_grid.into();
        let fill_value = fill_value.into();
        let (codec_options, metadata_options) = {
            let config = global_config();
            (config.codec_options(), config.array_metadata_options())
        };

        Self {
            data_type,
            chunk_grid,
            chunk_key_encoding: DefaultChunkKeyEncoding::default().into(),
            fill_value,
            array_to_array_codecs: Vec::default(),
            array_to_bytes_codec: None,
            bytes_to_bytes_codecs: Vec::default(),
            attributes: serde_json::Map::default(),
            storage_transformers: StorageTransformerChain::default(),
            dimension_names: None,
            additional_fields: AdditionalFieldsV3::default(),
            #[cfg(feature = "sharding")]
            subchunk_shape: None,
            codec_options,
            metadata_options,
        }
    }

    /// Create a new builder copying the configuration of an existing array.
    #[must_use]
    pub fn from_array<T: ?Sized>(array: &Array<T>) -> Self {
        let mut builder = Self::new(
            array.shape().to_vec(),
            array.chunk_grid().metadata(),
            array.data_type().clone(),
            array.fill_value_metadata(),
        );
        let additional_fields = match array.metadata() {
            ArrayMetadata::V2(_metadata) => AdditionalFieldsV3::default(),
            ArrayMetadata::V3(metadata) => metadata.additional_fields.clone(),
        };

        builder.array_to_array_codecs = array.codecs().array_to_array_codecs().to_vec();
        builder.array_to_bytes_codec = Some(array.codecs().array_to_bytes_codec().clone());
        builder.bytes_to_bytes_codecs = array.codecs().bytes_to_bytes_codecs().to_vec();

        builder
            .additional_fields(additional_fields)
            .attributes(array.attributes().clone())
            .chunk_key_encoding(array.chunk_key_encoding().clone())
            .dimension_names(array.dimension_names().clone())
            .storage_transformers(array.storage_transformers().clone());
        builder
    }

    /// Set the shape.
    pub fn shape(&mut self, shape: impl Into<ArrayShape>) -> &mut Self {
        let shape = shape.into();
        let chunk_grid_metadata = match &self.chunk_grid {
            ArrayBuilderChunkGridMaybe::ChunkGrid(chunk_grid) => {
                ArrayBuilderChunkGridMetadata::from(chunk_grid.as_chunk_grid().metadata())
            }
            ArrayBuilderChunkGridMaybe::Metadata(_array_shape, chunk_grid_metadata) => {
                chunk_grid_metadata.clone()
            }
        };
        self.chunk_grid = (shape, chunk_grid_metadata).into();
        self
    }

    /// Set the data type.
    pub fn data_type(&mut self, data_type: impl Into<ArrayBuilderDataType>) -> &mut Self {
        self.data_type = data_type.into();
        self
    }

    /// Set the chunk grid metadata.
    pub fn chunk_grid_metadata(
        &mut self,
        chunk_grid_metadata: impl Into<ArrayBuilderChunkGridMetadata>,
    ) -> &mut Self {
        let array_shape = match &self.chunk_grid {
            ArrayBuilderChunkGridMaybe::ChunkGrid(chunk_grid) => {
                chunk_grid.as_chunk_grid().array_shape()
            }
            ArrayBuilderChunkGridMaybe::Metadata(array_shape, _chunk_grid_metadata) => array_shape,
        };
        let chunk_grid_metadata = chunk_grid_metadata.into();
        self.chunk_grid = (array_shape.to_vec(), chunk_grid_metadata).into();
        self
    }

    /// Set the chunk grid. This may also change the array shape.
    pub fn chunk_grid(&mut self, chunk_grid: impl Into<ArrayBuilderChunkGrid>) -> &mut Self {
        let chunk_grid = chunk_grid.into();
        self.chunk_grid = chunk_grid.into();
        self
    }

    /// Set the fill value.
    pub fn fill_value(&mut self, fill_value: impl Into<ArrayBuilderFillValue>) -> &mut Self {
        self.fill_value = fill_value.into();
        self
    }

    /// Set the chunk key encoding.
    ///
    /// If left unmodified, the array will use `default` chunk key encoding with the `/` chunk key separator.
    pub fn chunk_key_encoding(
        &mut self,
        chunk_key_encoding: impl Into<ChunkKeyEncoding>,
    ) -> &mut Self {
        self.chunk_key_encoding = chunk_key_encoding.into();
        self
    }

    /// Set the chunk key encoding to `default` with `separator`.
    ///
    /// If left unmodified, the array will use `default` chunk key encoding with the `/` chunk key separator.
    pub fn chunk_key_encoding_default_separator(
        &mut self,
        separator: ChunkKeySeparator,
    ) -> &mut Self {
        self.chunk_key_encoding = DefaultChunkKeyEncoding::new(separator).into();
        self
    }

    /// Set the array-to-array codecs.
    ///
    /// If left unmodified, the array will have no array-to-array codecs.
    pub fn array_to_array_codecs(
        &mut self,
        array_to_array_codecs: Vec<Arc<dyn ArrayToArrayCodecTraits>>,
    ) -> &mut Self {
        self.array_to_array_codecs = array_to_array_codecs;
        self
    }

    /// Set the array-to-bytes codec.
    ///
    /// If left unmodified, the array will default to using the `bytes` codec with native endian encoding.
    pub fn array_to_bytes_codec(
        &mut self,
        array_to_bytes_codec: Arc<dyn ArrayToBytesCodecTraits>,
    ) -> &mut Self {
        self.array_to_bytes_codec = Some(array_to_bytes_codec);
        self
    }

    /// Set the bytes-to-bytes codecs.
    ///
    /// If left unmodified, the array will have no bytes-to-bytes codecs.
    pub fn bytes_to_bytes_codecs(
        &mut self,
        bytes_to_bytes_codecs: Vec<Arc<dyn BytesToBytesCodecTraits>>,
    ) -> &mut Self {
        self.bytes_to_bytes_codecs = bytes_to_bytes_codecs;
        self
    }

    /// Set the subchunk (inner chunk) shape for sharding.
    ///
    /// When set, the array will use the `sharding` codec.
    /// The chunk shape is the shard shape, and `subchunk_shape` is the shape of the inner chunks within each shard.
    ///
    /// If left unmodified or set to `None`, the array will not use sharding, unless configured manually via [`array_to_bytes_codec`](Self::array_to_bytes_codec).
    ///
    /// The subchunk shape must have all non-zero elements (validated during build).
    ///
    /// # Sharding Configuration
    ///
    /// This method uses a default [`ShardingCodecBuilder`](super::codec::ShardingCodecBuilder) configuration:
    /// - No array-to-array codecs preceding the sharding codec
    /// - No bytes-to-bytes codecs following the sharding codec
    /// - The shard index is encoded with `crc32c` checksum (if the `crc32c` feature is enabled)
    ///
    /// The codecs specified via [`array_to_array_codecs`](Self::array_to_array_codecs),
    /// [`array_to_bytes_codec`](Self::array_to_bytes_codec), and
    /// [`bytes_to_bytes_codecs`](Self::bytes_to_bytes_codecs) are used internally
    /// for encoding the inner chunks within each shard.
    ///
    /// For more advanced usage (e.g., compressing an entire shard), set
    /// [`array_to_bytes_codec`](Self::array_to_bytes_codec) explicitly with a sharding codec
    /// built using [`ShardingCodecBuilder`](super::codec::ShardingCodecBuilder).
    ///
    /// # Example
    /// ```rust
    /// # use zarrs::array::{ArrayBuilder, DataType, data_type};
    /// # let store = std::sync::Arc::new(zarrs::storage::store::MemoryStore::new());
    /// let array = ArrayBuilder::new(
    ///     vec![64, 64],    // array shape
    ///     vec![16, 16],    // chunk (shard) shape
    ///     data_type::float32(),
    ///     0.0f32,
    /// )
    /// .subchunk_shape(vec![4, 4])  // inner chunk shape within each shard
    /// .build(store, "/array")
    /// .unwrap();
    /// ```
    #[cfg(feature = "sharding")]
    pub fn subchunk_shape(&mut self, subchunk_shape: impl Into<Option<ArrayShape>>) -> &mut Self {
        self.subchunk_shape = subchunk_shape.into();
        self
    }

    /// Set the user defined attributes.
    ///
    /// If left unmodified, the user defined attributes of the array will be empty.
    pub fn attributes(
        &mut self,
        attributes: serde_json::Map<String, serde_json::Value>,
    ) -> &mut Self {
        self.attributes = attributes;
        self
    }

    /// Return a mutable reference to the attributes.
    pub fn attributes_mut(&mut self) -> &mut serde_json::Map<String, serde_json::Value> {
        &mut self.attributes
    }

    /// Set the additional fields.
    ///
    /// Set additional fields not defined in the Zarr specification.
    /// Use this cautiously. In general, store user defined attributes using [`ArrayBuilder::attributes`].
    ///
    /// `zarrs` and other implementations are expected to error when opening an array with unsupported additional fields, unless they are a JSON object containing `"must_understand": false`.
    pub fn additional_fields(&mut self, additional_fields: AdditionalFieldsV3) -> &mut Self {
        self.additional_fields = additional_fields;
        self
    }

    /// Set the dimension names.
    ///
    /// If left unmodified, all dimension names are "unnamed".
    pub fn dimension_names<I, D>(&mut self, dimension_names: Option<I>) -> &mut Self
    where
        I: IntoIterator<Item = D>,
        D: IntoDimensionName,
    {
        if let Some(dimension_names) = dimension_names {
            self.dimension_names = Some(
                dimension_names
                    .into_iter()
                    .map(IntoDimensionName::into_dimension_name)
                    .collect(),
            );
        } else {
            self.dimension_names = None;
        }
        self
    }

    /// Set the storage transformers.
    ///
    /// If left unmodified, there are no storage transformers.
    pub fn storage_transformers(
        &mut self,
        storage_transformers: StorageTransformerChain,
    ) -> &mut Self {
        self.storage_transformers = storage_transformers;
        self
    }

    /// Get the metadata of an array that would be created with the current builder state.
    ///
    /// # Errors
    /// Returns an [`ArrayCreateError`] if this metadata is invalid/unsupported by `zarrs`.
    pub fn build_metadata(&self) -> Result<ArrayMetadataV3, ArrayCreateError> {
        let chunk_grid = match &self.chunk_grid {
            ArrayBuilderChunkGridMaybe::ChunkGrid(chunk_grid) => chunk_grid.as_chunk_grid().clone(),
            ArrayBuilderChunkGridMaybe::Metadata(array_shape, metadata) => {
                ChunkGrid::from_metadata(&metadata.to_metadata()?, array_shape)
                    .map_err(ArrayCreateError::ChunkGridCreateError)?
            }
        };
        let data_type = self.data_type.to_data_type()?;
        let fill_value = match &self.fill_value.0 {
            ArrayBuilderFillValueImpl::FillValue(fill_value) => fill_value.clone(),
            ArrayBuilderFillValueImpl::Metadata(fill_value_metadata) => {
                // ArrayBuilder is always for V3 arrays
                data_type.fill_value_v3(fill_value_metadata).map_err(|_| {
                    ArrayCreateError::InvalidFillValueMetadata {
                        data_type_name: data_type
                            .name_v3()
                            .map_or_else(String::new, Cow::into_owned),
                        fill_value_metadata: fill_value_metadata.clone(),
                    }
                })?
            }
        };
        if let Some(dimension_names) = &self.dimension_names
            && dimension_names.len() != chunk_grid.dimensionality()
        {
            return Err(ArrayCreateError::InvalidDimensionNames(
                dimension_names.len(),
                chunk_grid.dimensionality(),
            ));
        }

        let array_to_bytes_codec = self
            .array_to_bytes_codec
            .clone()
            .unwrap_or_else(|| super::codec::default_array_to_bytes_codec(&data_type));

        // If subchunk_shape is set, wrap the codec chain with a sharding codec
        #[cfg(feature = "sharding")]
        let codec_chain = if let Some(subchunk_shape) = &self.subchunk_shape {
            use super::codec::array_to_bytes::sharding::ShardingCodecBuilder;

            // Validate and convert ArrayShape to ChunkShape (all elements must be non-zero)
            let subchunk_shape: ChunkShape = subchunk_shape
                .iter()
                .copied()
                .map(NonZeroU64::try_from)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| ArrayCreateError::InvalidSubchunkShape(subchunk_shape.clone()))?;

            let mut sharding_builder = ShardingCodecBuilder::new(subchunk_shape, &data_type);
            sharding_builder
                .array_to_array_codecs(self.array_to_array_codecs.clone())
                .array_to_bytes_codec(array_to_bytes_codec.clone())
                .bytes_to_bytes_codecs(self.bytes_to_bytes_codecs.clone());

            CodecChain::new(vec![], Arc::new(sharding_builder.build()), vec![])
        } else {
            CodecChain::new(
                self.array_to_array_codecs.clone(),
                array_to_bytes_codec,
                self.bytes_to_bytes_codecs.clone(),
            )
        };

        #[cfg(not(feature = "sharding"))]
        let codec_chain = CodecChain::new(
            self.array_to_array_codecs.clone(),
            array_to_bytes_codec,
            self.bytes_to_bytes_codecs.clone(),
        );

        // Create data type metadata
        let data_type_name = data_type
            .name_v3()
            .map_or_else(String::new, Cow::into_owned);
        let data_type_configuration = data_type.configuration_v3();
        let data_type_metadata = if data_type_configuration.is_empty() {
            MetadataV3::new(data_type_name.clone())
        } else {
            MetadataV3::new_with_configuration(data_type_name.clone(), data_type_configuration)
        };

        Ok(ArrayMetadataV3::new(
            chunk_grid.array_shape().to_vec(),
            chunk_grid.metadata(),
            data_type_metadata,
            data_type.metadata_fill_value(&fill_value).map_err(|_| {
                ArrayCreateError::InvalidFillValue {
                    data_type_name,
                    fill_value,
                }
            })?,
            codec_chain.create_metadatas(self.metadata_options.codec_metadata_options()),
        )
        .with_attributes(self.attributes.clone())
        .with_additional_fields(self.additional_fields.clone())
        .with_chunk_key_encoding(self.chunk_key_encoding.metadata())
        .with_dimension_names(self.dimension_names.clone())
        .with_storage_transformers(self.storage_transformers.create_metadatas()))
    }

    /// Build into an [`Array`].
    ///
    /// # Errors
    ///
    /// Returns [`ArrayCreateError`] if there is an error creating the array.
    /// This can be due to a storage error, an invalid path, or a problem with array configuration.
    pub fn build<TStorage: ?Sized>(
        &self,
        storage: Arc<TStorage>,
        path: &str,
    ) -> Result<Array<TStorage>, ArrayCreateError> {
        let path: NodePath = path.try_into()?;
        let array_metadata = ArrayMetadata::V3(self.build_metadata()?);
        Ok(
            Array::new_with_metadata(storage, path.as_str(), array_metadata)?
                .with_metadata_options(self.metadata_options)
                .with_codec_options(self.codec_options),
        )
    }

    /// Build into an [`Arc<Array>`].
    ///
    /// # Errors
    ///
    /// Returns [`ArrayCreateError`] if there is an error creating the array.
    /// This can be due to a storage error, an invalid path, or a problem with array configuration.
    pub fn build_arc<TStorage: ?Sized>(
        &self,
        storage: Arc<TStorage>,
        path: &str,
    ) -> Result<Arc<Array<TStorage>>, ArrayCreateError> {
        Ok(Arc::new(self.build(storage, path)?))
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

    use zarrs_data_type::FillValue;

    use super::*;
    use crate::array::chunk_grid::RegularChunkGrid;
    use crate::array::chunk_key_encoding::V2ChunkKeyEncoding;
    use crate::array::{ChunkGrid, data_type};
    use zarrs_chunk_grid::ChunkGridTraits;
    use zarrs_metadata::FillValueMetadata;
    use zarrs_metadata::v3::MetadataV3;
    use zarrs_metadata_ext::chunk_grid::regular::RegularChunkGridConfiguration;
    use zarrs_storage::storage_adapter::usage_log::UsageLogStorageAdapter;
    use zarrs_storage::store::MemoryStore;

    #[test]
    fn array_builder() {
        let mut builder = ArrayBuilder::new(vec![8, 8], [2, 2], data_type::int8(), 0i8);

        // Coverage
        builder.shape(vec![8, 8]);
        builder.data_type(data_type::int8());
        // builder.chunk_grid(vec![2, 2].try_into().unwrap());
        builder.chunk_grid_metadata([2, 2]);
        builder.fill_value(0i8);

        builder.dimension_names(["y", "x"].into());

        let mut attributes = serde_json::Map::new();
        attributes.insert("key".to_string(), "value".into());
        builder.attributes(attributes.clone());

        let mut additional_fields = AdditionalFieldsV3::new();
        let additional_field = serde_json::Map::new();
        additional_fields.insert("key".to_string(), additional_field.into());
        builder.additional_fields(additional_fields.clone());

        builder.chunk_key_encoding(V2ChunkKeyEncoding::new_dot());
        builder.chunk_key_encoding_default_separator(ChunkKeySeparator::Dot); // overrides previous
        let log_writer = Arc::new(std::sync::Mutex::new(std::io::stdout()));

        let storage = Arc::new(MemoryStore::new());
        let storage = Arc::new(UsageLogStorageAdapter::new(storage, log_writer, || {
            chrono::Utc::now().format("[%T%.3f] ").to_string()
        }));
        println!("{:?}", builder.build(storage.clone(), "/"));
        let array = builder.build(storage, "/").unwrap();
        assert_eq!(array.shape(), &[8, 8]);
        assert_eq!(*array.data_type(), data_type::int8());
        assert_eq!(array.chunk_grid_shape(), &vec![4, 4]);
        assert_eq!(array.fill_value(), &FillValue::from(0i8));
        assert_eq!(
            array.dimension_names(),
            &Some(vec![Some("y".to_string()), Some("x".to_string())])
        );
        assert_eq!(array.attributes(), &attributes);
        if let ArrayMetadata::V3(metadata) = array.metadata() {
            assert_eq!(metadata.additional_fields, additional_fields);
        }

        let builder2 = array.builder();
        assert_eq!(builder.data_type, builder2.data_type);
        assert_eq!(builder.fill_value, builder2.fill_value);
        assert_eq!(builder.attributes, builder2.attributes);
        assert_eq!(builder.dimension_names, builder2.dimension_names);
        assert_eq!(builder.additional_fields, builder2.additional_fields);
    }

    #[test]
    fn array_builder_invalid() {
        let storage = Arc::new(MemoryStore::new());
        // Invalid chunk shape
        let builder = ArrayBuilder::new(vec![8, 8], vec![2, 2, 2], data_type::int8(), 0i8);
        assert!(builder.build(storage.clone(), "/").is_err());
        // Invalid fill value, but okay when interpreted as fill value metadata
        let builder = ArrayBuilder::new(vec![8, 8], vec![2, 2], data_type::int8(), 0i16);
        assert!(builder.build(storage.clone(), "/").is_ok());
        // Strictly invalid fill value
        let builder = ArrayBuilder::new(vec![8, 8], vec![2, 2], data_type::int8(), vec![0, 0]);
        assert!(builder.build(storage.clone(), "/").is_err());
        // Invalid dimension names
        let mut builder = ArrayBuilder::new(vec![8, 8], vec![2, 2], data_type::int8(), 0i8);
        builder.dimension_names(["z", "y", "x"].into());
        assert!(builder.build(storage.clone(), "/").is_err());
    }

    #[test]
    fn array_builder_invalid_fill_value_metadata_error() {
        let storage = Arc::new(MemoryStore::new());
        // Use a fill value metadata that is incompatible with the data type
        let builder = ArrayBuilder::new(vec![8, 8], vec![2, 2], data_type::string(), 123);
        let err = builder.build(storage, "/").unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid fill value metadata for data type `string`: 123"
        );
    }

    #[test]
    fn array_builder_variants_array_shape() {
        ArrayBuilder::new(vec![8, 8], vec![2, 2], data_type::int8(), 0i8)
            .build_metadata()
            .unwrap();
        ArrayBuilder::new([8, 8], vec![2, 2], data_type::int8(), 0i8)
            .build_metadata()
            .unwrap();
        ArrayBuilder::new([8, 8], vec![2, 2], data_type::int8(), 0i8)
            .build_metadata()
            .unwrap();
        ArrayBuilder::new([8, 8].as_slice(), vec![2, 2], data_type::int8(), 0i8)
            .build_metadata()
            .unwrap();
    }

    #[test]
    fn array_builder_variants_data_type() {
        ArrayBuilder::new(vec![8, 8], vec![2, 2], data_type::int8(), 0i8)
            .build_metadata()
            .unwrap();
        ArrayBuilder::new(vec![8, 8], vec![2, 2], "int8", 0i8)
            .build_metadata()
            .unwrap();
        ArrayBuilder::new(vec![8, 8], vec![2, 2], r#"{"name":"int8"}"#, 0i8)
            .build_metadata()
            .unwrap();
        ArrayBuilder::new(
            vec![8, 8],
            vec![2, 2],
            r#"{"name":"int8"}"#.to_string(),
            0i8,
        )
        .build_metadata()
        .unwrap();
        ArrayBuilder::new(
            vec![8, 8],
            vec![2, 2],
            r#"{"name":"int8", "configuration":{},"must_understand":true}"#,
            0i8,
        )
        .build_metadata()
        .unwrap();
        ArrayBuilder::new(vec![8, 8], vec![2, 2], MetadataV3::new("int8"), 0i8)
            .build_metadata()
            .unwrap();
    }

    #[test]
    fn array_builder_variants_chunk_grid() {
        assert!(
            ArrayBuilder::new(vec![8, 8], vec![0, 0], data_type::int8(), 0i8)
                .build_metadata()
                .is_err()
        );
        assert!(
            ArrayBuilder::new(vec![8, 8], "regular", data_type::int8(), 0i8)
                .build_metadata()
                .is_err()
        );
        assert!(
            ArrayBuilder::new(vec![8, 8], "{", data_type::int8(), 0i8)
                .build_metadata()
                .is_err()
        );
        ArrayBuilder::new(vec![8, 8], vec![2, 2], data_type::int8(), 0i8)
            .build_metadata()
            .unwrap();
        ArrayBuilder::new(vec![8, 8], [2, 2], data_type::int8(), 0i8)
            .build_metadata()
            .unwrap();
        ArrayBuilder::new(vec![8, 8], [2, 2], data_type::int8(), 0i8)
            .build_metadata()
            .unwrap();
        ArrayBuilder::new(vec![8, 8], [2, 2].as_slice(), data_type::int8(), 0i8)
            .build_metadata()
            .unwrap();
        let nz2 = NonZeroU64::new(2).unwrap();
        ArrayBuilder::new(vec![8, 8], vec![nz2, nz2], data_type::int8(), 0i8)
            .build_metadata()
            .unwrap();
        ArrayBuilder::new(vec![8, 8], [nz2, nz2], data_type::int8(), 0i8)
            .build_metadata()
            .unwrap();
        ArrayBuilder::new(vec![8, 8], [nz2, nz2], data_type::int8(), 0i8)
            .build_metadata()
            .unwrap();
        ArrayBuilder::new(vec![8, 8], [nz2, nz2].as_slice(), data_type::int8(), 0i8)
            .build_metadata()
            .unwrap();
        ArrayBuilder::new(
            vec![8, 8],
            r#"{"name":"regular","configuration":{"chunk_shape":[2,2]}}"#,
            data_type::int8(),
            0i8,
        )
        .build_metadata()
        .unwrap();
        ArrayBuilder::new(
            vec![8, 8],
            r#"{"name":"regular","configuration":{"chunk_shape":[2,2]}}"#.to_string(),
            data_type::int8(),
            0i8,
        )
        .build_metadata()
        .unwrap();
        ArrayBuilder::new(
            vec![8, 8],
            MetadataV3::new_with_configuration(
                "regular",
                RegularChunkGridConfiguration {
                    chunk_shape: vec![NonZeroU64::new(2).unwrap(); 2],
                },
            ),
            data_type::int8(),
            0i8,
        )
        .build_metadata()
        .unwrap();

        ArrayBuilder::new_with_chunk_grid(
            RegularChunkGrid::new(vec![8, 8], vec![NonZeroU64::new(2).unwrap(); 2]).unwrap(),
            data_type::int8(),
            0i8,
        )
        .build_metadata()
        .unwrap();
        let chunk_grid: Arc<dyn ChunkGridTraits> = Arc::new(
            RegularChunkGrid::new(vec![4, 4], vec![NonZeroU64::new(2).unwrap(); 2]).unwrap(),
        );
        ArrayBuilder::new_with_chunk_grid(chunk_grid, data_type::int8(), 0i8)
            .build_metadata()
            .unwrap();
        ArrayBuilder::new_with_chunk_grid(
            ChunkGrid::new(
                RegularChunkGrid::new(vec![8, 8], vec![NonZeroU64::new(2).unwrap(); 2]).unwrap(),
            ),
            data_type::int8(),
            0i8,
        )
        .build_metadata()
        .unwrap();
    }

    #[test]
    fn array_builder_variants_fill_value() {
        ArrayBuilder::new(vec![8, 8], vec![2, 2], data_type::int8(), 0i8)
            .build_metadata()
            .unwrap();
        ArrayBuilder::new(vec![8, 8], vec![2, 2], data_type::int8(), 0i16)
            .build_metadata()
            .unwrap(); // 0i16 -> 0 metadata -> 0i8
        ArrayBuilder::new(
            vec![8, 8],
            vec![2, 2],
            data_type::int8(),
            FillValue::new(vec![0u8]),
        )
        .build_metadata()
        .unwrap();
        ArrayBuilder::new(
            vec![8, 8],
            vec![2, 2],
            data_type::int8(),
            FillValue::from(0u8),
        )
        .build_metadata()
        .unwrap();
        ArrayBuilder::new(
            vec![8, 8],
            vec![2, 2],
            data_type::int8(),
            FillValueMetadata::Number(serde_json::Number::from(0u8)),
        )
        .build_metadata()
        .unwrap();
        ArrayBuilder::new(vec![8, 8], vec![2, 2], data_type::float32(), f32::NAN)
            .build_metadata()
            .unwrap();
        ArrayBuilder::new(vec![8, 8], vec![2, 2], data_type::float32(), "NaN")
            .build_metadata()
            .unwrap();
        ArrayBuilder::new(vec![8, 8], vec![2, 2], data_type::float32(), "Infinity")
            .build_metadata()
            .unwrap();
        ArrayBuilder::new(vec![8, 8], vec![2, 2], data_type::float32(), "-Infinity")
            .build_metadata()
            .unwrap();
        let ab = ArrayBuilder::new(vec![8, 8], vec![2, 2], data_type::float32(), "0x7fc00000");
        assert_eq!(
            ab.build_metadata().unwrap().fill_value,
            FillValueMetadata::from("NaN")
        );
        let ab = ArrayBuilder::new(
            vec![8, 8],
            vec![2, 2],
            data_type::float32(),
            f32::from_bits(0x7fc00001),
        ); // non-standard NaN
        assert_eq!(
            ab.build_metadata().unwrap().fill_value,
            FillValueMetadata::from("0x7fc00001")
        );
    }
}
