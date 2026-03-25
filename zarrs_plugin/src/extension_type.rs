use std::fmt::Debug;

/// Marker trait for extension types.
pub trait ExtensionType: Debug + Default {}

/// The *data type* extension type.
#[derive(Debug, Copy, Clone, Default)]
pub struct ExtensionTypeDataType;

/// The *chunk grid* extension type.
#[derive(Debug, Copy, Clone, Default)]
pub struct ExtensionTypeChunkGrid;

/// The *chunk key encoding* extension type.
#[derive(Debug, Copy, Clone, Default)]
pub struct ExtensionTypeChunkKeyEncoding;

/// The *codec* extension type.
#[derive(Debug, Copy, Clone, Default)]
pub struct ExtensionTypeCodec;

/// The *storage transformer* extension type.
#[derive(Debug, Copy, Clone, Default)]
pub struct ExtensionTypeStorageTransformer;

impl ExtensionType for ExtensionTypeDataType {}
impl ExtensionType for ExtensionTypeChunkGrid {}
impl ExtensionType for ExtensionTypeChunkKeyEncoding {}
impl ExtensionType for ExtensionTypeCodec {}
impl ExtensionType for ExtensionTypeStorageTransformer {}
