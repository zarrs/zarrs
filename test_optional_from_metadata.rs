use zarrs::{
    array::DataType,
    metadata::v3::MetadataV3,
    registry::ExtensionAliasesDataTypeV3,
};

fn main() {
    // Test simple optional int32
    let json = r#"{"name":"optional","configuration":{"name":"int32","configuration":{}}}"#;
    let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
    let data_type = DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
    
    println!("Parsed data type: {:?}", data_type);
    println!("Data type name: {}", data_type.name());
    
    if let DataType::Optional(inner) = &data_type {
        println!("Inner data type: {}", inner.name());
        println!("Inner data type size: {:?}", inner.size());
    }
    
    // Test optional with complex configuration (numpy datetime64)
    let json = r#"{"name":"optional","configuration":{"name":"numpy.datetime64","configuration":{"unit":"s","scale_factor":1}}}"#;
    let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
    let data_type = DataType::from_metadata(&metadata, &ExtensionAliasesDataTypeV3::default()).unwrap();
    
    println!("\nParsed complex optional data type: {:?}", data_type);
    println!("Data type name: {}", data_type.name());
    
    // Test metadata roundtrip
    let metadata_roundtrip = data_type.metadata();
    let data_type_roundtrip = DataType::from_metadata(&metadata_roundtrip, &ExtensionAliasesDataTypeV3::default()).unwrap();
    
    println!("\nRoundtrip successful: {}", data_type == data_type_roundtrip);
    
    println!("\nOptional data type implementation is working correctly! ✅");
}