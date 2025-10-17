//! Test the Element and ElementOwned implementation for Option<T>

use zarrs::array::{ArrayBytes, DataType, Element, ElementOwned};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Test Option<i32>
    let data_type = DataType::Optional(Box::new(DataType::Int32));
    
    // Test data with Some and None values
    let elements = vec![Some(42i32), None, Some(-123i32), None, Some(0i32)];
    
    println!("Original elements: {:?}", elements);
    
    // Convert to ArrayBytes
    let array_bytes = Option::<i32>::into_array_bytes(&data_type, &elements)?;
    println!("Array bytes created successfully");
    
    // Convert back to elements
    let recovered_elements = Option::<i32>::from_array_bytes(&data_type, array_bytes)?;
    
    println!("Recovered elements: {:?}", recovered_elements);
    
    // Verify they match
    assert_eq!(elements, recovered_elements);
    println!("✓ Round-trip successful!");
    
    // Test Option<u8>
    let data_type_u8 = DataType::Optional(Box::new(DataType::UInt8));
    let elements_u8 = vec![Some(255u8), None, Some(0u8), Some(128u8)];
    
    println!("\nTesting Option<u8>:");
    println!("Original elements: {:?}", elements_u8);
    
    let array_bytes_u8 = Option::<u8>::into_array_bytes(&data_type_u8, &elements_u8)?;
    let recovered_elements_u8 = Option::<u8>::from_array_bytes(&data_type_u8, array_bytes_u8)?;
    
    println!("Recovered elements: {:?}", recovered_elements_u8);
    assert_eq!(elements_u8, recovered_elements_u8);
    println!("✓ Round-trip successful!");
    
    Ok(())
}