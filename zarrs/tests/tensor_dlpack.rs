#![allow(missing_docs)]
#![cfg(feature = "dlpack")]

use dlpark::ffi::{DLDataType, DLDataTypeCode};
use dlpark::{DlpackFlags, versioned};
use zarrs::array::{ArrayBuilder, ArraySubset, Tensor, TensorError, data_type};
use zarrs_storage::store::MemoryStore;

fn test_tensor() -> Tensor<'static> {
    let store = MemoryStore::new();
    let array = ArrayBuilder::new(vec![4, 4], vec![2, 2], data_type::float32(), -1.0f32)
        .build(store.into(), "/")
        .unwrap();
    array
        .store_chunk(&[0, 0], &[0.0f32, 1.0, 2.0, 3.0])
        .unwrap();
    array
        .retrieve_chunks(&ArraySubset::new_with_shape(vec![1, 2]))
        .unwrap()
}

#[test]
fn tensor_into_static_shared_does_not_copy() {
    let bytes = vec![0u8; 4 * size_of::<f32>()];
    let ptr = bytes.as_ptr();
    let tensor = Tensor::new(bytes, data_type::float32(), vec![2, 2]);
    // A `Vec` is adopted as shared bytes, so `into_static` retains the allocation
    assert_eq!(tensor.into_static().bytes().as_ptr(), ptr);
}

#[test]
fn tensor_borrowed_into_static_dlpack() {
    let elements = [1.0f32, 2.0, 3.0, 4.0];
    let bytes: Vec<u8> = elements.iter().flat_map(|f| f.to_ne_bytes()).collect();
    let tensor = Tensor::new(&bytes[..], data_type::float32(), vec![2, 2]);
    let dlpack = tensor.into_static().into_dlpack().unwrap();

    // `into_static` copied the borrowed bytes, so the export outlives them
    drop(bytes);
    assert_eq!(dlpack.shape().unwrap(), &[2, 2]);
    assert_eq!(dlpack.cpu_data_slice::<f32>().unwrap(), &elements);
}

#[test]
fn tensor_dlpack_versioned() {
    let dlpack: versioned::Dlpack = test_tensor().into_dlpack().unwrap();

    assert_eq!(dlpack.shape().unwrap(), &[2, 4]);
    assert_eq!(dlpack.strides().unwrap().unwrap(), &[4, 1]);
    assert_eq!(dlpack.num_bytes().unwrap(), 8 * size_of::<f32>());
    assert_eq!(
        dlpack.cpu_data_slice::<f32>().unwrap(),
        &[0.0f32, 1.0, -1.0, -1.0, 2.0, 3.0, -1.0, -1.0]
    );
}

#[test]
fn tensor_dlpack_data_types() {
    // (data type, expected code, expected bits)
    let data_types = [
        (data_type::bool(), DLDataTypeCode::BOOL, 8),
        (data_type::int2(), DLDataTypeCode::INT, 2),
        (data_type::int4(), DLDataTypeCode::INT, 4),
        (data_type::int8(), DLDataTypeCode::INT, 8),
        (data_type::int16(), DLDataTypeCode::INT, 16),
        (data_type::int32(), DLDataTypeCode::INT, 32),
        (data_type::int64(), DLDataTypeCode::INT, 64),
        (data_type::uint2(), DLDataTypeCode::UINT, 2),
        (data_type::uint4(), DLDataTypeCode::UINT, 4),
        (data_type::uint8(), DLDataTypeCode::UINT, 8),
        (data_type::uint16(), DLDataTypeCode::UINT, 16),
        (data_type::uint32(), DLDataTypeCode::UINT, 32),
        (data_type::uint64(), DLDataTypeCode::UINT, 64),
        (data_type::float16(), DLDataTypeCode::FLOAT, 16),
        (data_type::float32(), DLDataTypeCode::FLOAT, 32),
        (data_type::float64(), DLDataTypeCode::FLOAT, 64),
        (data_type::bfloat16(), DLDataTypeCode::BFLOAT, 16),
        (data_type::float8_e3m4(), DLDataTypeCode::FLOAT8_E3M4, 8),
        (data_type::float8_e4m3(), DLDataTypeCode::FLOAT8_E4M3, 8),
        (
            data_type::float8_e4m3b11fnuz(),
            DLDataTypeCode::FLOAT8_E4M3B11FNUZ,
            8,
        ),
        (
            data_type::float8_e4m3fnuz(),
            DLDataTypeCode::FLOAT8_E4M3FNUZ,
            8,
        ),
        (data_type::float8_e5m2(), DLDataTypeCode::FLOAT8_E5M2, 8),
        (
            data_type::float8_e5m2fnuz(),
            DLDataTypeCode::FLOAT8_E5M2FNUZ,
            8,
        ),
        (
            data_type::float8_e8m0fnu(),
            DLDataTypeCode::FLOAT8_E8M0FNU,
            8,
        ),
        (data_type::float6_e2m3fn(), DLDataTypeCode::FLOAT6_E2M3FN, 6),
        (data_type::float6_e3m2fn(), DLDataTypeCode::FLOAT6_E3M2FN, 6),
        (data_type::float4_e2m1fn(), DLDataTypeCode::FLOAT4_E2M1FN, 4),
        (data_type::complex_float16(), DLDataTypeCode::COMPLEX, 32),
        (data_type::complex64(), DLDataTypeCode::COMPLEX, 64),
        (data_type::complex_float32(), DLDataTypeCode::COMPLEX, 64),
        (data_type::complex128(), DLDataTypeCode::COMPLEX, 128),
        (data_type::complex_float64(), DLDataTypeCode::COMPLEX, 128),
    ];

    for (data_type, code, bits) in data_types {
        let element_size = data_type.fixed_size().unwrap();
        let tensor = Tensor::new(vec![0u8; element_size], data_type.clone(), vec![1]);
        let dlpack = tensor
            .into_dlpack()
            .unwrap_or_else(|err| panic!("{data_type} should be supported: {err}"));
        let dtype = dlpack.tensor().dtype;
        assert!(
            dtype.matches(DLDataType::scalar(code, bits)),
            "{data_type} mapped to {dtype:?}"
        );
        // The DLPack element size must match the zarrs in-memory element size
        assert_eq!(data_type.fixed_size(), Some(dtype.element_size()));
    }
}

#[test]
fn tensor_dlpack_subbyte_flag() {
    // `zarrs` pads sub-byte elements to a byte, so the padded flag must be set
    let tensor = Tensor::new(vec![0u8; 4], data_type::float4_e2m1fn(), vec![4]);
    let dlpack: versioned::Dlpack = tensor.into_dlpack().unwrap();
    assert_eq!(dlpack.flags(), DlpackFlags::IS_SUBBYTE_TYPE_PADDED);

    // Byte-sized data types are neither packed nor padded
    let tensor = Tensor::new(vec![0u8; 4], data_type::float8_e4m3(), vec![4]);
    let dlpack: versioned::Dlpack = tensor.into_dlpack().unwrap();
    assert_eq!(dlpack.flags(), DlpackFlags::empty());
}

#[test]
fn tensor_dlpack_unsupported_data_type() {
    // `complex_bfloat16` needs a data type code that postdates DLPack 1.3
    let tensor = Tensor::new(vec![0u8; 4], data_type::complex_bfloat16(), vec![1]);
    assert!(tensor.into_dlpack().is_err());

    // Complex subfloats have no DLPack data type code
    let tensor = Tensor::new(vec![0u8; 2], data_type::complex_float8_e4m3(), vec![1]);
    assert!(tensor.into_dlpack().is_err());

    // Variable-sized data types cannot be represented
    let tensor = Tensor::new(vec![0u8; 8], data_type::string(), vec![1]);
    assert!(tensor.into_dlpack().is_err());
}

#[test]
fn tensor_dlpack_unsupported_shape() {
    // A dimension that does not fit in an i64
    let tensor = Tensor::new(vec![], data_type::uint8(), vec![u64::MAX]);
    assert!(tensor.into_dlpack().is_err());

    // A shape whose compact strides overflow an i64
    let tensor = Tensor::new(vec![], data_type::uint8(), vec![2, i64::MAX as u64]);
    assert!(tensor.into_dlpack().is_err());
}

#[test]
fn tensor_dlpack_insufficient_bytes() {
    // The bytes must cover the shape, otherwise the managed tensor is out of bounds
    let tensor = Tensor::new(vec![0u8; 4], data_type::float32(), vec![100]);
    assert!(matches!(
        tensor.into_dlpack(),
        Err(TensorError::InsufficientBytes {
            expected: 400,
            actual: 4
        })
    ));

    // Exactly enough bytes is accepted
    let tensor = Tensor::new(vec![0u8; 400], data_type::float32(), vec![100]);
    assert!(tensor.into_dlpack().is_ok());

    // A zero element tensor needs no bytes
    let tensor = Tensor::new(vec![], data_type::float32(), vec![0]);
    assert!(tensor.into_dlpack().is_ok());
}
