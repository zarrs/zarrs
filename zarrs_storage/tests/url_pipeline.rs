//! Integration tests for URL pipeline functionality.

use zarrs_storage::url_pipeline::{parse_and_create, parse_url_pipeline};
use zarrs_storage::ReadableStorageTraits;

#[test]
fn test_parse_file_url() {
    let pipeline = parse_url_pipeline("file:///tmp/data").unwrap();
    assert_eq!(pipeline.root.scheme, "file");
    assert_eq!(pipeline.root.path, "/tmp/data");
    assert!(pipeline.adapters.is_empty());
}

#[test]
fn test_parse_http_url() {
    let pipeline = parse_url_pipeline("http://example.com/data").unwrap();
    assert_eq!(pipeline.root.scheme, "http");
    assert_eq!(pipeline.root.path, "example.com/data");
    assert!(pipeline.adapters.is_empty());
}

#[test]
fn test_parse_https_url() {
    let pipeline = parse_url_pipeline("https://example.com/data").unwrap();
    assert_eq!(pipeline.root.scheme, "https");
    assert_eq!(pipeline.root.path, "example.com/data");
    assert!(pipeline.adapters.is_empty());
}

#[test]
fn test_parse_s3_url() {
    let pipeline = parse_url_pipeline("s3://bucket/path/to/data").unwrap();
    assert_eq!(pipeline.root.scheme, "s3");
    assert_eq!(pipeline.root.path, "bucket/path/to/data");
    assert!(pipeline.adapters.is_empty());
}

#[test]
fn test_parse_gs_url() {
    let pipeline = parse_url_pipeline("gs://bucket/path/to/data").unwrap();
    assert_eq!(pipeline.root.scheme, "gs");
    assert_eq!(pipeline.root.path, "bucket/path/to/data");
    assert!(pipeline.adapters.is_empty());
}

#[test]
fn test_parse_file_zip_pipeline() {
    let pipeline = parse_url_pipeline("file:///tmp/data.zip|zip:path/inside").unwrap();
    assert_eq!(pipeline.root.scheme, "file");
    assert_eq!(pipeline.root.path, "/tmp/data.zip");
    assert_eq!(pipeline.adapters.len(), 1);
    assert_eq!(pipeline.adapters[0].scheme, "zip");
    assert_eq!(pipeline.adapters[0].path, "path/inside");
}

#[test]
fn test_parse_http_zip_pipeline() {
    let pipeline = parse_url_pipeline("http://example.com/data.zip|zip:").unwrap();
    assert_eq!(pipeline.root.scheme, "http");
    assert_eq!(pipeline.adapters.len(), 1);
    assert_eq!(pipeline.adapters[0].scheme, "zip");
    assert_eq!(pipeline.adapters[0].path, "");
}

#[test]
fn test_parse_nested_zip_pipeline() {
    let pipeline = parse_url_pipeline("file:///tmp/outer.zip|zip:inner.zip|zip:data").unwrap();
    assert_eq!(pipeline.root.scheme, "file");
    assert_eq!(pipeline.adapters.len(), 2);
    assert_eq!(pipeline.adapters[0].scheme, "zip");
    assert_eq!(pipeline.adapters[0].path, "inner.zip");
    assert_eq!(pipeline.adapters[1].scheme, "zip");
    assert_eq!(pipeline.adapters[1].path, "data");
}

#[test]
fn test_parse_s3_zip_pipeline() {
    let pipeline = parse_url_pipeline("s3://bucket/data.zip|zip:array").unwrap();
    assert_eq!(pipeline.root.scheme, "s3");
    assert_eq!(pipeline.root.path, "bucket/data.zip");
    assert_eq!(pipeline.adapters.len(), 1);
    assert_eq!(pipeline.adapters[0].scheme, "zip");
    assert_eq!(pipeline.adapters[0].path, "array");
}

#[test]
fn test_parse_with_query_params() {
    let pipeline = parse_url_pipeline("file:///tmp/data?readonly=true").unwrap();
    assert!(pipeline.root.query.contains_key("readonly"));
    assert_eq!(
        pipeline.root.query.get("readonly"),
        Some(&"true".to_string())
    );
}

#[test]
fn test_parse_adapter_with_query() {
    let pipeline = parse_url_pipeline("file:///tmp/data.zip|zip:path?extract=all").unwrap();
    assert!(pipeline.adapters[0].query.contains_key("extract"));
}

#[test]
fn test_parse_with_fragment() {
    let pipeline = parse_url_pipeline("file:///tmp/data#metadata").unwrap();
    assert_eq!(pipeline.root.fragment, Some("metadata".to_string()));
}

#[test]
fn test_parse_empty_pipeline_fails() {
    let result = parse_url_pipeline("");
    assert!(result.is_err());
}

#[test]
fn test_parse_invalid_root_scheme_still_parses() {
    // With the flexible parser, unknown schemes are allowed
    // They will fail later when trying to create a store if not registered
    let result = parse_url_pipeline("invalid://path");
    assert!(result.is_ok());
    assert_eq!(result.unwrap().root.scheme, "invalid");
}

#[test]
fn test_parse_memory_url() {
    let pipeline = parse_url_pipeline("memory://").unwrap();
    assert_eq!(pipeline.root.scheme, "memory");
}

// Store creation tests
#[test]
fn test_create_memory_store() {
    let store = parse_and_create("memory://").unwrap();
    assert!(store.get(&"test".try_into().unwrap()).is_ok());
}

#[test]
fn test_create_unsupported_scheme_fails() {
    let result = parse_and_create("unsupported://path");
    assert!(result.is_err());
}

// Test URL encoding
#[test]
fn test_parse_url_with_spaces() {
    let pipeline = parse_url_pipeline("file:///tmp/my%20data").unwrap();
    assert_eq!(pipeline.root.path, "/tmp/my%20data");
}

#[test]
fn test_parse_url_with_special_chars() {
    let pipeline = parse_url_pipeline("file:///tmp/data%2Bfile").unwrap();
    assert_eq!(pipeline.root.path, "/tmp/data%2Bfile");
}

// Complex pipeline tests
#[test]
fn test_complex_pipeline_parsing() {
    let url = "s3://my-bucket/path/to/outer.zip|zip:middle.zip|zip:inner/data";
    let pipeline = parse_url_pipeline(url).unwrap();

    assert_eq!(pipeline.root.scheme, "s3");
    assert_eq!(pipeline.root.path, "my-bucket/path/to/outer.zip");
    assert_eq!(pipeline.adapters.len(), 2);
    assert_eq!(pipeline.adapters[0].scheme, "zip");
    assert_eq!(pipeline.adapters[0].path, "middle.zip");
    assert_eq!(pipeline.adapters[1].scheme, "zip");
    assert_eq!(pipeline.adapters[1].path, "inner/data");
}

#[test]
fn test_pipeline_with_multiple_query_params() {
    let pipeline = parse_url_pipeline("file:///tmp/data?param1=value1&param2=value2").unwrap();
    assert_eq!(
        pipeline.root.query.get("param1"),
        Some(&"value1".to_string())
    );
    assert_eq!(
        pipeline.root.query.get("param2"),
        Some(&"value2".to_string())
    );
}

// Windows-specific path tests (conditional compilation)
#[cfg(target_os = "windows")]
#[test]
fn test_parse_windows_file_url() {
    let pipeline = parse_url_pipeline("file:///C:/Users/test/data").unwrap();
    assert_eq!(pipeline.root.path, "C:/Users/test/data");
}

// Test various adapter schemes
#[test]
fn test_parse_zarr3_adapter() {
    let pipeline = parse_url_pipeline("file:///tmp/data|zarr3:array").unwrap();
    assert_eq!(pipeline.adapters[0].scheme, "zarr3");
    assert_eq!(pipeline.adapters[0].path, "array");
}

#[test]
fn test_parse_zarr2_adapter() {
    let pipeline = parse_url_pipeline("file:///tmp/data|zarr2:array").unwrap();
    assert_eq!(pipeline.adapters[0].scheme, "zarr2");
    assert_eq!(pipeline.adapters[0].path, "array");
}

#[test]
fn test_parse_gzip_adapter() {
    let pipeline = parse_url_pipeline("file:///tmp/data.gz|gzip:").unwrap();
    assert_eq!(pipeline.adapters[0].scheme, "gzip");
}

#[test]
fn test_parse_zstd_adapter() {
    let pipeline = parse_url_pipeline("file:///tmp/data.zst|zstd:").unwrap();
    assert_eq!(pipeline.adapters[0].scheme, "zstd");
}

// Edge cases
#[test]
fn test_parse_url_with_empty_path() {
    let pipeline = parse_url_pipeline("file:///").unwrap();
    assert_eq!(pipeline.root.path, "/");
}

#[test]
fn test_parse_url_with_port() {
    let pipeline = parse_url_pipeline("http://example.com:8080/data").unwrap();
    assert!(pipeline.root.path.contains("example.com"));
}

#[test]
fn test_parse_adapter_without_colon_fails() {
    let result = parse_url_pipeline("file:///tmp/data|zip");
    assert!(result.is_err());
}

#[test]
fn test_parse_very_long_pipeline() {
    let url = "file:///tmp/a.zip|zip:b.zip|zip:c.zip|zip:d.zip|zip:e.zip";
    let pipeline = parse_url_pipeline(url).unwrap();
    assert_eq!(pipeline.adapters.len(), 4);
    assert!(pipeline.adapters.iter().all(|a| a.scheme == "zip"));
}

// Registry tests
#[test]
fn test_registry_singleton() {
    use zarrs_storage::url_pipeline::get_global_registry;
    let reg1 = get_global_registry();
    let reg2 = get_global_registry();
    assert!(std::ptr::eq(reg1, reg2));
}

#[test]
fn test_register_custom_root_store() {
    use std::sync::Arc;
    use zarrs_storage::store::MemoryStore;
    use zarrs_storage::url_pipeline::register_root_store;

    register_root_store("test", |_component| Ok(Arc::new(MemoryStore::new())));

    let store = parse_and_create("test://").unwrap();
    assert!(store.get(&"test".try_into().unwrap()).is_ok());
}

#[test]
fn test_url_normalization() {
    // Test that different representations of the same URL are handled consistently
    let pipeline1 = parse_url_pipeline("file:///tmp/data").unwrap();
    let pipeline2 = parse_url_pipeline("file:///tmp/data/").unwrap();

    assert_eq!(pipeline1.root.scheme, pipeline2.root.scheme);
    // Paths may differ due to trailing slash, which is expected
}
