#![allow(missing_docs)]
#![cfg(feature = "url-pipeline")]

// Referencing this type ensures `zarrs_object_store` is actually linked into this test binary.
// `inventory`-based self-registration relies on the registering crate's rlib being linked in; if
// nothing in the binary names any of its items, rustc/the linker may drop it entirely and the
// scheme registrations in `url_pipeline_support.rs` would silently never run.
#[allow(dead_code)]
fn _ensure_linked(_store: zarrs_object_store::AsyncObjectStore<object_store::memory::InMemory>) {}

#[tokio::test]
async fn unregistered_scheme_still_errors() {
    let result = zarrs_url_pipeline::try_resolve_async_readable("does-not-exist://foo").await;
    assert!(result.is_err());
}

#[tokio::test]
#[cfg(feature = "aws")]
async fn s3_scheme_is_registered_but_errors_without_a_bucket() {
    // No host/bucket is present, so `object_store::parse_url_opts` fails while parsing the
    // scheme-specific part, rather than with "no plugin registered for s3" — proving the `s3:`
    // scheme is claimed by `zarrs_object_store`'s plugin (match-time success) when the `aws`
    // feature is enabled, with the failure coming from a later stage (create-time), not scheme
    // lookup.
    let Err(err) = zarrs_url_pipeline::try_resolve_async_readable("s3://").await else {
        panic!("expected an error resolving s3://");
    };
    let err = err.to_string();
    assert!(
        !err.to_lowercase().contains("no plugin"),
        "expected a parse_url_opts-level error, got: {err}"
    );
}

#[tokio::test]
#[cfg(not(feature = "aws"))]
async fn s3_scheme_is_unregistered_without_the_aws_feature() {
    // With the `aws` feature disabled, `is_object_store_scheme` must not claim `s3:` at all, so
    // resolution fails at scheme lookup (`PluginUnsupportedError`, "... is not supported") rather
    // than reaching `object_store::parse_url_opts`.
    let Err(err) = zarrs_url_pipeline::try_resolve_async_readable("s3://").await else {
        panic!("expected an error resolving s3://");
    };
    assert!(
        err.to_string().to_lowercase().contains("not supported"),
        "expected a scheme-lookup error, got: {err}"
    );
}

#[tokio::test]
async fn http_scheme_resolves_and_fails_to_connect_on_first_request() {
    // `HttpBuilder::build()` performs no I/O, so resolving the pipeline stage itself always
    // succeeds for a syntactically valid `http:` URL; the connection is only attempted on the
    // first actual request. Driving a real `get()` against a port immediately closed after
    // binding guarantees a fast `ECONNREFUSED` (rather than a slow connection timeout), proving
    // the full pipeline -> object_store::parse_url_opts -> HttpBuilder -> request path is wired
    // up end-to-end.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);

    let storage = zarrs_url_pipeline::try_resolve_async_readable(&format!(
        "http://127.0.0.1:{port}/nonexistent-path"
    ))
    .await
    .expect("resolving the stage performs no I/O and should succeed");
    let key = zarrs_storage::StoreKey::new("a/b").unwrap();
    assert!(storage.get(&key).await.is_err());
}
