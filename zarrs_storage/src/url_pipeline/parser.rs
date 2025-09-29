//! URL pipeline parser.

use super::UrlPipelineError;
use std::collections::HashMap;

/// A component of a URL pipeline (root URL or adapter).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UrlComponent {
    /// The scheme (e.g., "file", "http", "zip", "zarr3")
    pub scheme: String,
    /// The path component
    pub path: String,
    /// Query parameters
    pub query: HashMap<String, String>,
    /// Fragment parameters
    pub fragment: Option<String>,
}

impl UrlComponent {
    /// Create a new URL component.
    #[must_use]
    pub fn new(scheme: String, path: String) -> Self {
        Self {
            scheme,
            path,
            query: HashMap::new(),
            fragment: None,
        }
    }

    /// Add a query parameter.
    #[must_use]
    pub fn with_query(mut self, key: String, value: String) -> Self {
        self.query.insert(key, value);
        self
    }

    /// Add a fragment.
    #[must_use]
    pub fn with_fragment(mut self, fragment: String) -> Self {
        self.fragment = Some(fragment);
        self
    }
}

/// A parsed URL pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UrlPipeline {
    /// The root URL component (file://, http://, s3://, etc.)
    pub root: UrlComponent,
    /// Adapter components (zip:, zarr3:, etc.) in order
    pub adapters: Vec<UrlComponent>,
}

impl UrlPipeline {
    /// Create a new URL pipeline.
    #[must_use]
    pub fn new(root: UrlComponent) -> Self {
        Self {
            root,
            adapters: Vec::new(),
        }
    }

    /// Add an adapter to the pipeline.
    #[must_use]
    pub fn with_adapter(mut self, adapter: UrlComponent) -> Self {
        self.adapters.push(adapter);
        self
    }
}

/// Parse a URL pipeline string.
///
/// # Errors
///
/// Returns an error if the URL is invalid or the pipeline structure is malformed.
pub fn parse_url_pipeline(url: &str) -> Result<UrlPipeline, UrlPipelineError> {
    let parts: Vec<&str> = url.split('|').collect();

    if parts.is_empty() {
        return Err(UrlPipelineError::InvalidPipeline(
            "empty URL pipeline".to_string(),
        ));
    }

    // Parse root URL
    let root = parse_component(parts[0], true)?;
    let mut pipeline = UrlPipeline::new(root);

    // Parse adapters
    for part in &parts[1..] {
        let adapter = parse_component(part, false)?;
        pipeline = pipeline.with_adapter(adapter);
    }

    Ok(pipeline)
}

/// Parse a single URL component.
fn parse_component(s: &str, is_root: bool) -> Result<UrlComponent, UrlPipelineError> {
    // Handle different URL formats
    if is_root {
        parse_root_url(s)
    } else {
        parse_adapter_url(s)
    }
}

/// Parse a root URL (file://, http://, s3://, gs://, etc.).
fn parse_root_url(s: &str) -> Result<UrlComponent, UrlPipelineError> {
    let url = url::Url::parse(s).map_err(|e| {
        UrlPipelineError::InvalidUrl(format!("failed to parse root URL '{s}': {e}"))
    })?;

    let scheme = url.scheme().to_string();
    let mut path = String::new();

    // Build path based on scheme
    match scheme.as_str() {
        "file" => {
            // For file URLs, use the path
            path = url.path().to_string();
            // On Windows, handle the leading slash
            #[cfg(windows)]
            if path.starts_with('/') && path.len() > 2 && path.chars().nth(2) == Some(':') {
                path = path[1..].to_string();
            }
        }
        "http" | "https" => {
            // For HTTP, reconstruct the full URL without the scheme
            if let Some(host) = url.host_str() {
                path = format!("{}{}", host, url.path());
                if let Some(query) = url.query() {
                    path.push('?');
                    path.push_str(query);
                }
            }
        }
        "s3" | "s3+http" | "s3+https" | "gs" | "az" => {
            // For cloud storage, format is scheme://bucket/path
            if let Some(host) = url.host_str() {
                path = format!("{}{}", host, url.path());
            }
        }
        "memory" => {
            // For memory store, path is just the host (if any)
            if let Some(host) = url.host_str() {
                path = host.to_string();
            } else {
                path = String::new();
            }
        }
        _ => {
            // For other schemes, allow them through - they may be registered by external stores
            // Use host + path if available
            if let Some(host) = url.host_str() {
                path = format!("{}{}", host, url.path());
            } else {
                path = url.path().to_string();
            }
        }
    }

    let mut component = UrlComponent::new(scheme, path);

    // Parse query parameters
    if let Some(query) = url.query() {
        for pair in query.split('&') {
            if let Some((key, value)) = pair.split_once('=') {
                component.query.insert(
                    urlencoding::decode(key)
                        .map_err(|e| UrlPipelineError::InvalidUrl(e.to_string()))?
                        .to_string(),
                    urlencoding::decode(value)
                        .map_err(|e| UrlPipelineError::InvalidUrl(e.to_string()))?
                        .to_string(),
                );
            }
        }
    }

    // Parse fragment
    if let Some(fragment) = url.fragment() {
        component.fragment = Some(fragment.to_string());
    }

    Ok(component)
}

/// Parse an adapter URL (zip:, zarr3:, zarr2:, gzip:, zstd:, etc.).
fn parse_adapter_url(s: &str) -> Result<UrlComponent, UrlPipelineError> {
    // Adapter format: scheme:path or scheme:path?query#fragment
    let (scheme_and_path, fragment) = if let Some((main, frag)) = s.split_once('#') {
        (main, Some(frag.to_string()))
    } else {
        (s, None)
    };

    let (scheme_and_path, query_string) =
        if let Some((main, query)) = scheme_and_path.split_once('?') {
            (main, Some(query))
        } else {
            (scheme_and_path, None)
        };

    let Some((scheme, path)) = scheme_and_path.split_once(':') else {
        return Err(UrlPipelineError::InvalidUrl(format!(
            "adapter URL must have scheme: '{s}'"
        )));
    };

    let mut component = UrlComponent::new(scheme.to_string(), path.to_string());

    // Parse query parameters
    if let Some(query) = query_string {
        for pair in query.split('&') {
            if let Some((key, value)) = pair.split_once('=') {
                component.query.insert(
                    urlencoding::decode(key)
                        .map_err(|e| UrlPipelineError::InvalidUrl(e.to_string()))?
                        .to_string(),
                    urlencoding::decode(value)
                        .map_err(|e| UrlPipelineError::InvalidUrl(e.to_string()))?
                        .to_string(),
                );
            }
        }
    }

    component.fragment = fragment;

    Ok(component)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_file_url() {
        let pipeline = parse_url_pipeline("file:///path/to/data").unwrap();
        assert_eq!(pipeline.root.scheme, "file");
        assert_eq!(pipeline.root.path, "/path/to/data");
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
    fn test_parse_s3_url() {
        let pipeline = parse_url_pipeline("s3://bucket/path/to/data").unwrap();
        assert_eq!(pipeline.root.scheme, "s3");
        assert_eq!(pipeline.root.path, "bucket/path/to/data");
        assert!(pipeline.adapters.is_empty());
    }

    #[test]
    fn test_parse_pipeline_with_zip() {
        let pipeline = parse_url_pipeline("file:///path/to/data.zip|zip:inner/path").unwrap();
        assert_eq!(pipeline.root.scheme, "file");
        assert_eq!(pipeline.root.path, "/path/to/data.zip");
        assert_eq!(pipeline.adapters.len(), 1);
        assert_eq!(pipeline.adapters[0].scheme, "zip");
        assert_eq!(pipeline.adapters[0].path, "inner/path");
    }

    #[test]
    fn test_parse_nested_pipeline() {
        let pipeline =
            parse_url_pipeline("s3://bucket/outer.zip|zip:inner.zip|zip:data|zarr3:array").unwrap();
        assert_eq!(pipeline.root.scheme, "s3");
        assert_eq!(pipeline.adapters.len(), 3);
        assert_eq!(pipeline.adapters[0].scheme, "zip");
        assert_eq!(pipeline.adapters[1].scheme, "zip");
        assert_eq!(pipeline.adapters[2].scheme, "zarr3");
    }

    #[test]
    fn test_parse_with_query() {
        let pipeline = parse_url_pipeline("file:///data.zip?mode=r|zip:path").unwrap();
        assert!(pipeline.root.query.contains_key("mode"));
        assert_eq!(pipeline.root.query.get("mode"), Some(&"r".to_string()));
    }

    #[test]
    fn test_parse_adapter_with_empty_path() {
        let pipeline = parse_url_pipeline("file:///data.zip|zip:").unwrap();
        assert_eq!(pipeline.adapters[0].path, "");
    }
}
