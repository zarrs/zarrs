//! Parsing of `url-pipeline` style pipeline strings.
//!
//! A pipeline is a `root-sub-url` optionally followed by one or more `|`-separated
//! `adapter-sub-url`s, each of the form `scheme:scheme-specific-part`.
//!
//! This parser only splits the pipeline into `(scheme, rest)` segments; it does not attempt to
//! decompose `rest` into authority/path/query, since different schemes need different treatment
//! (e.g. `s3://bucket/path` has an authority, `zip:path/within/zip.zarr/` does not). Each plugin's
//! `create_fn` is responsible for interpreting its own `rest`.
//!
//! A literal `|` inside a segment must be percent-encoded (`%7C`) since the spec does not define
//! an escaping mechanism for the pipe delimiter itself.

use crate::error::PipelineError;

/// A single parsed `scheme:rest` pipeline segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineSegment {
    /// The lowercased scheme (canonical form per the spec).
    pub scheme: String,
    /// The raw scheme-specific part, unparsed.
    pub rest: String,
}

/// Split a pipeline string into its raw `|`-separated segments.
#[must_use]
pub fn split_pipeline(pipeline: &str) -> Vec<&str> {
    pipeline.split('|').collect()
}

/// Parse a single `scheme:rest` segment.
///
/// # Errors
/// Returns [`PipelineError::InvalidSegment`] if the segment has no `:` or an empty scheme.
pub fn parse_segment(segment: &str) -> Result<PipelineSegment, PipelineError> {
    let Some(colon) = segment.find(':') else {
        return Err(PipelineError::InvalidSegment(
            segment.to_string(),
            "missing ':' separating scheme from scheme-specific-part".to_string(),
        ));
    };
    let (scheme, rest) = segment.split_at(colon);
    if scheme.is_empty() {
        return Err(PipelineError::InvalidSegment(
            segment.to_string(),
            "scheme must not be empty".to_string(),
        ));
    }
    // Strip the leading ':' from `rest`.
    let rest = &rest[1..];
    Ok(PipelineSegment {
        scheme: scheme.to_lowercase(),
        rest: rest.to_string(),
    })
}

/// Parse a full pipeline string into an ordered list of segments.
///
/// The first segment is the root; any remaining segments are adapters applied in order.
///
/// # Errors
/// Returns [`PipelineError::EmptyPipeline`] if `pipeline` is empty, or
/// [`PipelineError::InvalidSegment`] if any segment fails to parse.
pub fn parse_pipeline(pipeline: &str) -> Result<Vec<PipelineSegment>, PipelineError> {
    if pipeline.is_empty() {
        return Err(PipelineError::EmptyPipeline);
    }
    split_pipeline(pipeline)
        .into_iter()
        .map(parse_segment)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_root() {
        let segments = parse_pipeline("memory://").unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].scheme, "memory");
        assert_eq!(segments[0].rest, "//");
    }

    #[test]
    fn parses_spec_example_zip_zarr3() {
        let segments =
            parse_pipeline("s3://bucket/path/to/archive.zip|zip:path/within/zip.zarr/|zarr3:")
                .unwrap();
        assert_eq!(
            segments,
            vec![
                PipelineSegment {
                    scheme: "s3".to_string(),
                    rest: "//bucket/path/to/archive.zip".to_string()
                },
                PipelineSegment {
                    scheme: "zip".to_string(),
                    rest: "path/within/zip.zarr/".to_string()
                },
                PipelineSegment {
                    scheme: "zarr3".to_string(),
                    rest: String::new()
                },
            ]
        );
    }

    #[test]
    fn parses_spec_example_ocdbt() {
        let segments = parse_pipeline(
            "file:///tmp/dataset.ocdbt/|ocdbt://2025-01-01T01:23:45.678Z/path/within/database",
        )
        .unwrap();
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].scheme, "file");
        assert_eq!(segments[0].rest, "///tmp/dataset.ocdbt/");
        assert_eq!(segments[1].scheme, "ocdbt");
        assert_eq!(
            segments[1].rest,
            "//2025-01-01T01:23:45.678Z/path/within/database"
        );
    }

    #[test]
    fn parses_spec_example_icechunk() {
        let segments = parse_pipeline(
            "s3+https://example.com/path/to/database.icechunk/|icechunk://tag.v5/path/to/node/|zarr3:",
        )
        .unwrap();
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0].scheme, "s3+https");
        assert_eq!(segments[1].scheme, "icechunk");
        assert_eq!(segments[1].rest, "//tag.v5/path/to/node/");
        assert_eq!(segments[2].scheme, "zarr3");
    }

    #[test]
    fn scheme_is_case_insensitive_and_lowercased() {
        let segments = parse_pipeline("S3://bucket/key").unwrap();
        assert_eq!(segments[0].scheme, "s3");
    }

    #[test]
    fn vendor_scheme_token_is_kept_whole() {
        let segment = parse_segment("zarr-python.foo:bar").unwrap();
        assert_eq!(segment.scheme, "zarr-python.foo");
        assert_eq!(segment.rest, "bar");
    }

    #[test]
    fn empty_pipeline_errors() {
        assert!(matches!(
            parse_pipeline(""),
            Err(PipelineError::EmptyPipeline)
        ));
    }

    #[test]
    fn missing_colon_errors() {
        assert!(matches!(
            parse_segment("not-a-scheme"),
            Err(PipelineError::InvalidSegment(_, _))
        ));
    }

    #[test]
    fn empty_scheme_errors() {
        assert!(matches!(
            parse_segment(":rest"),
            Err(PipelineError::InvalidSegment(_, _))
        ));
    }
}
