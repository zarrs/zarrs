//! Serving the helper protocol (see [`zarrs_regression_testing::schema::Request`]) with the current `zarrs` or a previous release, so other tools can use `zarrs` as an implementation adapter.

use std::io::Read;
use std::panic::AssertUnwindSafe;

use zarrs_regression_testing::schema::{Info, Request, Response};

use crate::helper;
use crate::native::native_metadata;
use crate::releases::Release;
use crate::run::{read_current, write_current};

/// A `zarrs` to serve requests with.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Zarrs {
    Current,
    Release(Release),
}

impl std::str::FromStr for Zarrs {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "current" {
            return Ok(Self::Current);
        }
        s.parse()
            .map(Self::Release)
            .map_err(|err| format!("{err} or `current`"))
    }
}

/// Read a JSON array of requests from stdin and write a JSON array of responses to stdout.
///
/// # Errors
/// Returns an error if the requests cannot be read or parsed, a helper fails to build, or the responses cannot be written.
pub(crate) fn serve(zarrs: Zarrs) -> Result<(), String> {
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .map_err(|err| format!("read requests: {err}"))?;
    let requests: Vec<Request> =
        serde_json::from_str(&input).map_err(|err| format!("parse requests: {err}"))?;
    let responses = match zarrs {
        Zarrs::Current => serve_current(&requests),
        Zarrs::Release(release) => serve_release(release, requests)?,
    };
    serde_json::to_writer(std::io::stdout().lock(), &responses)
        .map_err(|err| format!("write responses: {err}"))
}

fn info(zarrs: Zarrs) -> Response {
    let (version, notes) = match zarrs {
        Zarrs::Current => (
            current_version(),
            Some(format!(
                "the zarrs of zarrs_regression_testing {}",
                env!("CARGO_PKG_VERSION")
            )),
        ),
        Zarrs::Release(release) => (release.to_string(), None),
    };
    Response::Info(Info {
        implementation: "zarrs".to_string(),
        version,
        notes,
    })
}

/// The version of the current `zarrs`, with its commit as build metadata if known (e.g. `0.24.0-dev+0123abcd`, or `0.24.0-dev+0123abcd.modified` with changes not committed).
fn current_version() -> String {
    let version = zarrs::version::version_str();
    match crate::commit() {
        Some((commit, modified)) => format!(
            "{version}+{}{}",
            &commit[..commit.len().min(8)],
            if modified { ".modified" } else { "" }
        ),
        None => version.to_string(),
    }
}

/// Requests are handled in order, as they may read arrays written by earlier requests.
fn serve_current(requests: &[Request]) -> Vec<Response> {
    std::panic::set_hook(Box::new(|_| {}));
    requests
        .iter()
        .map(|request| {
            let response = std::panic::catch_unwind(AssertUnwindSafe(|| match request {
                Request::Info => info(Zarrs::Current),
                Request::Write {
                    path,
                    metadata,
                    shape,
                    data,
                } => match write_current(path, metadata, shape, data) {
                    Ok(()) => Response::Ok(None),
                    Err(err) => Response::Err(err),
                },
                Request::Read { path, shape } => match read_current(path, shape) {
                    Ok(data) => Response::Ok(Some(data)),
                    Err(err) => Response::Err(err),
                },
            }));
            response.unwrap_or_else(|panic| {
                let message = panic
                    .downcast_ref::<&str>()
                    .map(|message| (*message).to_string())
                    .or_else(|| panic.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "unknown".to_string());
                Response::Err(format!("panic: {message}"))
            })
        })
        .collect()
}

fn serve_release(release: Release, mut requests: Vec<Request>) -> Result<Vec<Response>, String> {
    let binary = helper::build(release)?;
    for request in &mut requests {
        if let Request::Write { metadata, .. } = request {
            *metadata = native_metadata(release, metadata);
        }
    }
    let helper_requests: Vec<helper::Request> = requests
        .iter()
        .filter_map(|request| match request {
            Request::Info => None,
            Request::Write {
                path,
                metadata,
                shape,
                data,
            } => Some(helper::Request::Write {
                path: path.clone(),
                metadata,
                shape,
                data,
            }),
            Request::Read { path, shape } => Some(helper::Request::Read {
                path: path.clone(),
                shape,
            }),
        })
        .collect();
    let mut helper_responses = helper::run(&binary, &helper_requests).into_iter();
    Ok(requests
        .iter()
        .map(|request| match request {
            Request::Info => info(Zarrs::Release(release)),
            _ => match helper_responses.next() {
                Some(Ok(data)) => Response::Ok(data),
                Some(Err(err)) => Response::Err(err),
                None => Response::Err("missing helper response".to_string()),
            },
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cases;

    #[test]
    fn serve_current_roundtrip() {
        let data_types = cases::data_types();
        let combinations: Vec<_> = cases::combinations(&cases::codec_kinds(), &data_types)
            .into_iter()
            .filter(|combination| {
                combination.codec.to_string() == "gzip"
                    && data_types[combination.data_type].label == "int16"
            })
            .collect();
        let case = cases::sample_cases(&combinations, &data_types, 1, 0)
            .unwrap()
            .remove(0);
        let dir = std::env::temp_dir().join(format!(
            "zarrs_regression_testing_serve_{}",
            std::process::id()
        ));
        let responses = serve_current(&[
            Request::Info,
            Request::Write {
                path: dir.clone(),
                metadata: case.metadata.clone(),
                shape: case.shape.clone(),
                data: case.data.clone(),
            },
            Request::Read {
                path: dir.clone(),
                shape: case.shape.clone(),
            },
        ]);
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(matches!(&responses[0], Response::Info(info) if info.implementation == "zarrs"));
        assert_eq!(responses[1], Response::Ok(None));
        assert_eq!(responses[2], Response::Ok(Some(case.data)));
    }
}
