//! Conversion of cases and results to the machine-readable interface (see [`zarrs_regression_testing::schema`]).

use zarrs_regression_testing::schema;

use crate::cases::{Case, CodecKind, Combination, DataTypeCase};
use crate::run::CaseResult;
use crate::summary::Run;
use crate::{format, introduced};

/// The manifest of `cases` of `combinations` of `data_types`.
pub(crate) fn manifest(
    meta: schema::Meta,
    data_types: &[DataTypeCase],
    combinations: &[Combination],
    cases: &[Case],
) -> schema::Manifest {
    let mut codecs: Vec<CodecKind> = Vec::new();
    for combination in combinations {
        if !codecs.contains(&combination.codec) {
            codecs.push(combination.codec);
        }
    }
    schema::Manifest {
        schema: schema::SCHEMA,
        meta,
        codecs: codecs
            .iter()
            .map(|&codec| schema::Codec {
                label: codec.to_string(),
                introduced: introduced::codec(codec).map(str::to_string),
                note: introduced::codec_note(codec).map(str::to_string),
            })
            .collect(),
        data_types: data_types
            .iter()
            .map(|data_type| schema::DataType {
                label: data_type.label.to_string(),
                introduced: Some(introduced::data_type(data_type.label).to_string()),
                note: introduced::data_type_note(data_type.label).map(str::to_string),
                element_size: data_type.values.element_size(),
                numeric: data_type.values.numeric(),
            })
            .collect(),
        combinations: combinations
            .iter()
            .map(|combination| schema::Combination {
                codec: codecs
                    .iter()
                    .position(|&codec| codec == combination.codec)
                    .expect("collected above"),
                data_type: combination.data_type,
            })
            .collect(),
        cases: cases
            .iter()
            .map(|case| schema::Case {
                combination: case.combination,
                shape: case.shape.clone(),
                chunk_shape: case.chunk_shape.clone(),
                metadata: case.metadata.clone(),
                data: case.data.clone(),
                elements: format::elements(
                    &case.data,
                    &data_types[combinations[case.combination].data_type],
                ),
                lossy: case.lossy,
            })
            .collect(),
    }
}

/// The outcomes of a case: the current `zarrs` is subject `0`, and the releases follow in order.
fn outcomes(result: &CaseResult) -> schema::Outcomes {
    let read = |writer, reader, status: &schema::Status| schema::Read {
        writer,
        reader,
        status: status.clone(),
    };
    let mut outcomes = schema::Outcomes {
        writes: vec![result.current_write.clone()],
        reads: vec![read(0, 0, &result.current_roundtrip)],
        non_conformances: vec![vec![]],
    };
    for (index, release) in result.releases.iter().enumerate() {
        let subject = index + 1;
        outcomes.writes.push(release.write.clone());
        outcomes.reads.extend([
            read(0, subject, &release.forward),
            read(subject, 0, &release.backward),
            read(subject, subject, &release.roundtrip),
        ]);
        outcomes
            .non_conformances
            .push(release.non_conformances.clone());
    }
    outcomes
}

/// The results of a run: the current `zarrs` (the reference) and the releases (newest first).
pub(crate) fn results(manifest: schema::Manifest, run: &Run) -> schema::Results {
    let subject = |version: String, label: String, reference| schema::Subject {
        implementation: "zarrs".to_string(),
        version,
        label,
        reference,
    };
    let subjects = std::iter::once(subject(
        zarrs::version::version_str().to_string(),
        "current".to_string(),
        true,
    ))
    .chain(
        run.releases
            .iter()
            .map(|release| subject(release.to_string(), release.to_string(), false)),
    )
    .collect();
    schema::Results {
        manifest,
        subjects,
        outcomes: run.results.iter().map(outcomes).collect(),
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::cases;
    use crate::releases::RELEASES;
    use crate::run::ReleaseResult;

    #[test]
    fn results_roundtrip() {
        let data_types = cases::data_types();
        let combinations: Vec<_> = cases::combinations(&cases::codec_kinds(), &data_types)
            .into_iter()
            .filter(|combination| {
                ["optional<optional<float32>>", "string", "int16"]
                    .contains(&data_types[combination.data_type].label)
            })
            .collect();
        let cases = cases::sample_cases(&combinations, &data_types, 1, 0).unwrap();
        let results: Vec<_> = cases
            .iter()
            .map(|_| CaseResult {
                current_write: schema::Write::Ok,
                current_roundtrip: schema::Status::Ok,
                releases: vec![ReleaseResult {
                    write: schema::Write::Error("unsupported".to_string()),
                    forward: schema::Status::Fail("decoded bytes differ".to_string()),
                    backward: schema::Status::NotWritten,
                    roundtrip: schema::Status::NotWritten,
                    non_conformances: vec!["non-conformant".to_string()],
                }],
            })
            .collect();
        let run = Run {
            data_types: &data_types,
            combinations: &combinations,
            cases: &cases,
            results: &results,
            releases: &RELEASES[..1],
            work_dir: Path::new("work"),
        };
        let meta = schema::Meta {
            generator: schema::Generator {
                name: "test".to_string(),
                version: "0.0.0".to_string(),
                repository: String::new(),
                commit: None,
                modified: false,
            },
            date: "2026-10-09".to_string(),
            seed: 0,
            samples: 1,
            filter: None,
            reproduce: String::new(),
        };
        let manifest = manifest(meta, &data_types, &combinations, &cases);
        let results = super::results(manifest, &run);
        assert_eq!(results.subjects[0].label, "current");
        assert_eq!(
            results.outcomes[0].read(0, 1),
            Some(&schema::Status::Fail("decoded bytes differ".to_string()))
        );
        let json = serde_json::to_value(&results).unwrap();
        assert_eq!(json["outcomes"][0]["writes"][1]["error"], "unsupported");
        let roundtrip: schema::Results = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(serde_json::to_value(&roundtrip).unwrap(), json);
    }
}
