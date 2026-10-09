//! Summarising results.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::path::Path;

use zarrs_regression_testing::schema::{self, REJECTS_CONFORMANT, Status};

use crate::cases::{Case, Combination, DataTypeCase};
use crate::releases::Release;
use crate::run::{CaseResult, case_dir};

/// The inputs and results of a run.
pub(crate) struct Run<'a> {
    pub(crate) data_types: &'a [DataTypeCase],
    pub(crate) combinations: &'a [Combination],
    pub(crate) cases: &'a [Case],
    pub(crate) results: &'a [CaseResult],
    /// The tested releases, newest first.
    pub(crate) releases: &'a [Release],
    pub(crate) work_dir: &'a Path,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum FailureKind {
    /// Current cannot read back data it wrote.
    CurrentToCurrent,
    /// A release cannot read data written by current.
    CurrentToRelease,
    /// Current cannot read data written by a release.
    ReleaseToCurrent,
    /// Current cannot write data that a release can.
    CurrentCannotWrite,
}

impl FailureKind {
    /// The direction of the failure, independent of the release.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::CurrentToCurrent => "current→current",
            Self::CurrentToRelease => "current→release",
            Self::ReleaseToCurrent => "release→current",
            Self::CurrentCannotWrite => "current cannot write",
        }
    }
}

/// The compatibility of a combination with the latest release.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CombinationStatus {
    /// Current and the latest release read each other's data.
    Compatible,
    /// Current reads back its own data, but the latest release cannot.
    New,
    /// Neither current nor the latest release can read back their own data.
    Unsupported,
    /// Anything else.
    Other,
}

/// A group of combinations with the same compatibility bounds.
pub(crate) struct CompatibilityRow {
    /// The bound of current→release compatibility.
    pub(crate) forward: String,
    /// The bound of release→current compatibility.
    pub(crate) backward: String,
    /// The combinations as merged codecs and data types.
    pub(crate) combinations: Vec<(String, String)>,
}

/// A failure indicating a regression or bug.
pub(crate) struct Failure {
    pub(crate) case: usize,
    pub(crate) kind: FailureKind,
    /// The release involved, or [`None`] if current failed to read its own data.
    pub(crate) release: Option<Release>,
    pub(crate) message: String,
}

impl Failure {
    pub(crate) fn direction(&self) -> String {
        let release = self
            .release
            .map(|release| release.to_string())
            .unwrap_or_default();
        match self.kind {
            FailureKind::CurrentToCurrent if self.release.is_some() => {
                format!("current→current (as {release})")
            }
            FailureKind::CurrentToCurrent => "current→current".to_string(),
            FailureKind::CurrentToRelease => format!("current→{release}"),
            FailureKind::ReleaseToCurrent => format!("{release}→current"),
            FailureKind::CurrentCannotWrite => format!("current cannot write (unlike {release})"),
        }
    }

    /// The writer of the data (`current` or a release), i.e. its work subdirectory.
    pub(crate) fn writer(&self) -> String {
        match (self.kind, self.release) {
            (FailureKind::ReleaseToCurrent, Some(release)) => release.to_string(),
            _ => "current".to_string(),
        }
    }
}

impl Run<'_> {
    pub(crate) fn label(&self, combination: usize) -> (String, &'static str) {
        let combination = &self.combinations[combination];
        (
            combination.codec.to_string(),
            self.data_types[combination.data_type].label,
        )
    }

    /// Failures that indicate a regression or bug, and known issues.
    ///
    /// For a release that can write and read back a case itself, it is a failure if:
    /// - current cannot read data written by the release,
    /// - the release cannot read data written by current, or
    /// - current cannot write the case.
    ///
    /// It is also a failure if current cannot read back data it wrote, unless the latest release cannot either (a known issue).
    ///
    /// It is a known issue rather than a failure if a release cannot read data written by current, and it wrote the case non-conformantly in a way that also makes it reject conformant data (see [`REJECTS_CONFORMANT`]).
    #[must_use]
    pub(crate) fn failures(&self) -> (Vec<Failure>, Vec<Failure>) {
        let mut failures = Vec::new();
        let mut known_issues = Vec::new();
        for (case, result) in self.results.iter().enumerate() {
            let failure = |kind, release, message: &String| Failure {
                case,
                kind,
                release,
                message: message.clone(),
            };
            if let Status::Fail(message) = &result.current_roundtrip {
                if matches!(result.releases[0].roundtrip, Status::Fail(_)) {
                    let latest = Some(self.releases[0]);
                    known_issues.push(failure(FailureKind::CurrentToCurrent, latest, message));
                } else {
                    failures.push(failure(FailureKind::CurrentToCurrent, None, message));
                }
            }
            for (&release, release_result) in self.releases.iter().zip(&result.releases) {
                if release_result.roundtrip != Status::Ok {
                    continue;
                }
                let release = Some(release);
                if let Status::Fail(message) = &release_result.backward {
                    failures.push(failure(FailureKind::ReleaseToCurrent, release, message));
                }
                if let Status::Fail(message) = &release_result.forward {
                    let failure = failure(FailureKind::CurrentToRelease, release, message);
                    let rejects_conformant =
                        release_result
                            .non_conformances
                            .iter()
                            .any(|non_conformance| {
                                REJECTS_CONFORMANT.contains(&non_conformance.as_str())
                            });
                    if rejects_conformant {
                        known_issues.push(failure);
                    } else {
                        failures.push(failure);
                    }
                }
                if let schema::Write::Error(message) = &result.current_write {
                    failures.push(failure(FailureKind::CurrentCannotWrite, release, message));
                }
            }
        }
        (failures, known_issues)
    }

    /// Format failures with details, grouped by combination and direction (one example each unless `verbose`).
    #[must_use]
    pub(crate) fn format_failures(&self, failures: &[&Failure], verbose: bool) -> String {
        let mut grouped: BTreeMap<(usize, String), Vec<&Failure>> = BTreeMap::new();
        for failure in failures {
            let combination = self.cases[failure.case].combination;
            grouped
                .entry((combination, failure.direction()))
                .or_default()
                .push(failure);
        }
        let mut out = String::new();
        for ((combination, direction), failures) in grouped {
            let (codec, data_type) = self.label(combination);
            let samples = self
                .cases
                .iter()
                .filter(|case| case.combination == combination)
                .count();
            let _ = writeln!(
                out,
                "✗ {direction}: {codec} {data_type} ({}/{samples} samples)",
                failures.len()
            );
            let failures = if verbose {
                &failures[..]
            } else {
                &failures[..1]
            };
            for failure in failures {
                let case = &self.cases[failure.case];
                let case_dir = case_dir(self.work_dir, &failure.writer(), failure.case);
                let _ = writeln!(out, "    {}", failure.message);
                let _ = writeln!(out, "    {}", case.describe());
                let _ = writeln!(out, "    {}", case_dir.display());
            }
        }
        out
    }

    /// Format failures concisely: the releases affected for each codec and data type.
    #[must_use]
    pub(crate) fn format_failures_concise(&self, failures: &[&Failure]) -> String {
        let items = failures.iter().map(|failure| {
            let combination = self.cases[failure.case].combination;
            (failure.kind, combination, failure.release)
        });
        let mut out = String::new();
        for (kind, releases, codecs, data_types) in self.grouped_rows(items) {
            let direction = kind.label();
            let _ = writeln!(out, "{direction:<20} {releases:<18} {codecs}: {data_types}");
        }
        out
    }

    /// Format the non-conformances of data written by releases (see [`crate::conformance`]): the releases affected for each codec and data type.
    #[must_use]
    pub(crate) fn format_non_conformances(&self) -> String {
        let items =
            self.cases
                .iter()
                .zip(self.results)
                .flat_map(|(case, result)| {
                    self.releases.iter().zip(&result.releases).flat_map(
                        move |(&release, result)| {
                            result.non_conformances.iter().map(move |non_conformance| {
                                (non_conformance.as_str(), case.combination, Some(release))
                            })
                        },
                    )
                });
        let mut out = String::new();
        for (non_conformance, releases, codecs, data_types) in self.grouped_rows(items) {
            let _ = writeln!(
                out,
                "{releases:<18} {codecs}: {data_types} [{non_conformance}]"
            );
        }
        out
    }

    /// Group `(key, combination, release)` items by key and the releases affected, as rows of `(key, releases, codecs, data types)` with merged codecs and data types.
    fn grouped_rows<K: Ord + Clone>(
        &self,
        items: impl IntoIterator<Item = (K, usize, Option<Release>)>,
    ) -> Vec<(K, String, String, String)> {
        // (key, combination) -> releases
        let mut releases: BTreeMap<(K, usize), BTreeSet<Release>> = BTreeMap::new();
        for (key, combination, release) in items {
            releases
                .entry((key, combination))
                .or_default()
                .extend(release);
        }
        // (key, releases) -> codec -> data types
        let mut groups: BTreeMap<(K, String), BTreeMap<String, Vec<&str>>> = BTreeMap::new();
        for ((key, combination), affected) in releases {
            let affected: Vec<bool> = self
                .releases
                .iter()
                .map(|release| affected.contains(release))
                .collect();
            let (codec, data_type) = self.label(combination);
            groups
                .entry((key, self.ranges(&affected).join(", ")))
                .or_default()
                .entry(codec)
                .or_default()
                .push(data_type);
        }
        groups
            .into_iter()
            .flat_map(|((key, releases), codecs)| {
                self.merge_codecs(codecs)
                    .into_iter()
                    .map(move |(codecs, data_types)| {
                        (key.clone(), releases.clone(), codecs, data_types)
                    })
            })
            .collect()
    }

    /// Whether `direction` is compatible for all `results` (of a combination), for every release.
    fn compatible(
        &self,
        results: &[&CaseResult],
        direction: impl Fn(&CaseResult, usize) -> &Status,
    ) -> Vec<bool> {
        (0..self.releases.len())
            .map(|release| {
                results
                    .iter()
                    .all(|result| *direction(result, release) == Status::Ok)
            })
            .collect()
    }

    /// Describe the releases matching `include` (in the order of [`Run::releases`]) as ranges, e.g. `0.13–0.20`.
    pub(crate) fn ranges(&self, include: &[bool]) -> Vec<String> {
        let mut ranges = Vec::new();
        let mut index = 0;
        while index < include.len() {
            if include[index] {
                let start = index;
                while index + 1 < include.len() && include[index + 1] {
                    index += 1;
                }
                ranges.push(if start == index {
                    self.releases[start].to_string()
                } else {
                    format!("{}–{}", self.releases[index], self.releases[start])
                });
            }
            index += 1;
        }
        ranges
    }

    /// Describe the contiguous range of compatible releases from the newest, and any others.
    fn bound(&self, compatible: &[bool]) -> (usize, String) {
        let contiguous = compatible.iter().take_while(|&&ok| ok).count();
        let mut description = if contiguous == 0 {
            "none".to_string()
        } else {
            format!("{}+", self.releases[contiguous - 1])
        };
        let mut others = compatible.to_vec();
        others[..contiguous].fill(false);
        let others = self.ranges(&others);
        if !others.is_empty() {
            let _ = write!(description, " (+{})", others.join(", "));
        }
        (contiguous, description)
    }

    /// Merge codecs with identical data types, and abbreviate:
    /// - the data types of a codec as `all data types` if every applicable data type is included, and
    /// - codecs as `name(*)` if every variant of a codec is included (e.g. `blosc(*)`).
    fn merge_codecs(&self, codecs: BTreeMap<String, Vec<&str>>) -> Vec<(String, String)> {
        let all_combinations = |codec: &str| {
            self.combinations
                .iter()
                .filter(|combination| combination.codec.to_string() == codec)
                .count()
        };
        let family = |codec: &str| codec.split('(').next().unwrap_or_default().to_string();
        let all_variants = |family_name: &str| {
            self.combinations
                .iter()
                .map(|combination| combination.codec.to_string())
                .filter(|codec| family(codec) == family_name)
                .collect::<BTreeSet<_>>()
                .len()
        };

        let mut by_data_types: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for (codec, data_types) in codecs {
            let data_types = if data_types.len() == all_combinations(&codec) {
                "all data types".to_string()
            } else {
                data_types.join(" ")
            };
            by_data_types.entry(data_types).or_default().push(codec);
        }
        by_data_types
            .into_iter()
            .map(|(data_types, codecs)| {
                let mut families: BTreeMap<String, Vec<String>> = BTreeMap::new();
                for codec in codecs {
                    families.entry(family(&codec)).or_default().push(codec);
                }
                let codecs: Vec<String> = families
                    .into_iter()
                    .flat_map(|(family, codecs)| {
                        if codecs.len() > 1 && codecs.len() == all_variants(&family) {
                            vec![format!("{family}(*)")]
                        } else {
                            codecs
                        }
                    })
                    .collect();
                (codecs.join(", "), data_types)
            })
            .collect()
    }

    /// The results of each combination.
    pub(crate) fn by_combination(&self) -> Vec<Vec<&CaseResult>> {
        let mut by_combination: Vec<Vec<&CaseResult>> = vec![vec![]; self.combinations.len()];
        for (case, result) in self.cases.iter().zip(self.results) {
            by_combination[case.combination].push(result);
        }
        by_combination
    }

    /// The compatibility of a combination (with `results`) with the latest (first) release.
    #[must_use]
    pub(crate) fn status(results: &[&CaseResult]) -> CombinationStatus {
        let all = |status: fn(&CaseResult) -> &Status| {
            results.iter().all(|result| *status(result) == Status::Ok)
        };
        let current = all(|result| &result.current_roundtrip);
        let release = all(|result| &result.releases[0].roundtrip);
        let forward = all(|result| &result.releases[0].forward);
        let backward = all(|result| &result.releases[0].backward);
        match (current, release, forward && backward) {
            (_, _, true) => CombinationStatus::Compatible,
            (true, false, false) => CombinationStatus::New,
            (false, false, false) => CombinationStatus::Unsupported,
            _ => CombinationStatus::Other,
        }
    }

    /// Count combinations by their compatibility with the latest (first) release.
    #[must_use]
    pub(crate) fn counts(&self) -> String {
        let (mut compatible, mut new, mut unsupported, mut other) = (0, 0, 0, 0);
        for results in self.by_combination() {
            match Self::status(&results) {
                CombinationStatus::Compatible => compatible += 1,
                CombinationStatus::New => new += 1,
                CombinationStatus::Unsupported => unsupported += 1,
                CombinationStatus::Other => other += 1,
            }
        }
        format!(
            "{compatible} combinations compatible with {} in both directions, {new} new/fixed in current, {unsupported} unsupported, {other} other",
            self.releases[0]
        )
    }

    /// How far back compatibility extends for a combination (with `results`), as the number of contiguous compatible releases from the newest and a description, for current→release and release→current.
    ///
    /// Returns [`None`] if neither current nor any release can write the combination.
    #[must_use]
    pub(crate) fn bounds(
        &self,
        results: &[&CaseResult],
    ) -> Option<((usize, String), (usize, String))> {
        let current_writes = results
            .iter()
            .any(|result| result.current_write == schema::Write::Ok);
        let any_release_writes = results.iter().any(|result| {
            result
                .releases
                .iter()
                .any(|release| release.backward != Status::NotWritten)
        });
        if !current_writes && !any_release_writes {
            return None;
        }
        let forward = if current_writes {
            self.bound(
                &self.compatible(results, |result, release| &result.releases[release].forward),
            )
        } else {
            (0, "n/a".to_string())
        };
        let backward = self.bound(&self.compatible(results, |result, release| {
            &result.releases[release].backward
        }));
        Some((forward, backward))
    }

    /// Group combinations by how far back data compatibility extends, and count the unsupported combinations.
    #[must_use]
    pub(crate) fn compatibility_rows(&self) -> (Vec<CompatibilityRow>, usize) {
        // (forward, backward) -> codec -> data types
        type Bounds = (usize, usize, String, String);
        let mut groups: BTreeMap<Bounds, BTreeMap<String, Vec<&str>>> = BTreeMap::new();
        let mut unsupported = 0;
        for (combination, results) in self.by_combination().into_iter().enumerate() {
            let Some(((forward_count, forward), (backward_count, backward))) =
                self.bounds(&results)
            else {
                unsupported += 1;
                continue;
            };
            let (codec, data_type) = self.label(combination);
            groups
                .entry((
                    usize::MAX - forward_count,
                    usize::MAX - backward_count,
                    forward,
                    backward,
                ))
                .or_default()
                .entry(codec)
                .or_default()
                .push(data_type);
        }
        let rows = groups
            .into_iter()
            .map(|((_, _, forward, backward), codecs)| CompatibilityRow {
                forward,
                backward,
                combinations: self.merge_codecs(codecs),
            })
            .collect();
        (rows, unsupported)
    }

    /// Summarise how far back data compatibility extends for each combination.
    #[must_use]
    pub(crate) fn compatibility(&self) -> String {
        let (rows, unsupported) = self.compatibility_rows();
        let mut out = String::new();
        let _ = writeln!(
            out,
            "current→release: the oldest release that reads data written by current zarrs (as do all newer releases)"
        );
        let _ = writeln!(
            out,
            "release→current: the oldest release whose data current zarrs reads (as for all newer releases)"
        );
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "{:<18} {:<18} combinations",
            "current→release", "release→current"
        );
        for row in rows {
            for (index, (codecs, data_types)) in row.combinations.into_iter().enumerate() {
                let (forward, backward) = if index == 0 {
                    (row.forward.as_str(), row.backward.as_str())
                } else {
                    ("", "")
                };
                let _ = writeln!(out, "{forward:<18} {backward:<18} {codecs}: {data_types}");
            }
        }
        if unsupported > 0 {
            let _ = writeln!(
                out,
                "\n{unsupported} combinations are not supported by current zarrs or any tested release"
            );
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bound_contiguous_and_others() {
        let releases = [
            Release(23),
            Release(22),
            Release(21),
            Release(20),
            Release(19),
        ];
        let run = Run {
            data_types: &[],
            combinations: &[],
            cases: &[],
            results: &[],
            releases: &releases,
            work_dir: Path::new(""),
        };
        assert_eq!(run.bound(&[true; 5]).1, "0.19+");
        assert_eq!(run.bound(&[false; 5]).1, "none");
        assert_eq!(run.bound(&[true, true, false, false, false]).1, "0.22+");
        assert_eq!(
            run.bound(&[true, false, true, true, false]),
            (1, "0.23+ (+0.20–0.21)".to_string())
        );
        assert_eq!(
            run.bound(&[false, true, false, false, true]).1,
            "none (+0.22, 0.19)"
        );
    }
}
