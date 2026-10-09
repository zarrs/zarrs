//! Previous `zarrs` releases tested for data compatibility.

/// A previous minor release of `zarrs` (`0.<minor>`, resolved to the latest patch release).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct Release(pub(crate) u32);

/// Tested releases, newest first.
///
/// The first entry is the latest release, which is tested in CI.
/// Add new releases to the front of this list when they are published, and set [`Release::publish_time`] of the previous latest release.
pub(crate) const RELEASES: &[Release] = &[
    Release(23),
    Release(22),
    Release(21),
    Release(20),
    Release(19),
    Release(18),
    Release(17),
    Release(16),
    Release(15),
    Release(14),
    Release(13),
    Release(12),
    Release(11),
    Release(10),
    // Releases prior to the "Changes after Provisional Acceptance" of the Zarr V3 specification (excluding the removal of implicit groups),
    // the last of which was renaming the `endian` codec to `bytes` (zarr-specs#263, merged 2024-01-11).
    // Release(9),
    // Release(8),
    // Release(7),
    // Release(6),
    // Release(5),
    // Release(4),
    // Release(3),
    // Release(2),
];

/// The toolchain for releases that fail to compile with recent toolchains (see [`Release::toolchain`]).
pub(crate) const OLD_TOOLCHAIN: &str = "1.86.0";

impl std::fmt::Display for Release {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0.{}", self.0)
    }
}

impl std::str::FromStr for Release {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        RELEASES
            .iter()
            .find(|release| release.to_string() == s)
            .copied()
            .ok_or_else(|| format!("unknown zarrs `{s}`: expected a tested release (e.g. `0.23`)"))
    }
}

impl Release {
    /// The latest time at which registry packages are considered when resolving the dependencies of this release.
    ///
    /// This is the publish time of the last (non-yanked) patch release, so that the helper is built with dependencies available at the time.
    /// It is `None` for the latest release, which is built with the latest dependencies.
    ///
    /// Cargo only respects the current yank state, so a release cannot be resolved if a dependency version required at the time has since been yanked.
    /// In that case, the cutoff is the earliest time at which the dependencies resolve.
    pub(crate) fn publish_time(self) -> Option<&'static str> {
        match self.0 {
            23.. => None,
            22 => Some("2025-11-29T08:12:35Z"),
            21 => Some("2025-06-19T12:42:22Z"),
            20 => Some("2025-06-01T09:28:18Z"),
            19 => Some("2025-02-13T00:09:54Z"),
            18 => Some("2024-12-29T23:27:00Z"),
            17 => Some("2024-10-17T20:42:06Z"),
            16 => Some("2024-08-22T10:14:07Z"),
            // `bytes` 1.6.0 has been yanked and was replaced by 1.6.1 two days after the last patch release
            15 => Some("2024-07-14T00:00:00Z"),
            14 => Some("2024-05-16T07:21:34Z"),
            13 => Some("2024-05-07T22:25:43Z"),
            12 => Some("2024-03-17T00:33:36Z"),
            11 => Some("2024-02-06T00:26:32Z"),
            // `futures-util` 0.3.29 and 0.3.30 (required by 0.7 to 0.10) have been yanked, so the dependencies only resolve from Oct 2024 (0.3.31)
            7..=10 => Some("2024-10-16T00:00:00Z"),
            6 => Some("2023-11-16T10:45:29Z"),
            5 => Some("2023-10-09T22:27:01Z"),
            4 => Some("2023-10-06T01:04:05Z"),
            3 => Some("2023-09-27T11:17:26Z"),
            _ => Some("2023-09-25T06:02:04Z"),
        }
    }

    /// The toolchain to build the helper for this release with, or [`None`] for the current toolchain.
    ///
    /// Releases prior to 0.16 fail to compile the sharding codec with Rust 1.87+, where `u64::is_multiple_of` takes precedence over `num::Integer::is_multiple_of`.
    pub(crate) fn toolchain(self) -> Option<&'static str> {
        (self.0 < 16).then_some(OLD_TOOLCHAIN)
    }

    /// The `zarrs` features enabled in the helper for this release.
    pub(crate) fn features(self) -> Vec<&'static str> {
        let minor = self.0;
        let mut features = vec!["blosc", "gzip", "crc32c", "sharding", "transpose", "zstd"];
        if (7..=10).contains(&minor) {
            // Broken feature gating in these releases
            features.extend(["async", "ndarray"]);
        }
        let since = [
            (6, "bitround"),
            (6, "zfp"),
            (11, "bz2"),
            (11, "pcodec"),
            (16, "gdeflate"),
            (17, "filesystem"),
            (19, "fletcher32"),
            (20, "zlib"),
            (21, "float8"),
            (22, "adler32"),
            (23, "microfloat"),
        ];
        features.extend(
            since
                .into_iter()
                .filter(|(since, _)| minor >= *since)
                .map(|(_, feature)| feature),
        );
        features
    }

    /// The helper adapter source for this release (bridges API differences between releases).
    pub(crate) fn adapter(self) -> String {
        let adapter = match self.0 {
            23.. => include_str!("../helper/adapter_v0_23.rs"),
            20..=22 => include_str!("../helper/adapter_v0_20.rs"),
            17..=19 => include_str!("../helper/adapter_v0_17.rs"),
            16 => include_str!("../helper/adapter_v0_16.rs"),
            15 | 4 | 5 => include_str!("../helper/adapter_v0_4.rs"),
            11..=14 => include_str!("../helper/adapter_v0_11.rs"),
            6..=10 => include_str!("../helper/adapter_v0_6.rs"),
            _ => include_str!("../helper/adapter_v0_2.rs"),
        };
        if self.0 >= 16 {
            format!(
                "{adapter}\n{}",
                include_str!("../helper/adapter_v0_16_common.rs")
            )
        } else {
            adapter.to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publish_times() {
        assert_eq!(RELEASES[0].publish_time(), None);
        for release in &RELEASES[1..] {
            assert!(
                release.publish_time().is_some(),
                "set the publish time of {release}"
            );
        }
    }
}
