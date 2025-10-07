//! A filesystem store for the [`zarrs`](https://docs.rs/zarrs/latest/zarrs/index.html) crate.
//!
//! This implementation is conformant with the filesystem store defined in the Zarr V3 specification: <https://zarr-specs.readthedocs.io/en/latest/v3/stores/filesystem/index.html>.
//!
//! ## Licence
//! `zarrs_filesystem` is licensed under either of
//! - the Apache License, Version 2.0 [LICENSE-APACHE](https://docs.rs/crate/zarrs_filesystem/latest/source/LICENCE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0> or
//! - the MIT license [LICENSE-MIT](https://docs.rs/crate/zarrs_filesystem/latest/source/LICENCE-MIT) or <http://opensource.org/licenses/MIT>, at your option.

use zarrs_storage::{
    byte_range::{ByteOffset, ByteRange, ByteRangeIterator},
    store_set_partial_many, Bytes, ListableStorageTraits, MaybeBytesIterator, OffsetBytesIterator,
    ReadableStorageTraits, StorageError, StoreKey, StoreKeyError, StoreKeys, StoreKeysPrefixes,
    StorePrefix, StorePrefixes, WritableStorageTraits,
};

use bytes::BytesMut;
use parking_lot::RwLock; // TODO: std::sync::RwLock with Rust 1.78+
use thiserror::Error;
use walkdir::WalkDir;

use std::{
    collections::HashMap,
    fs::OpenOptions,
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

#[cfg(target_os = "linux")]
use libc::O_DIRECT;
#[cfg(target_os = "linux")]
use std::os::{
    fd::AsRawFd,
    unix::fs::{MetadataExt, OpenOptionsExt},
};

// // Register the store.
// inventory::submit! {
//     ReadableStorePlugin::new("file", |uri| Ok(Arc::new(create_store_filesystem(uri)?)))
// }
// inventory::submit! {
//     WritableStorePlugin::new("file", |uri| Ok(Arc::new(create_store_filesystem(uri)?)))
// }
// inventory::submit! {
//     ListableStorePlugin::new("file", |uri| Ok(Arc::new(create_store_filesystem(uri)?)))
// }
// inventory::submit! {
//     ReadableWritableStorePlugin::new("file", |uri| Ok(Arc::new(create_store_filesystem(uri)?)))
// }

// #[allow(clippy::similar_names)]
// fn create_store_filesystem(uri: &str) -> Result<FilesystemStore, StorePluginCreateError> {
//     let url = url::Url::parse(uri)?;
//     let path = std::path::PathBuf::from(url.path());
//     FilesystemStore::new(path).map_err(|e| StorePluginCreateError::Other(e.to_string()))
// }

/// For `O_DIRECT`, we need a buffer that is aligned to the page size and is a
/// multiple of the page size.
fn bytes_aligned(size: usize) -> BytesMut {
    let align = page_size::get();
    let mut bytes = BytesMut::with_capacity(size + 2 * align);
    let offset = bytes.as_ptr().align_offset(align);
    bytes.split_off(offset)
}

/// Options for use with [`FilesystemStore`]
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub struct FilesystemStoreOptions {
    direct_io: bool,
}

impl FilesystemStoreOptions {
    /// Set whether or not to enable direct I/O. Needs support from the
    /// operating system (currently only Linux) and file system.
    pub fn direct_io(&mut self, direct_io: bool) -> &mut Self {
        self.direct_io = direct_io;
        self
    }
}

/// A synchronous file system store.
///
/// See <https://zarr-specs.readthedocs.io/en/latest/v3/stores/filesystem/index.html>.
#[derive(Debug)]
pub struct FilesystemStore {
    base_path: PathBuf,
    sort: bool,
    readonly: bool,
    options: FilesystemStoreOptions,
    files: Mutex<HashMap<StoreKey, Arc<RwLock<()>>>>,
}

impl FilesystemStore {
    /// Create a new file system store at a given `base_path`.
    ///
    /// # Errors
    /// Returns a [`FilesystemStoreCreateError`] if `base_directory`:
    ///   - is not valid, or
    ///   - it points to an existing file rather than a directory.
    pub fn new<P: AsRef<Path>>(base_path: P) -> Result<Self, FilesystemStoreCreateError> {
        Self::new_with_options(base_path, FilesystemStoreOptions::default())
    }

    /// Create a new file system store at a given `base_path` and `options`.
    ///
    /// # Errors
    /// Returns a [`FilesystemStoreCreateError`] if `base_directory`:
    ///   - is not valid, or
    ///   - it points to an existing file rather than a directory.
    pub fn new_with_options<P: AsRef<Path>>(
        base_path: P,
        options: FilesystemStoreOptions,
    ) -> Result<Self, FilesystemStoreCreateError> {
        let base_path = base_path.as_ref().to_path_buf();
        if base_path.to_str().is_none() {
            return Err(FilesystemStoreCreateError::InvalidBasePath(base_path));
        }

        let readonly = if base_path.exists() {
            // the path already exists, check if it is read only
            let md = std::fs::metadata(&base_path).map_err(FilesystemStoreCreateError::IOError)?;
            md.permissions().readonly()
        } else {
            // the path does not exist, so try and create it. If this succeeds, the filesystem is not read only
            std::fs::create_dir_all(&base_path).map_err(FilesystemStoreCreateError::IOError)?;
            std::fs::remove_dir(&base_path)?;
            false
        };

        Ok(Self {
            base_path,
            sort: false,
            options,
            readonly,
            files: Mutex::default(),
        })
    }

    /// Makes the store sort directories/files when walking.
    #[must_use]
    pub const fn sorted(mut self) -> Self {
        self.sort = true;
        self
    }

    /// Maps a [`StoreKey`] to a filesystem [`PathBuf`].
    #[must_use]
    pub fn key_to_fspath(&self, key: &StoreKey) -> PathBuf {
        let mut path = self.base_path.clone();
        if !key.as_str().is_empty() {
            path.push(key.as_str().strip_prefix('/').unwrap_or(key.as_str()));
        }
        path
    }

    /// Maps a filesystem [`PathBuf`] to a [`StoreKey`].
    fn fspath_to_key(&self, path: &std::path::Path) -> Result<StoreKey, StoreKeyError> {
        let path = pathdiff::diff_paths(path, &self.base_path)
            .ok_or_else(|| StoreKeyError::from(path.to_str().unwrap_or_default().to_string()))?;
        let path_str = path.to_string_lossy();
        #[cfg(target_os = "windows")]
        {
            StoreKey::new(path_str.replace('\\', "/"))
        }
        #[cfg(not(target_os = "windows"))]
        {
            StoreKey::new(path_str)
        }
    }

    /// Maps a store [`StorePrefix`] to a filesystem [`PathBuf`].
    #[must_use]
    pub fn prefix_to_fs_path(&self, prefix: &StorePrefix) -> PathBuf {
        let mut path = self.base_path.clone();
        path.push(prefix.as_str());
        path
    }

    fn get_file_mutex(&self, key: &StoreKey) -> Arc<RwLock<()>> {
        let mut files = self.files.lock().unwrap();
        let file = files
            .entry(key.clone())
            .or_insert_with(|| Arc::new(RwLock::default()))
            .clone();
        drop(files);
        file
    }

    fn set_impl(
        &self,
        key: &StoreKey,
        value: &[u8],
        offset: ByteOffset,
        truncate: bool,
    ) -> Result<(), StorageError> {
        let file = self.get_file_mutex(key);
        let _lock = file.write();

        // Create directories
        let key_path = self.key_to_fspath(key);
        if let Some(parent) = key_path.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let mut flags = OpenOptions::new();
        flags.write(true).create(true).truncate(truncate);

        // TODO: for now, only Linux support; also no support for `offset != 0`
        let enable_direct =
            cfg!(target_os = "linux") && self.options.direct_io && offset == 0 && !value.is_empty();

        // If `value` is already page-size aligned, we don't need to copy.
        let need_copy = value.as_ptr().align_offset(page_size::get()) != 0
            || value.len() % page_size::get() != 0;

        #[cfg(target_os = "linux")]
        if enable_direct {
            flags.custom_flags(O_DIRECT);
        }

        let mut file = flags.open(key_path)?;

        // Write
        if enable_direct {
            if need_copy {
                let mut buf = bytes_aligned(value.len());
                buf.extend_from_slice(value);

                // Pad to page size
                let pad_size = buf.len().next_multiple_of(page_size::get()) - buf.len();
                buf.extend(std::iter::repeat(0).take(pad_size));

                file.write_all(&buf)?;
            } else {
                file.write_all(value)?;
            }

            // Truncate again to requested size
            file.set_len(value.len() as u64)?;
        } else {
            file.seek(SeekFrom::Start(offset))?;
            file.write_all(value)?;
        }

        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[allow(clippy::too_many_lines)]
    fn get_partial_many_direct_io<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        use std::collections::BTreeMap;

        // Lock and open the file
        let file = self.get_file_mutex(key);
        let _lock = file.read();
        let mut flags = OpenOptions::new();
        flags.read(true);
        flags.custom_flags(O_DIRECT);
        let file = match flags.open(self.key_to_fspath(key)) {
            Ok(file) => file,
            Err(err) => {
                if err.kind() == std::io::ErrorKind::NotFound {
                    return Ok(None);
                }
                return Err(err.into());
            }
        };

        let file_size: u64 = file.metadata()?.size();
        let ps = page_size::get() as u64;
        let fd = file.as_raw_fd();

        // Find intersected pages
        let byte_ranges = byte_ranges.collect::<Vec<ByteRange>>();
        let intersected_pages_coalesced =
            direct_io::coalesce_byte_ranges_with_page_size(file_size, &byte_ranges, ps);

        // Read intersected pages
        let mut page_bytes = BTreeMap::new();
        for pages in intersected_pages_coalesced {
            let num_pages = pages.0.end - pages.0.start;
            let offset = pages.0.start * ps;
            let length = usize::try_from(num_pages * ps).unwrap();
            let mut buf = bytes_aligned(length);
            let ret = unsafe {
                libc::pread(
                    fd,
                    buf.as_mut_ptr().cast::<libc::c_void>(),
                    length,
                    libc::off_t::try_from(offset).unwrap(),
                )
            };
            if ret < 0 {
                return Err(std::io::Error::last_os_error().into());
            }
            unsafe {
                buf.set_len(length);
            }
            page_bytes.insert(offset, buf.freeze());
        }

        // Extract the requested byte ranges
        let out = byte_ranges
            .into_iter()
            .map(|byte_range| {
                use std::ops::Bound;

                let start = byte_range.start(file_size);
                let length = usize::try_from(byte_range.length(file_size)).unwrap();

                if (start + length as u64) > file_size {
                    // NOTE: Could put this check earlier
                    return Err(zarrs_storage::byte_range::InvalidByteRangeError::new(
                        byte_range, file_size,
                    )
                    .into());
                }

                // Find the first element in page_bytes that is >= start_page_aligned
                let (intersected_pages_start, bytes) = page_bytes
                    .range((Bound::Unbounded, Bound::Included(&start)))
                    .next_back()
                    .ok_or_else(|| {
                        StorageError::Other(format!(
                            "Could not find intersected pages for byte range {byte_range}"
                        ))
                    })?;
                // TODO: use upper bound when btree_cursors stabilises
                // let (intersected_pages_start, bytes) = page_bytes
                //     .upper_bound(Bound::Included(&start))
                //     .peek_prev()
                //     .ok_or_else(|| {
                //         StorageError::Other(format!(
                //             "Could not find intersected pages for byte range {byte_range}"
                //         ))
                //     })?;

                let offset = usize::try_from(start - *intersected_pages_start).unwrap();
                Ok(bytes.slice(offset..offset + length))
            })
            .collect::<Vec<_>>();
        Ok(Some(Box::new(out.into_iter())))
    }
}

impl ReadableStorageTraits for FilesystemStore {
    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        #[cfg(target_os = "linux")]
        if self.options.direct_io {
            return self.get_partial_many_direct_io(key, byte_ranges);
        }

        // Lock and open the file
        let file = self.get_file_mutex(key);
        let _lock = file.read();
        let mut flags = OpenOptions::new();
        flags.read(true);
        let mut file = match flags.open(self.key_to_fspath(key)) {
            Ok(file) => file,
            Err(err) => {
                if err.kind() == std::io::ErrorKind::NotFound {
                    return Ok(None);
                }
                return Err(err.into());
            }
        };

        let out = byte_ranges
            .map(|byte_range| {
                // Seek
                match byte_range {
                    ByteRange::FromStart(offset, _) => file.seek(SeekFrom::Start(offset)),
                    ByteRange::Suffix(length) => {
                        file.seek(SeekFrom::End(-(i64::try_from(length).unwrap())))
                    }
                }?;

                // Read
                match byte_range {
                    ByteRange::FromStart(_, None) => {
                        let mut buffer = Vec::new();
                        file.read_to_end(&mut buffer)?;
                        Ok(Bytes::from(buffer))
                    }
                    ByteRange::FromStart(_, Some(length)) | ByteRange::Suffix(length) => {
                        let length = usize::try_from(length).unwrap();
                        let mut buffer = vec![0; length];
                        file.read_exact(&mut buffer)?;
                        Ok(Bytes::from(buffer))
                    }
                }
            })
            .collect::<Vec<_>>();

        Ok(Some(Box::new(out.into_iter())))
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        let key_path = self.key_to_fspath(key);
        std::fs::metadata(key_path).map_or_else(|_| Ok(None), |metadata| Ok(Some(metadata.len())))
    }

    fn supports_get_partial(&self) -> bool {
        true
    }
}

impl WritableStorageTraits for FilesystemStore {
    fn set(&self, key: &StoreKey, value: Bytes) -> Result<(), StorageError> {
        if self.readonly {
            Err(StorageError::ReadOnly)
        } else {
            Self::set_impl(self, key, &value, 0, true)
        }
    }

    fn set_partial_many(
        &self,
        key: &StoreKey,
        offset_values: OffsetBytesIterator,
    ) -> Result<(), StorageError> {
        if self.readonly {
            return Err(StorageError::ReadOnly);
        }

        store_set_partial_many(self, key, offset_values)
    }

    fn erase(&self, key: &StoreKey) -> Result<(), StorageError> {
        if self.readonly {
            return Err(StorageError::ReadOnly);
        }

        let file = self.get_file_mutex(key);
        let _lock = file.write();

        let key_path = self.key_to_fspath(key);
        let result = std::fs::remove_file(key_path);
        if let Err(err) = result {
            match err.kind() {
                std::io::ErrorKind::NotFound => Ok(()),
                _ => Err(err.into()),
            }
        } else {
            Ok(())
        }
    }

    fn erase_prefix(&self, prefix: &StorePrefix) -> Result<(), StorageError> {
        if self.readonly {
            return Err(StorageError::ReadOnly);
        }

        let _lock = self.files.lock(); // lock all operations

        let prefix_path = self.prefix_to_fs_path(prefix);
        let result = std::fs::remove_dir_all(prefix_path);
        if let Err(err) = result {
            match err.kind() {
                std::io::ErrorKind::NotFound => Ok(()),
                _ => Err(err.into()),
            }
        } else {
            Ok(())
        }
    }

    fn supports_set_partial(&self) -> bool {
        true
    }
}

impl ListableStorageTraits for FilesystemStore {
    fn list(&self) -> Result<StoreKeys, StorageError> {
        Ok(WalkDir::new(&self.base_path)
            .sort_by_file_name()
            .into_iter()
            .filter_map(std::result::Result::ok)
            .filter(|v| v.path().is_file())
            .filter_map(|v| self.fspath_to_key(v.path()).ok())
            .collect())
    }

    fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        Ok(WalkDir::new(self.prefix_to_fs_path(prefix))
            .sort_by_file_name()
            .into_iter()
            .filter_map(std::result::Result::ok)
            .filter(|v| v.path().is_file())
            .filter_map(|v| self.fspath_to_key(v.path()).ok())
            .collect())
    }

    fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        let prefix_path = self.prefix_to_fs_path(prefix);
        let mut keys: StoreKeys = vec![];
        let mut prefixes: StorePrefixes = vec![];
        let dir = std::fs::read_dir(prefix_path);
        if let Ok(dir) = dir {
            for entry in dir {
                let entry = entry?;
                let fs_path = entry.path();
                let path = fs_path.file_name().unwrap();
                if fs_path.is_dir() {
                    prefixes.push(StorePrefix::new(
                        prefix.as_str().to_string() + path.to_str().unwrap() + "/",
                    )?);
                } else {
                    keys.push(StoreKey::new(
                        prefix.as_str().to_owned() + path.to_str().unwrap(),
                    )?);
                }
            }
        }
        if self.sort {
            keys.sort();
            prefixes.sort();
        }

        Ok(StoreKeysPrefixes::new(keys, prefixes))
    }

    fn size(&self) -> Result<u64, StorageError> {
        Ok(WalkDir::new(&self.base_path)
            .into_iter()
            .filter_map(std::result::Result::ok)
            .filter_map(|v| {
                if v.path().is_file() {
                    Some(std::fs::metadata(v.path()).unwrap().len())
                } else {
                    None
                }
            })
            .sum())
    }

    fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        let mut size = 0;
        for key in self.list_prefix(prefix)? {
            if let Some(size_key) = self.size_key(&key)? {
                size += size_key;
            }
        }
        Ok(size)
    }
}

/// A filesystem store creation error.
#[derive(Debug, Error)]
pub enum FilesystemStoreCreateError {
    /// An IO error.
    #[error(transparent)]
    IOError(#[from] std::io::Error),
    /// The path is not valid on this system.
    #[error("base path {0} is not valid")]
    InvalidBasePath(PathBuf),
}

#[cfg(target_os = "linux")]
mod direct_io {
    /// A range of intersected pages, with `Ord` tailored for coalescing.
    #[derive(Eq, PartialEq, Debug)]
    pub(super) struct IntersectedPages(pub std::ops::Range<u64>);

    impl Ord for IntersectedPages {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            // Increasing order for start, decreasing order for end
            self.0
                .start
                .cmp(&other.0.start)
                .then_with(|| other.0.end.cmp(&self.0.end))
        }
    }

    impl PartialOrd for IntersectedPages {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    pub(super) fn coalesce_byte_ranges_with_page_size(
        file_size: u64,
        byte_ranges: &[zarrs_storage::byte_range::ByteRange],
        page_size: u64,
    ) -> impl Iterator<Item = IntersectedPages> {
        use itertools::Itertools;

        // Find intersected pages
        let intersected_pages: std::collections::BTreeSet<IntersectedPages> = byte_ranges
            .iter()
            .map(|range| {
                let start_page = range.start(file_size) / page_size;
                let end_page = range.end(file_size).div_ceil(page_size);
                IntersectedPages(start_page..end_page)
            })
            .collect();

        // Determine the pages to read (joining neighbouring pages)
        intersected_pages.into_iter().coalesce(|a, b| {
            if a.0.end >= b.0.start {
                Ok(IntersectedPages(a.0.start..b.0.end.max(a.0.end)))
            } else {
                Err((a, b))
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::direct_io::*;
    use zarrs_storage::byte_range::ByteRange;

    #[test]
    fn test_coalesce_byte_ranges_with_page_size() {
        let ps = 4;
        let file_size = 64;
        let byte_ranges = vec![
            ByteRange::FromStart(5, Some(2)),  // 1
            ByteRange::FromStart(0, Some(1)),  // 0
            ByteRange::FromStart(30, Some(4)), // 7-8
            ByteRange::Suffix(4),              // 15
            ByteRange::FromStart(8, Some(4)),  // 2
            ByteRange::FromStart(8, Some(8)),  // 2-3
            ByteRange::Suffix(7),              // 14-15
        ];
        let pages: Vec<_> =
            coalesce_byte_ranges_with_page_size(file_size, &byte_ranges, ps).collect();
        let expected = vec![
            IntersectedPages(0..4),
            IntersectedPages(7..9),
            IntersectedPages(14..16),
        ];
        assert_eq!(pages, expected);
    }
}
