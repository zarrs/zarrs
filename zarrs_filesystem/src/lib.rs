//! A filesystem store for the [`zarrs`](https://docs.rs/zarrs/latest/zarrs/index.html) crate.
//!
//! This implementation is conformant with the filesystem store defined in the Zarr V3 specification: <https://zarr-specs.readthedocs.io/en/latest/v3/stores/filesystem/index.html>.
//!
//! ## Licence
//! `zarrs_filesystem` is licensed under either of
//! - the Apache License, Version 2.0 [LICENSE-APACHE](https://docs.rs/crate/zarrs_filesystem/latest/source/LICENCE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0> or
//! - the MIT license [LICENSE-MIT](https://docs.rs/crate/zarrs_filesystem/latest/source/LICENCE-MIT) or <http://opensource.org/licenses/MIT>, at your option.

use bytes::BytesMut;
use std::sync::RwLock;
use thiserror::Error;
use walkdir::WalkDir;
use zarrs_storage::byte_range::{ByteOffset, ByteRange, ByteRangeIterator, InvalidByteRangeError};
use zarrs_storage::{
    store_set_partial_many, Bytes, ListableStorageTraits, MaybeBytesIterator, OffsetBytesIterator,
    ReadableStorageTraits, StorageError, StoreKey, StoreKeyError, StoreKeys, StoreKeysPrefixes,
    StorePrefix, StorePrefixes, WritableStorageTraits,
};

#[cfg(target_os = "linux")]
mod direct_io;
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[cfg(target_os = "linux")]
use direct_io::{MetadataExt, OpenOptionsExt, O_DIRECT};

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

        Ok(Self {
            base_path,
            sort: false,
            options,
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
                buf.extend(std::iter::repeat_n(0, pad_size));

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
        use std::os::unix::fs::FileExt;

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
            let bytes_read = {
                let buf = buf.spare_capacity_mut();
                let buf = unsafe {
                    std::slice::from_raw_parts_mut(buf.as_mut_ptr().cast::<u8>(), length)
                };
                file.read_at(buf, offset as u64)?
            };
            unsafe {
                buf.set_len(bytes_read);
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
        let file_size = file.metadata()?.len();

        let out = byte_ranges
            .map(|byte_range| {
                // Seek
                match byte_range {
                    ByteRange::FromStart(offset, _) => file.seek(SeekFrom::Start(offset)),
                    ByteRange::Suffix(length) => {
                        file.seek(SeekFrom::End(-(i64::try_from(length).unwrap())))
                    }
                }?;

                // Get read length
                let length = match byte_range {
                    ByteRange::FromStart(offset, None) => {
                        file_size.checked_sub(offset).ok_or_else(|| {
                            StorageError::from(InvalidByteRangeError::new(byte_range, file_size))
                        })?
                    }
                    ByteRange::FromStart(_, Some(length)) | ByteRange::Suffix(length) => length,
                };
                let length = usize::try_from(length).unwrap();

                // Read
                let mut buffer = Vec::with_capacity(length);
                let spare = buffer.spare_capacity_mut();
                // SAFETY: We're reading into uninitialised memory, which is safe because
                // read_exact will fill all bytes or return an error
                let slice = unsafe {
                    std::slice::from_raw_parts_mut(spare.as_mut_ptr().cast::<u8>(), length)
                };
                file.read_exact(slice)?;
                // SAFETY: read_exact succeeded, so all bytes are now initialised
                unsafe {
                    buffer.set_len(length);
                }
                Ok(Bytes::from(buffer))
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
        Self::set_impl(self, key, &value, 0, true)
    }

    fn set_partial_many(
        &self,
        key: &StoreKey,
        offset_values: OffsetBytesIterator,
    ) -> Result<(), StorageError> {
        store_set_partial_many(self, key, offset_values)
    }

    fn erase(&self, key: &StoreKey) -> Result<(), StorageError> {
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
