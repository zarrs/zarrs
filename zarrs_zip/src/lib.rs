//! A storage adapter for `zip` files for the [`zarrs`](https://docs.rs/zarrs/latest/zarrs/index.html) crate.
//!
//! ```
//! # use std::path::PathBuf;
//! # use std::sync::Arc;
//! use zarrs_storage::StoreKey;
//! use zarrs_filesystem::FilesystemStore;
//! use zarrs_zip::ZipStorageAdapter;
//!
//! let fs_root = PathBuf::from("/path/to/a/directory");
//! # let fs_root = PathBuf::from("tests/");
//! let fs_store = Arc::new(FilesystemStore::new(&fs_root)?);
//! let zip_key = StoreKey::new("zarr.zip")?;
//! let zip_store = Arc::new(ZipStorageAdapter::new(fs_store, zip_key)?);
//! # Ok::<_, Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Licence
//! `zarrs_zip` is licensed under either of
//! - the Apache License, Version 2.0 [LICENSE-APACHE](https://docs.rs/crate/zarrs_zip/latest/source/LICENCE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0> or
//! - the MIT license [LICENSE-MIT](https://docs.rs/crate/zarrs_zip/latest/source/LICENCE-MIT) or <http://opensource.org/licenses/MIT>, at your option.

use zarrs_storage::{
    byte_range::{extract_byte_ranges_read_seek, ByteRangeIterator},
    Bytes, ListableStorageTraits, MaybeBytesIterator, ReadableStorageTraits, StorageError,
    StorageValueIO, StoreKey, StoreKeys, StoreKeysPrefixes, StorePrefix, StorePrefixes,
};

use itertools::Itertools;
use std::sync::Mutex;
use thiserror::Error;
use zip::{result::ZipError, ZipArchive};

use std::{path::PathBuf, sync::Arc};

/// A zip storage adapter.
pub struct ZipStorageAdapter<TStorage: ?Sized> {
    size: u64,
    zip_archive: Mutex<ZipArchive<StorageValueIO<TStorage>>>,
    zip_path: PathBuf,
}

impl<TStorage: ?Sized + ReadableStorageTraits> ZipStorageAdapter<TStorage> {
    /// Create a new zip storage adapter.
    ///
    /// # Errors
    /// Returns a [`ZipStorageAdapterCreateError`] if the store value at `key` is not a valid zip file.
    pub fn new(
        storage: Arc<TStorage>,
        key: StoreKey,
    ) -> Result<Self, ZipStorageAdapterCreateError> {
        Self::new_with_path(storage, key, "")
    }

    /// Create a new zip storage adapter to `path` within the zip file.
    ///
    /// # Errors
    /// Returns a [`ZipStorageAdapterCreateError`] if the store value at `key` is not a valid zip file.
    pub fn new_with_path<T: Into<PathBuf>>(
        storage: Arc<TStorage>,
        key: StoreKey,
        path: T,
    ) -> Result<Self, ZipStorageAdapterCreateError> {
        let zip_path = path.into();
        let size = storage
            .size_key(&key)?
            .ok_or::<ZipStorageAdapterCreateError>(
                StorageError::UnknownKeySize(key.clone()).into(),
            )?;
        let storage_io = StorageValueIO::new(storage, key, size);
        let zip_archive = Mutex::new(
            ZipArchive::new(storage_io)
                .map_err(|err| ZipStorageAdapterCreateError::ZipError(err.to_string()))?,
        );
        Ok(Self {
            size,
            zip_archive,
            zip_path,
        })
    }

    fn key_str_to_zip_path(&self, key: &str) -> String {
        let mut zip_name = self.zip_path.clone();
        zip_name.push(key);

        let mut zip_name = zip_name.to_string_lossy();
        if cfg!(windows) {
            zip_name = zip_name.replace('\\', "/").into();
        }
        zip_name.to_string()
    }

    fn get_impl<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        let mut zip_archive = self.zip_archive.lock().unwrap();
        let mut file = {
            let zip_file = zip_archive.by_name_seek(&self.key_str_to_zip_path(key.as_str()));
            match zip_file {
                Ok(zip_file) => zip_file,
                Err(err) => match err {
                    ZipError::FileNotFound => return Ok(None),
                    _ => return Err(StorageError::Other(err.to_string())),
                },
            }
        };

        let out = Box::new(
            extract_byte_ranges_read_seek(&mut file, byte_ranges)?
                .into_iter()
                .map(|b| Ok(Bytes::from(b))),
        );
        Ok(Some(out))
    }

    fn zip_file_strip_prefix<'a>(&self, name: &'a str) -> Option<&'a str> {
        name.strip_prefix(self.zip_path.to_str().unwrap())
            .filter(|&name| !name.is_empty())
    }
}

impl<TStorage: ?Sized + ReadableStorageTraits> ReadableStorageTraits
    for ZipStorageAdapter<TStorage>
{
    fn get_partial_values_key<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        self.get_impl(key, byte_ranges)
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        let mut zip_archive = self.zip_archive.lock().unwrap();
        let file = zip_archive.by_name(key.as_str());
        match file {
            Ok(file) => Ok(Some(file.compressed_size())),
            Err(err) => match err {
                ZipError::FileNotFound => Ok(None),
                _ => Err(StorageError::Other(err.to_string())),
            },
        }
    }
}

impl<TStorage: ?Sized + ReadableStorageTraits> ListableStorageTraits
    for ZipStorageAdapter<TStorage>
{
    fn list(&self) -> Result<StoreKeys, StorageError> {
        Ok(self
            .zip_archive
            .lock()
            .unwrap()
            .file_names()
            .filter_map(|name| self.zip_file_strip_prefix(name))
            .filter_map(|v| StoreKey::try_from(v).ok())
            .sorted()
            .collect())
    }

    fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        let mut zip_archive = self.zip_archive.lock().unwrap();
        let file_names: Vec<String> = zip_archive
            .file_names()
            .filter_map(|name| self.zip_file_strip_prefix(name))
            .map(std::string::ToString::to_string)
            .collect();
        Ok(file_names
            .into_iter()
            .filter_map(|name| {
                if name.starts_with(prefix.as_str()) {
                    if let Ok(file) = zip_archive.by_name(&self.key_str_to_zip_path(&name)) {
                        if file.is_file() {
                            let name = name.strip_suffix('/').unwrap_or(&name);
                            if let Ok(store_key) = StoreKey::try_from(name) {
                                return Some(store_key);
                            }
                        }
                    }
                }
                None
            })
            .sorted()
            .collect())
    }

    fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        let zip_archive = self.zip_archive.lock().unwrap();
        let mut keys: StoreKeys = vec![];
        let mut prefixes: StorePrefixes = vec![];
        for name in zip_archive
            .file_names()
            .filter_map(|name| self.zip_file_strip_prefix(name))
        {
            if name.starts_with(prefix.as_str()) {
                if name.ends_with('/') {
                    if let Ok(store_prefix) = StorePrefix::try_from(name) {
                        if let Some(parent) = store_prefix.parent() {
                            if &parent == prefix {
                                prefixes.push(store_prefix);
                            }
                        }
                    }
                } else if let Ok(store_key) = StoreKey::try_from(name) {
                    let parent = store_key.parent();
                    if &parent == prefix {
                        keys.push(store_key);
                    }
                }
            }
        }
        keys.sort();
        prefixes.sort();

        Ok(StoreKeysPrefixes::new(keys, prefixes))
    }

    fn size(&self) -> Result<u64, StorageError> {
        Ok(self.size)
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

/// A zip store creation error.
#[derive(Debug, Error)]
pub enum ZipStorageAdapterCreateError {
    /// An IO error.
    #[error(transparent)]
    IOError(#[from] std::io::Error),
    /// An existing directory.
    #[error("{0} is an existing directory, not a zip file")]
    ExistingDir(PathBuf),
    /// A zip error.
    #[error("{0}")]
    ZipError(String),
    /// A storage error.
    #[error(transparent)]
    StorageError(#[from] StorageError),
}

#[cfg(test)]
mod tests {
    use walkdir::WalkDir;

    use zarrs_filesystem::FilesystemStore;
    use zarrs_storage::WritableStorageTraits;

    use super::*;
    use std::{
        error::Error,
        fs::File,
        io::{Read, Write},
        path::Path,
    };

    // https://github.com/zip-rs/zip/blob/master/examples/write_dir.rs
    fn zip_dir<I: Iterator<Item = walkdir::DirEntry>>(
        it: I,
        prefix: &str,
        writer: File,
        method: zip::CompressionMethod,
    ) -> zip::result::ZipResult<()> {
        let mut zip = zip::ZipWriter::new(writer);
        let options = zip::write::SimpleFileOptions::default().compression_method(method);
        let mut buffer = Vec::new();
        for entry in it {
            let path = entry.path();
            let name = path.strip_prefix(Path::new(prefix)).unwrap();
            if path.is_file() {
                #[allow(deprecated)]
                zip.start_file_from_path(name, options)?;
                let mut f = File::open(path)?;
                f.read_to_end(&mut buffer)?;
                zip.write_all(&buffer)?;
                buffer.clear();
            } else if !name.as_os_str().is_empty() {
                #[allow(deprecated)]
                zip.add_directory_from_path(name, options)?;
            }
        }
        zip.finish()?;
        Result::Ok(())
    }

    fn zip_write(path: &Path) -> Result<(), Box<dyn Error>> {
        let tmp_path = tempfile::TempDir::new()?;
        let tmp_path = tmp_path.path();
        let store = FilesystemStore::new(tmp_path)?.sorted();
        store.set(&"a/b/zarr.json".try_into()?, vec![0, 1, 2, 3].into())?;
        store.set(&"a/c/zarr.json".try_into()?, vec![].into())?;
        store.set(&"a/d/e/zarr.json".try_into()?, vec![].into())?;
        store.set(&"a/f/g/zarr.json".try_into()?, vec![].into())?;
        store.set(&"a/f/h/zarr.json".try_into()?, vec![].into())?;
        store.set(&"b/zarr.json".try_into()?, vec![].into())?;
        store.set(&"b/c/d/zarr.json".try_into()?, vec![].into())?;
        store.set(&"c/zarr.json".try_into()?, vec![].into())?;

        let walkdir = WalkDir::new(tmp_path);

        let file = File::create(path).unwrap();
        zip_dir(
            &mut walkdir.into_iter().filter_map(std::result::Result::ok),
            tmp_path.to_str().unwrap(),
            file,
            zip::CompressionMethod::Stored,
        )?;

        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn zip_root() -> Result<(), Box<dyn Error>> {
        let path = tempfile::TempDir::new()?;
        let mut path = path.path().to_path_buf();
        let store = FilesystemStore::new(path.clone())?;

        path.push("test.zip");
        zip_write(&path)?;

        let store = Arc::new(ZipStorageAdapter::new(
            store.into(),
            StoreKey::new("test.zip")?,
        )?);

        assert_eq!(
            store.list()?,
            &[
                "a/b/zarr.json".try_into()?,
                "a/c/zarr.json".try_into()?,
                "a/d/e/zarr.json".try_into()?,
                "a/f/g/zarr.json".try_into()?,
                "a/f/h/zarr.json".try_into()?,
                "b/c/d/zarr.json".try_into()?,
                "b/zarr.json".try_into()?,
                "c/zarr.json".try_into()?,
            ]
        );
        assert_eq!(
            store.list_prefix(&"a/".try_into()?)?,
            &[
                "a/b/zarr.json".try_into()?,
                "a/c/zarr.json".try_into()?,
                "a/d/e/zarr.json".try_into()?,
                "a/f/g/zarr.json".try_into()?,
                "a/f/h/zarr.json".try_into()?,
            ]
        );
        assert_eq!(
            store.list_prefix(&"a/d/".try_into()?)?,
            &["a/d/e/zarr.json".try_into()?]
        );
        assert_eq!(
            store.list_prefix(&"".try_into()?)?,
            &[
                "a/b/zarr.json".try_into()?,
                "a/c/zarr.json".try_into()?,
                "a/d/e/zarr.json".try_into()?,
                "a/f/g/zarr.json".try_into()?,
                "a/f/h/zarr.json".try_into()?,
                "b/c/d/zarr.json".try_into()?,
                "b/zarr.json".try_into()?,
                "c/zarr.json".try_into()?,
            ]
        );

        let list = store.list_dir(&"a/".try_into()?)?;
        assert_eq!(list.keys(), &[]);
        assert_eq!(
            list.prefixes(),
            &[
                "a/b/".try_into()?,
                "a/c/".try_into()?,
                "a/d/".try_into()?,
                "a/f/".try_into()?,
            ]
        );

        assert_eq!(
            store.get(&"a/b/zarr.json".try_into()?)?.unwrap(),
            vec![0, 1, 2, 3]
        );
        assert_eq!(
            store.get(&"a/c/zarr.json".try_into()?)?.unwrap(),
            Vec::<u8>::new().as_slice()
        );

        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn zip_path() -> Result<(), Box<dyn Error>> {
        let path = tempfile::TempDir::new()?;
        let mut path = path.path().to_path_buf();
        let store = FilesystemStore::new(path.clone())?;
        path.push("test.zip");
        zip_write(&path)?;

        let store = Arc::new(ZipStorageAdapter::new_with_path(
            store.into(),
            StoreKey::new("test.zip")?,
            "a/",
        )?);

        assert_eq!(
            store.list()?,
            &[
                "b/zarr.json".try_into()?,
                "c/zarr.json".try_into()?,
                "d/e/zarr.json".try_into()?,
                "f/g/zarr.json".try_into()?,
                "f/h/zarr.json".try_into()?,
            ]
        );
        assert_eq!(store.list_prefix(&"a/".try_into()?)?, &[]);
        assert_eq!(
            store.list_prefix(&"d/".try_into()?)?,
            &["d/e/zarr.json".try_into()?]
        );
        assert_eq!(
            store.list_prefix(&"".try_into()?)?,
            &[
                "b/zarr.json".try_into()?,
                "c/zarr.json".try_into()?,
                "d/e/zarr.json".try_into()?,
                "f/g/zarr.json".try_into()?,
                "f/h/zarr.json".try_into()?,
            ]
        );

        let list = store.list_dir(&"".try_into()?)?;
        assert_eq!(list.keys(), &[]);
        assert_eq!(
            list.prefixes(),
            &[
                "b/".try_into()?,
                "c/".try_into()?,
                "d/".try_into()?,
                "f/".try_into()?,
            ]
        );

        assert_eq!(
            store.get(&"b/zarr.json".try_into()?)?.unwrap(),
            vec![0, 1, 2, 3]
        );
        // assert_eq!(store.get(&"c/zarr.json".try_into()?)?, Vec::<u8>::new().as_slice());

        Ok(())
    }
}
