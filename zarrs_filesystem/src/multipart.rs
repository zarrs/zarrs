//! The multipart writer of a [`FilesystemStore`].

use std::fs::{File, OpenOptions};
use std::io::{ErrorKind, IoSlice, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use zarrs_storage::{CowBytes, StorageError, StoreKey, StoreWriter};

use crate::FilesystemStore;

/// The suffix of the staging file of a multipart write.
pub(crate) const STAGING_SUFFIX: &str = ".zarrs-part";

/// Returns true if `file_name` is the name of the staging file of a multipart write.
pub(crate) fn is_staging_file(file_name: &std::ffi::OsStr) -> bool {
    file_name
        .to_str()
        .is_some_and(|name| name.starts_with('.') && name.ends_with(STAGING_SUFFIX))
}

/// The number of the next staging file, unique within this process.
static STAGING_NUMBER: AtomicU64 = AtomicU64::new(0);

/// The maximum number of parts written with one vectored write.
const MAX_SLICES: usize = 64;

/// A multipart writer of a [`FilesystemStore`].
///
/// The parts are appended to a staging file, `.{name}.{n}.zarrs-part`, next to the file of the key.
/// On finish, the staging file is synchronised and renamed to the file of the key, so the value is replaced
/// atomically. Aborting or dropping the writer removes the staging file.
pub(crate) struct FilesystemStoreWriter<'s> {
    store: &'s FilesystemStore,
    key: StoreKey,
    staging_path: PathBuf,
    /// The staging file, or [`None`] once the writer is finished or aborted.
    file: Option<File>,
}

impl<'s> FilesystemStoreWriter<'s> {
    pub(crate) fn new(store: &'s FilesystemStore, key: &StoreKey) -> Result<Self, StorageError> {
        let key_path = store.key_to_fspath(key);
        let (Some(parent), Some(name)) = (key_path.parent(), key_path.file_name()) else {
            return Err(StorageError::Other(format!(
                "the key {key} has no file for a multipart write"
            )));
        };
        let name = name.to_string_lossy();
        if !parent.exists() {
            std::fs::create_dir_all(parent)?;
        }
        loop {
            let number = STAGING_NUMBER.fetch_add(1, Ordering::Relaxed);
            let staging_path = parent.join(format!(".{name}.{number}{STAGING_SUFFIX}"));
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&staging_path)
            {
                Ok(file) => {
                    return Ok(Self {
                        store,
                        key: key.clone(),
                        staging_path,
                        file: Some(file),
                    });
                }
                // Left by another process, or an interrupted write
                Err(err) if err.kind() == ErrorKind::AlreadyExists => {}
                Err(err) => return Err(err.into()),
            }
        }
    }

    fn file(&mut self) -> &mut File {
        self.file
            .as_mut()
            .expect("the file is open until the writer is consumed")
    }

    /// Remove the staging file.
    fn remove(&mut self) -> std::io::Result<()> {
        if self.file.take().is_some() {
            std::fs::remove_file(&self.staging_path)
        } else {
            Ok(())
        }
    }
}

/// Write all of `slices` to `file`, in order.
fn write_all_vectored(file: &mut File, mut slices: &mut [IoSlice<'_>]) -> std::io::Result<()> {
    while !slices.is_empty() {
        match file.write_vectored(slices) {
            Ok(0) => return Err(ErrorKind::WriteZero.into()),
            Ok(written) => IoSlice::advance_slices(&mut slices, written),
            Err(err) if err.kind() == ErrorKind::Interrupted => {}
            Err(err) => return Err(err),
        }
    }
    Ok(())
}

impl StoreWriter for FilesystemStoreWriter<'_> {
    fn write(&mut self, part: CowBytes<'_>) -> Result<(), StorageError> {
        self.file().write_all(&part)?;
        Ok(())
    }

    fn write_many(&mut self, parts: Vec<CowBytes<'_>>) -> Result<(), StorageError> {
        let file = self.file();
        let mut slices = [IoSlice::new(&[]); MAX_SLICES];
        let mut parts = parts.iter().filter(|part| !part.is_empty()).peekable();
        while parts.peek().is_some() {
            let mut len = 0;
            for (slice, part) in slices.iter_mut().zip(parts.by_ref()) {
                *slice = IoSlice::new(part);
                len += 1;
            }
            write_all_vectored(file, &mut slices[..len])?;
        }
        Ok(())
    }

    fn finish(mut self: Box<Self>) -> Result<(), StorageError> {
        let result = (|| {
            // Surface write-back errors before dropping the file, which ignores close errors.
            self.file().sync_all()?;
            self.store.invalidate_handle(&self.key);
            std::fs::rename(&self.staging_path, self.store.key_to_fspath(&self.key))
        })();
        match result {
            Ok(()) => {
                self.file = None;
                Ok(())
            }
            Err(err) => {
                // Dropping the writer removes the staging file.
                Err(err.into())
            }
        }
    }

    fn abort(mut self: Box<Self>) -> Result<(), StorageError> {
        Ok(self.remove()?)
    }
}

impl Drop for FilesystemStoreWriter<'_> {
    fn drop(&mut self) {
        let _ = self.remove();
    }
}
