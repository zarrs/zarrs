#![allow(missing_docs)]

use std::error::Error;
use std::sync::Arc;

use zarrs_filesystem::{FilesystemStore, FilesystemStoreOptions};
use zarrs_storage::storage_adapter::atomic_write::AtomicWriteStorageAdapter;
use zarrs_storage::{
    Bytes, ListableStorageTraits, ReadableStorageTraits, StoreKey, WritableStorageTraits,
};

#[cfg(target_os = "linux")]
fn try_open_direct_io(path: &str) -> std::io::Result<std::fs::File> {
    use std::os::fd::FromRawFd;

    use libc::{open, O_DIRECT, O_RDONLY};

    let c_path = std::ffi::CString::new(path).unwrap();
    unsafe {
        let fd = open(c_path.as_ptr(), O_RDONLY | O_DIRECT);
        if fd < 0 {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(std::fs::File::from_raw_fd(fd))
        }
    }
}

#[cfg(target_os = "linux")]
fn create_direct_io_fs() -> Result<FilesystemStore, Box<dyn Error>> {
    use zarrs_filesystem::FilesystemStoreOptions;

    let path = tempfile::TempDir::new()?;
    let mut opts = FilesystemStoreOptions::default();
    opts.direct_io(true);

    Ok(FilesystemStore::new_with_options(path.path(), opts)?.sorted())
}

#[test]
#[cfg_attr(miri, ignore)]
fn filesystem() -> Result<(), Box<dyn Error>> {
    let path = tempfile::TempDir::new()?;
    let store = FilesystemStore::new(path.path())?.sorted();
    // let store = FilesystemStore::new("tests/data/store")?.sorted();
    zarrs_storage::store_test::store_write(&store)?;
    zarrs_storage::store_test::store_read(&store)?;
    zarrs_storage::store_test::store_list(&store)?;
    zarrs_storage::store_test::store_list_size(&store)?;
    Ok(())
}

#[test]
fn atomic_write_adapter() -> Result<(), Box<dyn Error>> {
    let path = tempfile::TempDir::new()?;
    let mut options = FilesystemStoreOptions::default();
    options.file_handle_cache_size(2);
    let store = Arc::new(FilesystemStore::new_with_options(path.path(), options)?.sorted());
    let store = AtomicWriteStorageAdapter::new(store);
    let key = "a/b".try_into()?;
    let temporary_key = AtomicWriteStorageAdapter::<FilesystemStore>::temporary_key(&key)?;

    store.set(&key, Bytes::from_static(b"first"))?;
    assert_eq!(store.get(&key)?, Some(Bytes::from_static(b"first")));
    assert!(!store.list()?.contains(&temporary_key));

    store.set(&key, Bytes::from_static(b"second"))?;
    assert_eq!(store.get(&key)?, Some(Bytes::from_static(b"second")));
    assert!(!store.list()?.contains(&temporary_key));

    store.set_partial(&key, 1, Bytes::from_static(b"X"))?;
    assert_eq!(store.get(&key)?, Some(Bytes::from_static(b"sXcond")));
    assert!(!store.list()?.contains(&temporary_key));

    store.set(&temporary_key, Bytes::from_static(b"incomplete"))?;
    let error = store.set(&key, Bytes::from_static(b"third")).unwrap_err();
    assert_eq!(
        error.to_string(),
        format!("temporary key {temporary_key} already exists")
    );
    assert_eq!(store.get(&key)?, Some(Bytes::from_static(b"sXcond")));
    assert_eq!(
        store.get(&temporary_key)?,
        Some(Bytes::from_static(b"incomplete"))
    );

    let error = store
        .set(&StoreKey::root(), Bytes::from_static(b"root"))
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "atomic writes do not support the root store key"
    );
    assert!(!path.path().join(".tmp").exists());
    Ok(())
}

#[test]
fn atomic_write_adapter_leaves_temporary_key_on_rename_failure() -> Result<(), Box<dyn Error>> {
    let path = tempfile::TempDir::new()?;
    let store = Arc::new(FilesystemStore::new(path.path())?.sorted());
    let adapter = AtomicWriteStorageAdapter::new(store.clone());
    let key = "a/b".try_into()?;
    let temporary_key = AtomicWriteStorageAdapter::<FilesystemStore>::temporary_key(&key)?;

    std::fs::create_dir_all(store.key_to_fspath(&key))?;

    assert!(adapter
        .set(&key, Bytes::from_static(b"replacement"))
        .is_err());
    assert!(store.key_to_fspath(&key).is_dir());
    assert_eq!(
        store.get(&temporary_key)?,
        Some(Bytes::from_static(b"replacement"))
    );
    assert!(store.list()?.contains(&temporary_key));
    Ok(())
}

#[cfg(target_os = "linux")]
#[test]
// #[cfg_attr(miri, ignore)]
fn direct_io_store_test() -> Result<(), Box<dyn Error>> {
    let tmpfile = tempfile::NamedTempFile::new()?;
    if try_open_direct_io(tmpfile.path().to_str().unwrap()).is_err() {
        // Skip this test if direct I/O is not supported
        return Ok(());
    }

    let store: FilesystemStore = create_direct_io_fs()?;
    zarrs_storage::store_test::store_write(&store)?;
    zarrs_storage::store_test::store_read(&store)?;
    zarrs_storage::store_test::store_list(&store)?;
    zarrs_storage::store_test::store_list_size(&store)?;
    Ok(())
}

#[test]
#[cfg_attr(miri, ignore)]
fn filesystem_handle_cache() -> Result<(), Box<dyn Error>> {
    use zarrs_filesystem::FilesystemStoreOptions;

    let path = tempfile::TempDir::new()?;
    let mut opts = FilesystemStoreOptions::default();
    opts.file_handle_cache_size(16);
    let store = FilesystemStore::new_with_options(path.path(), opts)?.sorted();
    zarrs_storage::store_test::store_write(&store)?;
    zarrs_storage::store_test::store_read(&store)?;
    zarrs_storage::store_test::store_list(&store)?;
    zarrs_storage::store_test::store_list_size(&store)?;
    Ok(())
}

#[test]
#[cfg_attr(miri, ignore)]
fn filesystem_handle_cache_invalidation() -> Result<(), Box<dyn Error>> {
    use zarrs_filesystem::FilesystemStoreOptions;
    use zarrs_storage::{ReadableStorageTraits, StoreKey, WritableStorageTraits};

    let path = tempfile::TempDir::new()?;
    let mut opts = FilesystemStoreOptions::default();
    opts.file_handle_cache_size(16);
    let store = FilesystemStore::new_with_options(path.path(), opts)?;

    let key: StoreKey = "a/b".try_into()?;
    store.set(&key, vec![0u8; 4].into())?;
    assert_eq!(store.get(&key)?.unwrap(), vec![0u8; 4]);

    // Overwrite with a different size; the cached handle must not serve stale bytes or size
    store.set(&key, vec![1u8; 8].into())?;
    assert_eq!(store.get(&key)?.unwrap(), vec![1u8; 8]);

    // Erase; the cached handle must not resurrect the key
    store.erase(&key)?;
    assert!(store.get(&key)?.is_none());

    // Erase prefix invalidates too
    store.set(&key, vec![2u8; 4].into())?;
    assert_eq!(store.get(&key)?.unwrap(), vec![2u8; 4]);
    store.erase_prefix(&"a/".try_into()?)?;
    assert!(store.get(&key)?.is_none());

    Ok(())
}

/// Prove the cached handle is actually reused: delete the file behind the store's back and check
/// that reads still succeed from the retained file handle (POSIX unlink semantics).
#[cfg(unix)]
#[test]
#[cfg_attr(miri, ignore)]
fn filesystem_handle_cache_reuse() -> Result<(), Box<dyn Error>> {
    use zarrs_filesystem::FilesystemStoreOptions;
    use zarrs_storage::{ReadableStorageTraits, StoreKey, WritableStorageTraits};

    let path = tempfile::TempDir::new()?;
    let mut opts = FilesystemStoreOptions::default();
    opts.file_handle_cache_size(16);
    let store = FilesystemStore::new_with_options(path.path(), opts)?;

    let key: StoreKey = "a/b".try_into()?;
    store.set(&key, vec![1u8; 4].into())?;
    assert_eq!(store.get(&key)?.unwrap(), vec![1u8; 4]);

    // Delete the file behind the store's back; a cached-handle read still succeeds
    std::fs::remove_file(store.key_to_fspath(&key))?;
    assert_eq!(store.get(&key)?.unwrap(), vec![1u8; 4]);

    // Erasing through the store drops the handle; the key is then gone
    store.erase(&key)?;
    assert!(store.get(&key)?.is_none());

    Ok(())
}

#[cfg(target_os = "linux")]
#[test]
fn direct_io_coalescing_test() -> Result<(), Box<dyn Error>> {
    use zarrs_storage::byte_range::ByteRange;
    use zarrs_storage::{Bytes, ReadableStorageTraits, WritableStorageTraits};
    let tmpfile = tempfile::NamedTempFile::new()?;
    if try_open_direct_io(tmpfile.path().to_str().unwrap()).is_err() {
        // Skip this test if direct I/O is not supported
        return Ok(());
    }

    let store: FilesystemStore = create_direct_io_fs()?;

    let ps = page_size::get();
    let base_vec: Bytes = (0..(ps * 10) + 15)
        .map(|i| (i % 256) as u8)
        .collect::<Vec<u8>>()
        .into();

    // Test out scenarios of page coalescing:
    // A disjoint prefix
    let prefix: Bytes = base_vec.get(1..11).unwrap().to_owned().into(); // prefix needing only first page

    // 2 fully overlapping ranges i.e., one contains the other
    let suffix: Bytes = base_vec
        .get((base_vec.len() - 1500)..)
        .unwrap()
        .to_owned()
        .into();
    let small_suffix: Bytes = base_vec
        .get((base_vec.len() - 15)..)
        .unwrap()
        .to_owned()
        .into(); // suffix fitting within one page

    // Consecutive chunks with no overlap but are contiguous i.e., pages 2..3 and 3..4
    let chunk_consecutive_1: Bytes = base_vec.get((ps * 2)..(ps * 3)).unwrap().to_owned().into();
    let chunk_consecutive_2: Bytes = base_vec.get((ps * 3)..(ps * 4)).unwrap().to_owned().into();

    // Partially overlapping chunks i.e., pages 5..7 and 6..8
    let chunk_overlap_1: Bytes = base_vec.get((ps * 6)..(ps * 8)).unwrap().to_owned().into();
    let chunk_overlap_2: Bytes = base_vec
        .get((ps * 6) - 1..(ps * 7))
        .unwrap()
        .to_owned()
        .into();

    store.set(&"big_buff".try_into()?, base_vec)?;
    // Mix up ordering of requests to ensure returned order is independent of the underlying coalescing operation
    let expected = vec![
        prefix,
        suffix,
        chunk_consecutive_1,
        small_suffix,
        chunk_overlap_1,
        chunk_consecutive_2,
        chunk_overlap_2,
    ];
    let result = store
        .get_partial_many(
            &"big_buff".try_into()?,
            Box::new(
                [
                    ByteRange::FromStart(1, Some(10)),
                    ByteRange::Suffix(1500),
                    ByteRange::FromStart(
                        (ps * 2).try_into().unwrap(),
                        Some(ps.try_into().unwrap()),
                    ),
                    ByteRange::Suffix(15),
                    ByteRange::FromStart(
                        (ps * 6).try_into().unwrap(),
                        Some((ps * 2).try_into().unwrap()),
                    ),
                    ByteRange::FromStart(
                        (ps * 3).try_into().unwrap(),
                        Some(ps.try_into().unwrap()),
                    ),
                    ByteRange::FromStart(
                        (ps * 6 - 1).try_into().unwrap(),
                        Some((ps + 1).try_into().unwrap()),
                    ),
                ]
                .into_iter(),
            ),
        )
        .unwrap()
        .unwrap()
        .collect::<Result<Vec<_>, _>>()?;
    expected.into_iter().zip(result).for_each(|(e, r)| {
        assert_eq!(
            e,
            r,
            "errored with expected length {} and result length {}",
            e.len(),
            r.len()
        );
    });
    Ok(())
}
