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

#[cfg(target_os = "linux")]
#[test]
#[cfg_attr(miri, ignore)]
fn filesystem_sync_errors() -> Result<(), Box<dyn Error>> {
    use std::io::Write;
    use std::os::unix::fs::symlink;

    use zarrs_storage::StorageError;

    // /dev/null accepts writes but rejects fsync, so a failure must come from syncing.
    let mut file = std::fs::OpenOptions::new().write(true).open("/dev/null")?;
    file.write_all(b"value")?;
    assert_eq!(
        file.sync_all().unwrap_err().raw_os_error(),
        Some(libc::EINVAL)
    );

    let assert_sync_error = |result: Result<(), StorageError>| match result.unwrap_err() {
        StorageError::IOError(error) => {
            assert_eq!(error.raw_os_error(), Some(libc::EINVAL));
        }
        error => panic!("expected a sync I/O error, got {error}"),
    };

    for cache_size in [0, 16] {
        let path = tempfile::TempDir::new()?;
        let mut options = FilesystemStoreOptions::default();
        options.file_handle_cache_size(cache_size);
        let store = FilesystemStore::new_with_options(path.path(), options)?;
        let key: StoreKey = "null".try_into()?;
        symlink("/dev/null", store.key_to_fspath(&key))?;

        assert_sync_error(store.set(&key, Bytes::from_static(b"value").into()));
        assert_sync_error(store.set(&key, Bytes::new().into()));

        // Partial writes go to the key in place (no read of it first), so they reach
        // /dev/null directly and must surface its sync error too.
        let partial_key: StoreKey = "partial".try_into()?;
        symlink("/dev/null", store.key_to_fspath(&partial_key))?;
        let offset_values = [
            (0, Bytes::from_static(b"first").into()),
            (8, Bytes::from_static(b"last").into()),
        ];
        assert_sync_error(
            store.set_partial_many(&partial_key, Box::new(offset_values.into_iter())),
        );
    }
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

    store.set(&key, Bytes::from_static(b"first").into())?;
    assert_eq!(store.get(&key)?, Some(Bytes::from_static(b"first")));
    assert!(!store.list()?.contains(&temporary_key));

    store.set(&key, Bytes::from_static(b"second").into())?;
    assert_eq!(store.get(&key)?, Some(Bytes::from_static(b"second")));
    assert!(!store.list()?.contains(&temporary_key));

    store.set_partial(&key, 1, Bytes::from_static(b"X").into())?;
    assert_eq!(store.get(&key)?, Some(Bytes::from_static(b"sXcond")));
    assert!(!store.list()?.contains(&temporary_key));

    store.set(&temporary_key, Bytes::from_static(b"incomplete").into())?;
    let error = store
        .set(&key, Bytes::from_static(b"third").into())
        .unwrap_err();
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
        .set(&StoreKey::root(), Bytes::from_static(b"root").into())
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
        .set(&key, Bytes::from_static(b"replacement").into())
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

fn get_partial_out_of_range(store: &FilesystemStore) -> Result<(), Box<dyn Error>> {
    use zarrs_storage::byte_range::ByteRange;

    let key = StoreKey::new("a")?;
    store.set(&key, vec![0, 1, 2].into())?;
    for byte_range in [
        ByteRange::Suffix(4),
        ByteRange::FromStart(4, None),
        ByteRange::FromStart(2, Some(u64::MAX)),
        ByteRange::FromStart(u64::MAX, Some(2)),
    ] {
        assert!(store.get_partial(&key, byte_range).is_err());
    }
    assert_eq!(
        store.get_partial(&key, ByteRange::Suffix(3))?,
        Some(vec![0, 1, 2].into())
    );
    Ok(())
}

#[test]
#[cfg_attr(miri, ignore)]
fn filesystem_get_partial_out_of_range() -> Result<(), Box<dyn Error>> {
    let path = tempfile::TempDir::new()?;
    get_partial_out_of_range(&FilesystemStore::new(path.path())?)
}

#[cfg(target_os = "linux")]
#[test]
#[cfg_attr(miri, ignore)]
fn direct_io_get_partial_out_of_range() -> Result<(), Box<dyn Error>> {
    let tmpfile = tempfile::NamedTempFile::new()?;
    if try_open_direct_io(tmpfile.path().to_str().unwrap()).is_err() {
        // Skip this test if direct I/O is not supported
        return Ok(());
    }
    get_partial_out_of_range(&create_direct_io_fs()?)
}

#[test]
#[cfg_attr(miri, ignore)]
fn filesystem_set_partial_in_place() -> Result<(), Box<dyn Error>> {
    // A partial write updates the file in place: the rest is kept, a gap past the end is
    // zero-filled, and a missing key is created. With a handle cache, a later read sees it.
    for cache_size in [0, 16] {
        let path = tempfile::TempDir::new()?;
        let mut opts = FilesystemStoreOptions::default();
        opts.file_handle_cache_size(cache_size);
        let store = FilesystemStore::new_with_options(path.path(), opts)?;

        let key: StoreKey = "a/b".try_into()?;
        store.set(&key, vec![9u8; 8].into())?;
        assert_eq!(store.get(&key)?.unwrap(), Bytes::from(vec![9u8; 8])); // cache the handle
        store.set_partial(&key, 2, vec![1, 2].into())?;
        store.set_partial(&key, 10, vec![3].into())?;
        assert_eq!(
            store.get(&key)?.unwrap(),
            Bytes::from(vec![9, 9, 1, 2, 9, 9, 9, 9, 0, 0, 3])
        );

        let missing: StoreKey = "c/d".try_into()?;
        store.set_partial(&missing, 3, vec![7].into())?;
        assert_eq!(store.get(&missing)?.unwrap(), Bytes::from(vec![0, 0, 0, 7]));
    }
    Ok(())
}

#[cfg(unix)]
#[test]
#[cfg_attr(miri, ignore)]
fn filesystem_set_partial_does_not_read_the_file() -> Result<(), Box<dyn Error>> {
    // A partial write must not read the value back (e.g. to rewrite it whole): appending an
    // inner chunk to a shard would then cost the whole shard. A write-only file proves it.
    use std::os::unix::fs::PermissionsExt;

    let path = tempfile::TempDir::new()?;
    let store = FilesystemStore::new(path.path())?;
    let key: StoreKey = "shard".try_into()?;
    let head: Vec<u8> = (0..1_000_000u32).map(|i| (i % 251) as u8).collect();
    store.set(&key, head.clone().into())?;
    let file = path.path().join("shard");
    std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o200))?;
    store.set_partial(&key, 1_000_000, vec![1u8; 4096].into())?;
    std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o600))?;

    let bytes = store.get(&key)?.unwrap();
    assert_eq!(bytes.len(), 1_004_096);
    assert_eq!(&bytes[..1_000_000], &head[..]);
    assert!(bytes[1_000_000..].iter().all(|&b| b == 1));
    Ok(())
}

#[cfg(target_os = "linux")]
#[test]
#[cfg_attr(miri, ignore)]
fn direct_io_set_partial_keeps_the_file_length() -> Result<(), Box<dyn Error>> {
    let tmpfile = tempfile::NamedTempFile::new()?;
    if try_open_direct_io(tmpfile.path().to_str().unwrap()).is_err() {
        // Skip this test if direct I/O is not supported
        return Ok(());
    }

    // The direct I/O path of `set` truncates to the written length; a partial write at offset
    // 0 must not take it.
    let store: FilesystemStore = create_direct_io_fs()?;
    let key: StoreKey = "a".try_into()?;
    store.set(&key, vec![5u8; 8192].into())?;
    store.set_partial(&key, 0, vec![1, 2].into())?;
    let bytes = store.get(&key)?.unwrap();
    assert_eq!(bytes.len(), 8192);
    assert_eq!(&bytes[..2], &[1, 2]);
    assert!(bytes[2..].iter().all(|&b| b == 5));
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

    // A range batch must invalidate both cached contents and the cached size.
    store.set_partial_many(
        &key,
        Box::new(
            [
                (1, Bytes::from_static(b"XY").into()),
                (10, Bytes::from_static(b"Z").into()),
            ]
            .into_iter(),
        ),
    )?;
    assert_eq!(
        store.get(&key)?.unwrap(),
        vec![1, b'X', b'Y', 1, 1, 1, 1, 1, 0, 0, b'Z']
    );
    assert_eq!(store.size_key(&key)?, Some(11));

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

    store.set(&"big_buff".try_into()?, base_vec.into())?;
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

/// The names of the files in `dir` and its subdirectories.
fn files_on_disk(dir: &std::path::Path) -> Vec<String> {
    let mut files: Vec<_> = walkdir_files(dir)
        .into_iter()
        .map(|path| {
            path.strip_prefix(dir)
                .unwrap()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    files.sort();
    files
}

fn walkdir_files(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut files = vec![];
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            files.extend(walkdir_files(&path));
        } else {
            files.push(path);
        }
    }
    files
}

#[test]
#[cfg_attr(miri, ignore)]
fn filesystem_multipart() -> Result<(), Box<dyn Error>> {
    let path = tempfile::TempDir::new()?;
    let store = FilesystemStore::new(path.path())?.sorted();
    assert!(store.supports_multipart());
    zarrs_storage::store_test::store_multipart(&store)?;
    zarrs_storage::store_test::store_multipart_patch(&store)?;
    assert_eq!(files_on_disk(path.path()), Vec::<String>::new());
    Ok(())
}

#[test]
#[cfg_attr(miri, ignore)]
fn filesystem_multipart_staging_file() -> Result<(), Box<dyn Error>> {
    let path = tempfile::TempDir::new()?;
    let mut options = FilesystemStoreOptions::default();
    options.file_handle_cache_size(4);
    let store = FilesystemStore::new_with_options(path.path(), options)?.sorted();
    let key: StoreKey = "a/b".try_into()?;
    store.set(&key, Bytes::from_static(b"old").into())?;
    assert_eq!(store.get(&key)?, Some(Bytes::from_static(b"old")));

    // Many parts, more than one vectored write, including empty parts
    let parts: Vec<Vec<u8>> = (0..200u8).map(|i| vec![i; usize::from(i % 3)]).collect();
    let expected: Vec<u8> = parts.concat();
    let mut writer = store.set_multipart(&key, None)?;
    writer.write_many(parts.iter().map(Into::into).collect())?;
    writer.write(Bytes::from_static(b"end").into())?;

    // The staging file is hidden from listing, and the old value is intact
    let files = files_on_disk(path.path());
    assert_eq!(files.len(), 2);
    assert!(files
        .iter()
        .any(|file| file.starts_with("a/.b.") && file.ends_with(".zarrs-part")));
    assert_eq!(store.list()?, vec![key.clone()]);
    assert_eq!(store.list_prefix(&"a/".try_into()?)?, vec![key.clone()]);
    assert_eq!(
        store.list_dir(&"a/".try_into()?)?.keys(),
        &vec![key.clone()]
    );
    assert_eq!(store.size()?, 3);
    assert_eq!(store.get(&key)?, Some(Bytes::from_static(b"old")));

    writer.finish()?;
    assert_eq!(files_on_disk(path.path()), vec!["a/b".to_string()]);
    assert_eq!(
        store.get(&key)?,
        Some(Bytes::from([expected, b"end".to_vec()].concat()))
    );
    Ok(())
}

#[test]
#[cfg_attr(miri, ignore)]
fn filesystem_multipart_finish_failure_removes_the_staging_file() -> Result<(), Box<dyn Error>> {
    let path = tempfile::TempDir::new()?;
    let store = FilesystemStore::new(path.path())?.sorted();
    let key: StoreKey = "a/b".try_into()?;
    std::fs::create_dir_all(store.key_to_fspath(&key).join("c"))?;
    let mut writer = store.set_multipart(&key, None)?;
    writer.write(Bytes::from_static(b"value").into())?;
    assert!(writer.finish().is_err());
    assert_eq!(files_on_disk(path.path()), Vec::<String>::new());
    Ok(())
}

#[cfg(target_os = "linux")]
#[test]
#[cfg_attr(miri, ignore)]
fn direct_io_multipart() -> Result<(), Box<dyn Error>> {
    let store = create_direct_io_fs()?;
    assert!(!store.supports_multipart());
    zarrs_storage::store_test::store_multipart(&store)?;
    Ok(())
}

#[test]
#[cfg_attr(miri, ignore)]
fn atomic_write_adapter_multipart() -> Result<(), Box<dyn Error>> {
    let path = tempfile::TempDir::new()?;
    let store = Arc::new(FilesystemStore::new(path.path())?.sorted());
    let store = AtomicWriteStorageAdapter::new(store);
    assert!(store.supports_multipart_patch());
    zarrs_storage::store_test::store_multipart(&store)?;
    zarrs_storage::store_test::store_multipart_patch(&store)?;
    assert_eq!(files_on_disk(path.path()), Vec::<String>::new());
    Ok(())
}

#[test]
#[cfg_attr(miri, ignore)]
fn filesystem_multipart_patch() -> Result<(), Box<dyn Error>> {
    let path = tempfile::TempDir::new()?;
    let store = FilesystemStore::new(path.path())?.sorted();
    assert!(store.supports_multipart_patch());
    let key: StoreKey = "a".try_into()?;
    let mut writer = store.set_multipart(&key, None)?;
    // A header written last
    writer.write(Bytes::from_static(&[0, 0]).into())?;
    writer.write_many(vec![
        Bytes::from_static(b"bc").into(),
        Bytes::from_static(b"d").into(),
    ])?;
    writer.patch(0, Bytes::from_static(b"HH").into())?;
    writer.patch(3, Bytes::from_static(b"C").into())?;
    assert!(writer.patch(4, Bytes::from_static(b"XX").into()).is_err());
    writer.finish()?;
    assert_eq!(store.get(&key)?, Some(Bytes::from_static(b"HHbCd")));
    Ok(())
}
