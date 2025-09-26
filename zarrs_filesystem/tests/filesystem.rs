#![allow(missing_docs)]

use zarrs_filesystem::FilesystemStore;

use std::error::Error;

#[cfg(target_os = "linux")]
fn try_open_direct_io(path: &str) -> std::io::Result<std::fs::File> {
    use libc::{open, O_DIRECT, O_RDONLY};
    use std::os::fd::FromRawFd;

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

#[test]
#[cfg_attr(miri, ignore)]
fn filesystem() -> Result<(), Box<dyn Error>> {
    let path = tempfile::TempDir::new()?;
    let store = FilesystemStore::new(path.path())?.sorted();
    // let store = FilesystemStore::new("tests/data/store")?.sorted();
    zarrs_storage::store_test::store_write(&store)?;
    zarrs_storage::store_test::store_read(&store)?;
    zarrs_storage::store_test::store_list(&store)?;
    Ok(())
}

#[cfg(target_os = "linux")]
#[test]
// #[cfg_attr(miri, ignore)]
fn direct_io() -> Result<(), Box<dyn Error>> {
    use zarrs_filesystem::FilesystemStoreOptions;
    use zarrs_storage::{
        byte_range::ByteRange, Bytes, ReadableStorageTraits, WritableStorageTraits,
    };

    let tmpfile = tempfile::NamedTempFile::new()?;
    if try_open_direct_io(tmpfile.path().to_str().unwrap()).is_err() {
        // Skip this test if direct I/O is not supported
        return Ok(());
    }

    let path = tempfile::TempDir::new()?;
    let mut opts = FilesystemStoreOptions::default();
    opts.direct_io(true);

    let store = FilesystemStore::new_with_options(path.path(), opts)?.sorted();
    zarrs_storage::store_test::store_write(&store)?;
    zarrs_storage::store_test::store_read(&store)?;
    zarrs_storage::store_test::store_list(&store)?;

    // Test out fetching different kinds of non-page aligned reads against a larger file.
    let ps = page_size::get();
    let base_vec: Bytes = (0..(ps * 5) + 15)
        .map(|i| (i % 256) as u8)
        .collect::<Vec<u8>>()
        .into();
    let prefix: Bytes = base_vec.get(1..11).unwrap().to_owned().into(); // prefix
    let suffix: Bytes = base_vec
        .get((base_vec.len() - 1500)..)
        .unwrap()
        .to_owned()
        .into(); // suffix large enough to trigger large double ps, see comment
    let small_suffix: Bytes = base_vec.get((ps * 5)..).unwrap().to_owned().into(); // suffix to fit in one page
    let chunk: Bytes = base_vec.get(1..ps + 3).unwrap().to_owned().into(); // > ps request
    let chunk_2: Bytes = base_vec.get((5)..(5 + (ps * 2))).unwrap().to_owned().into(); // > 2 * ps request

    store.set(&"big_buff".try_into()?, base_vec.into())?;
    let expected = vec![prefix, suffix, chunk, chunk_2, small_suffix];
    let result = store
        .get_partial_many(
            &"big_buff".try_into()?,
            Box::new(
                [
                    ByteRange::FromStart(1, Some(10)),
                    ByteRange::Suffix(1500),
                    ByteRange::FromStart(1, Some((ps + 2) as u64)),
                    ByteRange::FromStart(5, Some((ps * 2) as u64)),
                    ByteRange::Suffix(15),
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
            "{}",
            format!(
                "errored with expected length {} and result length {}",
                e.len(),
                r.len()
            )
        )
    });
    Ok(())
}
