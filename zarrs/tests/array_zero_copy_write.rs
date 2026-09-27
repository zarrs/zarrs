#![allow(missing_docs)]

//! Tests that a chunk write reuses the caller's buffer when the codec chain is pass-through.
//!
//! Pointer equality is a sound proof of zero copy here: `Bytes::slice_ref` preserves the data
//! pointer, and a fresh allocation cannot share an address with a buffer that is still alive.

use std::collections::BTreeMap;
use std::error::Error;
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use zarrs::array::codec::BytesCodec;
use zarrs::array::{Array, ArrayBuilder, ArrayBytes, CowBytes, data_type};
use zarrs::storage::store::MemoryStore;
use zarrs::storage::{
    ListableStorageTraits, MaybeBytes, MaybeBytesIterator, OffsetBytesIterator,
    ReadableStorageTraits, StorageError, StoreKey, StoreKeys, StoreKeysPrefixes, StorePrefix,
    WritableStorageTraits,
};
use zarrs_storage::byte_range::ByteRangeIterator;

type TestResult = Result<(), Box<dyn Error>>;

/// A store that records the address and length of every value passed to `set`.
#[derive(Debug, Default)]
struct PtrRecordingStore {
    inner: MemoryStore,
    ptrs: Mutex<BTreeMap<StoreKey, (usize, usize)>>,
}

impl PtrRecordingStore {
    fn recorded(&self, key: &str) -> Option<(usize, usize)> {
        let key = StoreKey::new(key).unwrap();
        self.ptrs.lock().unwrap().get(&key).copied()
    }
}

impl WritableStorageTraits for PtrRecordingStore {
    fn set(&self, key: &StoreKey, value: CowBytes<'_>) -> Result<(), StorageError> {
        self.ptrs
            .lock()
            .unwrap()
            .insert(key.clone(), (value.as_ptr() as usize, value.len()));
        self.inner.set(key, value)
    }

    fn set_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        offset_values: OffsetBytesIterator<'a>,
    ) -> Result<(), StorageError> {
        self.inner.set_partial_many(key, offset_values)
    }

    fn erase(&self, key: &StoreKey) -> Result<(), StorageError> {
        self.inner.erase(key)
    }

    fn erase_prefix(&self, prefix: &StorePrefix) -> Result<(), StorageError> {
        self.inner.erase_prefix(prefix)
    }

    fn supports_set_partial(&self) -> bool {
        self.inner.supports_set_partial()
    }
}

impl ReadableStorageTraits for PtrRecordingStore {
    fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        self.inner.get(key)
    }

    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        self.inner.get_partial_many(key, byte_ranges)
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        self.inner.size_key(key)
    }

    fn supports_get_partial(&self) -> bool {
        self.inner.supports_get_partial()
    }
}

impl ListableStorageTraits for PtrRecordingStore {
    fn list(&self) -> Result<StoreKeys, StorageError> {
        self.inner.list()
    }

    fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        self.inner.list_prefix(prefix)
    }

    fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        self.inner.list_dir(prefix)
    }

    fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        self.inner.size_prefix(prefix)
    }
}

/// An array with a pass-through codec chain: fixed length data type, no bytes-to-bytes codecs.
fn passthrough_array() -> (Array<PtrRecordingStore>, Arc<PtrRecordingStore>) {
    let store = Arc::new(PtrRecordingStore::default());
    let builder = ArrayBuilder::new(vec![4, 4], vec![2, 2], data_type::uint8(), 0u8);
    let array = builder.build(store.clone(), "/array").unwrap();
    (array, store)
}

/// Writing from a borrowed slice must reach the store without an intervening copy.
///
/// `MemoryStore` retains its buffers so it must take ownership, but a store that writes and
/// discards (such as `FilesystemStore`) sees the caller's memory directly. This asserts the
/// value handed to `set` still points at the caller's buffer.
#[test]
fn zero_copy_write_from_borrowed_slice() -> TestResult {
    let (array, store) = passthrough_array();

    let data = vec![1u8, 2, 3, 4];
    let (expected_ptr, expected_len) = (data.as_ptr() as usize, data.len());

    array.store_chunk(&[0, 0], data.as_slice())?;

    assert_eq!(
        store.recorded("array/c/0/0"),
        Some((expected_ptr, expected_len)),
        "a borrowed slice should reach the store without a copy"
    );

    let retrieved: Vec<u8> = array.retrieve_chunk(&[0, 0])?;
    assert_eq!(retrieved, vec![1u8, 2, 3, 4]);
    drop(data);
    Ok(())
}

/// The read-path mirror of [`zero_copy_passthrough_chain`]: decoding a chunk must share the
/// buffer held by the store rather than copying it.
#[test]
fn zero_copy_read_passthrough_chain() -> TestResult {
    let (array, store) = passthrough_array();
    array.store_chunk(&[0, 0], &Bytes::from(vec![1u8, 2, 3, 4]))?;

    // The address of the buffer the store holds, which must outlive the read below.
    let stored = store.get(&StoreKey::new("array/c/0/0").unwrap())?.unwrap();
    let stored_ptr = stored.as_ptr() as usize;

    let retrieved: ArrayBytes = array.retrieve_chunk(&[0, 0])?;
    let retrieved = retrieved.into_fixed()?;

    assert_eq!(
        retrieved.as_ptr() as usize,
        stored_ptr,
        "a pass-through chain should decode without copying the stored buffer"
    );
    assert_eq!(&retrieved[..], &[1u8, 2, 3, 4]);
    drop(stored);
    Ok(())
}

#[test]
fn zero_copy_passthrough_chain() -> TestResult {
    let (array, store) = passthrough_array();

    // A non-fill-value chunk, so that it is actually written.
    let data = Bytes::from(vec![1u8, 2, 3, 4]);
    let (expected_ptr, expected_len) = (data.as_ptr() as usize, data.len());

    array.store_chunk(&[0, 0], &data)?;

    // `data` is still alive here, so an equal address cannot be a coincidental reallocation.
    assert_eq!(
        store.recorded("array/c/0/0"),
        Some((expected_ptr, expected_len)),
        "pass-through chain should store the caller's buffer without a copy"
    );

    // The write must still be correct.
    let retrieved: Vec<u8> = array.retrieve_chunk(&[0, 0])?;
    assert_eq!(retrieved, vec![1u8, 2, 3, 4]);
    drop(data);
    Ok(())
}

#[test]
#[cfg(feature = "gzip")]
fn compressed_chain_copies() -> TestResult {
    let store = Arc::new(PtrRecordingStore::default());
    let mut builder = ArrayBuilder::new(vec![4, 4], vec![2, 2], data_type::uint8(), 0u8);
    builder.bytes_to_bytes_codecs(vec![Arc::new(zarrs::array::codec::GzipCodec::new(5)?)]);
    let array = builder.build(store.clone(), "/array")?;

    let data = Bytes::from(vec![1u8, 2, 3, 4]);
    let data_ptr = data.as_ptr() as usize;

    array.store_chunk(&[0, 0], &data)?;

    // A compressor must produce a new buffer; documents that the fast path does not apply.
    let (recorded_ptr, _) = store.recorded("array/c/0/0").expect("chunk was written");
    assert_ne!(
        recorded_ptr, data_ptr,
        "a compressed chain cannot store the caller's buffer"
    );

    let retrieved: Vec<u8> = array.retrieve_chunk(&[0, 0])?;
    assert_eq!(retrieved, vec![1u8, 2, 3, 4]);
    drop(data);
    Ok(())
}

/// The owned `Vec` path was already zero-copy; lock that in against future regressions.
#[test]
fn zero_copy_owned_vec() -> TestResult {
    let (array, store) = passthrough_array();

    let data = vec![1u8, 2, 3, 4];
    let (expected_ptr, expected_len) = (data.as_ptr() as usize, data.len());

    array.store_chunk(&[0, 0], data)?;

    assert_eq!(
        store.recorded("array/c/0/0"),
        Some((expected_ptr, expected_len)),
        "an owned Vec should be moved into the store, not copied"
    );
    Ok(())
}

/// A borrowed slice with no shared owner still works, just with a copy.
#[test]
fn borrowed_slice_still_correct() -> TestResult {
    let (array, _store) = passthrough_array();
    array.store_chunk(&[0, 0], &[1u8, 2, 3, 4])?;
    let retrieved: Vec<u8> = array.retrieve_chunk(&[0, 0])?;
    assert_eq!(retrieved, vec![1u8, 2, 3, 4]);
    Ok(())
}

/// An array whose `bytes` codec swaps byte order, so the data type must rewrite its input.
fn byte_swapping_array() -> (Array<PtrRecordingStore>, Arc<PtrRecordingStore>) {
    // The endianness that is *not* native, so encode and decode always byte swap.
    let non_native = if cfg!(target_endian = "little") {
        BytesCodec::big()
    } else {
        BytesCodec::little()
    };
    let store = Arc::new(PtrRecordingStore::default());
    let mut builder = ArrayBuilder::new(vec![4], vec![2], data_type::uint16(), 0u16);
    builder.array_to_bytes_codec(Arc::new(non_native));
    let array = builder.build(store.clone(), "/array").unwrap();
    (array, store)
}

/// Byte swapping must still swap when the input is shared rather than owned.
///
/// The data type reverses each component after taking ownership, a path that previously only
/// ever saw an owned buffer. This asserts on the *encoded* bytes rather than a round trip,
/// because encode and decode are the same operation here, so a broken swap would cancel out.
#[test]
fn byte_swap_encodes_from_shared_bytes() -> TestResult {
    let (array, store) = byte_swapping_array();

    let elements = [0x0102u16, 0x0304];
    let mut raw = Vec::new();
    for e in elements {
        raw.extend_from_slice(&e.to_ne_bytes());
    }
    array.store_chunk(&[0], &Bytes::from(raw.clone()))?;

    // Each 2-byte component must be reversed relative to the in-memory representation.
    let stored = store.get(&StoreKey::new("array/c/0").unwrap())?.unwrap();
    let expected: Vec<u8> = raw
        .as_chunks::<2>()
        .0
        .iter()
        .flat_map(|c| [c[1], c[0]])
        .collect();
    assert_eq!(
        &stored[..],
        &expected[..],
        "components must be byte swapped"
    );

    // And the values must still round-trip back to what was written.
    let retrieved: Vec<u16> = array.retrieve_chunk(&[0])?;
    assert_eq!(retrieved, elements);
    Ok(())
}

/// Byte swapping cannot be zero copy on read, because the swap must write somewhere.
#[test]
fn byte_swap_read_copies() -> TestResult {
    let (array, store) = byte_swapping_array();
    array.store_chunk(&[0], &Bytes::from(vec![1u8, 2, 3, 4]))?;

    // The address of the buffer the store holds, which must outlive the read below.
    let stored = store.get(&StoreKey::new("array/c/0").unwrap())?.unwrap();
    let stored_ptr = stored.as_ptr() as usize;

    let retrieved: ArrayBytes = array.retrieve_chunk(&[0])?;
    let retrieved = retrieved.into_fixed()?;

    assert_ne!(
        retrieved.as_ptr() as usize,
        stored_ptr,
        "byte swapping must produce a new buffer"
    );

    drop(stored);
    Ok(())
}

/// The asynchronous write path must share the same fast path.
/// `retrieve_encoded_chunk` shares the store's buffer rather than copying it into a `Vec`.
///
/// This is what lets the encoded chunk cache hold the store's allocation directly.
#[test]
fn retrieve_encoded_chunk_shares_the_stored_buffer() -> TestResult {
    let (array, store) = passthrough_array();
    array.store_chunk(&[0, 0], &Bytes::from(vec![1u8, 2, 3, 4]))?;

    // The address of the buffer the store holds, which must outlive the read below.
    let stored = store.get(&StoreKey::new("array/c/0/0").unwrap())?.unwrap();
    let stored_ptr = stored.as_ptr() as usize;

    let encoded = array.retrieve_encoded_chunk(&[0, 0])?.unwrap();

    assert_eq!(
        encoded.as_ptr() as usize,
        stored_ptr,
        "retrieving an encoded chunk should not copy the stored buffer"
    );
    assert_eq!(&encoded[..], &[1u8, 2, 3, 4]);
    drop(stored);
    Ok(())
}

#[cfg(feature = "async")]
mod r#async {
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex};

    use bytes::Bytes;
    use zarrs::array::{ArrayBuilder, CowBytes, data_type};
    use zarrs::storage::store::AsyncMemoryStore;
    use zarrs::storage::{
        AsyncWritableStorageTraits, OffsetBytesIterator, StorageError, StoreKey, StorePrefix,
    };

    use super::TestResult;

    #[derive(Debug, Default)]
    struct AsyncPtrRecordingStore {
        inner: AsyncMemoryStore,
        ptrs: Mutex<BTreeMap<StoreKey, (usize, usize)>>,
    }

    #[async_trait::async_trait]
    impl AsyncWritableStorageTraits for AsyncPtrRecordingStore {
        async fn set(&self, key: &StoreKey, value: CowBytes<'_>) -> Result<(), StorageError> {
            self.ptrs
                .lock()
                .unwrap()
                .insert(key.clone(), (value.as_ptr() as usize, value.len()));
            self.inner.set(key, value).await
        }

        async fn set_partial_many<'a>(
            &'a self,
            key: &StoreKey,
            offset_values: OffsetBytesIterator<'a>,
        ) -> Result<(), StorageError> {
            self.inner.set_partial_many(key, offset_values).await
        }

        async fn erase(&self, key: &StoreKey) -> Result<(), StorageError> {
            self.inner.erase(key).await
        }

        async fn erase_prefix(&self, prefix: &StorePrefix) -> Result<(), StorageError> {
            self.inner.erase_prefix(prefix).await
        }

        fn supports_set_partial(&self) -> bool {
            self.inner.supports_set_partial()
        }
    }

    #[tokio::test]
    async fn async_zero_copy_passthrough_chain() -> TestResult {
        let store = Arc::new(AsyncPtrRecordingStore::default());
        let builder = ArrayBuilder::new(vec![4, 4], vec![2, 2], data_type::uint8(), 0u8);
        let array = builder.build(store.clone(), "/array")?;

        let data = Bytes::from(vec![1u8, 2, 3, 4]);
        let (expected_ptr, expected_len) = (data.as_ptr() as usize, data.len());

        array.async_store_chunk(&[0, 0], &data).await?;

        let key = StoreKey::new("array/c/0/0")?;
        assert_eq!(
            store.ptrs.lock().unwrap().get(&key).copied(),
            Some((expected_ptr, expected_len)),
            "async pass-through chain should store the caller's buffer without a copy"
        );
        drop(data);
        Ok(())
    }
}
