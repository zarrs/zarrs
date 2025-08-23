use std::sync::Arc;

use auto_impl::auto_impl;
use itertools::Itertools;

use super::{
    byte_range::ByteRange, Bytes, MaybeBytes, StorageError, StoreKey, StoreKeyOffsetValue,
    StoreKeyRange, StoreKeys, StoreKeysPrefixes, StorePrefix, StorePrefixes,
};

/// Readable storage traits.
#[auto_impl(Arc)]
pub trait ReadableStorageTraits: Send + Sync {
    /// Retrieve the value (bytes) associated with a given [`StoreKey`].
    ///
    /// Returns [`None`] if the key is not found.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying storage error.
    fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        Ok(self
            .get_partial_values_key(key, &mut [ByteRange::FromStart(0, None)].into_iter())?
            .map(|mut v| v.remove(0)))
    }

    /// Retrieve partial bytes from a list of byte ranges for a store key.
    ///
    /// Returns [`None`] if the key is not found.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying storage error.
    fn get_partial_values_key(
        &self,
        key: &StoreKey,
        byte_ranges: &mut (dyn Iterator<Item = ByteRange> + Send),
    ) -> Result<Option<Vec<Bytes>>, StorageError>;

    /// Retrieve partial bytes from a list of [`StoreKeyRange`].
    ///
    /// # Parameters
    /// * `key_ranges`: ordered set of ([`StoreKey`], [`ByteRange`]) pairs. A key may occur multiple times with different ranges.
    ///
    /// # Output
    /// A a list of values in the order of the `key_ranges`. It will be [`None`] for missing keys.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying storage error.
    fn get_partial_values(
        &self,
        key_ranges: &[StoreKeyRange],
    ) -> Result<Vec<MaybeBytes>, StorageError> {
        self.get_partial_values_batched_by_key(key_ranges)
    }

    /// Return the size in bytes of the value at `key`.
    ///
    /// Returns [`None`] if the key is not found.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying storage error.
    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError>;

    /// A utility method with the same input and output as [`get_partial_values`](ReadableStorageTraits::get_partial_values) that internally calls [`get_partial_values_key`](ReadableStorageTraits::get_partial_values_key) with byte ranges grouped by key.
    ///
    /// Readable storage can use this function in the implementation of [`get_partial_values`](ReadableStorageTraits::get_partial_values) if that is optimal.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying storage error.
    fn get_partial_values_batched_by_key(
        &self,
        key_ranges: &[StoreKeyRange],
    ) -> Result<Vec<MaybeBytes>, StorageError> {
        let mut out: Vec<MaybeBytes> = Vec::with_capacity(key_ranges.len());
        let mut last_key = None;
        let mut byte_ranges_key = Vec::new();
        for key_range in key_ranges {
            if last_key.is_none() {
                last_key = Some(&key_range.key);
            }
            let last_key_val = last_key.unwrap();

            if key_range.key != *last_key_val {
                // Found a new key, so do a batched get of the byte ranges of the last key
                let bytes = (self.get_partial_values_key(
                    last_key.unwrap(),
                    &mut byte_ranges_key.iter().copied(),
                )?)
                .map_or_else(
                    || vec![None; byte_ranges_key.len()],
                    |partial_values| partial_values.into_iter().map(Some).collect(),
                );
                out.extend(bytes);
                last_key = Some(&key_range.key);
                byte_ranges_key.clear();
            }

            byte_ranges_key.push(key_range.byte_range);
        }

        if !byte_ranges_key.is_empty() {
            // Get the byte ranges of the last key
            let bytes = (self
                .get_partial_values_key(last_key.unwrap(), &mut byte_ranges_key.iter().copied())?)
            .map_or_else(
                || vec![None; byte_ranges_key.len()],
                |partial_values| partial_values.into_iter().map(Some).collect(),
            );
            out.extend(bytes);
        }

        Ok(out)
    }
}

/// Listable storage traits.
#[auto_impl(Arc)]
pub trait ListableStorageTraits: Send + Sync {
    /// Retrieve all [`StoreKeys`] in the store.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying error with the store.
    fn list(&self) -> Result<StoreKeys, StorageError>;

    /// Retrieve all [`StoreKeys`] with a given [`StorePrefix`].
    ///
    /// # Errors
    /// Returns a [`StorageError`] if the prefix is not a directory or there is an underlying error with the store.
    fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError>;

    /// Retrieve all [`StoreKeys`] and [`StorePrefix`] which are direct children of [`StorePrefix`].
    ///
    /// # Errors
    /// Returns a [`StorageError`] if the prefix is not a directory or there is an underlying error with the store.
    ///
    fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError>;

    /// Return the size in bytes of all keys under `prefix`.
    ///
    /// # Errors
    /// Returns a `StorageError` if the store does not support `size()` or there is an underlying error with the store.
    fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError>;

    /// Return the total size in bytes of the storage.
    ///
    /// # Errors
    /// Returns a `StorageError` if the store does not support `size()` or there is an underlying error with the store.
    fn size(&self) -> Result<u64, StorageError> {
        self.size_prefix(&StorePrefix::root())
    }
}

/// Set partial values for a store.
///
/// This method reads entire values, updates them, and replaces them.
/// Stores can use this internally if they do not support updating/appending without replacement.
///
/// # Errors
/// Returns a [`StorageError`] if an underlying store operation fails.
///
/// # Panics
/// Panics if a key ends beyond `usize::MAX`.
pub fn store_set_partial_values<T: ReadableWritableStorageTraits>(
    store: &T,
    key_offset_values: &[StoreKeyOffsetValue],
    // truncate: bool,
) -> Result<(), StorageError> {
    // Group by key
    key_offset_values
        .iter()
        .chunk_by(|key_offset_value| key_offset_value.key())
        .into_iter()
        .map(|(key, group)| (key.clone(), group.into_iter().cloned().collect::<Vec<_>>()))
        .try_for_each(|(key, group)| {
            // Lock the store key
            // let mutex = store.mutex(&key)?;
            // let _lock = mutex.lock();

            // Read the store key
            let bytes = store.get(&key)?.unwrap_or_default();
            let mut bytes = Vec::<u8>::from(bytes);

            // Convert to a mutable vector of the required length
            let end_max = group
                .iter()
                .map(|key_offset_value| {
                    usize::try_from(
                        key_offset_value.offset() + key_offset_value.value().len() as u64,
                    )
                    .unwrap()
                })
                .max()
                .unwrap();
            if bytes.len() < end_max {
                bytes.resize_with(end_max, Default::default);
            }
            // else if truncate {
            //     bytes.truncate(end_max);
            // };

            // Update the store key
            for key_offset_value in group {
                let start = usize::try_from(key_offset_value.offset()).unwrap();
                bytes[start..start + key_offset_value.value().len()]
                    .copy_from_slice(key_offset_value.value());
            }

            // Write the store key
            store.set(&key, Bytes::from(bytes))
        })?;
    Ok(())
}

/// Writable storage traits.
#[auto_impl(Arc)]
pub trait WritableStorageTraits: Send + Sync {
    /// Store bytes at a [`StoreKey`].
    ///
    /// # Errors
    /// Returns a [`StorageError`] on failure to store.
    fn set(&self, key: &StoreKey, value: Bytes) -> Result<(), StorageError>;

    /// Store bytes according to a list of [`StoreKeyOffsetValue`].
    ///
    /// # Errors
    /// Returns a [`StorageError`] on failure to store.
    fn set_partial_values(
        &self,
        key_offset_values: &[StoreKeyOffsetValue],
    ) -> Result<(), StorageError>;

    /// Erase a [`StoreKey`].
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying storage error.
    fn erase(&self, key: &StoreKey) -> Result<(), StorageError>;

    /// Erase a list of [`StoreKey`].
    ///
    /// # Errors
    /// Returns a [`StorageError`] if there is an underlying storage error.
    fn erase_values(&self, keys: &[StoreKey]) -> Result<(), StorageError> {
        keys.iter().try_for_each(|key| self.erase(key))?;
        Ok(())
    }

    /// Erase all [`StoreKey`] under [`StorePrefix`].
    ///
    /// # Errors
    /// Returns a [`StorageError`] is the prefix is not in the store, or the erase otherwise fails.
    fn erase_prefix(&self, prefix: &StorePrefix) -> Result<(), StorageError>;
}

/// A supertrait of [`ReadableStorageTraits`] and [`WritableStorageTraits`].
pub trait ReadableWritableStorageTraits: ReadableStorageTraits + WritableStorageTraits {
    /// Return a readable version of the store.
    fn readable(self: Arc<Self>) -> Arc<dyn ReadableStorageTraits>;

    /// Return a writable version of the store.
    fn writable(self: Arc<Self>) -> Arc<dyn WritableStorageTraits>;
}

impl<T> ReadableWritableStorageTraits for T
where
    T: ReadableStorageTraits + WritableStorageTraits + 'static,
{
    fn readable(self: Arc<Self>) -> Arc<dyn ReadableStorageTraits> {
        self.clone()
    }

    fn writable(self: Arc<Self>) -> Arc<dyn WritableStorageTraits> {
        self.clone()
    }
}

/// A supertrait of [`ReadableStorageTraits`] and [`ListableStorageTraits`].
pub trait ReadableListableStorageTraits: ReadableStorageTraits + ListableStorageTraits {
    /// Return a readable version of the store.
    fn readable(self: Arc<Self>) -> Arc<dyn ReadableStorageTraits>;

    /// Return a listable version of the store.
    fn listable(self: Arc<Self>) -> Arc<dyn ListableStorageTraits>;
}

impl<T> ReadableListableStorageTraits for T
where
    T: ReadableStorageTraits + ListableStorageTraits + 'static,
{
    fn readable(self: Arc<Self>) -> Arc<dyn ReadableStorageTraits> {
        self.clone()
    }

    fn listable(self: Arc<Self>) -> Arc<dyn ListableStorageTraits> {
        self.clone()
    }
}

/// A supertrait of [`ReadableWritableStorageTraits`] and [`ListableStorageTraits`].
pub trait ReadableWritableListableStorageTraits:
    ReadableWritableStorageTraits + ListableStorageTraits
{
    /// Return a readable and writable version of the store.
    fn readable_writable(self: Arc<Self>) -> Arc<dyn ReadableWritableStorageTraits>;

    /// Return a readable and listable version of the store.
    fn readable_listable(self: Arc<Self>) -> Arc<dyn ReadableListableStorageTraits>;

    /// Return a listable version of the store.
    fn listable(self: Arc<Self>) -> Arc<dyn ListableStorageTraits>;
}

impl<T> ReadableWritableListableStorageTraits for T
where
    T: ReadableWritableStorageTraits + ListableStorageTraits + 'static,
{
    fn readable_writable(self: Arc<Self>) -> Arc<dyn ReadableWritableStorageTraits> {
        self.clone()
    }

    fn readable_listable(self: Arc<Self>) -> Arc<dyn ReadableListableStorageTraits> {
        self.clone()
    }

    fn listable(self: Arc<Self>) -> Arc<dyn ListableStorageTraits> {
        self.clone()
    }
}

/// Discover the children of a store prefix.
///
/// # Errors
/// Returns a [`StorageError`] if there is an underlying error with the store.
pub fn discover_children<TStorage: ?Sized + ReadableStorageTraits + ListableStorageTraits>(
    storage: &Arc<TStorage>,
    prefix: &StorePrefix,
) -> Result<StorePrefixes, StorageError> {
    let children: Result<Vec<_>, _> = storage
        .list_dir(prefix)?
        .prefixes()
        .iter()
        .filter(|v| !v.as_str().starts_with("__"))
        .map(|v| StorePrefix::new(v.as_str()))
        .collect();
    Ok(children?)
}
