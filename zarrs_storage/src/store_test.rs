use std::error::Error;

use crate::byte_range::ByteRange;
#[cfg(feature = "async")]
use crate::{AsyncListableStorageTraits, AsyncReadableStorageTraits, AsyncWritableStorageTraits};
use crate::{ListableStorageTraits, ReadableStorageTraits, StorePrefix, WritableStorageTraits};

ambisync::scoped! {
#![defaults(
    sync(fns("async_{}"), types("Async{}")),
    async(feature = "async"),
)]

#[ambisync]
#[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
/// Create a store with the following data
/// ```text
/// - a/
///   - b [0, 1, 2, 3]
///   - c [0]
///   - d/
///     - e
///   - f/
///     - g
///     - h
/// - i/
///   - j/
///     - k [0, 1]
/// ```
pub async fn async_store_write<T: AsyncWritableStorageTraits>(
    store: &T,
) -> Result<(), Box<dyn Error>> {
    store.erase_prefix(&StorePrefix::root()).await?;

    store
        .set(&"a/b".try_into()?, vec![255, 255, 255].into())
        .await?;
    store
        .set_partial(&"a/b".try_into()?, 1, vec![1, 2].into())
        .await?;
    store
        .set_partial(&"a/b".try_into()?, 3, vec![3].into())
        .await?;

    store
        .set_partial(&"a/b".try_into()?, 0, vec![0].into())
        .await?;

    store.set(&"a/c".try_into()?, vec![0].into()).await?;
    store.set(&"a/d/e".try_into()?, vec![].into()).await?;
    store.set(&"a/f/g".try_into()?, vec![].into()).await?;
    store.set(&"a/f/h".try_into()?, vec![].into()).await?;
    store.set(&"i/j/k".try_into()?, vec![0, 1].into()).await?;

    store.set(&"erase".try_into()?, vec![].into()).await?;
    store.erase(&"erase".try_into()?).await?;
    store.erase(&"erase".try_into()?).await?; // succeeds

    store
        .set(&"erase_many_0".try_into()?, vec![].into())
        .await?;
    store
        .set(&"erase_many_1".try_into()?, vec![].into())
        .await?;
    store
        .erase_many(&["erase_many_0".try_into()?, "erase_many_1".try_into()?])
        .await?;

    store
        .set(&"erase_prefix/0".try_into()?, vec![].into())
        .await?;
    store
        .set(&"erase_prefix/1".try_into()?, vec![].into())
        .await?;
    store.erase_prefix(&"erase_prefix/".try_into()?).await?;

    Ok(())
}

#[ambisync]
#[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
/// Read from the store and check the data matches the expected values after [`async_store_write`].
pub async fn async_store_read<T: AsyncReadableStorageTraits>(
    store: &T,
) -> Result<(), Box<dyn Error>> {
    assert!(store.get(&"notfound".try_into()?).await?.is_none());
    assert!(store.size_key(&"notfound".try_into()?).await?.is_none());
    assert_eq!(
        store.get(&"a/b".try_into()?).await?,
        Some(vec![0, 1, 2, 3].into())
    );
    assert_eq!(store.size_key(&"a/b".try_into()?).await?, Some(4));
    assert_eq!(store.size_key(&"a/c".try_into()?).await?, Some(1));
    assert_eq!(store.size_key(&"i/j/k".try_into()?).await?, Some(2));
    let partial_values = store
        .get_partial_many(
            &"a/b".try_into()?,
            Box::new([ByteRange::FromStart(1, Some(1)), ByteRange::Suffix(1)].into_iter()),
        )
        .await?
        .unwrap();
    let partial_values = ambisync::alt!(
        sync => partial_values.collect::<Result<Vec<_>, _>>()?,
        async => {
            use futures::TryStreamExt;
            partial_values.try_collect::<Vec<_>>().await?
        },
    );
    assert_eq!(partial_values, vec![vec![1], vec![3]]);
    assert_eq!(
        store
            .get_partial(&"a/b".try_into()?, ByteRange::FromStart(1, None))
            .await?,
        Some(vec![1, 2, 3].into())
    );
    assert_eq!(
        store
            .get_partial(&"a/b".try_into()?, ByteRange::Suffix(2))
            .await?,
        Some(vec![2, 3].into())
    );
    assert_eq!(
        store
            .get_partial(&"i/j/k".try_into()?, ByteRange::FromStart(1, Some(1)))
            .await?,
        Some(vec![1].into())
    );
    assert!(store
        .get_partial(&"a/b".try_into()?, ByteRange::FromStart(1, Some(10)))
        .await
        .is_err());
    assert_eq!(
        store
            .get_partial(&"notfound".try_into()?, ByteRange::FromStart(1, Some(10)))
            .await
            .unwrap(),
        None
    );

    Ok(())
}

#[ambisync]
#[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
/// List the store and check the data matches the expected values after [`async_store_write`].
pub async fn async_store_list<T: AsyncListableStorageTraits>(
    store: &T,
) -> Result<(), Box<dyn Error>> {
    assert_eq!(
        store.list().await?,
        &[
            "a/b".try_into()?,
            "a/c".try_into()?,
            "a/d/e".try_into()?,
            "a/f/g".try_into()?,
            "a/f/h".try_into()?,
            "i/j/k".try_into()?
        ]
    );

    assert_eq!(
        store.list_prefix(&"".try_into()?).await?,
        &[
            "a/b".try_into()?,
            "a/c".try_into()?,
            "a/d/e".try_into()?,
            "a/f/g".try_into()?,
            "a/f/h".try_into()?,
            "i/j/k".try_into()?
        ]
    );

    assert_eq!(
        store.list_prefix(&"a/".try_into()?).await?,
        &[
            "a/b".try_into()?,
            "a/c".try_into()?,
            "a/d/e".try_into()?,
            "a/f/g".try_into()?,
            "a/f/h".try_into()?
        ]
    );
    assert_eq!(
        store.list_prefix(&"i/".try_into()?).await?,
        &["i/j/k".try_into()?]
    );
    assert_eq!(store.list_prefix(&"notfound/".try_into()?).await?, &[]);

    {
        let list_dir = store.list_dir(&"a/".try_into()?).await?;
        assert_eq!(list_dir.keys(), &["a/b".try_into()?, "a/c".try_into()?,]);
        assert_eq!(
            list_dir.prefixes(),
            &["a/d/".try_into()?, "a/f/".try_into()?,]
        );
    }
    {
        let list_dir = store.list_dir(&"notfound/".try_into()?).await?;
        assert_eq!(list_dir.keys(), &[]);
        assert_eq!(list_dir.prefixes(), &[]);
    }
    Ok(())
}

#[ambisync]
/// Check that size aggregation methods return the expected number of bytes after [`store_write`].
///
/// This test is not applicable to stores that perform compression or transformation of values.
#[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
pub async fn async_store_list_size<T: AsyncListableStorageTraits>(
    store: &T,
) -> Result<(), Box<dyn Error>> {
    assert_eq!(store.size().await?, 7);
    assert_eq!(store.size_prefix(&"a/".try_into()?).await?, 5);
    assert_eq!(store.size_prefix(&"i/".try_into()?).await?, 2);
    assert_eq!(store.size_prefix(&"notfound/".try_into()?).await?, 0);
    Ok(())
}

}
