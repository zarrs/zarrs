//! Standard trait conveniences retained from main without new API dependencies.
use std::sync::Arc;
use zarrs::array::{ArrayBuilder, data_type};

#[test]
fn cloned_builder_is_independent() {
    let mut original = ArrayBuilder::new(vec![4, 4], vec![2, 2], data_type::uint8(), 7u8);
    original
        .attributes_mut()
        .insert("source".into(), true.into());
    original.dimension_names(Some(["y", "x"]));
    let mut cloned = original.clone();
    cloned.shape(vec![8, 8]);
    cloned
        .attributes_mut()
        .insert("source".into(), false.into());
    let original = original.build(Arc::new(()), "/original").unwrap();
    let cloned = cloned.build(Arc::new(()), "/cloned").unwrap();
    assert_eq!(original.shape(), &[4, 4]);
    assert_eq!(cloned.shape(), &[8, 8]);
    assert_eq!(original.attributes()["source"], true);
    assert_eq!(cloned.attributes()["source"], false);
    assert_eq!(original.dimension_names(), cloned.dimension_names());
    assert_eq!(original.fill_value(), cloned.fill_value());
}

#[test]
fn array_clone_accepts_non_clone_and_unsized_storage() {
    struct NonClone;
    let array = ArrayBuilder::new(vec![4], vec![2], data_type::uint8(), 3u8)
        .build(Arc::new(NonClone), "/array")
        .unwrap()
        .with_codec_options(zarrs::array::CodecOptions::default().with_concurrent_target(3));
    let mut cloned = array.clone();
    assert!(Arc::ptr_eq(&array.storage(), &cloned.storage()));
    assert_eq!(array.metadata(), cloned.metadata());
    cloned.attributes_mut().insert("cloned".into(), true.into());
    assert!(!array.attributes().contains_key("cloned"));
    let storage: Arc<dyn std::any::Any> = Arc::new(NonClone);
    let dynamic = array.with_storage(storage);
    assert!(Arc::ptr_eq(&dynamic.storage(), &dynamic.clone().storage()));
}

#[test]
fn metadata_retrieve_traits() {
    use zarrs::config::MetadataRetrieveVersion;
    fn bounds<T: std::fmt::Debug + Clone + Copy + Default>() {}
    bounds::<MetadataRetrieveVersion>();
    assert!(matches!(
        MetadataRetrieveVersion::default(),
        MetadataRetrieveVersion::Default
    ));
}

#[test]
fn group_metadata_options_are_copy() {
    use zarrs::config::MetadataConvertVersion;
    use zarrs::group::GroupMetadataOptions;
    fn bounds<T: Copy>() {}
    bounds::<GroupMetadataOptions>();
    let options =
        GroupMetadataOptions::default().with_metadata_convert_version(MetadataConvertVersion::V3);
    let copied = options;
    assert!(matches!(
        options.metadata_convert_version(),
        MetadataConvertVersion::V3
    ));
    assert!(matches!(
        copied.metadata_convert_version(),
        MetadataConvertVersion::V3
    ));
}

#[test]
fn tensor_clone_preserves_cow_semantics() {
    use std::borrow::Cow;
    use zarrs::array::Tensor;
    fn bounds<T: Clone + std::fmt::Debug>() {}
    bounds::<Tensor>();
    let owned = Tensor::new(vec![1, 2, 3], data_type::uint8(), vec![3]);
    let cloned = owned.clone();
    assert_eq!(owned.as_parts(), cloned.as_parts());
    assert_ne!(owned.bytes().as_ptr(), cloned.bytes().as_ptr());
    assert!(matches!(cloned.into_parts().0, Cow::Owned(_)));
    static BYTES: &[u8] = &[4, 5];
    let borrowed = Tensor::new(Cow::Borrowed(BYTES), data_type::uint8(), vec![2]);
    let cloned = borrowed.clone();
    assert_eq!(borrowed.bytes().as_ptr(), cloned.bytes().as_ptr());
    assert!(matches!(cloned.into_parts().0, Cow::Borrowed(_)));
}
