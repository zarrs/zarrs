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
