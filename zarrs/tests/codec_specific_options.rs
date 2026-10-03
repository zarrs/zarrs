#![allow(missing_docs)]

//! Codec-specific options reach codecs nested in other codecs.

use std::num::NonZeroU64;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use zarrs::array::codec::{BytesCodec, OptionalCodec, ShardingCodecBuilder, VlenCodec};
use zarrs::array::{
    ArrayToBytesCodecTraits, CodecChain, CodecCreateError, CodecMetadataOptions,
    CodecSpecificOptions, CodecTraits, DataType, Endianness, FillValue,
    UnboundArrayToBytesCodecTraits, data_type,
};
use zarrs::metadata::Configuration;
use zarrs_codec::{PartialDecoderCapability, PartialEncoderCapability};
use zarrs_metadata_ext::codec::vlen::{VlenIndexDataType, VlenIndexLocation};
use zarrs_plugin::ZarrVersion;

/// An option detected by [`ProbeCodec`].
struct ProbeOptions;

/// A `bytes` codec that counts how many times it is bound with [`ProbeOptions`].
#[derive(Debug)]
struct ProbeCodec {
    inner: BytesCodec,
    bound_with_options: Arc<AtomicUsize>,
}

zarrs_plugin::impl_extension_aliases!(ProbeCodec, v3: "zarrs.test.probe");

impl CodecTraits for ProbeCodec {
    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        None
    }

    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        self.inner.partial_decoder_capability()
    }

    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        self.inner.partial_encoder_capability()
    }
}

impl UnboundArrayToBytesCodecTraits for ProbeCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn UnboundArrayToBytesCodecTraits> {
        self
    }

    fn with_context(
        &self,
        data_type: DataType,
        fill_value: FillValue,
        codec_specific_options: &CodecSpecificOptions,
    ) -> Result<Arc<dyn ArrayToBytesCodecTraits>, CodecCreateError> {
        if codec_specific_options
            .get_option::<ProbeOptions>()
            .is_some()
        {
            self.bound_with_options.fetch_add(1, Ordering::SeqCst);
        }
        self.inner
            .with_context(data_type, fill_value, codec_specific_options)
    }
}

#[test]
fn codec_specific_options_reach_nested_codecs() -> Result<(), CodecCreateError> {
    let bound_with_options = Arc::new(AtomicUsize::new(0));
    let probe = |endianness: Option<Endianness>| {
        Arc::new(ProbeCodec {
            inner: BytesCodec::new(endianness),
            bound_with_options: bound_with_options.clone(),
        })
    };
    let probe_chain = |endianness: Option<Endianness>| {
        Arc::new(CodecChain::new(vec![], probe(endianness), vec![]))
    };
    let options = CodecSpecificOptions::default().with_option(ProbeOptions);
    let bind = |codec: Arc<dyn UnboundArrayToBytesCodecTraits>,
                data_type: DataType,
                fill_value: FillValue|
     -> Result<usize, CodecCreateError> {
        bound_with_options.store(0, Ordering::SeqCst);
        CodecChain::new(vec![], codec, vec![]).with_context(data_type, fill_value, &options)?;
        Ok(bound_with_options.load(Ordering::SeqCst))
    };
    let little = Some(Endianness::Little);

    // Index and data codecs
    let vlen = VlenCodec::new(
        probe_chain(little),
        probe_chain(None),
        VlenIndexDataType::UInt64,
        VlenIndexLocation::Start,
    );
    assert_eq!(
        bind(Arc::new(vlen), data_type::string(), FillValue::from(""))?,
        2
    );

    // Mask and data codecs
    let optional = OptionalCodec::new(probe_chain(None), probe_chain(little));
    assert_eq!(
        bind(
            Arc::new(optional),
            data_type::uint16().to_optional(),
            FillValue::from(None::<u16>),
        )?,
        2
    );

    // Inner and index codecs
    let sharding =
        ShardingCodecBuilder::new(vec![NonZeroU64::new(2).unwrap()], &data_type::uint16())
            .array_to_bytes_codec(probe(little))
            .index_array_to_bytes_codec(probe(little))
            .build_arc();
    assert_eq!(
        bind(sharding, data_type::uint16(), FillValue::from(0u16))?,
        2
    );
    Ok(())
}
