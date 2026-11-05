use crate::{
    BytesRepresentation, BytesToBytesCodecTraits, CodecError, CodecOptions, ComputeResources,
    CowBytes,
};

/// Immediate or deferred encoded input for an array-to-bytes decoder.
#[derive(Debug)]
pub enum ArrayBytesDecodeIntoInput<'a> {
    /// Already-produced encoded bytes.
    Bytes(CowBytes<'a>),
    /// One pending bytes-to-bytes decode.
    Deferred(BytesDecodeSource<'a>),
}

impl<'a> From<CowBytes<'a>> for ArrayBytesDecodeIntoInput<'a> {
    fn from(bytes: CowBytes<'a>) -> Self {
        Self::Bytes(bytes)
    }
}

impl<'a> ArrayBytesDecodeIntoInput<'a> {
    /// Resolve this input without cloning its bytes.
    ///
    /// # Errors
    /// Returns the producer's decoding error.
    pub fn into_bytes(
        self,
        options: &CodecOptions,
        resources: &ComputeResources,
    ) -> Result<CowBytes<'a>, CodecError> {
        match self {
            Self::Bytes(bytes) => Ok(bytes),
            Self::Deferred(source) => source.decode(options, resources),
        }
    }
}

/// A single-use bytes-to-bytes decode whose output placement is not yet chosen.
///
/// Both decoding routes consume the source, retaining ownership of its encoded
/// bytes and ensuring that only one route can execute.
///
/// Sources cannot be cloned:
/// ```compile_fail
/// use zarrs_codec::BytesDecodeSource;
/// fn duplicate(source: BytesDecodeSource<'_>) {
///     let _ = source.clone();
/// }
/// ```
///
/// Nor can a consumed source be decoded again:
/// ```compile_fail
/// use zarrs_codec::{BytesDecodeSource, CodecOptions, ComputeResources};
/// fn twice(source: BytesDecodeSource<'_>, options: &CodecOptions, resources: &ComputeResources) {
///     let _ = source.decode(options, resources);
///     let _ = source.decode(options, resources);
/// }
/// ```
#[derive(Debug)]
pub struct BytesDecodeSource<'a> {
    codec: &'a dyn BytesToBytesCodecTraits,
    bytes: CowBytes<'a>,
    decoded_representation: BytesRepresentation,
}

impl<'a> BytesDecodeSource<'a> {
    /// Construct a source without cloning the encoded bytes.
    #[must_use]
    pub fn new(
        codec: &'a dyn BytesToBytesCodecTraits,
        bytes: CowBytes<'a>,
        decoded_representation: BytesRepresentation,
    ) -> Self {
        Self {
            codec,
            bytes,
            decoded_representation,
        }
    }

    /// Return the expected decoded representation.
    #[must_use]
    pub fn decoded_representation(&self) -> &BytesRepresentation {
        &self.decoded_representation
    }

    /// Whether the producer avoids a full-size decoded intermediate and copy.
    ///
    /// This does not promise to avoid decoder state or workspace allocations.
    #[must_use]
    pub fn is_decode_into_efficient(&self) -> bool {
        self.codec.is_decode_into_efficient()
    }

    /// Consume this source and decode into `output`.
    ///
    /// On error, `output` may have been partially written.
    ///
    /// # Errors
    /// Returns a decoding error, including when decoded bytes do not fill `output`.
    pub fn decode_into(
        self,
        output: &mut [u8],
        options: &CodecOptions,
        resources: &ComputeResources,
    ) -> Result<(), CodecError> {
        self.codec.decode_into(
            self.bytes,
            &self.decoded_representation,
            output,
            options,
            resources,
        )
    }

    /// Consume this source and decode normally.
    ///
    /// # Errors
    /// Returns the producer's decoding error.
    pub fn decode(
        self,
        options: &CodecOptions,
        resources: &ComputeResources,
    ) -> Result<CowBytes<'a>, CodecError> {
        self.codec
            .decode(self.bytes, &self.decoded_representation, options, resources)
    }
}

#[cfg(test)]
mod tests {
    //! The default `BytesToBytesCodecTraits::decode_into` fallback.
    //!
    //! Run under Miri with `MIRIFLAGS=-Zmiri-ignore-leaks`: Rayon's
    //! `use_current_thread` pools retain their thread-local registry allocations.

    use crate::*;
    use std::borrow::Cow;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use zarrs_plugin::{ExtensionName, ZarrVersion};

    #[derive(Debug, Default)]
    struct Producer {
        normal: AtomicUsize,
    }

    impl ExtensionName for Producer {
        fn name(&self, _: ZarrVersion) -> Option<Cow<'static, str>> {
            None
        }
    }

    impl CodecTraits for Producer {
        fn partial_decoder_capability(&self) -> PartialDecoderCapability {
            PartialDecoderCapability {
                partial_read: false,
                partial_decode: false,
            }
        }
        fn partial_encoder_capability(&self) -> PartialEncoderCapability {
            PartialEncoderCapability {
                partial_encode: false,
            }
        }

        fn configuration(
            &self,
            _: ZarrVersion,
            _: &CodecMetadataOptions,
        ) -> Option<zarrs_metadata::Configuration> {
            None
        }
    }

    impl BytesToBytesCodecTraits for Producer {
        fn recommended_concurrency(
            &self,
            _: &BytesRepresentation,
        ) -> Result<RecommendedConcurrency, CodecError> {
            Ok(RecommendedConcurrency::new_maximum(1))
        }
        fn into_dyn(self: Arc<Self>) -> Arc<dyn BytesToBytesCodecTraits> {
            self
        }
        fn encoded_representation(&self, repr: &BytesRepresentation) -> BytesRepresentation {
            *repr
        }
        fn encode<'a>(
            &self,
            bytes: CowBytes<'a>,
            _: &CodecOptions,
            _: &ComputeResources,
        ) -> Result<CowBytes<'a>, CodecError> {
            Ok(bytes)
        }
        fn decode<'a>(
            &self,
            bytes: CowBytes<'a>,
            repr: &BytesRepresentation,
            _options: &CodecOptions,
            resources: &ComputeResources,
        ) -> Result<CowBytes<'a>, CodecError> {
            self.normal.fetch_add(1, Ordering::Relaxed);
            assert_eq!(*repr, BytesRepresentation::FixedSize(3));
            assert_eq!(resources.compute_target(), 7);
            Ok(bytes)
        }
    }

    fn options() -> CodecOptions {
        CodecOptions::default()
    }

    fn resources() -> ComputeResources {
        // Default resources query Rayon. Keep that query on a single-thread local
        // pool so Miri does not start persistent global worker threads.
        thread_local! {
            static POOL: rayon::ThreadPool = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .use_current_thread()
                .build()
                .unwrap();
        }
        POOL.with(|pool| pool.install(|| ComputeResources::default().with_compute_target(7)))
    }

    #[test]
    fn byte_decode_fallback_validates_length_before_copying() {
        let producer = Producer::default();
        for len in [2, 3, 4] {
            let mut output = vec![0xa5; len];
            let result = producer.decode_into(
                vec![1, 2, 3].into(),
                &BytesRepresentation::FixedSize(3),
                &mut output,
                &options(),
                &resources(),
            );
            assert_eq!(result.is_ok(), len == 3);
            assert_eq!(
                output,
                if len == 3 {
                    vec![1, 2, 3]
                } else {
                    vec![0xa5; len]
                }
            );
        }
        assert_eq!(producer.normal.load(Ordering::Relaxed), 3);
        assert!(!producer.is_decode_into_efficient());
    }
}
