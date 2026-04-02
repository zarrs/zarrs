#[cfg(not(target_arch = "wasm32"))]
#[path = "blosc_codec_via_blosc_src.rs"]
mod blosc_codec_impl;

#[cfg(target_arch = "wasm32")]
#[path = "blosc_codec_via_blusc.rs"]
mod blosc_codec_impl;

pub use blosc_codec_impl::BloscCodec;
