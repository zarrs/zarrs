//! `MaybeSend` and `MaybeSync` traits for cross-platform async code.
//!
//! Reference: <https://github.com/iced-rs/iced/blob/master/futures/src/maybe.rs>

#[cfg(not(target_arch = "wasm32"))]
mod platform {
    /// A marker trait that enforces `Send` only on native platforms.
    ///
    pub use core::marker::Send as MaybeSend;
    /// A marker trait that enforces `Sync` only on native platforms.
    ///
    pub use core::marker::Sync as MaybeSync;
}

#[cfg(target_arch = "wasm32")]
mod platform {
    /// A marker trait that enforces `Send` only on native platforms.
    pub trait MaybeSend {}

    impl<T> MaybeSend for T {}

    /// A marker trait that enforces `Sync` only on native platforms.
    pub trait MaybeSync {}

    impl<T> MaybeSync for T {}
}

pub use platform::{MaybeSend, MaybeSync};
