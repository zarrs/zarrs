//! `MaybeSend` and `MaybeSync` traits for cross-platform async code.
//!
//! Reference: <https://github.com/iced-rs/iced/blob/master/futures/src/maybe.rs>

#[cfg(not(target_arch = "wasm32"))]
mod platform {
    /// An extension trait that enforces `Send` only on native platforms.
    ///
    /// Useful for writing cross-platform async code!
    pub trait MaybeSend: Send {}

    impl<T> MaybeSend for T where T: Send {}

    /// An extension trait that enforces `Sync` only on native platforms.
    ///
    /// Useful for writing cross-platform async code!
    pub trait MaybeSync: Sync {}

    impl<T> MaybeSync for T where T: Sync {}
}

#[cfg(target_arch = "wasm32")]
mod platform {
    /// An extension trait that enforces `Send` only on native platforms.
    ///
    /// Useful for writing cross-platform async code!
    pub trait MaybeSend {}

    impl<T> MaybeSend for T {}

    /// An extension trait that enforces `Sync` only on native platforms.
    ///
    /// Useful for writing cross-platform async code!
    pub trait MaybeSync {}

    impl<T> MaybeSync for T {}
}

pub use platform::{MaybeSend, MaybeSync};