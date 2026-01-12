#[cfg(doc)]
use crate::{ExtensionAliases, ExtensionNameStatic};

/// A macro to implement [`ExtensionAliases`], and [`ExtensionNameStatic`] for an extension type.
///
/// This macro generates:
/// - Two static `LazyLock<RwLock<ExtensionAliasesConfig>>` variables (one for V3, one for V2)
/// - `ExtensionAliases<ZarrVersion3>` implementation
/// - `ExtensionAliases<ZarrVersion2>` implementation
/// - `ExtensionNameStatic` implementation (which enables the blanket `ExtensionName` impl)
///
/// The blanket implementation of [`ExtensionAliases`] is automatically available for types
/// that implement both `ExtensionAliases<ZarrVersion2>` and `ExtensionAliases<ZarrVersion3>`.
///
/// The static variable names are derived from the type name by converting to `SCREAMING_SNAKE_CASE`
/// and appending `_ALIASES_V3` or `_ALIASES_V2`.
///
/// # Variants
///
/// All variants require explicit `v3:` and/or `v2:` labels. A version can be omitted entirely
/// if the extension does not support that Zarr version.
///
/// ## V3-only (no V2 support)
/// ```ignore
/// impl_extension_aliases!(MyCodec, v3: "my_codec");
/// impl_extension_aliases!(MyCodec, v3: "my_codec", ["alias1"]);
/// impl_extension_aliases!(MyCodec, v3: "my_codec", ["alias1"], [regex]);
/// ```
///
/// ## V2-only (no V3 support)
/// ```ignore
/// impl_extension_aliases!(MyCodec, v2: "my_codec");
/// impl_extension_aliases!(MyCodec, v2: "my_codec", ["alias1"]);
/// impl_extension_aliases!(MyCodec, v2: "my_codec", ["alias1"], [regex]);
/// ```
///
/// ## Both versions
/// ```ignore
/// impl_extension_aliases!(MyCodec, v3: "v3_name", v2: "v2_name");
/// impl_extension_aliases!(MyCodec, v3: "v3_name", ["v3_alias"], v2: "v2_name");
/// impl_extension_aliases!(MyCodec, v3: "v3_name", ["v3_alias"], v2: "v2_name", ["v2_alias"]);
/// impl_extension_aliases!(MyCodec, v3: "v3_name", ["v3_alias"], [v3_regex], v2: "v2_name", ["v2_alias"], [v2_regex]);
/// ```
#[macro_export]
macro_rules! impl_extension_aliases {
    // ============== V3-ONLY VARIANTS ==============

    // V3-only with name only
    ($type:ident, v3: $v3_default:expr) => {
        $crate::impl_extension_aliases!(@v3_impl $type, $v3_default, [], []);
        $crate::impl_extension_aliases!(@unsupported $type, V2);
        $crate::impl_extension_aliases!(@name_static $type);
    };

    // V3-only with string aliases
    ($type:ident, v3: $v3_default:expr, [$($v3_alias:expr),* $(,)?]) => {
        $crate::impl_extension_aliases!(@v3_impl $type, $v3_default, [$($v3_alias),*], []);
        $crate::impl_extension_aliases!(@unsupported $type, V2);
        $crate::impl_extension_aliases!(@name_static $type);
    };

    // V3-only with string aliases and regex
    ($type:ident, v3: $v3_default:expr, [$($v3_alias:expr),* $(,)?], [$($v3_regex:expr),* $(,)?]) => {
        $crate::impl_extension_aliases!(@v3_impl $type, $v3_default, [$($v3_alias),*], [$($v3_regex),*]);
        $crate::impl_extension_aliases!(@unsupported $type, V2);
        $crate::impl_extension_aliases!(@name_static $type);
    };

    // ============== V2-ONLY VARIANTS ==============

    // V2-only with name only
    ($type:ident, v2: $v2_default:expr) => {
        $crate::impl_extension_aliases!(@unsupported $type, V3);
        $crate::impl_extension_aliases!(@v2_impl $type, $v2_default, [], []);
        $crate::impl_extension_aliases!(@name_static $type);
    };

    // V2-only with string aliases
    ($type:ident, v2: $v2_default:expr, [$($v2_alias:expr),* $(,)?]) => {
        $crate::impl_extension_aliases!(@unsupported $type, V3);
        $crate::impl_extension_aliases!(@v2_impl $type, $v2_default, [$($v2_alias),*], []);
        $crate::impl_extension_aliases!(@name_static $type);
    };

    // V2-only with string aliases and regex
    ($type:ident, v2: $v2_default:expr, [$($v2_alias:expr),* $(,)?], [$($v2_regex:expr),* $(,)?]) => {
        $crate::impl_extension_aliases!(@unsupported $type, V3);
        $crate::impl_extension_aliases!(@v2_impl $type, $v2_default, [$($v2_alias),*], [$($v2_regex),*]);
        $crate::impl_extension_aliases!(@name_static $type);
    };

    // ============== BOTH VERSIONS VARIANTS ==============

    // Both versions, name only
    ($type:ident, v3: $v3_default:expr, v2: $v2_default:expr) => {
        $crate::impl_extension_aliases!(@v3_impl $type, $v3_default, [], []);
        $crate::impl_extension_aliases!(@v2_impl $type, $v2_default, [], []);
        $crate::impl_extension_aliases!(@name_static $type);
    };

    // Both versions, V3 with aliases
    ($type:ident, v3: $v3_default:expr, [$($v3_alias:expr),* $(,)?], v2: $v2_default:expr) => {
        $crate::impl_extension_aliases!(@v3_impl $type, $v3_default, [$($v3_alias),*], []);
        $crate::impl_extension_aliases!(@v2_impl $type, $v2_default, [], []);
        $crate::impl_extension_aliases!(@name_static $type);
    };

    // Both versions, both with aliases (no regex)
    ($type:ident,
     v3: $v3_default:expr, [$($v3_alias:expr),* $(,)?],
     v2: $v2_default:expr, [$($v2_alias:expr),* $(,)?]
    ) => {
        $crate::impl_extension_aliases!(@v3_impl $type, $v3_default, [$($v3_alias),*], []);
        $crate::impl_extension_aliases!(@v2_impl $type, $v2_default, [$($v2_alias),*], []);
        $crate::impl_extension_aliases!(@name_static $type);
    };

    // Full form with regex patterns
    ($type:ident,
     v3: $v3_default:expr, [$($v3_alias:expr),* $(,)?], [$($v3_regex:expr),* $(,)?],
     v2: $v2_default:expr, [$($v2_alias:expr),* $(,)?], [$($v2_regex:expr),* $(,)?]
    ) => {
        $crate::impl_extension_aliases!(@v3_impl $type, $v3_default, [$($v3_alias),*], [$($v3_regex),*]);
        $crate::impl_extension_aliases!(@v2_impl $type, $v2_default, [$($v2_alias),*], [$($v2_regex),*]);
        $crate::impl_extension_aliases!(@name_static $type);
    };

    // ============== INTERNAL HELPERS ==============

    // Internal: V3 implementation
    (@v3_impl $type:ident, $v3_default:expr, [$($v3_alias:expr),*], [$($v3_regex:expr),*]) => {
        $crate::paste::paste! {
            static [<$type:upper _ALIASES_V3>]: ::std::sync::LazyLock<::std::sync::RwLock<$crate::ExtensionAliasesConfig>> =
                ::std::sync::LazyLock::new(|| {
                    ::std::sync::RwLock::new($crate::ExtensionAliasesConfig::new(
                        $v3_default,
                        ::std::vec![$(::std::borrow::Cow::Borrowed($v3_alias)),*],
                        ::std::vec![$($v3_regex),*],
                    ))
                });

            impl $crate::ExtensionAliases<$crate::ZarrVersion3> for $type {
                fn aliases() -> ::std::sync::RwLockReadGuard<'static, $crate::ExtensionAliasesConfig> {
                    [<$type:upper _ALIASES_V3>].read().unwrap()
                }

                fn aliases_mut() -> ::std::sync::RwLockWriteGuard<'static, $crate::ExtensionAliasesConfig> {
                    [<$type:upper _ALIASES_V3>].write().unwrap()
                }
            }
        }
    };

    // Internal: V2 implementation
    (@v2_impl $type:ident, $v2_default:expr, [$($v2_alias:expr),*], [$($v2_regex:expr),*]) => {
        $crate::paste::paste! {
            static [<$type:upper _ALIASES_V2>]: ::std::sync::LazyLock<::std::sync::RwLock<$crate::ExtensionAliasesConfig>> =
                ::std::sync::LazyLock::new(|| {
                    ::std::sync::RwLock::new($crate::ExtensionAliasesConfig::new(
                        $v2_default,
                        ::std::vec![$(::std::borrow::Cow::Borrowed($v2_alias)),*],
                        ::std::vec![$($v2_regex),*],
                    ))
                });

            impl $crate::ExtensionAliases<$crate::ZarrVersion2> for $type {
                fn aliases() -> ::std::sync::RwLockReadGuard<'static, $crate::ExtensionAliasesConfig> {
                    [<$type:upper _ALIASES_V2>].read().unwrap()
                }

                fn aliases_mut() -> ::std::sync::RwLockWriteGuard<'static, $crate::ExtensionAliasesConfig> {
                    [<$type:upper _ALIASES_V2>].write().unwrap()
                }
            }
        }
    };

    // Internal: unsupported V3
    (@unsupported $type:ident, V3) => {
        $crate::paste::paste! {
            static [<$type:upper _ALIASES_V3>]: ::std::sync::LazyLock<::std::sync::RwLock<$crate::ExtensionAliasesConfig>> =
                ::std::sync::LazyLock::new(|| {
                    ::std::sync::RwLock::new($crate::ExtensionAliasesConfig::new(
                        "",
                        ::std::vec![],
                        ::std::vec![],
                    ))
                });

            impl $crate::ExtensionAliases<$crate::ZarrVersion3> for $type {
                fn aliases() -> ::std::sync::RwLockReadGuard<'static, $crate::ExtensionAliasesConfig> {
                    [<$type:upper _ALIASES_V3>].read().unwrap()
                }

                fn aliases_mut() -> ::std::sync::RwLockWriteGuard<'static, $crate::ExtensionAliasesConfig> {
                    [<$type:upper _ALIASES_V3>].write().unwrap()
                }

                fn matches_name(_name: &str) -> bool {
                    false
                }
            }
        }
    };

    // Internal: unsupported V2
    (@unsupported $type:ident, V2) => {
        $crate::paste::paste! {
            static [<$type:upper _ALIASES_V2>]: ::std::sync::LazyLock<::std::sync::RwLock<$crate::ExtensionAliasesConfig>> =
                ::std::sync::LazyLock::new(|| {
                    ::std::sync::RwLock::new($crate::ExtensionAliasesConfig::new(
                        "",
                        ::std::vec![],
                        ::std::vec![],
                    ))
                });

            impl $crate::ExtensionAliases<$crate::ZarrVersion2> for $type {
                fn aliases() -> ::std::sync::RwLockReadGuard<'static, $crate::ExtensionAliasesConfig> {
                    [<$type:upper _ALIASES_V2>].read().unwrap()
                }

                fn aliases_mut() -> ::std::sync::RwLockWriteGuard<'static, $crate::ExtensionAliasesConfig> {
                    [<$type:upper _ALIASES_V2>].write().unwrap()
                }

                fn matches_name(_name: &str) -> bool {
                    false
                }
            }
        }
    };

    // Internal: ExtensionNameStatic implementation
    (@name_static $type:ident) => {
        $crate::paste::paste! {
            impl $crate::ExtensionNameStatic for $type {
                const DEFAULT_NAME_FN: fn($crate::ZarrVersion) -> ::core::option::Option<::std::borrow::Cow<'static, str>> = |version| {
                    match version {
                        $crate::ZarrVersion::V2 => {
                            let aliases = [<$type:upper _ALIASES_V2>].read().unwrap();
                            if aliases.default_name.is_empty() {
                                ::core::option::Option::None
                            } else {
                                ::core::option::Option::Some(aliases.default_name.clone())
                            }
                        }
                        $crate::ZarrVersion::V3 => {
                            let aliases = [<$type:upper _ALIASES_V3>].read().unwrap();
                            if aliases.default_name.is_empty() {
                                ::core::option::Option::None
                            } else {
                                ::core::option::Option::Some(aliases.default_name.clone())
                            }
                        }
                    }
                };
            }
        }
    };
}
