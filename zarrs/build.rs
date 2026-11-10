#![allow(missing_docs)]

use std::path::Path;
use std::{env, fs};

fn main() {
    println!("cargo::rerun-if-changed=Cargo.toml");

    let major_version = env::var("CARGO_PKG_VERSION_MAJOR").unwrap();
    let minor_version = env::var("CARGO_PKG_VERSION_MINOR").unwrap();
    let patch_version = env::var("CARGO_PKG_VERSION_PATCH").unwrap();
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("version.rs");

    let version = format!(
        r"
pub(crate) const VERSION_MAJOR: u32 = {major_version};
pub(crate) const VERSION_MINOR: u32 = {minor_version};
pub(crate) const VERSION_PATCH: u32 = {patch_version};
"
    );
    if fs::read_to_string(&dest_path).ok().as_deref() != Some(version.as_str()) {
        fs::write(dest_path, version).unwrap();
    }
}
