TOOLCHAIN := "nightly"
export RUST_BACKTRACE := "0"

# Display the available recipes
help:
    @just --list --unsorted

# Build (cargo check) with all / default / no default features
build:
    cargo +{{TOOLCHAIN}} check
    cargo +{{TOOLCHAIN}} check --all-features
    cargo +{{TOOLCHAIN}} check --no-default-features

# Test with all features
test:
    cargo +{{TOOLCHAIN}} test --all-features
    cargo +{{TOOLCHAIN}} test --all-features --examples

# Format with rustfmt
fmt:
    cargo +{{TOOLCHAIN}} fmt

# Lint with clippy
clippy:
    cargo +{{TOOLCHAIN}} clippy --all-features -- -D warnings
    cargo +{{TOOLCHAIN}} clippy --all-features --all-targets -- -A clippy::pedantic

# Generate documentation
doc:
    RUSTDOCFLAGS="-D warnings --cfg docsrs" cargo +{{TOOLCHAIN}} doc -Z unstable-options -Z rustdoc-scrape-examples --all-features --no-deps

# Build/test/clippy/doc/check formatting - recommended before a PR
check: build test clippy doc
    cargo +{{TOOLCHAIN}} fmt --all -- --check

# Build (WASM)
build_wasm:
    cargo check -p zarrs --target wasm32-unknown-unknown --no-default-features --features "ndarray crc32c gzip sharding transpose async"

# Build/clippy (WASM)
check_wasm: build_wasm
    cargo clippy -p zarrs --target wasm32-unknown-unknown --no-default-features --features "ndarray crc32c gzip sharding transpose async" -- -A clippy::arc_with_non_send_sync

# Run clippy with extra lints
_clippy_extra:
    cargo +{{TOOLCHAIN}} clippy --all-features -- -D warnings -W clippy::nursery -A clippy::significant-drop-tightening -A clippy::significant-drop-in-scrutinee

_miri:
    MIRIFLAGS="-Zmiri-disable-isolation -Zmiri-ignore-leaks -Zmiri-tree-borrows" cargo +{{TOOLCHAIN}} miri test -p zarrs --all-features

_coverage_install:
    cargo install cargo-llvm-cov --locked

_coverage_report:
    cargo +{{TOOLCHAIN}} llvm-cov --all-features --doctests --html

_coverage_file:
    cargo +{{TOOLCHAIN}} llvm-cov --all-features --doctests --lcov --output-path lcov.info

# Initialize snapshot test data submodule
init_snapshots:
    git submodule update --init zarrs/tests/data/snapshots

# Test codec snapshots
test_snapshots:
    cargo +{{TOOLCHAIN}} test --all-features -p zarrs --test codec_snapshot_tests

# Clean up generated snapshot files
clean_snapshots:
    rm -rf zarrs/tests/data/snapshots/*

# Update codec snapshots (requires submodule to be initialized)
update_snapshots: clean_snapshots
    UPDATE_SNAPSHOTS=1 cargo +{{TOOLCHAIN}} test --all-features -p zarrs --test codec_snapshot_tests
    @echo "Snapshot data updated. Check and commit as required."

# Add newly supported codec snapshots (previously unsupported combinations that now work)
add_snapshots:
    ADD_SNAPSHOTS=1 cargo +{{TOOLCHAIN}} test --all-features -p zarrs --test codec_snapshot_tests
    @echo "Newly supported snapshots added. Check and commit as required."
