# Contributing to zarrs

Thank you for your interest in contributing to `zarrs`!

## How to Contribute

- **Issues:** Report bugs or suggest enhancements by opening an issue on GitHub.
Please check existing issues first.
- **Pull Requests (PRs):** Code and documentation contributions are welcomed via PRs.
For a major change, please open an issue first to discuss the proposal.

## Issue Guidelines

When opening an issue, please provide:

- **Clear Title:** A brief summary of the issue, feature request, proposal, etc.
- **Bug Reports:** Include the `zarrs` version, steps to reproduce, expected vs. actual behavior, and any relevant logs or errors.
- **Feature Requests:** Describe the feature, its use case, and why it's needed.

## Pull Request Guidelines

- **Add Tests**: Add tests for any new functionality or bug fixes.
- **Code Quality**: Run `just check`[^1] to build `zarrs`, run `rustfmt` and `clippy`, and ensure all tests pass. Or alternatively:
  - Build with
    - `cargo check`
    - `cargo check --all-features`
    - `cargo check --no-default-features`
  - Format your code using `cargo fmt`.
  - Lint with `cargo clippy --all-features -- -D warnings`.
  - Run tests using `cargo test --all-features`.
  - CI will run additional checks across multiple platforms and check semver and minimal versions.
- **Changelog**: Please include a suggested entry for the `CHANGELOG.md` in your PR description if the PR title is not sufficient.

---

Thanks again for contributing!

[^1]: Just a command runner - https://github.com/casey/just
