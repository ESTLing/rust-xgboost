# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

### Added
- Rust bindings for XGBoost 3.2.x C API
- `DMatrix` for loading and manipulating data matrices (dense, CSR, CSC, LibSVM, binary)
- `Booster` for training, prediction, evaluation, model save/load
- Flat string key-value parameter API (like Python's `xgb.train()`)
- Shared library auto-download from PyPI wheels via build.rs
- Cross-platform support: Linux (x86_64, aarch64), macOS (x86_64, aarch64), Windows (x86_64)

[0.1.0]: https://github.com/marcomq/rust-xgboost/releases/tag/v0.1.0
