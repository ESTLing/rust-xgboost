# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

### Added
- Rust bindings for XGBoost 3.2.x C API
- `DMatrix` for loading and manipulating data matrices:
  - `from_dense`, `from_csr`, `from_csc`, `load`, `load_binary`
  - Metadata: `num_rows`, `num_cols`, `shape`, `num_nonmissing`, `slice`
  - Labels/weights: `set_label`/`get_label`, `set_weight`/`get_weight`
  - Generic info: `get_float_info`/`set_float_info`, `get_uint_info`/`set_uint_info`
  - Serialization: `save` (binary format)
- `Booster` for training, prediction, evaluation, model persistence:
  - `train` with eval sets; `update` for single-iteration boosting
  - `predict`, `predict_margin`, `predict_leaf`, `predict_contributions`, `predict_interactions`
  - `evaluate` returning `HashMap<String, f32>`
  - `save`/`load` (JSON/UBJSON), `save_buffer`/`load_buffer` (raw bytes)
  - `dump_model` for model introspection
  - Attributes: `get_attribute`/`set_attribute`/`get_attribute_names`
  - Feature metadata: `get_feature_names`/`set_feature_names`, `FeatureMap`
- `PredictConfig` for fine-grained prediction control
- Flat string key-value parameter API (`&[(&str, &str)]`)
- Shared library auto-download from PyPI wheels (no cmake/ninja/submodule needed)
  - `XGBOOST_VERSION` env var to override version; `RUST_PYPI_INDEX` for mirror support
- Cross-platform: Linux (x86_64, aarch64), macOS (x86_64, aarch64), Windows (x86_64)

### Fixed
- `DMatrix::load` and `load_binary` correctly use `XGDMatrixCreateFromURI` with JSON config
- Deprecated `XGDMatrixCreateFromFile` replaced in all load paths
- Drop implementations safe against double-panic on cleanup failure

[0.1.0]: https://github.com/ESTLing/rust-xgboost/releases/tag/v0.1.0
