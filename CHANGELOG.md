# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.4] - 2026-06-19

### Added
- `examples/ranking/` — complete learning-to-rank demo: group/qid metadata, `rank:ndcg` objective, early stopping with NDCG maximization.

### Changed
- `DMatrix::set_float_info` / `set_uint_info` migrated from deprecated `XGDMatrixSetFloatInfo` / `XGDMatrixSetUIntInfo` to unified `XGDMatrixSetInfoFromInterface` (C API via `__array_interface__` JSON).

## [0.2.3] - 2026-06-19

### Added
- `PredictConfig::missing` — controls the missing-value sentinel in prediction config JSON (defaults to `f32::NAN`). Required by `XGBoosterPredictFromDense` for `inplace_predict`.

## [0.2.2] - 2026-06-08

### Fixed
- `EarlyStopping` with `save_best(true)` lost `best_iteration` / `best_score` attributes after training — `slice_trees` (which creates a new booster handle) was called after `set_attribute`, dropping the attributes. Reversed the call order so attributes are written on the pruned handle.

## [0.2.1] - 2026-06-08

### Added
- `inplace_predict(data, num_rows, config)` — predict directly from `&[f32]` without constructing `DMatrix`
- `EarlyStopping::with_save_best()` — prune over-trained trees after training, matching Python's `save_best=True`

### Changed
- `predict()` now takes `&PredictConfig` directly; `predict_with_config` removed
- `slice_trees()` internal method added on `Booster` for save_best tree pruning

### Fixed
- Test added: `save_best` predictions identical to `predict_with_best_epoch`

## [0.2.0] - 2024-06-08

### Added
- Training callback system (`TrainingCallback` trait):
  - `EvaluationMonitor` — prints metrics at fixed interval via `log::info!`
  - `EarlyStopping` — stops training when metric stops improving; writes `best_iteration`/`best_score` attributes
  - `add_callback()` setter — callbacks stored on `Booster`, `train()` stays at 3 params
- `EvalsLog` type — per-iteration metric history returned by `train()`, passed to callbacks
- `set_custom_metric()` — user-defined `(name, score)` pairs, visible to callbacks and early stopping
- `predict_with_best_epoch()` — convenience for early-stopping workflow
- `set_params()` — batch parameter setter on `Booster`
- `Booster::new(num_features)` — creates internal dummy matrix, no need to pass `&DMatrix` at construction
- `json_cstr!` macro — constructs XGBoost-compatible JSON `CString` with NaN handling

### Changed
- `train()` is now an instance method (`&mut self`) instead of a static constructor
- Callbacks and custom metrics moved from `train()` parameters to `Booster` fields (setters)
- `predict()` switched from `XGBoosterPredict` to `XGBoosterPredictFromDMatrix` C API
- `PredictConfig::as_json()` renamed to `as_cstr()`, returns `CString`
- C API output (`print!`/`println!`) replaced with `log` crate macros in public functions

## [0.1.0] - 2024-06-08

### Added
- Rust bindings for XGBoost 3.2.x C API
- `DMatrix` for loading and manipulating data matrices:
  - `from_dense`, `from_csr`, `from_csc`, `load`, `load_binary`
  - Metadata: `num_rows`, `num_cols`, `shape`, `num_nonmissing`, `slice`
  - Labels/weights: `set_label`/`get_label`, `set_weight`/`get_weight`
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

[0.2.4]: https://github.com/ESTLing/rust-xgboost/compare/v0.2.3...HEAD
[0.2.3]: https://github.com/ESTLing/rust-xgboost/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/ESTLing/rust-xgboost/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/ESTLing/rust-xgboost/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/ESTLing/rust-xgboost/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ESTLing/rust-xgboost/releases/tag/v0.1.0
