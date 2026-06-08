# Rust xgboost-rs vs Python XGBoost — API Gap Analysis

> **Python reference**:  (XGBoost 3.2.0)
> **Rust current**:  (xgboost-rs 0.1.0)

This report catalogs every Python public API surface that has **no Rust equivalent**,
along with the estimated effort and priority for each gap.

---

## 1. Booster — Missing Methods

| # | Python Method | Priority | Effort | Description |
|---|--------------|----------|--------|-------------|
| 1 | `inplace_predict()` | **High** | Medium | Predict without constructing DMatrix. Takes raw data + config JSON. |
| 2 | `eval()` / `eval_set()` | **High** | Small | Evaluate on a single or multiple DMatrices with custom metric support. Rust has `evaluate()` which returns raw scores but lacks iteration tracking, custom metric, and formatted output. |
| 3 | `boost()` | **High** | Medium | One boosting round with custom gradient/hessian. Needed for custom training loops. |
| 4 | `copy()` | Medium | Small | Deep-copy a booster. |
| 5 | `reset()` | Medium | Small | Reset booster to initial state. |
| 6 | `save_config()` / `load_config()` | Medium | Small | Save/load internal configuration as JSON string. |
| 7 | `get_score()` / `get_fscore()` | Medium | Small | ✅ `get_score(importance_type, feature_names)` — returns `Vec<(String, f32)>` sorted descending. |
| 8 | `get_split_value_histogram()` | Low | Medium | Split value histogram for a given feature. |
| 9 | `trees_to_dataframe()` | Low | Large | Export trees as pandas DataFrame. Requires pandas dependency or alternative. |
| 10 | `num_boosted_rounds()` / `num_features()` | Medium | Small | Model metadata accessors. `num_features` already available indirectly via `evaluate()`. |
| 11 | `best_iteration` / `best_score` | Medium | Small | ⬜ Attributes written by `EarlyStopping::after_training`; accessible via `get_attribute()`. No dedicated accessor yet. |
| 12 | `get_categories()` | Low | Medium | Get categorical feature information. |
| 13 | `__getitem__()` / `__iter__()` | Low | Medium | Slice/iterate individual trees from the model. |

### Booster: Parameter & Predict Notes

- **Rust `predict()`** is split into `predict`, `predict_margin`, `predict_leaf`, `predict_contributions`, `predict_interactions`. Python uses a single `predict()` with keyword flags. Both approaches are valid.
- `predict_with_config(config)` provides tree limiting (`iteration_end`), covering the `iteration_range` use case.
- `predict_with_best_epoch(epoch)` is a convenience for early stopping workflows.
- **Rust `dump_model()`** returns a `String`; Python returns `List[str]` via `get_dump()`. Rust also has `dump_model_vec()`. Parity is acceptable.
- **`set_param()` / `set_params()`** exist in both.
- **Attribute system**: Rust has `get_attribute`/`set_attribute`/`get_attribute_names`, Python has `attr`/`attributes`/`set_attr`. Covered.

---

## 2. DMatrix — Missing Methods

| # | Python Method | Priority | Effort | Description |
|---|--------------|----------|--------|-------------|
| 1 | `get_data()` | Low | Medium | Extract underlying data as scipy CSR. May not be practical in Rust without sparse matrix type. |
| 2 | `get_quantile_cut()` | Low | Medium | Get quantile cut points for histogram-based training. |
| 3 | `get_categories()` | Low | Medium | Get categorical feature data. |
| 4 | `set_base_margin()` / `get_base_margin()` | Medium | Small | Access base margin. Covered generically via `set_float_info`/`get_float_info` with `KEY_BASE_MARGIN`, but no convenience method. |
| 5 | `set_group()` / `get_group()` | Medium | Small | Group info for ranking. Same situation as base_margin — covered via generic info API but no convenience. |
| 6 | `set_info()` | Low | Small | Batch set multiple info fields at once. Convenience, not essential. |
| 7 | `feature_names` / `feature_types` properties | Low | Small | Get/set as properties. Rust uses `get_str_feature_info`/`set_str_feature_info`. |
| 8 | `set_float_info_npy2d()` | Low | Small | 2D float info (for multi-target). Niche. |

### DMatrix: Constructor Parity

Python DMatrix constructor dispatches on input type (numpy array, scipy sparse, file path, dataloader). Rust has explicit factory functions:
- `DMatrix::from_dense()` → numpy dense
- `DMatrix::from_csr()` → scipy CSR
- `DMatrix::from_csc()` → scipy CSC
- `DMatrix::load()` → file path

Missing input types:
- Python `pandas.DataFrame` / `cudf` / `polars` / `arrow.Table` — **not expected** in a C-API binding layer.
- `QuantileDMatrix` — see section 3.

---

## 3. Missing Classes (Entirely Absent)

| # | Python Class | Priority | Effort | Status | Description |
|---|--------------|----------|--------|--------|-------------|
| 1 | **`QuantileDMatrix`** | **High** | Large | ❌ | Memory-efficient DMatrix with built-in quantization. Required for large-scale datasets. Supports external memory via `DataIter`. Key features: `max_bin`, ref DMatrix for consistent binning, `max_quantile_batches`. |
| 2 | **`ExtMemQuantileDMatrix`** | Medium | Large | ❌ | External-memory QuantileDMatrix. Processes data in streaming batches. |
| 3 | **`DataIter`** | **High** | Medium | ❌ | Abstract data iterator for streaming/proxy DMatrix. Required for both QuantileDMatrix and ExtMemQuantileDMatrix. Provides `reset()`, `next()`, callback system. |
| 4 | **`TrainingCallback`** (hierarchy) | **High** | Medium | ✅ `TrainingCallback` trait, `EvaluationMonitor`, `EarlyStopping`. Missing: `TrainingCheckPoint`, `LearningRateScheduler`. |
| 5 | **`CVPack`** | Medium | Small | ❌ | Cross-validation fold helper. |
| 6 | **`RabitTracker`** | Low | Large | ❌ | Distributed training tracker (MPI-like). |
| 7 | **`FederatedTracker`** | Low | Large | ❌ | Federated learning tracker. |

---

## 4. Missing Functions

| # | Python Function | Priority | Effort | Description |
|---|-----------------|----------|--------|-------------|
| 1 | **`cv()`** | **High** | Large | Cross-validation. Needs fold splitting, custom objective/metric support, stratified folds, shuffle, seed. Outputs evaluation history. |
| 2 | `set_config()` / `get_config()` / `config_context()` | Medium | Small | Global XGBoost configuration (verbosity, nthread, etc.). Rust currently passes thread count via DMatrix config JSON. |
| 3 | `build_info()` | Low | Small | Return build info dict. |
| 4 | `plot_importance()` | Low | N/A | Matplotlib-based plotting. Out of scope for core bindings; would belong in a separate visualization crate. |
| 5 | `plot_tree()` | Low | N/A | Same as above. |
| 6 | `to_graphviz()` | Low | N/A | Graphviz tree visualization. Out of scope for core bindings. |

---

## 5. Missing Features in Existing Functions

### `Booster::train()` — Missing Parameters

| # | Feature | Priority | Effort | Status |
|---|---------|----------|--------|--------|
| 1 | `obj` — Custom objective function | **High** | Medium | ❌ |
| 2 | `feval` / `custom_metric` — Custom evaluation metric | **High** | Small | ✅ `set_custom_metric()` |
| 3 | `callbacks` — Training callback pipeline | **High** | Large | ✅ `TrainingCallback` trait + `add_callback()` |
| 4 | `xgb_model` — Continue training from existing model | Medium | Small | ❌ |
| 5 | `evals_result` — Store evaluation history | Medium | Small | ✅ `train()` returns `EvalsLog` |
| 6 | `early_stopping_rounds` — Built-in early stopping | **High** | Medium | ✅ `EarlyStopping` callback |
| 7 | `verbose_eval` — Control evaluation output frequency | Medium | Small | ✅ `EvaluationMonitor` callback |
| 8 | `maximize` — Direction for early stopping | Medium | Small | ✅ in `EarlyStopping` |

### `DMatrix::load()` / `load_binary()` — Missing Parameters

| # | Feature | Priority | Effort |
|---|---------|----------|--------|
| 1 | `data_split_mode` parameter | Low | Trivial — just add to JSON config |

### Feature Map

Python's `get_score()` with `importance_type` supports: `"weight"`, `"gain"`, `"cover"`, `"total_gain"`, `"total_cover"`. ✅ Implemented.

---

## 6. Distributed / Federated Learning

| # | Feature | Priority | Effort | Description |
|---|---------|----------|--------|-------------|
| 1 | `collective` module | Low | Large | MPI-like collective operations (`allreduce`, `broadcast`, `init`, `finalize`). |
| 2 | `federated` module | Low | Large | Federated learning server. |
| 3 | `tracker.RabitTracker` | Low | Large | Dask/Rabit distributed tracker. |

**Verdict**: Out of scope for the initial binding crate. These are advanced production features.

---

## 7. Prioritized Implementation Roadmap

### Phase 1: Core Completeness (Must Have)

| # | Feature | Status |
|---|---------|--------|
| 1 | Custom objective + custom metric in `Booster::train()` | ⬜ custom metric ✅, objective ❌ |
| 2 | `Booster::boost()` — single-step boosting with custom grad/hess | ❌ |
| 3 | `Booster::inplace_predict()` — predict from raw data | ❌ |
| 4 | `EarlyStopping` callback (+ `evals_result`, `early_stopping_rounds`, `maximize`, `verbose_eval`) | ✅ |
| 5 | `Booster::eval_set()` — multi-eval evaluation with iteration tracking | ✅ |

### Phase 2: Feature Parity (Should Have)

| # | Feature | Status |
|---|---------|--------|
| 6 | `QuantileDMatrix` — histogram-based memory-efficient DMatrix | ❌ |
| 7 | `DataIter` — streaming data iterator abstraction | ❌ |
| 8 | `Booster::get_score()` — feature importance | ✅ |
| 9 | `cv()` — cross-validation | ❌ |
| 10 | `copy()` / `reset()` — booster lifecycle | ❌ |
| 11 | `save_config()` / `load_config()` — config serialization | ❌ |
| 12 | `num_boosted_rounds()` / `best_iteration` / `best_score` — metadata access | ⬜ attributes written by `EarlyStopping`, no accessors yet |

### Phase 3: Advanced Features (Nice to Have)

13. **`ExtMemQuantileDMatrix`** — external memory
14. **`TrainingCheckPoint` / `LearningRateScheduler`** — remaining callbacks
15. **`get_split_value_histogram()`** — split analysis
16. **`set_config()` / `get_config()`** — global config
17. **`trees_to_dataframe()` / tree iteration** — model introspection

### Phase 4: Out of Scope for Core Crate

18. Distributed/federated learning
19. Plotting functions (separate crate)
20. Scikit-learn API wrappers (separate crate)

---

## 8. Notes on Design Philosophy Alignment

### Good Matches
- Flat `&[(&str, &str)]` params ↔ Python dict params
- `From<double>` for missing ↔ Python `np.nan`
- Separate factory methods `from_dense`/`from_csr`/`from_csc` vs Python's type-dispatch constructor (both valid for their language)
- `XGBResult<T>` ↔ Python exceptions

### Divergences to Consider
- Python uses `eval_set()` returning formatted string; Rust uses `evaluate()` returning `HashMap<String, f32>`. Consider aligning to one approach.
- Python `predict()` uses keyword flags; Rust uses separate methods. Both valid.
- Python Booster constructor can take `model_file`; Rust uses `Booster::load()`. Both valid.
