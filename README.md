# rust-xgboost

Rust bindings for the [XGBoost](https://xgboost.ai) gradient boosting library.

The shared library is downloaded automatically from
[PyPI](https://pypi.org/project/xgboost/) at build time — no system
packages or manual compilation needed.

## Highlights

- **Flat string key-value API** — params just like Python's `xgb.train()`
- **Training callbacks** — `EvaluationMonitor`, `EarlyStopping`, or your own impl of `TrainingCallback`
- **Custom evaluation metrics** — user-defined `(name, score)` pairs, visible to callbacks
- **Zero system deps** — XGBoost shared library auto-downloaded at build time

## Requirements

- **Rust 1.71+** (uses [`raw-dylib`](https://doc.rust-lang.org/reference/items/external-blocks.html#the-link-attribute) for Windows linking)
- **No system dependencies** — XGBoost headers and shared library are downloaded automatically

## Basic usage

```rust
use xgboost_rs::{Booster, DMatrix, EvaluationMonitor, EarlyStopping};

// Training data: 5 rows × 3 features
let x_train = &[1.0, 1.0, 1.0,  1.0, 1.0, 0.0,  1.0, 1.0, 1.0,
                0.0, 0.0, 0.0,  1.0, 1.0, 1.0];
let mut dtrain = DMatrix::from_dense(x_train, 5).unwrap();
dtrain.set_label(&[1.0, 1.0, 1.0, 0.0, 1.0]).unwrap();

let x_test = &[0.7, 0.9, 0.6];
let mut dtest = DMatrix::from_dense(x_test, 1).unwrap();
dtest.set_label(&[1.0]).unwrap();

// Create booster, set params, add callbacks
let mut booster = Booster::new(3).unwrap();
booster.set_params(&[
    ("max_depth", "2"),
    ("eta", "1.0"),
    ("objective", "binary:logistic"),
    ("eval_metric", "logloss"),
]).unwrap();

booster.add_callback(Box::new(EvaluationMonitor::new(1)));
booster.add_callback(Box::new(EarlyStopping::new(
    3, "test", "", false,
)));

let history = booster.train(&dtrain, 100, &[(&dtest, "test")]).unwrap();
println!("{:?}", booster.predict(&dtest).unwrap());
```

## XGBoost version

`xgboost-rs` targets **XGBoost 3.2.0** (tested on Windows). Versions 3.0.0 and 3.1.0
are also supported but untested. Set the `XGBOOST_VERSION` environment
variable to choose a version:

```bash
XGBOOST_VERSION=3.1.0 cargo build
```

## Using a PyPI mirror

Set `RUST_PYPI_INDEX` to a mirror URL for faster downloads (e.g. in China):

```bash
# Tsinghua mirror
export RUST_PYPI_INDEX=https://pypi.tuna.tsinghua.edu.cn
```

## Supported Platforms

| Platform    | Architecture | Status                          |
|-------------|-------------|---------------------------------|
| Linux       | x86_64      | Supported, not tested           |
| Linux       | aarch64     | Supported, not tested           |
| macOS       | x86_64      | Supported, not tested           |
| macOS       | aarch64     | Supported, not tested           |
| Windows     | x86_64      | Supported (Rust 1.71+ required) |

## License

MIT
