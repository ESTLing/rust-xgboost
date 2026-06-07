# rust-xgboost

Rust bindings for the [XGBoost](https://xgboost.ai) gradient boosting library.

The shared library is downloaded automatically from
[PyPI](https://pypi.org/project/xgboost/) at build time — no system
packages or manual compilation needed.

## Requirements

- **Rust 1.71+** (uses [`raw-dylib`](https://doc.rust-lang.org/reference/items/external-blocks.html#the-link-attribute) for Windows linking)
- **XGBoost 3.0.0 – 3.2.0** — these are the targeted versions. All are untested.
- **No system dependencies** — XGBoost headers and shared library are downloaded automatically

## Documentation

* [Documentation](https://docs.rs/xgboost-rs)

## Basic usage

```rust
use xgboost_rs::{DMatrix, Booster};

fn main() {
    // training matrix with 5 examples and 3 features
    let x_train = &[1.0, 1.0, 1.0,
                    1.0, 1.0, 0.0,
                    1.0, 1.0, 1.0,
                    0.0, 0.0, 0.0,
                    1.0, 1.0, 1.0];
    let mut dtrain = DMatrix::from_dense(x_train, 5).unwrap();
    dtrain.set_label(&[1.0, 1.0, 1.0, 0.0, 1.0]).unwrap();

    let x_test = &[0.7, 0.9, 0.6];
    let mut dtest = DMatrix::from_dense(x_test, 1).unwrap();
    dtest.set_label(&[1.0]).unwrap();

    let eval_sets = &[(&dtrain, "train"), (&dtest, "test")];

    // All parameters as string key-value pairs, just like Python
    let bst = Booster::train(
        &[("max_depth", "2"), ("eta", "1.0"), ("objective", "binary:logistic")],
        &dtrain,
        2,
        Some(eval_sets),
    ).unwrap();

    println!("{:?}", bst.predict(&dtest).unwrap());
}
```

## XGBoost version

`xgboost-rs` targets **XGBoost 3.0.0 – 3.2.0**. All versions are untested.
Set the `XGBOOST_VERSION` environment variable to choose a version:

```bash
XGBOOST_VERSION=3.2.0 cargo build
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
