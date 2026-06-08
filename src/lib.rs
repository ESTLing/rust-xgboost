//! Rust wrapper around the [XGBoost](https://xgboost.ai) machine learning library.
//!
//! Provides a high level interface for training machine learning models using
//! [gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting).
//!
//! Currently in the early stages of development, API is likely to be fairly unstable as new
//! features are added.
//!
//! # Basic usage example
//!
//! ```
//! use xgboost_rs::{DMatrix, Booster, EvaluationMonitor, EvalsLog};
//!
//! fn main() {
//!     let x_train = &[1.0, 1.0, 1.0,
//!                     1.0, 1.0, 0.0,
//!                     1.0, 1.0, 1.0,
//!                     0.0, 0.0, 0.0,
//!                     1.0, 1.0, 1.0];
//!     let mut dtrain = DMatrix::from_dense(x_train, 5).unwrap();
//!     dtrain.set_label(&[1.0, 1.0, 1.0, 0.0, 1.0]).unwrap();
//!
//!     let x_test = &[0.7, 0.9, 0.6];
//!     let mut dtest = DMatrix::from_dense(x_test, 1).unwrap();
//!     dtest.set_label(&[1.0]).unwrap();
//!
//!     let mut booster = Booster::new(2).unwrap();
//!     booster.set_params(&[
//!         ("max_depth", "2"), ("eta", "1.0"), ("objective", "binary:logistic"),
//!     ]).unwrap();
//!
//!     booster.add_callback(Box::new(EvaluationMonitor::new(1)));
//!     let history = booster.train(
//!         &dtrain,
//!         2,
//!         &[(&dtrain, "train"), (&dtest, "test")],
//!     ).unwrap();
//!
//!     println!("{:?}", booster.predict(&dtest, &PredictConfig::default()).unwrap());
//! }
//! ```
//!
#[macro_use]
extern crate log;
extern crate libc;
extern crate tempfile;
extern crate xgboost_sys;

macro_rules! xgb_call {
    ($x:expr) => {
        XGBError::check_return_value(unsafe { $x })
    };
}

// Sentinel string used as a placeholder for the JSON5 literal `NaN` in config values.
// `serde_json` serializes `f64::NAN` as `null`, but XGBoost's JSON parser expects the
// bare literal `NaN`. Used in `json_cstr!` — the macro replaces it after serialization.
const NAN_SENTINEL: &str = "\x00NaN";

/// Build an XGBoost-compatible JSON string from key-value pairs, returning a [`CString`].
///
/// After serialization, the `NAN_SENTINEL` placeholder is replaced with the bare
/// literal `NaN`, matching XGBoost's JSON parser expectations.
///
/// Values can be any type that implements `serde::Serialize`. Arrays use `[v1, v2, ...]` syntax.
///
/// [`CString`]: std::ffi::CString
///
/// # Examples
///
/// ```ignore
/// // DMatrix construction config
/// let config = json_cstr!("missing" => NAN_SENTINEL, "nthread" => 0);
/// // → {"missing":NaN,"nthread":0}
///
/// // __array_interface__ for dense data
/// let ai = json_cstr!(
///     "data" => [ptr as usize, false],
///     "shape" => [3, 2],
///     "typestr" => "<f4",
///     "version" => 3,
/// );
/// // → {"data":[1407000,false],"shape":[3,2],"typestr":"<f4","version":3}
/// ```
macro_rules! json_cstr {
    ($($key:expr => $val:expr),* $(,)?) => {{
        let value = serde_json::json!({$($key: $val),*});
        let json_cstr = value.to_string().replace("\"\\u0000NaN\"", "NaN");
        std::ffi::CString::new(json_cstr).unwrap()
    }};
}

mod error;
pub use error::{XGBError, XGBResult};

mod callback;
mod dmatrix;
pub use dmatrix::{DMatrix, KEY_GROUP, KEY_GROUP_PTR, KEY_LABEL, KEY_WEIGHT, KEY_BASE_MARGIN, KEY_LABEL_LOWER_BOUND, KEY_LABEL_UPPER_BOUND, KEY_QID};

mod booster;
pub use booster::{Booster, EarlyStopping, EvalsLog, EvaluationMonitor, FeatureMap, FeatureType, PredictConfig, PredictType, TrainingCallback};
use std::{ffi, path::Path};

#[cfg(not(target_os = "windows"))]
pub fn path_to_c_str<P: AsRef<Path>>(path: P) -> ffi::CString {
    use std::os::unix::ffi::OsStrExt;
    ffi::CString::new(path.as_ref().as_os_str().as_bytes()).unwrap()
}
#[cfg(target_os = "windows")]
pub fn path_to_c_str<P: AsRef<Path>>(path: P) -> ffi::CString {
    ffi::CString::new(path.as_ref().as_os_str().as_encoded_bytes()).unwrap()
}
