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
//! use xgb::{DMatrix, Booster};
//!
//! fn main() {
//!     let x_train = &[1.0, 1.0, 1.0,
//!                     1.0, 1.0, 0.0,
//!                     1.0, 1.0, 1.0,
//!                     0.0, 0.0, 0.0,
//!                     1.0, 1.0, 1.0];
//!     let mut dtrain = DMatrix::from_dense(x_train, 5).unwrap();
//!     dtrain.set_labels(&[1.0, 1.0, 1.0, 0.0, 1.0]).unwrap();
//!
//!     let x_test = &[0.7, 0.9, 0.6];
//!     let mut dtest = DMatrix::from_dense(x_test, 1).unwrap();
//!     dtest.set_labels(&[1.0]).unwrap();
//!
//!     let eval_sets = &[(&dtrain, "train"), (&dtest, "test")];
//!
//!     let bst = Booster::train(
//!         &[("max_depth", "2"), ("eta", "1.0"), ("objective", "binary:logistic")],
//!         &dtrain,
//!         2,
//!         Some(eval_sets),
//!     ).unwrap();
//!
//!     println!("{:?}", bst.predict(&dtest).unwrap());
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

mod error;
pub use error::{XGBError, XGBResult};

mod dmatrix;
pub use dmatrix::DMatrix;

mod booster;
pub use booster::{Booster, FeatureMap, FeatureType, PredictConfig, PredictType};
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
