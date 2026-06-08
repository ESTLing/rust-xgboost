use crate::dmatrix::DMatrix;
use crate::error::XGBError;
use std::collections::{BTreeMap, HashMap};
use std::io::{self, BufRead, BufReader, Write};
use std::os::raw;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::{ffi, fmt, fs::File, ptr, slice};

use super::XGBResult;

/// Used to control the return type of predictions made by C Booster API.
enum PredictOption {
    OutputMargin,
    PredictLeaf,
    PredictContribitions,
    //ApproximateContributions,
    PredictInteractions,
}

#[derive(Default, Debug, Clone)]
pub enum PredictType {
    #[default]
    Normal = 0,
    OutputMargin = 1,
    PredictContribitions = 2,
    PredictApproximateContributions = 3,
    PredictFeatureInteractions = 4,
    PredictApproximateFeatureInteractions = 5,
    PredictLeafTraining = 6,
}

#[derive(Default)]
pub struct PredictConfig {
    pub _type: PredictType,
    pub training: bool,
    pub iteration_begin: i64,
    pub iteration_end: i64,
    pub strict_shape: bool,
}

impl PredictConfig {
    /// returns 0 terminated json of the config, mainly for usage in predict_matrix
    pub fn as_json(&self) -> String {
        format!(
            "{{\"type\":{},\"training\":{},\"iteration_begin\":{},\"iteration_end\":{},\"strict_shape\":{}}}\0",
            self._type.clone() as usize,
            self.training,
            self.iteration_begin,
            self.iteration_end,
            self.strict_shape
        )
    }
}

impl PredictOption {
    /// Convert list of options into a bit mask.
    fn options_as_mask(options: &[PredictOption]) -> i32 {
        let mut option_mask = 0x00;
        for option in options {
            let value = match *option {
                PredictOption::OutputMargin => 0x01,
                PredictOption::PredictLeaf => 0x02,
                PredictOption::PredictContribitions => 0x04,
                //PredictOption::ApproximateContributions => 0x08,
                PredictOption::PredictInteractions => 0x10,
            };
            option_mask |= value;
        }

        option_mask
    }
}

/// Evaluation history, mapping dataset-name → metric-name → scores per iteration.
///
/// Returned by [`Booster::train`] and passed to [`TrainingCallback::after_iteration`].
///
/// ```
/// # use std::collections::BTreeMap;
/// # type EvalsLog = BTreeMap<String, BTreeMap<String, Vec<f32>>>;
/// // { "train": { "rmse": [0.12, 0.09, 0.08] } }
/// ```
pub type EvalsLog = BTreeMap<String, BTreeMap<String, Vec<f32>>>;

/// Trait for training callbacks, matching Python's
/// [`TrainingCallback`](https://xgboost.readthedocs.io/en/latest/python/callbacks.html).
///
/// Callbacks are called after each boosting iteration. Return `true` from
/// [`after_iteration`](TrainingCallback::after_iteration) to stop training early.
///
/// # Examples
///
/// ```
/// use xgboost_rs::{Booster, TrainingCallback, EvalsLog};
///
/// struct MyCallback;
///
/// impl TrainingCallback for MyCallback {
///     fn after_iteration(&mut self, _booster: &Booster, epoch: u32, evals_log: &EvalsLog) -> bool {
///         println!("Epoch {}: {:?}", epoch, evals_log);
///         false // never stop
///     }
/// }
/// ```
pub trait TrainingCallback {
    /// Called after each boosting iteration.
    ///
    /// * `booster` — the current model (read-only)
    /// * `epoch` — current iteration number (0-indexed)
    /// * `evals_log` — accumulated evaluation history up to this iteration
    ///
    /// Returns `true` to stop training early (early stopping).
    fn after_iteration(&mut self, booster: &Booster, epoch: u32, evals_log: &EvalsLog) -> bool;

    /// Called after training completes (even if stopped early).
    ///
    /// Receives `&mut Booster` so callbacks can write attributes (e.g. `best_iteration`).
    fn after_training(&mut self, _booster: &mut Booster) {}
}

/// Built-in callback that prints evaluation metrics at a fixed interval.
///
/// Matches Python's [`EvaluationMonitor`](https://xgboost.readthedocs.io/en/latest/python/callbacks.html).
///
/// # Examples
///
/// ```no_run
/// # use xgboost_rs::{Booster, DMatrix, EvaluationMonitor};
/// # let dtrain = DMatrix::from_dense(&[1.0], 1).unwrap();
/// # let mut booster = Booster::new(1).unwrap();
/// let mut monitor = EvaluationMonitor::new(1); // print every iteration
/// booster.add_callback(Box::new(monitor));
/// booster.train(&dtrain, 10, &[]).unwrap();
/// ```
pub struct EvaluationMonitor {
    period: usize,
}

impl EvaluationMonitor {
    /// Create a new monitor that prints evaluation metrics every `period` iterations.
    ///
    /// Use `period = 1` to print every iteration.
    pub fn new(period: usize) -> Self {
        assert!(period > 0, "EvaluationMonitor period must be greater than 0");
        Self { period }
    }
}

impl TrainingCallback for EvaluationMonitor {
    fn after_iteration(&mut self, _booster: &Booster, epoch: u32, evals_log: &EvalsLog) -> bool {
        if evals_log.is_empty() {
            return false;
        }

        if epoch as usize % self.period == 0 || self.period == 1 {
            let mut msg = format!("[{}]", epoch);
            for (data_name, metrics) in evals_log {
                for (metric_name, scores) in metrics {
                    if let Some(score) = scores.last() {
                        msg.push_str(&format!("\t{}-{}:{:.5}", data_name, metric_name, score));
                    }
                }
            }
            info!("{}", msg);
        }

        false // never stop training
    }
}

/// Built-in callback for early stopping, matching Python's
/// [`EarlyStopping`](https://xgboost.readthedocs.io/en/latest/python/callbacks.html).
///
/// Monitors a metric on a given eval dataset. If it does not improve for `rounds`
/// consecutive iterations, training stops.
///
/// If `metric_name` or `data_name` is `None`, the last one in [`EvalsLog`] is used.
///
/// # Examples
///
/// ```no_run
/// # use xgboost_rs::{Booster, DMatrix, EarlyStopping, EvaluationMonitor};
/// # let dtrain = DMatrix::from_dense(&[1.0, 2.0], 1).unwrap();
/// # let dtest = DMatrix::from_dense(&[1.0, 2.0], 1).unwrap();
/// let mut booster = Booster::new(2).unwrap();
/// booster.set_params(&[("max_depth", "2"), ("eta", "1.0"), ("objective", "binary:logistic")]).unwrap();
/// booster.set_param("eval_metric", "logloss").unwrap();
/// booster.add_callback(Box::new(EvaluationMonitor::new(1)));
/// booster.add_callback(Box::new(EarlyStopping::new(5, "test", "logloss", false)));
/// booster.train(&dtrain, 100, &[(&dtest, "test")]).unwrap();
/// ```
pub struct EarlyStopping {
    rounds: usize,
    metric_name: Option<String>,
    data_name: Option<String>,
    maximize: Option<bool>,
    min_delta: f32,
    current_rounds: usize,
    best_score: Option<f32>,
    best_epoch: Option<u32>,
}

impl EarlyStopping {
    /// Create an early-stopping callback.
    ///
    /// * `rounds` — consecutive rounds without improvement before stopping
    /// * `data_name` — eval dataset name to monitor (last dataset if empty string)
    /// * `metric_name` — metric to monitor (last metric if empty string)
    /// * `maximize` — `true` if higher is better, `false` if lower. `None` for auto-detect
    ///   (currently defaults to minimize).
    pub fn new(rounds: usize, data_name: &str, metric_name: &str, maximize: bool) -> Self {
        assert!(rounds > 0, "EarlyStopping rounds must be greater than 0");
        Self {
            rounds,
            data_name: if data_name.is_empty() { None } else { Some(data_name.to_string()) },
            metric_name: if metric_name.is_empty() { None } else { Some(metric_name.to_string()) },
            maximize: Some(maximize),
            min_delta: 0.0,
            current_rounds: 0,
            best_score: None,
            best_epoch: None,
        }
    }

    /// Set the minimum absolute change in score to qualify as an improvement.
    ///
    /// Defaults to `0.0` — any improvement counts.
    pub fn with_min_delta(mut self, delta: f32) -> Self {
        self.min_delta = delta;
        self
    }

    /// Returns the epoch with the best score, if any improvement was recorded.
    ///
    /// Only valid after training completes.
    pub fn best_epoch(&self) -> Option<u32> {
        self.best_epoch
    }
}

impl TrainingCallback for EarlyStopping {
    fn after_iteration(&mut self, _booster: &Booster, epoch: u32, evals_log: &EvalsLog) -> bool {
        if evals_log.is_empty() {
            return false;
        }

        // Pick data set: user-specified or last one
        let data_name = self
            .data_name
            .as_deref()
            .or_else(|| evals_log.keys().last().map(|s| s.as_str()))
            .unwrap_or("");
        let data_log = match evals_log.get(data_name) {
            Some(d) => d,
            None => return false,
        };

        // Pick metric: user-specified or last one
        let metric_name = self
            .metric_name
            .as_deref()
            .or_else(|| data_log.keys().last().map(|s| s.as_str()))
            .unwrap_or("");
        let scores = match data_log.get(metric_name) {
            Some(s) => s,
            None => return false,
        };

        let score = match scores.last() {
            Some(s) => *s,
            None => return false,
        };

        // Determine direction
        let maximize = self.maximize.unwrap_or(false);

        match self.best_score {
            None => {
                self.best_score = Some(score);
                self.best_epoch = Some(epoch);
                self.current_rounds = 0;
            }
            Some(best) => {
                let improved = if maximize {
                    score - self.min_delta > best
                } else {
                    score + self.min_delta < best
                };
                if improved {
                    self.best_score = Some(score);
                    self.best_epoch = Some(epoch);
                    self.current_rounds = 0;
                } else {
                    self.current_rounds += 1;
                }
            }
        }

        self.current_rounds >= self.rounds
    }

    fn after_training(&mut self, booster: &mut Booster) {
        // Write best_iteration attribute so predict() can limit trees
        if let Some(best_epoch) = self.best_epoch {
            // best_iteration is the epoch of the best score, stored as str
            let _ = booster.set_attribute("best_iteration", &best_epoch.to_string());
            if let Some(score) = self.best_score {
                let _ = booster.set_attribute("best_score", &score.to_string());
            }
        }
        // Reset for reuse
        self.current_rounds = 0;
        self.best_score = None;
        self.best_epoch = None;
    }
}

/// Core model in XGBoost, containing functions for training, evaluating and predicting.
///
/// Create with [`new`](struct.Booster.html#method.new), then call [`train`](struct.Booster.html#method.train).
///
/// For iterative control, use [`update`](struct.Booster.html#method.update) in a loop instead of `train`.
pub struct Booster {
    handle: xgboost_sys::BoosterHandle,
    _dummy_dmatrix: Option<DMatrix>,
    callbacks: Vec<Box<dyn TrainingCallback>>,
    custom_metric: Option<Box<dyn Fn(&[f32], &DMatrix) -> Vec<(String, f32)>>>,
}

impl Booster {
    /// Create a new Booster for training.
    ///
    /// `num_features` tells XGBoost the feature dimensionality. An internal dummy matrix
    /// is created and kept alive for the booster's lifetime.
    /// Use [`load`](Booster::load) instead if restoring a saved model.
    ///
    /// Set parameters via [`set_param`](Booster::set_param) or [`set_params`](Booster::set_params)
    /// before calling [`train`](Booster::train).
    pub fn new(num_features: usize) -> XGBResult<Self> {
        let dummy_data = vec![0.0f32; num_features];
        let dummy = DMatrix::from_dense(&dummy_data, 1)?;
        let dmats = [dummy.handle];
        let mut handle = ptr::null_mut();
        xgb_call!(xgboost_sys::XGBoosterCreate(dmats.as_ptr(), 1, &mut handle))?;
        Ok(Booster {
            handle,
            _dummy_dmatrix: Some(dummy),
            callbacks: Vec::new(),
            custom_metric: None,
        })
    }

    /// Save this Booster as a binary file at given path.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> XGBResult<()> {
        debug!("Writing Booster to: {}", path.as_ref().display());
        let fname = crate::path_to_c_str(path);
        xgb_call!(xgboost_sys::XGBoosterSaveModel(self.handle, fname.as_ptr()))
    }

    /// Save this Booster to a buffer.
    /// Format is "ubj" when binary, otherwise "json"
    pub fn save_buffer(&self, binary: bool) -> XGBResult<Vec<u8>> {
        trace!("Writing Booster to buffer");
        let config = format!("{{\"format\":\"{}\"}}", if binary { "ubj" } else { "json" });
        let mut out_len: xgboost_sys::bst_ulong = 0;
        let mut out_buffer = ptr::null();
        xgb_call!(xgboost_sys::XGBoosterSaveModelToBuffer(
            self.handle,
            config.as_bytes().as_ptr() as *const raw::c_char,
            &mut out_len,
            &mut out_buffer
        ))?;
        let buffer = unsafe { slice::from_raw_parts(out_buffer as *const u8, out_len as usize).to_vec() };
        Ok(buffer)
    }

    /// Load a Booster from a binary file at given path.
    pub fn load<P: AsRef<Path>>(path: P) -> XGBResult<Self> {
        debug!("Loading Booster from: {}", path.as_ref().display());

        // gives more control over error messages, avoids stack trace dump from C++
        if !path.as_ref().exists() {
            return Err(XGBError::new(format!("File not found: {}", path.as_ref().display())));
        }

        let fname = crate::path_to_c_str(path);
        let mut handle = ptr::null_mut();
        xgb_call!(xgboost_sys::XGBoosterCreate(ptr::null(), 0, &mut handle))?;
        xgb_call!(xgboost_sys::XGBoosterLoadModel(handle, fname.as_ptr()))?;
        Ok(Booster {
            handle,
            _dummy_dmatrix: None,
            callbacks: Vec::new(),
            custom_metric: None,
        })
    }

    /// Load a Booster directly from a buffer.
    pub fn load_buffer(bytes: &[u8]) -> XGBResult<Self> {
        debug!("Loading Booster from buffer (length = {})", bytes.len());

        let mut handle = ptr::null_mut();
        xgb_call!(xgboost_sys::XGBoosterCreate(ptr::null(), 0, &mut handle))?;
        xgb_call!(xgboost_sys::XGBoosterLoadModelFromBuffer(
            handle,
            bytes.as_ptr() as *const _,
            bytes.len() as u64
        ))?;
        Ok(Booster {
            handle,
            _dummy_dmatrix: None,
            callbacks: Vec::new(),
            custom_metric: None,
        })
    }

    /// Add a training callback.
    ///
    /// Callbacks are invoked in insertion order after each iteration. Any callback
    /// returning `true` from [`after_iteration`](TrainingCallback::after_iteration)
    /// stops training early.
    ///
    /// ```
    /// # use xgboost_rs::{Booster, EvaluationMonitor};
    /// # let mut booster = Booster::new(2).unwrap();
    /// booster.add_callback(Box::new(EvaluationMonitor::new(1)));
    /// ```
    pub fn add_callback(&mut self, cb: Box<dyn TrainingCallback>) {
        self.callbacks.push(cb);
    }

    /// Set a custom evaluation metric, matching Python's `custom_metric` / `feval`.
    ///
    /// The closure receives predictions and the eval matrix, and returns one or more
    /// `(metric_name, score)` pairs. Called every iteration on each eval set; results
    /// appear in [`EvalsLog`] alongside built-in metrics and are visible to callbacks.
    ///
    /// ```
    /// # use xgboost_rs::{Booster, DMatrix};
    /// # let mut booster = Booster::new(2).unwrap();
    /// booster.set_custom_metric(Box::new(|preds: &[f32], dmat: &DMatrix| -> Vec<(String, f32)> {
    ///     let labels = dmat.get_label().unwrap();
    ///     let mse = preds.iter().zip(labels)
    ///         .map(|(p, l)| (p - l).powi(2)).sum::<f32>() / preds.len() as f32;
    ///     vec![("mse".into(), mse)]
    /// }));
    /// ```
    pub fn set_custom_metric(&mut self, metric: Box<dyn Fn(&[f32], &DMatrix) -> Vec<(String, f32)>>) {
        self.custom_metric = Some(metric);
    }

    /// Train this model for a given number of boosting rounds.
    ///
    /// Evaluates on `eval_sets` every iteration, accumulating results into the returned
    /// [`EvalsLog`]. Callbacks (added via [`add_callback`](Booster::add_callback)) and
    /// custom metrics (set via [`set_custom_metric`](Booster::set_custom_metric)) are
    /// invoked automatically.
    ///
    /// # Parameters
    ///
    /// * `dtrain` — training data matrix
    /// * `boost_rounds` — number of boosting iterations
    /// * `eval_sets` — evaluation datasets with names, e.g. `&[(&dtest, "test")]`
    ///
    /// # Errors
    ///
    /// Returns [`XGBError`] if training or evaluation fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use xgboost_rs::{Booster, DMatrix, EvaluationMonitor};
    /// # let dtrain = DMatrix::from_dense(&[1.0, 2.0, 3.0], 1).unwrap();
    /// # let dtest = DMatrix::from_dense(&[1.0, 2.0, 3.0], 1).unwrap();
    /// let mut booster = Booster::new(3).unwrap();
    /// booster.set_params(&[
    ///     ("max_depth", "2"), ("eta", "1.0"), ("objective", "binary:logistic"),
    /// ]).unwrap();
    /// booster.add_callback(Box::new(EvaluationMonitor::new(1)));
    /// let history = booster.train(
    ///     &dtrain,
    ///     10,
    ///     &[(&dtest, "test")],
    /// ).unwrap();
    /// ```
    pub fn train(
        &mut self,
        dtrain: &DMatrix,
        boost_rounds: u32,
        eval_sets: &[(&DMatrix, &str)],
    ) -> XGBResult<EvalsLog> {
        let mut evals_log: EvalsLog = BTreeMap::new();

        for i in 0..boost_rounds {
            xgb_call!(xgboost_sys::XGBoosterUpdateOneIter(
                self.handle,
                i as i32,
                dtrain.handle
            ))?;

            // Evaluate on all eval sets
            let eval_results = self.eval_set(eval_sets, i as i32)?;
            for (data_name, metrics) in &eval_results {
                let entry = evals_log.entry(data_name.clone()).or_default();
                for (metric_name, score) in metrics {
                    entry.entry(metric_name.clone()).or_default().push(*score);
                }
            }

            // Compute custom metrics
            if let Some(ref metric) = self.custom_metric {
                for (dmat, name) in eval_sets {
                    let preds = self.predict(dmat)?;
                    let custom_metrics = metric(&preds, dmat);
                    let entry = evals_log.entry(name.to_string()).or_default();
                    for (metric_name, score) in custom_metrics {
                        entry.entry(metric_name).or_default().push(score);
                    }
                }
            }

            // Invoke callbacks
            let mut should_stop = false;
            let mut callbacks = std::mem::take(&mut self.callbacks);
            for cb in &mut *callbacks {
                if cb.after_iteration(self, i, &evals_log) {
                    should_stop = true;
                }
            }
            self.callbacks = callbacks;
            if should_stop {
                break;
            }
        }

        let mut callbacks = std::mem::take(&mut self.callbacks);
        for cb in &mut *callbacks {
            cb.after_training(self);
        }
        self.callbacks = callbacks;

        Ok(evals_log)
    }

    /// Update this model by training it for one round with given training matrix.
    ///
    /// Uses XGBoost's objective function that was specificed in this Booster's learning objective parameters.
    ///
    /// * `dtrain` - matrix to train the model with for a single iteration
    /// * `iteration` - current iteration number
    pub fn update(&mut self, dtrain: &DMatrix, iteration: i32) -> XGBResult<()> {
        xgb_call!(xgboost_sys::XGBoosterUpdateOneIter(
            self.handle,
            iteration,
            dtrain.handle
        ))
    }

    fn eval_set(&self, evals: &[(&DMatrix, &str)], iteration: i32) -> XGBResult<HashMap<String, HashMap<String, f32>>> {
        let (dmats, names) = {
            let mut dmats = Vec::with_capacity(evals.len());
            let mut names = Vec::with_capacity(evals.len());
            for (dmat, name) in evals {
                dmats.push(dmat);
                names.push(*name);
            }
            (dmats, names)
        };
        assert_eq!(dmats.len(), names.len());

        let mut s: Vec<xgboost_sys::DMatrixHandle> = dmats.iter().map(|x| x.handle).collect();

        // build separate arrays of C strings and pointers to them to ensure they live long enough
        let mut evnames: Vec<ffi::CString> = Vec::with_capacity(names.len());
        let mut evptrs: Vec<*const libc::c_char> = Vec::with_capacity(names.len());

        for name in &names {
            let cstr = ffi::CString::new(*name).unwrap();
            evptrs.push(cstr.as_ptr());
            evnames.push(cstr);
        }

        // shouldn't be necessary, but guards against incorrect array sizing
        evptrs.shrink_to_fit();

        let mut out_result = ptr::null();
        xgb_call!(xgboost_sys::XGBoosterEvalOneIter(
            self.handle,
            iteration,
            s.as_mut_ptr(),
            evptrs.as_mut_ptr(),
            dmats.len() as u64,
            &mut out_result
        ))?;
        let out = unsafe {
            ffi::CStr::from_ptr(out_result)
                .to_str()
                .map_err(|e| XGBError::new(format!("eval output not valid UTF-8: {}", e)))?
                .to_owned()
        };
        Ok(Booster::parse_eval_string(&out, &names))
    }

    /// Evaluate given matrix against this model using metrics defined in this model's parameters.
    ///
    /// See parameter::learning::EvaluationMetric for a full list.
    ///
    /// Returns a map of evaluation metric name to score.
    pub fn evaluate(&self, dmat: &DMatrix) -> XGBResult<HashMap<String, f32>> {
        let name = "default";
        let mut eval = self.eval_set(&[(dmat, name)], 0)?;
        let mut result = HashMap::new();
        eval.remove(name).unwrap().into_iter().for_each(|(k, v)| {
            result.insert(k.to_owned(), v);
        });

        Ok(result)
    }

    /// Get a string attribute that was previously set for this model.
    pub fn get_attribute(&self, key: &str) -> XGBResult<Option<String>> {
        let key = ffi::CString::new(key).unwrap();
        let mut out_buf = ptr::null();
        let mut success = 0;
        xgb_call!(xgboost_sys::XGBoosterGetAttr(
            self.handle,
            key.as_ptr(),
            &mut out_buf,
            &mut success
        ))?;
        if success == 0 {
            return Ok(None);
        }
        assert!(success == 1);

        let c_str: &ffi::CStr = unsafe { ffi::CStr::from_ptr(out_buf) };
        let out = c_str
            .to_str()
            .map_err(|e| XGBError::new(format!("attribute not valid UTF-8: {}", e)))?;
        Ok(Some(out.to_owned()))
    }

    /// Store a string attribute in this model with given key.
    pub fn set_attribute(&mut self, key: &str, value: &str) -> XGBResult<()> {
        let key = ffi::CString::new(key).unwrap();
        let value = ffi::CString::new(value).unwrap();
        xgb_call!(xgboost_sys::XGBoosterSetAttr(self.handle, key.as_ptr(), value.as_ptr()))
    }

    /// Get names of all attributes stored in this model. Values can then be fetched with calls to `get_attribute`.
    pub fn get_attribute_names(&self) -> XGBResult<Vec<String>> {
        let mut out_len = 0;
        let mut out = ptr::null_mut();
        xgb_call!(xgboost_sys::XGBoosterGetAttrNames(self.handle, &mut out_len, &mut out))?;
        if out_len > 0 {
            let out_ptr_slice = unsafe { slice::from_raw_parts(out, out_len as usize) };
            let out_vec = out_ptr_slice
                .iter()
                .map(|str_ptr| unsafe {
                    ffi::CStr::from_ptr(*str_ptr)
                        .to_str()
                        .map(|s| s.to_owned())
                        .map_err(|e| XGBError::new(format!("attribute name not valid UTF-8: {}", e)))
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(out_vec)
        } else {
            Ok(Vec::new())
        }
    }

    /// Get names of feature names stored in this model.
    pub fn get_feature_names(&self) -> XGBResult<Vec<String>> {
        self.get_feature_info("feature_name")
    }

    /// Get names of features stored in this model.
    pub fn get_feature_info(&self, field: &str) -> XGBResult<Vec<String>> {
        let mut out_len = 0;
        let mut out = ptr::null_mut();
        let field: ffi::CString = ffi::CString::new(field).unwrap();
        xgb_call!(xgboost_sys::XGBoosterGetStrFeatureInfo(
            self.handle,
            field.as_ptr(),
            &mut out_len,
            &mut out
        ))?;
        if out_len > 0 {
            let out_ptr_slice = unsafe { slice::from_raw_parts(out, out_len as usize) };
            let out_vec = out_ptr_slice
                .iter()
                .map(|str_ptr| unsafe {
                    ffi::CStr::from_ptr(*str_ptr)
                        .to_str()
                        .map(|s| s.to_owned())
                        .map_err(|e| XGBError::new(format!("attribute name not valid UTF-8: {}", e)))
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(out_vec)
        } else {
            Ok(Vec::new())
        }
    }

    /// Set names of features stored in this model.
    pub fn set_feature_names(&self, features: &Vec<&str>) -> XGBResult<()> {
        self.set_feature_info("feature_name", features)
    }

    /// Set names of features stored in this model.
    #[allow(clippy::unnecessary_cast)]
    pub fn set_feature_info(&self, field: &str, features: &Vec<&str>) -> XGBResult<()> {
        let field: ffi::CString = ffi::CString::new(field).unwrap();

        // We want zero terminated strings
        let c_temp_features: Vec<ffi::CString> = features.iter().map(|s| ffi::CString::new(*s).unwrap()).collect();
        let mut c_feature_ptr: Vec<*const raw::c_char> = c_temp_features
            .into_iter()
            .map(|s| s.into_raw() as *const raw::c_char)
            .collect();

        xgb_call!(xgboost_sys::XGBoosterSetStrFeatureInfo(
            self.handle,
            field.as_ptr(),
            c_feature_ptr.as_mut_ptr() as *mut *const raw::c_char,
            features.len() as u64
        ))
    }

    /// Predict results for given data.
    ///
    /// config_json should be a 0 terminated string, preferred created by PredictConfig::as_json
    /// Returns an array containing one entry per row in the given data and its shape as array.
    pub fn predict_matrix(&self, dmat: &DMatrix, config_json: &str) -> XGBResult<(Vec<f32>, Vec<u64>)> {
        let str_buffer: std::ffi::CString;
        let cfg = if !config_json.is_empty() && config_json.ends_with('\u{0}') {
            unsafe { std::ffi::CStr::from_ptr(config_json.as_ptr() as *const raw::c_char) }
        } else {
            str_buffer = std::ffi::CString::new(config_json).unwrap();
            str_buffer.as_c_str()
        };
        let mut out_shape = ptr::null();
        let mut out_shape_dim = 0;
        let mut out_result = ptr::null();
        xgb_call!(xgboost_sys::XGBoosterPredictFromDMatrix(
            self.handle,
            dmat.handle,
            cfg.as_ptr() as *const raw::c_char,
            &mut out_shape,
            &mut out_shape_dim,
            &mut out_result
        ))?;
        if out_result.is_null() {
            return Err(XGBError::new("predict_matrix: null result pointer".to_string()));
        }
        let shape = unsafe { slice::from_raw_parts(out_shape, out_shape_dim as usize).to_vec() };
        let mut data_size = 1;
        for dim in &shape {
            data_size *= dim;
        }
        let data = unsafe { slice::from_raw_parts(out_result, data_size as usize).to_vec() };

        Ok((data, shape))
    }

    /// Predict results for given data.
    ///
    /// Returns an array containing one entry per row in the given data.
    /// Uses old call to XGBoosterPredict
    pub fn predict(&self, dmat: &DMatrix) -> XGBResult<Vec<f32>> {
        let option_mask = PredictOption::options_as_mask(&[]);
        let ntree_limit = 0;
        let mut out_len = 0;
        let mut out_result = ptr::null();
        xgb_call!(xgboost_sys::XGBoosterPredict(
            self.handle,
            dmat.handle,
            option_mask,
            ntree_limit,
            0,
            &mut out_len,
            &mut out_result
        ))?;

        if out_result.is_null() {
            return Err(XGBError::new("predict: null result pointer".to_string()));
        }
        let data = unsafe { slice::from_raw_parts(out_result, out_len as usize).to_vec() };
        Ok(data)
    }

    /// Predict margin for given data.
    ///
    /// Returns an array containing one entry per row in the given data.
    pub fn predict_margin(&self, dmat: &DMatrix) -> XGBResult<Vec<f32>> {
        let option_mask = PredictOption::options_as_mask(&[PredictOption::OutputMargin]);
        let ntree_limit = 0;
        let mut out_len = 0;
        let mut out_result = ptr::null();
        xgb_call!(xgboost_sys::XGBoosterPredict(
            self.handle,
            dmat.handle,
            option_mask,
            ntree_limit,
            1,
            &mut out_len,
            &mut out_result
        ))?;
        if out_result.is_null() {
            return Err(XGBError::new("predict_margin: null result pointer".to_string()));
        }
        let data = unsafe { slice::from_raw_parts(out_result, out_len as usize).to_vec() };
        Ok(data)
    }

    /// Get predicted leaf index for each sample in given data.
    ///
    /// Returns an array of shape (number of samples, number of trees) as tuple of (data, num_rows).
    ///
    /// Note: the leaf index of a tree is unique per tree, so e.g. leaf 1 could be found in both tree 1 and tree 0.
    pub fn predict_leaf(&self, dmat: &DMatrix) -> XGBResult<(Vec<f32>, (usize, usize))> {
        let option_mask = PredictOption::options_as_mask(&[PredictOption::PredictLeaf]);
        let ntree_limit = 0;
        let mut out_len = 0;
        let mut out_result = ptr::null();
        xgb_call!(xgboost_sys::XGBoosterPredict(
            self.handle,
            dmat.handle,
            option_mask,
            ntree_limit,
            0,
            &mut out_len,
            &mut out_result
        ))?;
        if out_result.is_null() {
            return Err(XGBError::new("predict_leaf: null result pointer".to_string()));
        }

        let data = unsafe { slice::from_raw_parts(out_result, out_len as usize).to_vec() };
        let num_rows = dmat.num_rows();
        let num_cols = data.len() / num_rows;
        Ok((data, (num_rows, num_cols)))
    }

    /// Get feature contributions (SHAP values) for each prediction.
    ///
    /// The sum of all feature contributions is equal to the run untransformed margin value of the
    /// prediction.
    ///
    /// Returns an array of shape (number of samples, number of features + 1) as a tuple of
    /// (data, num_rows). The final column contains the bias term.
    pub fn predict_contributions(&self, dmat: &DMatrix) -> XGBResult<(Vec<f32>, (usize, usize))> {
        let option_mask = PredictOption::options_as_mask(&[PredictOption::PredictContribitions]);
        let ntree_limit = 0;
        let mut out_len = 0;
        let mut out_result = ptr::null();
        xgb_call!(xgboost_sys::XGBoosterPredict(
            self.handle,
            dmat.handle,
            option_mask,
            ntree_limit,
            0,
            &mut out_len,
            &mut out_result
        ))?;
        if out_result.is_null() {
            return Err(XGBError::new("predict_contributions: null result pointer".to_string()));
        }

        let data = unsafe { slice::from_raw_parts(out_result, out_len as usize).to_vec() };
        let num_rows = dmat.num_rows();
        let num_cols = data.len() / num_rows;
        Ok((data, (num_rows, num_cols)))
    }

    /// Get SHAP interaction values for each pair of features for each prediction.
    ///
    /// The sum of each row (or column) of the interaction values equals the corresponding SHAP
    /// value (from `predict_contributions`), and the sum of the entire matrix equals the raw
    /// untransformed margin value of the prediction.
    ///
    /// Returns an array of shape (number of samples, number of features + 1, number of features + 1).
    /// The final row and column contain the bias terms.
    pub fn predict_interactions(&self, dmat: &DMatrix) -> XGBResult<(Vec<f32>, (usize, usize, usize))> {
        let option_mask = PredictOption::options_as_mask(&[PredictOption::PredictInteractions]);
        let ntree_limit = 0;
        let mut out_len = 0;
        let mut out_result = ptr::null();
        xgb_call!(xgboost_sys::XGBoosterPredict(
            self.handle,
            dmat.handle,
            option_mask,
            ntree_limit,
            0,
            &mut out_len,
            &mut out_result
        ))?;
        if out_result.is_null() {
            return Err(XGBError::new("predict_interactions: null result pointer".to_string()));
        }

        let data = unsafe { slice::from_raw_parts(out_result, out_len as usize).to_vec() };
        let num_rows = dmat.num_rows();

        let dim = ((data.len() / num_rows) as f64).sqrt() as usize;
        Ok((data, (num_rows, dim, dim)))
    }

    /// Get a dump of this model as a string.
    ///
    /// * `with_statistics` - whether to include statistics in output dump
    /// * `feature_map` - if given, map feature IDs to feature names from given map
    pub fn dump_model(&self, with_statistics: bool, feature_map: Option<&FeatureMap>) -> XGBResult<String> {
        if let Some(fmap) = feature_map {
            let tmp_dir = match tempfile::tempdir() {
                Ok(dir) => dir,
                Err(err) => return Err(XGBError::new(err.to_string())),
            };

            let file_path = tmp_dir.path().join("fmap.json");
            let mut file: File = match File::create(&file_path) {
                Ok(f) => f,
                Err(err) => return Err(XGBError::new(err.to_string())),
            };

            for (feature_num, (feature_name, feature_type)) in &fmap.0 {
                writeln!(file, "{}\t{}\t{}", feature_num, feature_name, feature_type).unwrap();
            }

            self.dump_model_fmap(with_statistics, Some(&file_path))
        } else {
            self.dump_model_fmap(with_statistics, None)
        }
    }

    pub fn dump_model_vec(&self, with_statistics: bool) -> XGBResult<Vec<String>> {
        self.dump_model_fmap_vec(with_statistics, None)
    }

    fn dump_model_fmap(&self, with_statistics: bool, feature_map_path: Option<&PathBuf>) -> XGBResult<String> {
        Ok(self.dump_model_fmap_vec(with_statistics, feature_map_path)?.join("\n"))
    }

    fn dump_model_fmap_vec(&self, with_statistics: bool, feature_map_path: Option<&PathBuf>) -> XGBResult<Vec<String>> {
        let fmap = if let Some(path) = feature_map_path {
            crate::path_to_c_str(path)
        } else {
            ffi::CString::new("").unwrap()
        };
        let format = ffi::CString::new("text").unwrap();
        let mut out_len = 0;
        let mut out_dump_array = ptr::null_mut();
        xgb_call!(xgboost_sys::XGBoosterDumpModelEx(
            self.handle,
            fmap.as_ptr(),
            with_statistics as i32,
            format.as_ptr(),
            &mut out_len,
            &mut out_dump_array
        ))?;

        if out_len > 0 {
            let out_ptr_slice = unsafe { slice::from_raw_parts(out_dump_array, out_len as usize) };
            let out_vec: Vec<String> = out_ptr_slice
                .iter()
                .map(|str_ptr| unsafe {
                    ffi::CStr::from_ptr(*str_ptr)
                        .to_str()
                        .map(|s| s.to_owned())
                        .map_err(|e| XGBError::new(format!("attribute name not valid UTF-8: {}", e)))
                })
                .collect::<Result<Vec<_>, _>>()?;

            assert_eq!(out_len as usize, out_vec.len());
            Ok(out_vec)
        } else {
            Ok(Vec::new())
        }
    }

    pub fn set_param(&mut self, name: &str, value: &str) -> XGBResult<()> {
        let name = ffi::CString::new(name).unwrap();
        let value = ffi::CString::new(value).unwrap();
        xgb_call!(xgboost_sys::XGBoosterSetParam(
            self.handle,
            name.as_ptr(),
            value.as_ptr()
        ))
    }

    /// Set multiple parameters at once from key-value string pairs.
    ///
    /// ```
    /// # use xgboost_rs::{Booster, DMatrix};
    /// # let dtrain = DMatrix::from_dense(&[1.0], 1).unwrap();
    /// let mut booster = Booster::new(2).unwrap();
    /// booster.set_params(&[
    ///     ("max_depth", "2"),
    ///     ("eta", "1.0"),
    ///     ("objective", "binary:logistic"),
    /// ]).unwrap();
    /// ```
    pub fn set_params(&mut self, params: &[(&str, &str)]) -> XGBResult<()> {
        for (key, value) in params {
            self.set_param(key, value)?;
        }
        Ok(())
    }

    fn parse_eval_string(eval: &str, evnames: &[&str]) -> HashMap<String, HashMap<String, f32>> {
        let mut result: HashMap<String, HashMap<String, f32>> = HashMap::new();

        debug!("Parsing evaluation line: {}", &eval);
        for part in eval.split('\t').skip(1) {
            for evname in evnames {
                if part.starts_with(evname) {
                    let metric_parts: Vec<&str> = part[evname.len() + 1..].split(':').collect();
                    assert_eq!(metric_parts.len(), 2);
                    let metric = metric_parts[0];
                    let score = metric_parts[1]
                        .parse::<f32>()
                        .unwrap_or_else(|_| panic!("Unable to parse XGBoost metrics output: {}", eval));

                    let metric_map = result.entry(evname.to_string()).or_default();
                    metric_map.insert(metric.to_owned(), score);
                }
            }
        }

        debug!("result: {:?}", &result);
        result
    }
}

impl Drop for Booster {
    fn drop(&mut self) {
        if let Err(e) = xgb_call!(xgboost_sys::XGBoosterFree(self.handle)) {
            error!("XGBoosterFree failed in drop: {}", e);
        }
    }
}

/// Maps a feature index to a name and type, used when dumping models as text.
///
/// See [dump_model](struct.Booster.html#method.dump_model) for usage.
pub struct FeatureMap(BTreeMap<u32, (String, FeatureType)>);

impl FeatureMap {
    /// Read a `FeatureMap` from a file at given path.
    ///
    /// File should contain one feature definition per line, and be of the form:
    /// ```text
    /// <number>\t<name>\t<type>\n
    /// ```
    ///
    /// Type should be one of:
    /// * `i` - binary feature
    /// * `q` - quantitative feature
    /// * `int` - integer features
    ///
    /// E.g.:
    /// ```text
    /// 0   age int
    /// 1   is-parent?=yes  i
    /// 2   is-parent?=no   i
    /// 3   income  int
    /// ```
    pub fn from_file<P: AsRef<Path>>(path: P) -> io::Result<FeatureMap> {
        let file = File::open(path)?;
        let mut features: FeatureMap = FeatureMap(BTreeMap::new());

        for (i, line) in BufReader::new(&file).lines().enumerate() {
            let line = line?;
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() != 3 {
                let msg = format!(
                    "Unable to parse features from line {}, expected 3 tab separated values",
                    i + 1
                );
                return Err(io::Error::new(io::ErrorKind::InvalidData, msg));
            }

            assert_eq!(parts.len(), 3);
            let feature_num: u32 = match parts[0].parse() {
                Ok(num) => num,
                Err(err) => {
                    let msg = format!(
                        "Unable to parse features from line {}, could not parse feature number: {}",
                        i + 1,
                        err
                    );
                    return Err(io::Error::new(io::ErrorKind::InvalidData, msg));
                }
            };

            let feature_name = &parts[1];
            let feature_type = match FeatureType::from_str(parts[2]) {
                Ok(feature_type) => feature_type,
                Err(msg) => {
                    let msg = format!("Unable to parse features from line {}: {}", i + 1, msg);
                    return Err(io::Error::new(io::ErrorKind::InvalidData, msg));
                }
            };
            features.0.insert(feature_num, (feature_name.to_string(), feature_type));
        }
        Ok(features)
    }
}

/// Indicates the type of a feature, used when dumping models as text.
pub enum FeatureType {
    /// Binary indicator feature.
    Binary,

    /// Quantitative feature (e.g. age, time, etc.), can be missing.
    Quantitative,

    /// Integer feature (when hinted, decision boundary will be integer).
    Integer,
}

impl FromStr for FeatureType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "i" => Ok(FeatureType::Binary),
            "q" => Ok(FeatureType::Quantitative),
            "int" => Ok(FeatureType::Integer),
            _ => Err(format!(
                "unrecognised feature type '{}', must be one of: 'i', 'q', 'int'",
                s
            )),
        }
    }
}

impl fmt::Display for FeatureType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match self {
            FeatureType::Binary => "i",
            FeatureType::Quantitative => "q",
            FeatureType::Integer => "int",
        };
        write!(f, "{}", s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_booster_param() {
        let mut booster = Booster::new(2).expect("Creating Booster failed");
        let res = booster.set_param("key", "value");
        assert!(res.is_ok());
    }

    #[test]
    fn get_set_attr() {
        let mut booster = Booster::new(2).expect("Creating Booster failed");
        let attr = booster.get_attribute("foo").expect("Getting attribute failed");
        assert_eq!(attr, None);

        booster.set_attribute("foo", "bar").expect("Setting attribute failed");
        let attr = booster.get_attribute("foo").expect("Getting attribute failed");
        assert_eq!(attr, Some("bar".to_owned()));
    }

    #[test]
    fn save_and_load_from_buffer() {
        let mut booster = Booster::new(2).expect("Creating Booster failed");
        let attr = booster.get_attribute("foo").expect("Getting attribute failed");
        assert_eq!(attr, None);

        booster.set_attribute("foo", "bar").expect("Setting attribute failed");
        let attr = booster.get_attribute("foo").expect("Getting attribute failed");
        assert_eq!(attr, Some("bar".to_owned()));

        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("test-xgboost-model");
        booster.save(&path).expect("saving booster");
        drop(booster);
        let bytes = std::fs::read(&path).expect("read saved booster file");
        let booster = Booster::load_buffer(&bytes[..]).expect("load booster from buffer");
        let attr = booster.get_attribute("foo").expect("Getting attribute failed");
        assert_eq!(attr, Some("bar".to_owned()));

        let in_memory_bytes = booster.save_buffer(true).unwrap();
        let booster =
            Booster::load_buffer(&in_memory_bytes[..] as &[u8]).expect("load booster from memory only buffer");
        let attr = booster.get_attribute("foo").expect("Getting attribute failed");
        assert_eq!(attr, Some("bar".to_owned()));
    }

    #[test]
    fn get_attribute_names() {
        let mut booster = Booster::new(2).expect("Creating Booster failed");
        let attrs = booster.get_attribute_names().expect("Getting attributes failed");
        assert_eq!(attrs, Vec::<String>::new());

        booster.set_attribute("foo", "bar").expect("Setting attribute failed");
        booster
            .set_attribute("another", "another")
            .expect("Setting attribute failed");
        booster.set_attribute("4", "4").expect("Setting attribute failed");
        booster
            .set_attribute("an even longer attribute name?", "")
            .expect("Setting attribute failed");

        let mut expected = vec!["foo", "another", "4", "an even longer attribute name?"];
        expected.sort();
        let mut attrs = booster.get_attribute_names().expect("Getting attributes failed");
        attrs.sort();
        assert_eq!(attrs, expected);
    }

    #[test]
    fn get_set_feature_names() {
        let booster = Booster::new(2).expect("Creating Booster failed");
        let attrs = booster.get_feature_names().expect("Getting features failed");
        assert_eq!(attrs, Vec::<String>::new());
        let expected = vec!["foo", "another", "4"];
        booster.set_feature_names(&expected).expect("Setting features failed");
        let attrs = booster.get_feature_names().expect("Getting features failed");
        assert_eq!(attrs.len(), 3);
    }

    fn create_booster(params: &[(&str, &str)], num_features: usize) -> Booster {
        let mut booster = Booster::new(num_features).unwrap();
        for (key, value) in params {
            booster.set_param(key, value).unwrap();
        }
        booster
    }

    #[test]
    fn train_and_predict() {
        let data = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let num_rows = 4;
        let mut dtrain = DMatrix::from_dense(data, num_rows).unwrap();
        dtrain.set_label(&[0.0, 1.0, 0.0, 1.0]).unwrap();

        let params = &[("max_depth", "2"), ("eta", "1.0"), ("objective", "binary:logistic")];

        let mut booster = create_booster(params, 2);
        for i in 0..3 {
            booster.update(&dtrain, i).expect("update failed");
        }

        let preds = booster.predict(&dtrain).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn train_with_eval() {
        let data = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let num_rows = 4;
        let mut dtrain = DMatrix::from_dense(data, num_rows).unwrap();
        dtrain.set_label(&[0.0, 1.0, 0.0, 1.0]).unwrap();
        let mut dtest = DMatrix::from_dense(data, num_rows).unwrap();
        dtest.set_label(&[0.0, 1.0, 0.0, 1.0]).unwrap();

        let eval_sets = &[(&dtest, "test")];
        let mut bst = Booster::new(2).expect("Creating Booster failed");
        bst.set_params(&[("max_depth", "2"), ("eta", "1.0"), ("objective", "binary:logistic")])
            .expect("set_params failed");
        bst.train(&dtrain, 2, eval_sets).unwrap();

        let preds = bst.predict(&dtest).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn parse_eval_string() {
        let s = "[0]\ttrain-map@4-:0.5\ttrain-logloss:1.0\ttest-map@4-:0.25\ttest-logloss:0.75";
        let mut metrics = HashMap::new();

        let mut train_metrics = HashMap::new();
        train_metrics.insert("map@4-".to_owned(), 0.5);
        train_metrics.insert("logloss".to_owned(), 1.0);

        let mut test_metrics = HashMap::new();
        test_metrics.insert("map@4-".to_owned(), 0.25);
        test_metrics.insert("logloss".to_owned(), 0.75);

        metrics.insert("train".to_owned(), train_metrics);
        metrics.insert("test".to_owned(), test_metrics);
        assert_eq!(Booster::parse_eval_string(s, &["train", "test"]), metrics);
    }
}
