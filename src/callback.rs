//! Training callbacks, matching Python's
//! [`xgboost.callback`](https://xgboost.readthedocs.io/en/latest/python/callbacks.html) module.

use std::collections::BTreeMap;

/// Evaluation history, mapping dataset-name → metric-name → scores per iteration.
///
/// Returned by [`Booster::train`](crate::Booster::train) and passed to
/// [`TrainingCallback::after_iteration`].
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
    fn after_iteration(&mut self, booster: &crate::Booster, epoch: u32, evals_log: &EvalsLog) -> bool;

    /// Called after training completes (even if stopped early).
    ///
    /// Receives `&mut Booster` so callbacks can write attributes (e.g. `best_iteration`).
    fn after_training(&mut self, _booster: &mut crate::Booster) {}
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
    fn after_iteration(&mut self, _booster: &crate::Booster, epoch: u32, evals_log: &EvalsLog) -> bool {
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
    fn after_iteration(&mut self, _booster: &crate::Booster, epoch: u32, evals_log: &EvalsLog) -> bool {
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

    fn after_training(&mut self, booster: &mut crate::Booster) {
        // Write best_iteration attribute so predict() can limit trees
        if let Some(best_epoch) = self.best_epoch {
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
