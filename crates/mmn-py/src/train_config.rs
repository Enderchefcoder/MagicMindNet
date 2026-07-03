use mmn_train::TrainConfig;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn validate_optimizer_name(optimizer: &str) -> PyResult<()> {
    if mmn_train::VALID_OPTIMIZERS.contains(&optimizer) {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "Unknown optimizer {optimizer:?}. Valid options: \"adamw\", \"muon\", \"hybrid\"."
        )))
    }
}

/// Training settings shared by `Train`, `TrainClassifier`, and `TrainDiffusion`.
#[pyclass(name = "TrainConfig")]
pub struct PyTrainConfig {
    #[pyo3(get, set)]
    pub epochs: usize,
    #[pyo3(get, set)]
    pub batch_size: usize,
    #[pyo3(get, set)]
    pub cuda: bool,
    #[pyo3(get, set)]
    pub optimizer: String,
    #[pyo3(get, set)]
    pub learning_rate: f32,
    #[pyo3(get, set)]
    pub verbose: bool,
}

#[pymethods]
impl PyTrainConfig {
    #[new]
    #[pyo3(signature = (epochs=1, batch_size=8, cuda=false, optimizer="hybrid", learning_rate=3e-4, verbose=false))]
    pub fn new(
        epochs: usize,
        batch_size: usize,
        cuda: bool,
        optimizer: &str,
        learning_rate: f32,
        verbose: bool,
    ) -> PyResult<Self> {
        validate_optimizer_name(optimizer)?;
        Ok(Self {
            epochs,
            batch_size,
            cuda,
            optimizer: optimizer.to_string(),
            learning_rate,
            verbose,
        })
    }

    fn __repr__(&self) -> String {
        let py_bool = |b: bool| if b { "True" } else { "False" };
        format!(
            "TrainConfig(epochs={}, batch_size={}, cuda={}, optimizer={:?}, learning_rate={}, verbose={})",
            self.epochs,
            self.batch_size,
            py_bool(self.cuda),
            self.optimizer,
            self.learning_rate,
            py_bool(self.verbose)
        )
    }
}

impl PyTrainConfig {
    pub fn to_train_config(&self) -> TrainConfig {
        TrainConfig {
            epochs: self.epochs,
            batch_size: self.batch_size,
            cuda: self.cuda,
            optimizer: self.optimizer.clone(),
            learning_rate: self.learning_rate,
            verbose: self.verbose,
        }
    }
}

/// Build a `TrainConfig` from an optional base config plus keyword overrides
/// (shared by the `model.train(...)` convenience methods).
#[allow(clippy::too_many_arguments)]
pub(crate) fn resolve_train_config(
    base: Option<&PyTrainConfig>,
    epochs: Option<usize>,
    batch_size: Option<usize>,
    learning_rate: Option<f32>,
    optimizer: Option<&str>,
    cuda: Option<bool>,
    verbose: Option<bool>,
) -> PyResult<TrainConfig> {
    let mut cfg = base
        .map(PyTrainConfig::to_train_config)
        .unwrap_or_default();
    if let Some(e) = epochs {
        cfg.epochs = e;
    }
    if let Some(b) = batch_size {
        cfg.batch_size = b;
    }
    if let Some(lr) = learning_rate {
        cfg.learning_rate = lr;
    }
    if let Some(opt) = optimizer {
        cfg.optimizer = opt.to_string();
    }
    if let Some(c) = cuda {
        cfg.cuda = c;
    }
    if let Some(v) = verbose {
        cfg.verbose = v;
    }
    validate_optimizer_name(&cfg.optimizer)?;
    Ok(cfg)
}
