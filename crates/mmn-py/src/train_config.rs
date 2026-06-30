use mmn_train::TrainConfig;
use pyo3::prelude::*;

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
}

#[pymethods]
impl PyTrainConfig {
    #[new]
    #[pyo3(signature = (epochs=1, batch_size=8, cuda=false, optimizer="hybrid", learning_rate=3e-4))]
    pub fn new(
        epochs: usize,
        batch_size: usize,
        cuda: bool,
        optimizer: &str,
        learning_rate: f32,
    ) -> Self {
        Self {
            epochs,
            batch_size,
            cuda,
            optimizer: optimizer.to_string(),
            learning_rate,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TrainConfig(epochs={}, batch_size={}, cuda={}, optimizer={:?}, learning_rate={})",
            self.epochs, self.batch_size, self.cuda, self.optimizer, self.learning_rate
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
        }
    }
}
