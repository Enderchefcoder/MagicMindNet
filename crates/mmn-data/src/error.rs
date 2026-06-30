pub use mmn_core::MmnError;

pub fn data_missing_row(column: &str) -> MmnError {
    MmnError::DataMissingRow {
        message: format!("Dataset lacks row column: {column}"),
        fix: format!("Add column '{column}' to your data file or change the row parameter."),
        explanation: "The dataset loader expected a column that is not present in the file.".into(),
    }
}

pub fn data_mismatch(dataset_type: &str, model_type: &str) -> MmnError {
    MmnError::DataMismatch {
        message: format!("Dataset type {dataset_type} used on wrong model type {model_type}"),
        fix: "Use a dataset class that matches your model (e.g. DatasetQA for Chatbot).".into(),
        explanation: "Example: corpus dataset on diffusion model is not supported.".into(),
    }
}
