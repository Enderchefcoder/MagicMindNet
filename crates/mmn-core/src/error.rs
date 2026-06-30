use thiserror::Error;

#[derive(Debug, Error, Clone)]
pub enum MmnError {
    #[error("{message}")]
    Cpu {
        message: String,
        fix: String,
        explanation: String,
    },
    #[error("{message}")]
    Cuda {
        message: String,
        fix: String,
        explanation: String,
    },
    #[error("{message}")]
    DataMismatch {
        message: String,
        fix: String,
        explanation: String,
    },
    #[error("{message}")]
    DataMissingRow {
        message: String,
        fix: String,
        explanation: String,
    },
    #[error("{message}")]
    ModelMismatch {
        message: String,
        fix: String,
        explanation: String,
    },
    #[error("{message}")]
    Shape { message: String },
    #[error("{message}")]
    Other { message: String },
}

impl MmnError {
    pub fn message(&self) -> &str {
        match self {
            MmnError::Cpu { message, .. }
            | MmnError::Cuda { message, .. }
            | MmnError::DataMismatch { message, .. }
            | MmnError::DataMissingRow { message, .. }
            | MmnError::ModelMismatch { message, .. }
            | MmnError::Shape { message }
            | MmnError::Other { message } => message,
        }
    }

    pub fn fix(&self) -> Option<&str> {
        match self {
            MmnError::Cpu { fix, .. }
            | MmnError::Cuda { fix, .. }
            | MmnError::DataMismatch { fix, .. }
            | MmnError::DataMissingRow { fix, .. }
            | MmnError::ModelMismatch { fix, .. } => Some(fix),
            _ => None,
        }
    }

    pub fn explanation(&self) -> Option<&str> {
        match self {
            MmnError::Cpu { explanation, .. }
            | MmnError::Cuda { explanation, .. }
            | MmnError::DataMismatch { explanation, .. }
            | MmnError::DataMissingRow { explanation, .. }
            | MmnError::ModelMismatch { explanation, .. } => Some(explanation),
            _ => None,
        }
    }

    pub fn cpu_inaccessible(detail: impl Into<String>) -> Self {
        MmnError::Cpu {
            message: format!("CPU not accessible: {}", detail.into()),
            fix: "Ensure the process has permission to use CPU resources.".into(),
            explanation: "MagicMindNet could not allocate or compute on CPU memory.".into(),
        }
    }

    pub fn cuda_missing() -> Self {
        MmnError::Cuda {
            message: "GPU missing, but CUDA was requested.".into(),
            fix: "Install an NVIDIA GPU and CUDA toolkit, or set cuda=False in TrainConfig.".into(),
            explanation: "TrainConfig.cuda=True requires a CUDA-capable device and mmn built with the cuda feature.".into(),
        }
    }
}

pub type Result<T> = std::result::Result<T, MmnError>;
