use mmn_core::MmnError;
use pyo3::exceptions::{PyException, PyRuntimeError};
use pyo3::prelude::*;

macro_rules! create_exception {
    ($name:ident, $doc:expr) => {
        pyo3::create_exception!(magicmindnet_native, $name, PyException, $doc);
    };
}

create_exception!(CPUError, "CPU not accessible");
create_exception!(CUDAError, "GPU missing but CUDA set");
create_exception!(DataMismatchError, "Dataset type on wrong model");
create_exception!(DataMissingRowError, "Dataset lacks required row");
create_exception!(ModelMismatchError, "Model merge size mismatch");

pub fn mmn_err_to_py(e: MmnError) -> PyErr {
    let mut msg = e.message().to_string();
    if let Some(fix) = e.fix() {
        msg.push_str(&format!("\nFix: {fix}"));
    }
    if let Some(exp) = e.explanation() {
        msg.push_str(&format!("\nExplanation: {exp}"));
    }
    match e {
        MmnError::Cpu { .. } => PyErr::new::<CPUError, _>(msg),
        MmnError::Cuda { .. } => PyErr::new::<CUDAError, _>(msg),
        MmnError::DataMismatch { .. } => PyErr::new::<DataMismatchError, _>(msg),
        MmnError::DataMissingRow { .. } => PyErr::new::<DataMissingRowError, _>(msg),
        MmnError::ModelMismatch { .. } => PyErr::new::<ModelMismatchError, _>(msg),
        _ => PyErr::new::<PyRuntimeError, _>(msg),
    }
}
