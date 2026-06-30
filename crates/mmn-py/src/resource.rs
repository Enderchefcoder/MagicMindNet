use crate::errors::mmn_err_to_py;
use mmn_resource::{current_limit_percent, limit};
use pyo3::prelude::*;

#[pyfunction]
pub fn limit_resources(percent: &str) -> PyResult<()> {
    limit(percent).map_err(mmn_err_to_py)
}

#[pyfunction]
pub fn limit_percent() -> u8 {
    current_limit_percent()
}
