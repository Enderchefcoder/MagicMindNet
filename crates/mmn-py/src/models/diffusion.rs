use mmn_models::Diffusion;
use pyo3::prelude::*;

use crate::errors::mmn_err_to_py;

#[pyclass(name = "Diffusion")]
pub struct PyDiffusion {
    #[allow(dead_code)]
    inner: Diffusion,
}

#[pymethods]
impl PyDiffusion {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: Diffusion::new(),
        }
    }

    #[getter]
    fn latent_channels(&self) -> usize {
        self.inner.latent_channels
    }

    fn __repr__(&self) -> String {
        format!(
            "Diffusion(latent_channels={})",
            self.inner.latent_channels
        )
    }

    /// One random latent training step; returns True when the UNet output is finite.
    fn smoke_step(&self) -> PyResult<bool> {
        use mmn_core::Tensor;
        let x = Tensor::randn(&[1, 3, 8, 8], false);
        let out = self
            .inner
            .training_step(&x, 1)
            .map_err(mmn_err_to_py)?;
        Ok(out.data.iter().all(|v| v.is_finite()))
    }
}
