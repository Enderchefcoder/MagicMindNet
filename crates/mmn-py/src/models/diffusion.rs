use mmn_models::Diffusion;
use pyo3::prelude::*;

use crate::errors::mmn_err_to_py;

#[pyclass(name = "Diffusion")]
pub struct PyDiffusion {
    pub(crate) inner: Diffusion,
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

    /// MSE noise-prediction loss on a random latent (eval only; not comparable to image training).
    fn denoise_loss(&self, t: usize) -> PyResult<f32> {
        use mmn_core::Tensor;
        let x = Tensor::randn(&[1, 3, 8, 8], false);
        self.inner.denoise_loss(&x, t).map_err(mmn_err_to_py)
    }

    /// MSE noise-prediction loss for an on-disk RGB image (`[1,3,8,8]` NCHW).
    fn denoise_loss_on_image(&self, path: String, t: usize) -> PyResult<f32> {
        use std::path::Path;
        let x = mmn_data::rgb_nchw_tensor_from_image_path(Path::new(&path)).map_err(mmn_err_to_py)?;
        self.inner.denoise_loss(&x, t).map_err(mmn_err_to_py)
    }

    /// Reverse-diffusion sample as a flat 8×8×3 RGB patch (192 floats, `[0,1]`).
    #[pyo3(signature = (steps=8, seed=None))]
    fn sample_rgb_patch(&self, steps: usize, seed: Option<u64>) -> PyResult<Vec<f32>> {
        let img = self.inner.sample_image(steps, seed).map_err(mmn_err_to_py)?;
        let plane = mmn_data::VISION_PATCH_DIM;
        let mut flat = vec![0.0f32; mmn_data::VISION_RGB_DIM];
        for c in 0..mmn_data::VISION_RGB_CHANNELS {
            for i in 0..plane {
                let y = i / mmn_data::VISION_RGB_SPATIAL;
                let x = i % mmn_data::VISION_RGB_SPATIAL;
                flat[c * plane + i] = img.data[[0, c, y, x]];
            }
        }
        Ok(flat)
    }
}
