use mmn_models::Diffusion;
use mmn_train::{mean_denoise_loss, mean_denoise_loss_masked};
use pyo3::prelude::*;
use std::path::Path;

use crate::datasets::{PyDatasetImageEdit, PyDatasetImageGen};
use crate::errors::{mmn_err_to_py, DataMismatchError};
use crate::io::{
    expect_checkpoint_family, export_diffusion_to_path, import_diffusion_from_path,
};
use crate::train::train_diffusion_dispatch;
use crate::train_config::{resolve_train_config, PyTrainConfig};

/// A small latent-diffusion image model (VAE + UNet) for 8x8 RGB patches.
#[pyclass(name = "Diffusion")]
pub struct PyDiffusion {
    pub(crate) inner: Diffusion,
}

fn tensor_to_rgb_patch_flat(img: &mmn_core::Tensor) -> Vec<f32> {
    let plane = mmn_data::VISION_PATCH_DIM;
    let mut flat = vec![0.0f32; mmn_data::VISION_RGB_DIM];
    for c in 0..mmn_data::VISION_RGB_CHANNELS {
        for i in 0..plane {
            let y = i / mmn_data::VISION_RGB_SPATIAL;
            let x = i % mmn_data::VISION_RGB_SPATIAL;
            flat[c * plane + i] = img.data[[0, c, y, x]];
        }
    }
    flat
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

    #[getter]
    fn parameters(&self) -> usize {
        self.inner.parameters()
    }

    fn __repr__(&self) -> String {
        format!(
            "Diffusion(latent_channels={}, parameters={})",
            self.inner.latent_channels,
            self.inner.parameters()
        )
    }

    /// Save this diffusion model to `path` (`mmn-diffusion-v1` JSON).
    #[pyo3(signature = (path, format="safetensors"))]
    fn save(&self, path: &str, format: &str) -> PyResult<()> {
        export_diffusion_to_path(&self.inner, format, path)
    }

    /// Load a diffusion checkpoint; the file format is detected automatically.
    #[staticmethod]
    #[pyo3(signature = (path, format=None))]
    fn load(path: &str, format: Option<&str>) -> PyResult<Self> {
        let format = match format {
            Some(f) => f.to_string(),
            None => {
                expect_checkpoint_family(
                    path,
                    "Diffusion",
                    "Use Chatbot.load() / Classifier.load() or the universal ai.load().",
                )?;
                "safetensors".to_string()
            }
        };
        Ok(Self {
            inner: import_diffusion_from_path(&format, path)?,
        })
    }

    /// Train on a `DatasetImageGen` or `DatasetImageEdit`; returns one mean loss per epoch.
    #[pyo3(signature = (dataset, config=None, *, epochs=None, batch_size=None, learning_rate=None, optimizer=None, cuda=None, verbose=None))]
    #[allow(clippy::too_many_arguments)]
    fn train(
        &mut self,
        dataset: &Bound<'_, PyAny>,
        config: Option<&PyTrainConfig>,
        epochs: Option<usize>,
        batch_size: Option<usize>,
        learning_rate: Option<f32>,
        optimizer: Option<&str>,
        cuda: Option<bool>,
        verbose: Option<bool>,
    ) -> PyResult<Vec<f32>> {
        let cfg = resolve_train_config(
            config,
            epochs,
            batch_size,
            learning_rate,
            optimizer,
            cuda,
            verbose,
        )?;
        train_diffusion_dispatch(self, dataset, &cfg)
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
        let x = mmn_data::rgb_nchw_tensor_from_image_path(Path::new(&path)).map_err(mmn_err_to_py)?;
        self.inner.denoise_loss(&x, t).map_err(mmn_err_to_py)
    }

    /// Mask-weighted denoise loss for image + mask paths (`mask` 1 = repaint region).
    fn denoise_loss_on_image_masked(
        &self,
        image_path: String,
        mask_path: String,
        t: usize,
    ) -> PyResult<f32> {
        let x =
            mmn_data::rgb_nchw_tensor_from_image_path(Path::new(&image_path)).map_err(mmn_err_to_py)?;
        let mask = mmn_data::grayscale_mask_tensor_from_image_path(Path::new(&mask_path))
            .map_err(mmn_err_to_py)?;
        self.inner
            .denoise_loss_masked(&x, &mask, t)
            .map_err(mmn_err_to_py)
    }

    /// Reverse-diffusion sample as a flat 8×8×3 RGB patch (192 floats, `[0,1]`).
    #[pyo3(signature = (steps=8, seed=None))]
    fn sample_rgb_patch(&self, steps: usize, seed: Option<u64>) -> PyResult<Vec<f32>> {
        let img = self.inner.sample_image(steps, seed).map_err(mmn_err_to_py)?;
        Ok(tensor_to_rgb_patch_flat(&img))
    }

    /// Inpaint sample as flat RGB patch; preserves unmasked pixels from `image_path`.
    #[pyo3(signature = (image_path, mask_path, steps=8, seed=None))]
    fn sample_inpaint_rgb_patch(
        &self,
        image_path: String,
        mask_path: String,
        steps: usize,
        seed: Option<u64>,
    ) -> PyResult<Vec<f32>> {
        let x =
            mmn_data::rgb_nchw_tensor_from_image_path(Path::new(&image_path)).map_err(mmn_err_to_py)?;
        let mask = mmn_data::grayscale_mask_tensor_from_image_path(Path::new(&mask_path))
            .map_err(mmn_err_to_py)?;
        let img = self
            .inner
            .sample_image_inpaint(&x, &mask, steps, seed)
            .map_err(mmn_err_to_py)?;
        Ok(tensor_to_rgb_patch_flat(&img))
    }

    /// Sample an RGB patch and write an 8×8 PNG to `path`.
    #[pyo3(signature = (path, steps=8, seed=None))]
    fn sample_rgb_patch_to_png(
        &self,
        path: String,
        steps: usize,
        seed: Option<u64>,
    ) -> PyResult<()> {
        let img = self.inner.sample_image(steps, seed).map_err(mmn_err_to_py)?;
        mmn_data::write_rgb_nchw_tensor_to_png(&img, Path::new(&path)).map_err(mmn_err_to_py)
    }

    /// Inpaint sample and write an 8×8 PNG to `path`.
    #[pyo3(signature = (path, image_path, mask_path, steps=8, seed=None))]
    fn sample_inpaint_rgb_patch_to_png(
        &self,
        path: String,
        image_path: String,
        mask_path: String,
        steps: usize,
        seed: Option<u64>,
    ) -> PyResult<()> {
        let x =
            mmn_data::rgb_nchw_tensor_from_image_path(Path::new(&image_path)).map_err(mmn_err_to_py)?;
        let mask = mmn_data::grayscale_mask_tensor_from_image_path(Path::new(&mask_path))
            .map_err(mmn_err_to_py)?;
        let img = self
            .inner
            .sample_image_inpaint(&x, &mask, steps, seed)
            .map_err(mmn_err_to_py)?;
        mmn_data::write_rgb_nchw_tensor_to_png(&img, Path::new(&path)).map_err(mmn_err_to_py)
    }

    /// Mean denoise MSE over dataset rows at fixed timestep `t` (default 7).
    #[pyo3(signature = (dataset, t=7))]
    fn compute_mean_denoise_loss(&self, dataset: &Bound<'_, PyAny>, t: usize) -> PyResult<f32> {
        if let Ok(ds) = dataset.downcast::<PyDatasetImageGen>() {
            return mean_denoise_loss(&self.inner, &ds.borrow().inner, t).map_err(mmn_err_to_py);
        }
        if let Ok(ds) = dataset.downcast::<PyDatasetImageEdit>() {
            return mean_denoise_loss_masked(&self.inner, &ds.borrow().inner, t)
                .map_err(mmn_err_to_py);
        }
        Err(PyErr::new::<DataMismatchError, _>(
            "compute_mean_denoise_loss requires DatasetImageGen or DatasetImageEdit.\nFix: Use an image manifest dataset.\nExplanation: QA/Corpus/Classification datasets are not diffusion image data.".to_string(),
        ))
    }
}
