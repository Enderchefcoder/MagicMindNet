use mmn_models::vision_rgb_patch_from_image_path;
use pyo3::prelude::*;

use crate::errors::mmn_err_to_py;

#[pyfunction]
pub fn vision_rgb_patch_from_image_path_py(path: String) -> PyResult<Vec<f32>> {
    vision_rgb_patch_from_image_path(std::path::Path::new(&path)).map_err(mmn_err_to_py)
}
