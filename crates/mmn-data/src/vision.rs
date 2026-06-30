use mmn_core::MmnError;
use std::path::Path;

pub const VISION_RGB_SPATIAL: usize = 8;
pub const VISION_RGB_CHANNELS: usize = 3;
pub const VISION_RGB_DIM: usize = 64 * VISION_RGB_CHANNELS;
pub const VISION_PATCH_DIM: usize = 64;
/// Default 1×1 grid (single 8×8 patch). Use 2 for a 2×2 tile split (4 prefix tokens).
pub const DEFAULT_VISION_PATCH_GRID: usize = 1;
pub const MAX_VISION_PATCH_GRID: usize = 4;

fn rgb_tile_to_nchw_patch(rgb: &image::RgbImage, tx: usize, ty: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; VISION_RGB_DIM];
    for y in 0..VISION_RGB_SPATIAL {
        for x in 0..VISION_RGB_SPATIAL {
            let pixel = rgb.get_pixel(
                (tx * VISION_RGB_SPATIAL + x) as u32,
                (ty * VISION_RGB_SPATIAL + y) as u32,
            );
            let idx = y * VISION_RGB_SPATIAL + x;
            v[idx] = pixel[0] as f32 / 255.0;
            v[VISION_PATCH_DIM + idx] = pixel[1] as f32 / 255.0;
            v[2 * VISION_PATCH_DIM + idx] = pixel[2] as f32 / 255.0;
        }
    }
    v
}

fn validate_patch_grid(grid: usize) -> Result<(), MmnError> {
    if grid == 0 || grid > MAX_VISION_PATCH_GRID {
        return Err(MmnError::Shape {
            message: format!(
                "vision patch grid must be 1..={MAX_VISION_PATCH_GRID}, got {grid}"
            ),
        });
    }
    Ok(())
}

/// Split a resized RGB image into `grid×grid` tiles of 8×8×3 NCHW patches.
pub fn rgb_patches_from_rgb_image(rgb: &image::RgbImage, grid: usize) -> Result<Vec<Vec<f32>>, MmnError> {
    validate_patch_grid(grid)?;
    let side = VISION_RGB_SPATIAL * grid;
    let resized = image::imageops::resize(
        rgb,
        side as u32,
        side as u32,
        image::imageops::FilterType::Triangle,
    );
    let mut patches = Vec::with_capacity(grid * grid);
    for ty in 0..grid {
        for tx in 0..grid {
            patches.push(rgb_tile_to_nchw_patch(&resized, tx, ty));
        }
    }
    Ok(patches)
}

/// Decode bytes to RGB and emit `grid×grid` normalized vision patches.
pub fn rgb_patches_from_image_bytes(bytes: &[u8], grid: usize) -> Result<Vec<Vec<f32>>, MmnError> {
    let img = image::load_from_memory(bytes).map_err(|e| MmnError::Other {
        message: format!("failed to decode image: {e}"),
    })?;
    rgb_patches_from_rgb_image(&img.to_rgb8(), grid)
}

/// Load a single 8×8 RGB image as an NCHW tensor `[1, 3, 8, 8]` in `[0, 1]`.
pub fn rgb_nchw_tensor_from_image_bytes(bytes: &[u8]) -> Result<mmn_core::Tensor, MmnError> {
    let flat = rgb_patch_from_image_bytes(bytes)?;
    let mut planes = vec![0.0f32; VISION_RGB_CHANNELS * VISION_RGB_SPATIAL * VISION_RGB_SPATIAL];
    for c in 0..VISION_RGB_CHANNELS {
        let off = c * VISION_PATCH_DIM;
        for i in 0..VISION_PATCH_DIM {
            planes[c * VISION_PATCH_DIM + i] = flat[off + i];
        }
    }
    Ok(mmn_core::Tensor::from_array(
        ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[1, VISION_RGB_CHANNELS, VISION_RGB_SPATIAL, VISION_RGB_SPATIAL]),
            planes,
        )
        .unwrap(),
        false,
    ))
}

/// Load an on-disk image as `[1, 3, 8, 8]` NCHW.
pub fn rgb_nchw_tensor_from_image_path(path: &Path) -> Result<mmn_core::Tensor, MmnError> {
    let bytes = std::fs::read(path).map_err(|e| MmnError::Other {
        message: format!("failed to read image {}: {e}", path.display()),
    })?;
    rgb_nchw_tensor_from_image_bytes(&bytes)
}

/// Grayscale inpainting mask `[1, 1, 8, 8]` in `[0, 1]` from image bytes.
pub fn grayscale_mask_tensor_from_image_bytes(bytes: &[u8]) -> Result<mmn_core::Tensor, MmnError> {
    let img = image::load_from_memory(bytes).map_err(|e| MmnError::Other {
        message: format!("failed to decode mask image: {e}"),
    })?;
    let luma = img.to_luma8();
    let resized = image::imageops::resize(
        &luma,
        VISION_RGB_SPATIAL as u32,
        VISION_RGB_SPATIAL as u32,
        image::imageops::FilterType::Triangle,
    );
    let mut planes = vec![0.0f32; VISION_PATCH_DIM];
    for y in 0..VISION_RGB_SPATIAL {
        for x in 0..VISION_RGB_SPATIAL {
            planes[y * VISION_RGB_SPATIAL + x] =
                resized.get_pixel(x as u32, y as u32)[0] as f32 / 255.0;
        }
    }
    Ok(mmn_core::Tensor::from_array(
        ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[1, 1, VISION_RGB_SPATIAL, VISION_RGB_SPATIAL]),
            planes,
        )
        .unwrap(),
        false,
    ))
}

/// Load a mask image from disk as `[1, 1, 8, 8]`.
pub fn grayscale_mask_tensor_from_image_path(path: &Path) -> Result<mmn_core::Tensor, MmnError> {
    let bytes = std::fs::read(path).map_err(|e| MmnError::Other {
        message: format!("failed to read mask {}: {e}", path.display()),
    })?;
    grayscale_mask_tensor_from_image_bytes(&bytes)
}

/// Write `[1, 3, 8, 8]` RGB NCHW tensor to an 8×8 PNG (values clamped to `[0, 1]`).
pub fn write_rgb_nchw_tensor_to_png(t: &mmn_core::Tensor, path: &Path) -> Result<(), MmnError> {
    if t.shape != [1, VISION_RGB_CHANNELS, VISION_RGB_SPATIAL, VISION_RGB_SPATIAL] {
        return Err(MmnError::Shape {
            message: format!("write_rgb_nchw_tensor_to_png expected [1,3,8,8], got {:?}", t.shape),
        });
    }
    let mut img: image::RgbImage = image::ImageBuffer::new(
        VISION_RGB_SPATIAL as u32,
        VISION_RGB_SPATIAL as u32,
    );
    for y in 0..VISION_RGB_SPATIAL {
        for x in 0..VISION_RGB_SPATIAL {
            let r = (t.data[[0, 0, y, x]].clamp(0.0, 1.0) * 255.0) as u8;
            let g = (t.data[[0, 1, y, x]].clamp(0.0, 1.0) * 255.0) as u8;
            let b = (t.data[[0, 2, y, x]].clamp(0.0, 1.0) * 255.0) as u8;
            img.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
        }
    }
    img.save(path).map_err(|e| MmnError::Other {
        message: format!("failed to write png {}: {e}", path.display()),
    })
}

/// Resize an image to 8×8 RGB and flatten as NCHW planes in `[0, 1]`.
pub fn rgb_patch_from_image_bytes(bytes: &[u8]) -> Result<Vec<f32>, MmnError> {
    Ok(rgb_patches_from_image_bytes(bytes, DEFAULT_VISION_PATCH_GRID)?
        .into_iter()
        .next()
        .unwrap_or_else(|| vec![0.0; VISION_RGB_DIM]))
}

/// Load an on-disk image file into normalized RGB vision patches.
pub fn rgb_patches_from_image_path(path: &Path, grid: usize) -> Result<Vec<Vec<f32>>, MmnError> {
    let bytes = std::fs::read(path).map_err(|e| MmnError::Other {
        message: format!("failed to read image {}: {e}", path.display()),
    })?;
    rgb_patches_from_image_bytes(&bytes, grid)
}

/// Load a single 8×8 RGB patch from disk (`grid=1`).
pub fn rgb_patch_from_image_path(path: &Path) -> Result<Vec<f32>, MmnError> {
    Ok(rgb_patches_from_image_path(path, DEFAULT_VISION_PATCH_GRID)?
        .into_iter()
        .next()
        .unwrap_or_else(|| vec![0.0; VISION_RGB_DIM]))
}

/// Parse `image` column values: comma-separated paths or a JSON string array.
pub fn parse_image_path_list(raw: &str) -> Vec<String> {
    let trimmed = raw.trim();
    if trimmed.starts_with('[') {
        serde_json::from_str::<Vec<String>>(trimmed).unwrap_or_default()
    } else {
        trimmed
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect()
    }
}

/// Average NCHW RGB planes into a flat grayscale 8×8 patch.
pub fn grayscale_patch_from_rgb(rgb: &[f32]) -> Vec<f32> {
    let mut gray = vec![0.0f32; VISION_PATCH_DIM];
    if rgb.len() >= VISION_RGB_DIM {
        for i in 0..VISION_PATCH_DIM {
            gray[i] = (rgb[i] + rgb[VISION_PATCH_DIM + i] + rgb[2 * VISION_PATCH_DIM + i]) / 3.0;
        }
    }
    gray
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Luma, Rgb};

    #[test]
    fn rgb_patch_from_png_bytes_has_expected_length() {
        let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::new(VISION_RGB_SPATIAL as u32, VISION_RGB_SPATIAL as u32);
        for (x, y, pixel) in img.enumerate_pixels_mut() {
            *pixel = Rgb([(x * 32) as u8, (y * 32) as u8, 128]);
        }
        let mut buf = Vec::new();
        img.write_to(
            &mut std::io::Cursor::new(&mut buf),
            image::ImageFormat::Png,
        )
        .unwrap();
        let patch = rgb_patch_from_image_bytes(&buf).unwrap();
        assert_eq!(patch.len(), VISION_RGB_DIM);
        assert!(patch.iter().any(|&v| v > 0.0));
    }

    #[test]
    fn rgb_patches_grid_two_produces_four_tiles() {
        let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(16, 16);
        for (x, _y, pixel) in img.enumerate_pixels_mut() {
            let r = if x < 8 { 255 } else { 0 };
            *pixel = Rgb([r, 0, 0]);
        }
        let mut buf = Vec::new();
        img.write_to(
            &mut std::io::Cursor::new(&mut buf),
            image::ImageFormat::Png,
        )
        .unwrap();
        let patches = rgb_patches_from_image_bytes(&buf, 2).unwrap();
        assert_eq!(patches.len(), 4);
        assert!(patches[0][0] > 0.9);
        assert!(patches[3][0] < 0.1);
    }

    #[test]
    fn parse_image_path_list_accepts_comma_and_json() {
        assert_eq!(
            parse_image_path_list("a.png, b.png"),
            vec!["a.png".to_string(), "b.png".to_string()]
        );
        assert_eq!(
            parse_image_path_list(r#"["x.jpg","y.jpg"]"#),
            vec!["x.jpg".to_string(), "y.jpg".to_string()]
        );
    }

    #[test]
    fn rgb_patch_from_image_path_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("mmn_vision_patch_test.png");
        let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(4, 4);
        for (_x, _y, pixel) in img.enumerate_pixels_mut() {
            *pixel = Rgb([255, 0, 0]);
        }
        img.save(&path).unwrap();
        let patch = rgb_patch_from_image_path(&path).unwrap();
        assert_eq!(patch.len(), VISION_RGB_DIM);
        assert!(patch[0] > 0.9);
        assert!(patch[VISION_PATCH_DIM] < 0.1);
    }

    #[test]
    fn grayscale_mask_tensor_from_image_path_is_unit_interval() {
        let dir = std::env::temp_dir();
        let path = dir.join("mmn_mask_test.png");
        let mut img: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::new(4, 4);
        for (_x, _y, pixel) in img.enumerate_pixels_mut() {
            *pixel = Luma([200]);
        }
        img.save(&path).unwrap();
        let mask = grayscale_mask_tensor_from_image_path(&path).unwrap();
        assert_eq!(mask.shape, vec![1, 1, VISION_RGB_SPATIAL, VISION_RGB_SPATIAL]);
        for &v in mask.data.iter() {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn write_rgb_nchw_tensor_to_png_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("mmn_rgb_write_test.png");
        let mut data = ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&[1, 3, 8, 8]));
        data[[0, 0, 0, 0]] = 1.0;
        data[[0, 1, 1, 1]] = 0.5;
        let t = mmn_core::Tensor::from_array(data, false);
        write_rgb_nchw_tensor_to_png(&t, &path).unwrap();
        let loaded = rgb_nchw_tensor_from_image_path(&path).unwrap();
        assert!((loaded.data[[0, 0, 0, 0]] - 1.0).abs() < 0.05);
        assert!((loaded.data[[0, 1, 1, 1]] - 0.5).abs() < 0.05);
    }
}
