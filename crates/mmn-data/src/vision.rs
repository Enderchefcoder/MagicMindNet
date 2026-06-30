use mmn_core::MmnError;
use std::path::Path;

pub const VISION_RGB_SPATIAL: usize = 8;
pub const VISION_RGB_CHANNELS: usize = 3;
pub const VISION_RGB_DIM: usize = 64 * VISION_RGB_CHANNELS;
pub const VISION_PATCH_DIM: usize = 64;

/// Resize an image to 8×8 RGB and flatten as NCHW planes in `[0, 1]`.
pub fn rgb_patch_from_image_bytes(bytes: &[u8]) -> Result<Vec<f32>, MmnError> {
    let img = image::load_from_memory(bytes).map_err(|e| MmnError::Other {
        message: format!("failed to decode image: {e}"),
    })?;
    let rgb = img.to_rgb8();
    let resized = image::imageops::resize(
        &rgb,
        VISION_RGB_SPATIAL as u32,
        VISION_RGB_SPATIAL as u32,
        image::imageops::FilterType::Triangle,
    );
    let mut v = vec![0.0f32; VISION_RGB_DIM];
    for y in 0..VISION_RGB_SPATIAL {
        for x in 0..VISION_RGB_SPATIAL {
            let pixel = resized.get_pixel(x as u32, y as u32);
            let idx = y * VISION_RGB_SPATIAL + x;
            v[idx] = pixel[0] as f32 / 255.0;
            v[VISION_PATCH_DIM + idx] = pixel[1] as f32 / 255.0;
            v[2 * VISION_PATCH_DIM + idx] = pixel[2] as f32 / 255.0;
        }
    }
    Ok(v)
}

/// Load an on-disk image file into a normalized RGB vision patch.
pub fn rgb_patch_from_image_path(path: &Path) -> Result<Vec<f32>, MmnError> {
    let bytes = std::fs::read(path).map_err(|e| MmnError::Other {
        message: format!("failed to read image {}: {e}", path.display()),
    })?;
    rgb_patch_from_image_bytes(&bytes)
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
    use image::{ImageBuffer, Rgb};

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
}
