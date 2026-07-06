//! From-scratch Zarr v2 reader/writer — the numpy ecosystem's chunked
//! array store.
//!
//! Reads directory stores: `.zarray` JSON metadata per array (groups via
//! `.zgroup` walk), C-order chunk files named `0.0` / `0/0`, `zlib`
//! compression (the from-scratch inflate) or none, every numeric dtype,
//! `fill_value` for missing chunks. Writes zarr-python-readable v2 arrays
//! with zlib chunks. Blosc-compressed stores are rejected with a clear
//! re-encode hint.

use super::deflate::deflate;
use super::hdf5::{adler32, undo_deflate};
use super::npy::{decode_element, descr_item_size};
use super::NamedArray;
use mmn_core::MmnError;
use std::fs;
use std::path::{Path, PathBuf};

fn err(message: impl Into<String>) -> MmnError {
    MmnError::Other {
        message: message.into(),
    }
}

/// True when the directory is a Zarr v2 array or group store.
pub fn is_zarr_dir(path: &Path) -> bool {
    path.is_dir() && (path.join(".zarray").is_file() || path.join(".zgroup").is_file())
}

struct ZarrayMeta {
    shape: Vec<usize>,
    chunks: Vec<usize>,
    descr: String,
    fill_value: f32,
    compressor: Option<String>,
    order_c: bool,
    separator: char,
}

fn parse_zarray(path: &Path) -> Result<ZarrayMeta, MmnError> {
    let text = fs::read_to_string(path.join(".zarray"))
        .map_err(|e| err(format!("cannot read {}: {e}", path.join(".zarray").display())))?;
    let v: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| err(format!("zarr .zarray is not JSON: {e}")))?;
    if v["zarr_format"].as_u64() != Some(2) {
        return Err(err(format!(
            "zarr format {} not supported (2 is)",
            v["zarr_format"]
        )));
    }
    let dims = |key: &str| -> Result<Vec<usize>, MmnError> {
        v[key].as_array()
            .ok_or_else(|| err(format!("zarr .zarray missing {key}")))?
            .iter()
            .map(|d| {
                d.as_u64()
                    .map(|n| n as usize)
                    .ok_or_else(|| err(format!("zarr {key} entry is not an integer")))
            })
            .collect()
    };
    let shape = dims("shape")?;
    let chunks = dims("chunks")?;
    if chunks.len() != shape.len() || chunks.contains(&0) {
        return Err(err("zarr chunks/shape rank mismatch"));
    }
    let descr = v["dtype"]
        .as_str()
        .ok_or_else(|| err("zarr .zarray missing dtype"))?
        .to_string();
    let compressor = match &v["compressor"] {
        serde_json::Value::Null => None,
        c => Some(
            c["id"].as_str()
                .ok_or_else(|| err("zarr compressor has no id"))?
                .to_string(),
        ),
    };
    if let Some(id) = &compressor {
        if id != "zlib" && id != "gzip" {
            return Err(err(format!(
                "zarr compressor {id:?} not supported (zlib/none; re-encode with numcodecs.Zlib)"
            )));
        }
    }
    if !v["filters"].is_null() {
        return Err(err("zarr filters are not supported"));
    }
    let order_c = v["order"].as_str().unwrap_or("C") == "C";
    let separator = v["dimension_separator"]
        .as_str()
        .and_then(|s| s.chars().next())
        .unwrap_or('.');
    let fill_value = v["fill_value"].as_f64().unwrap_or(0.0) as f32;
    Ok(ZarrayMeta {
        shape,
        chunks,
        descr,
        fill_value,
        compressor,
        order_c,
        separator,
    })
}

/// Decode one chunk file into f32 values (chunk-local C order).
fn decode_chunk(
    meta: &ZarrayMeta,
    raw: &[u8],
    chunk_numel: usize,
) -> Result<Vec<f32>, MmnError> {
    let bytes = match meta.compressor.as_deref() {
        Some("zlib") | Some("gzip") => undo_deflate(raw)?,
        _ => raw.to_vec(),
    };
    let item = descr_item_size(&meta.descr)?;
    if bytes.len() != chunk_numel * item {
        return Err(err(format!(
            "zarr chunk has {} bytes, expected {}",
            bytes.len(),
            chunk_numel * item
        )));
    }
    bytes
        .chunks_exact(item)
        .map(|c| decode_element(&meta.descr, c))
        .collect()
}

/// Read one Zarr v2 array directory into `(shape, f32 values)`.
fn read_zarr_array(path: &Path) -> Result<(Vec<usize>, Vec<f32>), MmnError> {
    let meta = parse_zarray(path)?;
    if !meta.order_c {
        return Err(err("zarr Fortran-order arrays are not supported"));
    }
    let numel: usize = meta.shape.iter().product();
    let mut out = vec![meta.fill_value; numel];
    let rank = meta.shape.len();
    let grid: Vec<usize> = meta
        .shape
        .iter()
        .zip(&meta.chunks)
        .map(|(&s, &c)| s.div_ceil(c))
        .collect();
    let total_chunks: usize = grid.iter().product::<usize>().max(1);
    let chunk_numel: usize = meta.chunks.iter().product();
    // Row-major strides over the full array.
    let mut strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * meta.shape[i + 1];
    }
    let mut grid_coords = vec![0usize; rank];
    for _ in 0..total_chunks {
        let name: String = if rank == 0 {
            "0".to_string()
        } else {
            grid_coords
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(&meta.separator.to_string())
        };
        let chunk_path = path.join(&name);
        if chunk_path.is_file() {
            let raw = fs::read(&chunk_path)
                .map_err(|e| err(format!("cannot read zarr chunk {name}: {e}")))?;
            let values = decode_chunk(&meta, &raw, chunk_numel.max(1))?;
            // Scatter the chunk into the output (edge chunks are padded).
            let mut local = vec![0usize; rank];
            for value in values.iter().take(chunk_numel.max(1)) {
                let mut inside = true;
                let mut flat = 0usize;
                for d in 0..rank {
                    let global = grid_coords[d] * meta.chunks[d] + local[d];
                    if global >= meta.shape[d] {
                        inside = false;
                        break;
                    }
                    flat += global * strides[d];
                }
                if inside {
                    out[flat] = *value;
                } else if rank == 0 {
                    out[0] = *value;
                }
                // Increment chunk-local C-order coordinates.
                for d in (0..rank).rev() {
                    local[d] += 1;
                    if local[d] < meta.chunks[d] {
                        break;
                    }
                    local[d] = 0;
                }
            }
            if rank == 0 && !values.is_empty() {
                out[0] = values[0];
            }
        }
        // Next grid cell (C order).
        for d in (0..rank).rev() {
            grid_coords[d] += 1;
            if grid_coords[d] < grid[d] {
                break;
            }
            grid_coords[d] = 0;
        }
    }
    Ok((meta.shape, out))
}

fn walk_group(dir: &Path, prefix: &str, out: &mut Vec<NamedArray>) -> Result<(), MmnError> {
    if dir.join(".zarray").is_file() {
        let (shape, values) = read_zarr_array(dir)?;
        out.push((prefix.to_string(), shape, values));
        return Ok(());
    }
    let mut children: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| err(format!("cannot list zarr group {}: {e}", dir.display())))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    children.sort();
    for child in children {
        let name = child.file_name().unwrap_or_default().to_string_lossy().into_owned();
        let path = if prefix.is_empty() {
            name
        } else {
            format!("{prefix}/{name}")
        };
        if is_zarr_dir(&child) {
            walk_group(&child, &path, out)?;
        }
    }
    Ok(())
}

/// Read a Zarr v2 store (single array or group tree) into named arrays.
pub fn read_zarr_arrays(path: &str) -> Result<Vec<NamedArray>, MmnError> {
    let root = Path::new(path);
    if !is_zarr_dir(root) {
        return Err(err(format!(
            "{path} is not a zarr v2 store (no .zarray/.zgroup)"
        )));
    }
    let mut out = Vec::new();
    walk_group(root, "", out.as_mut())?;
    if out.is_empty() {
        return Err(err(format!("{path}: zarr store contains no arrays")));
    }
    Ok(out)
}

/// zlib-wrap a deflate stream (numcodecs `Zlib` framing).
fn zlib_compress(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x9C];
    out.extend_from_slice(&deflate(data));
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

/// Write one array as a Zarr v2 directory (single whole-array chunk,
/// `<f4`, zlib compression) that zarr-python opens.
fn write_zarr_array(dir: &Path, shape: &[usize], values: &[f32]) -> Result<(), MmnError> {
    fs::create_dir_all(dir).map_err(|e| err(e.to_string()))?;
    let chunks: Vec<usize> = if shape.is_empty() {
        vec![1]
    } else {
        shape.to_vec()
    };
    let shape_json: Vec<usize> = shape.to_vec();
    let meta = serde_json::json!({
        "chunks": chunks,
        "compressor": {"id": "zlib", "level": 4},
        "dtype": "<f4",
        "fill_value": 0.0,
        "filters": null,
        "order": "C",
        "shape": shape_json,
        "zarr_format": 2,
    });
    fs::write(dir.join(".zarray"), serde_json::to_string_pretty(&meta).unwrap())
        .map_err(|e| err(e.to_string()))?;
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let chunk_name = if shape.is_empty() {
        "0".to_string()
    } else {
        vec!["0"; shape.len()].join(".")
    };
    fs::write(dir.join(chunk_name), zlib_compress(&raw)).map_err(|e| err(e.to_string()))
}

/// Write named arrays as a Zarr v2 group store (one array per `/`-nested
/// member) that `zarr.open_group` reads.
pub fn write_zarr_arrays(path: &str, arrays: &[NamedArray]) -> Result<(), MmnError> {
    if arrays.is_empty() {
        return Err(err("zarr writer needs at least one array"));
    }
    let root = Path::new(path);
    fs::create_dir_all(root).map_err(|e| err(e.to_string()))?;
    fs::write(root.join(".zgroup"), "{\"zarr_format\": 2}")
        .map_err(|e| err(e.to_string()))?;
    for (name, shape, values) in arrays {
        let numel: usize = shape.iter().product();
        if numel != values.len() {
            return Err(err(format!(
                "array {name}: shape {shape:?} needs {numel} values, got {}",
                values.len()
            )));
        }
        let mut dir = root.to_path_buf();
        for part in name.split('/').filter(|p| !p.is_empty()) {
            dir = dir.join(part);
            // Intermediate group markers keep zarr-python's tree walk happy.
        }
        // Mark every intermediate directory as a group.
        let mut cursor = root.to_path_buf();
        let parts: Vec<&str> = name.split('/').filter(|p| !p.is_empty()).collect();
        for part in &parts[..parts.len().saturating_sub(1)] {
            cursor = cursor.join(part);
            fs::create_dir_all(&cursor).map_err(|e| err(e.to_string()))?;
            let marker = cursor.join(".zgroup");
            if !marker.is_file() {
                fs::write(&marker, "{\"zarr_format\": 2}").map_err(|e| err(e.to_string()))?;
            }
        }
        write_zarr_array(&dir, shape, values)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("mmn_zarr_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn write_read_roundtrip_nested_group() {
        let dir = tmp("rt");
        let arrays = vec![
            (
                "layer/kernel".to_string(),
                vec![2, 3],
                vec![1.0, -2.0, 3.5, 0.0, 4.25, -0.5],
            ),
            ("bias".to_string(), vec![2], vec![0.125, -9.0]),
        ];
        write_zarr_arrays(dir.to_str().unwrap(), &arrays).unwrap();
        assert!(is_zarr_dir(&dir));
        let mut back = read_zarr_arrays(dir.to_str().unwrap()).unwrap();
        back.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(back[0].0, "bias");
        assert_eq!(back[0].2, vec![0.125, -9.0]);
        assert_eq!(back[1].0, "layer/kernel");
        assert_eq!(back[1].1, vec![2, 3]);
        let _ = fs::remove_dir_all(&dir);
    }

    /// Multi-chunk store with edge padding and a missing chunk that falls
    /// back to fill_value — built by hand per the v2 spec.
    #[test]
    fn chunked_store_with_missing_chunk_reads() {
        let dir = tmp("chunks");
        fs::create_dir_all(&dir).unwrap();
        let meta = serde_json::json!({
            "chunks": [2, 3],
            "compressor": null,
            "dtype": "<f4",
            "fill_value": 7.0,
            "filters": null,
            "order": "C",
            "shape": [3, 4],
            "zarr_format": 2,
        });
        fs::write(dir.join(".zarray"), meta.to_string()).unwrap();
        // Chunk (0,0): rows 0-1, cols 0-2 -> values 1..6.
        let c00: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        fs::write(dir.join("0.0"), c00).unwrap();
        // Chunk (0,1): cols 3 (+2 padding cols) -> only col 3 lands.
        let c01: Vec<u8> = [10.0f32, 0.0, 0.0, 20.0, 0.0, 0.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        fs::write(dir.join("0.1"), c01).unwrap();
        // Chunks (1,0) and (1,1) missing -> fill_value row.
        let arrays = read_zarr_arrays(dir.to_str().unwrap()).unwrap();
        assert_eq!(arrays[0].1, vec![3, 4]);
        assert_eq!(
            arrays[0].2,
            vec![
                1.0, 2.0, 3.0, 10.0, //
                4.0, 5.0, 6.0, 20.0, //
                7.0, 7.0, 7.0, 7.0,
            ]
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn blosc_and_bad_stores_rejected() {
        let dir = tmp("blosc");
        fs::create_dir_all(&dir).unwrap();
        let meta = serde_json::json!({
            "chunks": [1], "compressor": {"id": "blosc", "cname": "lz4"},
            "dtype": "<f4", "fill_value": 0.0, "filters": null,
            "order": "C", "shape": [1], "zarr_format": 2,
        });
        fs::write(dir.join(".zarray"), meta.to_string()).unwrap();
        let e = read_zarr_arrays(dir.to_str().unwrap()).err().unwrap();
        assert!(e.message().contains("blosc"), "got: {}", e.message());
        let empty = tmp("empty");
        fs::create_dir_all(&empty).unwrap();
        assert!(read_zarr_arrays(empty.to_str().unwrap()).is_err());
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(&empty);
    }
}
