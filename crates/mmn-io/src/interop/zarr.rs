//! From-scratch Zarr v2 reader/writer — the numpy ecosystem's chunked
//! array store.
//!
//! Reads directory stores: `.zarray` JSON metadata per array (groups via
//! `.zgroup` walk), C-order chunk files named `0.0` / `0/0`, **blosc**
//! (zarr's default — decoded by the from-scratch LZ4/zlib Blosc frame
//! reader), `zlib`, or uncompressed chunks, every numeric dtype,
//! `fill_value` for missing chunks. Writes zarr-python-readable v2 arrays
//! with zlib chunks.

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

/// True when the directory is a Zarr v2 or v3 array/group store.
pub fn is_zarr_dir(path: &Path) -> bool {
    path.is_dir()
        && (path.join(".zarray").is_file()
            || path.join(".zgroup").is_file()
            || path.join("zarr.json").is_file())
}

/// Compression stage of the codec chain.
#[derive(PartialEq)]
enum ZarrCompressor {
    None,
    /// zlib-framed deflate (v2 `zlib`, blosc-internal zlib).
    Zlib,
    /// RFC-1952 gzip framing (v3 `gzip` codec).
    Gzip,
    Blosc,
    /// Raw zstd frames (zarr-python 3.x default for both v2 and v3).
    Zstd,
    /// numcodecs LZ4: 4-byte LE original size + one LZ4 block.
    Lz4,
}

/// v3 `sharding_indexed` configuration: outer chunk files hold a grid of
/// inner chunks plus a trailing/leading (offset, nbytes) index.
struct ShardingMeta {
    inner_chunks: Vec<usize>,
    inner_compressor: ZarrCompressor,
    index_at_end: bool,
    index_has_crc: bool,
}

struct ZarrayMeta {
    shape: Vec<usize>,
    chunks: Vec<usize>,
    descr: String,
    fill_value: f32,
    compressor: ZarrCompressor,
    order_c: bool,
    separator: char,
    /// v3 default chunk-key encoding prefixes keys with `c<sep>`.
    chunk_prefix: bool,
    sharding: Option<ShardingMeta>,
}

/// Parse a v3 codec chain into a compressor stage (shared by array codecs,
/// shard inner codecs, and shard index codecs).
fn parse_v3_codec_chain(codecs: &[serde_json::Value]) -> Result<(ZarrCompressor, bool), MmnError> {
    let mut compressor = ZarrCompressor::None;
    let mut has_crc = false;
    for codec in codecs {
        match codec["name"].as_str().unwrap_or("") {
            "bytes" => {
                if codec["configuration"]["endian"].as_str() == Some("big") {
                    return Err(err("zarr v3 big-endian bytes codec not supported"));
                }
            }
            "gzip" => compressor = ZarrCompressor::Gzip,
            "zlib" => compressor = ZarrCompressor::Zlib,
            "blosc" => compressor = ZarrCompressor::Blosc,
            "zstd" => compressor = ZarrCompressor::Zstd,
            "crc32c" => has_crc = true,
            other => {
                return Err(err(format!(
                    "zarr v3 codec {other:?} not supported (bytes/gzip/blosc/zstd/crc32c/sharding_indexed)"
                )))
            }
        }
    }
    Ok((compressor, has_crc))
}

/// Map a Zarr v3 `data_type` name to the npy descr this crate decodes.
fn v3_dtype_descr(name: &str) -> Result<String, MmnError> {
    Ok(match name {
        "float16" => "<f2".to_string(),
        "float32" => "<f4".to_string(),
        "float64" => "<f8".to_string(),
        "int8" => "|i1".to_string(),
        "int16" => "<i2".to_string(),
        "int32" => "<i4".to_string(),
        "int64" => "<i8".to_string(),
        "uint8" => "|u1".to_string(),
        "uint16" => "<u2".to_string(),
        "uint32" => "<u4".to_string(),
        "uint64" => "<u8".to_string(),
        "bool" => "|b1".to_string(),
        other => return Err(err(format!("zarr v3 data_type {other:?} not supported"))),
    })
}

/// Undo RFC-1952 gzip framing (header + deflate + CRC-32 + size).
fn undo_gzip(raw: &[u8]) -> Result<Vec<u8>, MmnError> {
    if raw.len() < 18 || raw[0] != 0x1F || raw[1] != 0x8B || raw[2] != 8 {
        return Err(err("zarr gzip chunk is not a gzip stream"));
    }
    let flags = raw[3];
    let mut pos = 10usize;
    if flags & 0x04 != 0 {
        // FEXTRA
        let len = u16::from_le_bytes([raw[pos], raw[pos + 1]]) as usize;
        pos += 2 + len;
    }
    for bit in [0x08u8, 0x10] {
        // FNAME / FCOMMENT: NUL-terminated strings.
        if flags & bit != 0 {
            while *raw.get(pos).ok_or_else(|| err("gzip header truncated"))? != 0 {
                pos += 1;
            }
            pos += 1;
        }
    }
    if flags & 0x02 != 0 {
        pos += 2; // FHCRC
    }
    let body = raw
        .get(pos..raw.len() - 8)
        .ok_or_else(|| err("gzip stream truncated"))?;
    let out = super::inflate::inflate(body)?;
    let stored_crc = u32::from_le_bytes([
        raw[raw.len() - 8],
        raw[raw.len() - 7],
        raw[raw.len() - 6],
        raw[raw.len() - 5],
    ]);
    if super::zip::crc32(&out) != stored_crc {
        return Err(err("gzip CRC-32 mismatch"));
    }
    Ok(out)
}

/// Parse Zarr v3 `zarr.json` array metadata.
fn parse_zarr_v3(path: &Path, v: &serde_json::Value) -> Result<ZarrayMeta, MmnError> {
    let dims = |value: &serde_json::Value, what: &str| -> Result<Vec<usize>, MmnError> {
        value
            .as_array()
            .ok_or_else(|| err(format!("zarr v3 missing {what}")))?
            .iter()
            .map(|d| {
                d.as_u64()
                    .map(|n| n as usize)
                    .ok_or_else(|| err(format!("zarr v3 {what} entry is not an integer")))
            })
            .collect()
    };
    let shape = dims(&v["shape"], "shape")?;
    if v["chunk_grid"]["name"].as_str() != Some("regular") {
        return Err(err("zarr v3 chunk grid must be regular"));
    }
    let chunks = dims(&v["chunk_grid"]["configuration"]["chunk_shape"], "chunk_shape")?;
    if chunks.len() != shape.len() || chunks.contains(&0) {
        return Err(err("zarr chunks/shape rank mismatch"));
    }
    let descr = v3_dtype_descr(
        v["data_type"]
            .as_str()
            .ok_or_else(|| err("zarr v3 missing data_type"))?,
    )?;
    // Codec chain: sharding wraps an inner chain; otherwise `bytes`
    // (endian) followed by at most one compressor.
    let empty: Vec<serde_json::Value> = Vec::new();
    let codecs = v["codecs"].as_array().unwrap_or(&empty);
    let mut compressor = ZarrCompressor::None;
    let mut sharding = None;
    if codecs.len() == 1 && codecs[0]["name"].as_str() == Some("sharding_indexed") {
        let config = &codecs[0]["configuration"];
        let inner_chunks = dims(&config["chunk_shape"], "shard chunk_shape")?;
        if inner_chunks.len() != shape.len()
            || inner_chunks.iter().zip(&chunks).any(|(&i, &c)| i == 0 || !c.is_multiple_of(i))
        {
            return Err(err("zarr v3 shard chunk_shape must evenly divide the shard"));
        }
        let (inner_compressor, _) = parse_v3_codec_chain(
            config["codecs"].as_array().map(|a| a.as_slice()).unwrap_or(&[]),
        )?;
        let (_, index_has_crc) = parse_v3_codec_chain(
            config["index_codecs"].as_array().map(|a| a.as_slice()).unwrap_or(&[]),
        )?;
        let index_at_end = config["index_location"].as_str().unwrap_or("end") == "end";
        sharding = Some(ShardingMeta {
            inner_chunks,
            inner_compressor,
            index_at_end,
            index_has_crc,
        });
    } else {
        let (parsed, _) = parse_v3_codec_chain(codecs)?;
        compressor = parsed;
    }
    let separator = v["chunk_key_encoding"]["configuration"]["separator"]
        .as_str()
        .and_then(|s| s.chars().next())
        .unwrap_or('/');
    let chunk_prefix = v["chunk_key_encoding"]["name"].as_str() != Some("v2");
    let fill_value = v["fill_value"].as_f64().unwrap_or(0.0) as f32;
    let _ = path;
    Ok(ZarrayMeta {
        shape,
        chunks,
        descr,
        fill_value,
        compressor,
        order_c: true,
        separator,
        chunk_prefix,
        sharding,
    })
}

fn parse_zarray(path: &Path) -> Result<ZarrayMeta, MmnError> {
    if !path.join(".zarray").is_file() && path.join("zarr.json").is_file() {
        let text = fs::read_to_string(path.join("zarr.json"))
            .map_err(|e| err(format!("cannot read {}: {e}", path.join("zarr.json").display())))?;
        let v: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| err(format!("zarr.json is not JSON: {e}")))?;
        return parse_zarr_v3(path, &v);
    }
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
    if chunks.len() != shape.len() || (!shape.is_empty() && chunks.contains(&0)) {
        return Err(err("zarr chunks/shape rank mismatch"));
    }
    let descr = v["dtype"]
        .as_str()
        .ok_or_else(|| err("zarr .zarray missing dtype"))?
        .to_string();
    let compressor = match &v["compressor"] {
        serde_json::Value::Null => ZarrCompressor::None,
        c => match c["id"].as_str().ok_or_else(|| err("zarr compressor has no id"))? {
            // v2 numcodecs Zlib/GZip both write zlib-framed deflate.
            "zlib" | "gzip" => ZarrCompressor::Zlib,
            "blosc" => ZarrCompressor::Blosc,
            "zstd" => ZarrCompressor::Zstd,
            "lz4" => ZarrCompressor::Lz4,
            other => {
                return Err(err(format!(
                    "zarr compressor {other:?} not supported (blosc/zstd/lz4/zlib/none)"
                )))
            }
        },
    };
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
        chunk_prefix: false,
        sharding: None,
    })
}

/// Decompress one payload with a given compressor stage.
fn decompress_payload(
    compressor: &ZarrCompressor,
    raw: &[u8],
) -> Result<Vec<u8>, MmnError> {
    Ok(match compressor {
        ZarrCompressor::Zlib => undo_deflate(raw)?,
        ZarrCompressor::Gzip => undo_gzip(raw)?,
        ZarrCompressor::Blosc => super::blosc::blosc_decompress(raw)?,
        ZarrCompressor::Zstd => super::zstd::zstd_decompress(raw)?,
        ZarrCompressor::Lz4 => {
            if raw.len() < 4 {
                return Err(err("zarr lz4 chunk missing its size header"));
            }
            let size = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) as usize;
            super::lz4::lz4_decompress_block(&raw[4..], size)?
        }
        ZarrCompressor::None => raw.to_vec(),
    })
}

/// Decode one v3 shard file into whole-shard f32 values: read the
/// (offset, nbytes) index, decode present inner chunks, scatter them into
/// the shard grid, fill the rest with `fill_value`.
fn decode_shard(
    meta: &ZarrayMeta,
    sharding: &ShardingMeta,
    raw: &[u8],
) -> Result<Vec<f32>, MmnError> {
    let rank = meta.chunks.len();
    let inner_grid: Vec<usize> = meta
        .chunks
        .iter()
        .zip(&sharding.inner_chunks)
        .map(|(&shard, &inner)| shard / inner)
        .collect();
    let n_inner: usize = inner_grid.iter().product::<usize>().max(1);
    let index_len = n_inner * 16 + if sharding.index_has_crc { 4 } else { 0 };
    if raw.len() < index_len {
        return Err(err("zarr shard smaller than its index"));
    }
    let index = if sharding.index_at_end {
        &raw[raw.len() - index_len..]
    } else {
        &raw[..index_len]
    };
    if sharding.index_has_crc {
        let body = &index[..index.len() - 4];
        let stored = u32::from_le_bytes(index[index.len() - 4..].try_into().unwrap());
        if super::tf_checkpoint::crc32c(body) != stored {
            return Err(err("zarr shard index CRC-32C mismatch"));
        }
    }
    let shard_numel: usize = meta.chunks.iter().product();
    let inner_numel: usize = sharding.inner_chunks.iter().product();
    let item = descr_item_size(&meta.descr)?;
    let mut out = vec![meta.fill_value; shard_numel];
    // Shard-local row-major strides.
    let mut strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * meta.chunks[i + 1];
    }
    let mut grid_coords = vec![0usize; rank];
    for inner_index in 0..n_inner {
        let offset = u64::from_le_bytes(index[inner_index * 16..inner_index * 16 + 8].try_into().unwrap());
        let nbytes = u64::from_le_bytes(index[inner_index * 16 + 8..inner_index * 16 + 16].try_into().unwrap());
        if offset != u64::MAX || nbytes != u64::MAX {
            let payload = raw
                .get(offset as usize..(offset + nbytes) as usize)
                .ok_or_else(|| err("zarr shard inner chunk out of bounds"))?;
            let bytes = decompress_payload(&sharding.inner_compressor, payload)?;
            if bytes.len() != inner_numel * item {
                return Err(err(format!(
                    "zarr inner chunk has {} bytes, expected {}",
                    bytes.len(),
                    inner_numel * item
                )));
            }
            let values: Vec<f32> = bytes
                .chunks_exact(item)
                .map(|c| decode_element(&meta.descr, c))
                .collect::<Result<_, _>>()?;
            // Scatter the inner chunk into the shard buffer.
            let mut local = vec![0usize; rank];
            for value in values.iter().take(inner_numel.max(1)) {
                let mut flat = 0usize;
                for d in 0..rank {
                    flat += (grid_coords[d] * sharding.inner_chunks[d] + local[d]) * strides[d];
                }
                out[flat] = *value;
                for d in (0..rank).rev() {
                    local[d] += 1;
                    if local[d] < sharding.inner_chunks[d] {
                        break;
                    }
                    local[d] = 0;
                }
            }
        }
        for d in (0..rank).rev() {
            grid_coords[d] += 1;
            if grid_coords[d] < inner_grid[d] {
                break;
            }
            grid_coords[d] = 0;
        }
    }
    Ok(out)
}

/// Decode one chunk file into f32 values (chunk-local C order).
fn decode_chunk(
    meta: &ZarrayMeta,
    raw: &[u8],
    chunk_numel: usize,
) -> Result<Vec<f32>, MmnError> {
    if let Some(sharding) = &meta.sharding {
        let values = decode_shard(meta, sharding, raw)?;
        if values.len() != chunk_numel {
            return Err(err("zarr shard size mismatch"));
        }
        return Ok(values);
    }
    let bytes = match meta.compressor {
        ZarrCompressor::Zlib => undo_deflate(raw)?,
        ZarrCompressor::Gzip => undo_gzip(raw)?,
        ZarrCompressor::Blosc => super::blosc::blosc_decompress(raw)?,
        ZarrCompressor::Zstd => super::zstd::zstd_decompress(raw)?,
        ZarrCompressor::Lz4 => {
            if raw.len() < 4 {
                return Err(err("zarr lz4 chunk missing its size header"));
            }
            let size =
                u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) as usize;
            super::lz4::lz4_decompress_block(&raw[4..], size)?
        }
        ZarrCompressor::None => raw.to_vec(),
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
        let mut name: String = if rank == 0 {
            "0".to_string()
        } else {
            grid_coords
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(&meta.separator.to_string())
        };
        if meta.chunk_prefix {
            // v3 default chunk-key encoding: keys live under "c<sep>...";
            // rank-0 arrays use the bare "c" key.
            name = if rank == 0 {
                "c".to_string()
            } else {
                format!("c{}{}", meta.separator, name)
            };
        }
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
    // v3 nodes declare themselves in zarr.json; arrays terminate the walk.
    let v3_is_array = dir.join("zarr.json").is_file()
        && fs::read_to_string(dir.join("zarr.json"))
            .ok()
            .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
            .is_some_and(|v| v["node_type"].as_str() == Some("array"));
    if dir.join(".zarray").is_file() || v3_is_array {
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

/// RFC-1952 gzip framing (the v3 `gzip` codec's on-disk form).
fn gzip_compress(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x1F, 0x8B, 8, 0, 0, 0, 0, 0, 0, 255];
    out.extend_from_slice(&deflate(data));
    out.extend_from_slice(&super::zip::crc32(data).to_le_bytes());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out
}

/// Write one array as a Zarr v2 directory (single whole-array chunk,
/// `<f4`, zlib compression) that zarr-python opens.
fn write_zarr_array(dir: &Path, shape: &[usize], values: &[f32]) -> Result<(), MmnError> {
    fs::create_dir_all(dir).map_err(|e| err(e.to_string()))?;
    // Rank-0 arrays use empty chunks and a single "0" chunk file (the
    // zarr-python convention).
    let chunks: Vec<usize> = shape.to_vec();
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

/// Write one array as a Zarr v3 directory: `zarr.json` array node plus a
/// single gzip-compressed whole-array chunk under `c/`.
fn write_zarr_array_v3(dir: &Path, shape: &[usize], values: &[f32]) -> Result<(), MmnError> {
    fs::create_dir_all(dir).map_err(|e| err(e.to_string()))?;
    // Rank-0 arrays use an empty chunk_shape and the bare chunk key "c".
    let chunks: Vec<usize> = shape.to_vec();
    let meta = serde_json::json!({
        "zarr_format": 3,
        "node_type": "array",
        "shape": shape,
        "data_type": "float32",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": chunks}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0.0,
        "codecs": [
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "gzip", "configuration": {"level": 4}},
        ],
        "attributes": {},
    });
    fs::write(dir.join("zarr.json"), serde_json::to_string_pretty(&meta).unwrap())
        .map_err(|e| err(e.to_string()))?;
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    if shape.is_empty() {
        // Rank-0: the chunk key is the bare "c" file.
        return fs::write(dir.join("c"), gzip_compress(&raw)).map_err(|e| err(e.to_string()));
    }
    let mut chunk_dir = dir.join("c");
    for _ in 0..shape.len() - 1 {
        chunk_dir = chunk_dir.join("0");
    }
    fs::create_dir_all(&chunk_dir).map_err(|e| err(e.to_string()))?;
    fs::write(chunk_dir.join("0"), gzip_compress(&raw)).map_err(|e| err(e.to_string()))
}

/// Write named arrays as a Zarr group store (one array per `/`-nested
/// member) that `zarr.open_group` reads. `zarr_format` selects v2
/// (zlib-compressed chunks) or v3 (`zarr.json` nodes, gzip codec).
pub fn write_zarr_arrays_format(
    path: &str,
    arrays: &[NamedArray],
    zarr_format: u8,
) -> Result<(), MmnError> {
    if arrays.is_empty() {
        return Err(err("zarr writer needs at least one array"));
    }
    if zarr_format != 2 && zarr_format != 3 {
        return Err(err(format!(
            "zarr_format {zarr_format} not supported (2 or 3)"
        )));
    }
    let root = Path::new(path);
    fs::create_dir_all(root).map_err(|e| err(e.to_string()))?;
    let group_marker = |dir: &Path| -> Result<(), MmnError> {
        if zarr_format == 2 {
            let marker = dir.join(".zgroup");
            if !marker.is_file() {
                fs::write(&marker, "{\"zarr_format\": 2}").map_err(|e| err(e.to_string()))?;
            }
        } else {
            let marker = dir.join("zarr.json");
            if !marker.is_file() {
                fs::write(
                    &marker,
                    "{\"zarr_format\": 3, \"node_type\": \"group\", \"attributes\": {}}",
                )
                .map_err(|e| err(e.to_string()))?;
            }
        }
        Ok(())
    };
    group_marker(root)?;
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
        }
        // Mark every intermediate directory as a group.
        let mut cursor = root.to_path_buf();
        let parts: Vec<&str> = name.split('/').filter(|p| !p.is_empty()).collect();
        for part in &parts[..parts.len().saturating_sub(1)] {
            cursor = cursor.join(part);
            fs::create_dir_all(&cursor).map_err(|e| err(e.to_string()))?;
            group_marker(&cursor)?;
        }
        if zarr_format == 2 {
            write_zarr_array(&dir, shape, values)?;
        } else {
            write_zarr_array_v3(&dir, shape, values)?;
        }
    }
    Ok(())
}

/// Write named arrays as a Zarr v2 group store (one array per `/`-nested
/// member) that `zarr.open_group` reads.
pub fn write_zarr_arrays(path: &str, arrays: &[NamedArray]) -> Result<(), MmnError> {
    write_zarr_arrays_format(path, arrays, 2)
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
    fn unsupported_codec_and_bad_stores_rejected() {
        // Blosc container with a zstd inner codec (flags bits 5-7 = 4).
        let dir = tmp("zstd");
        fs::create_dir_all(&dir).unwrap();
        let meta = serde_json::json!({
            "chunks": [1], "compressor": {"id": "blosc", "cname": "zstd"},
            "dtype": "<f4", "fill_value": 0.0, "filters": null,
            "order": "C", "shape": [1], "zarr_format": 2,
        });
        fs::write(dir.join(".zarray"), meta.to_string()).unwrap();
        let mut frame = vec![2u8, 1, 4 << 5, 4];
        frame.extend_from_slice(&4u32.to_le_bytes()); // nbytes
        frame.extend_from_slice(&4u32.to_le_bytes()); // blocksize
        frame.extend_from_slice(&26u32.to_le_bytes()); // cbytes
        frame.extend_from_slice(&20u32.to_le_bytes()); // bstart (after bstarts)
        frame.extend_from_slice(&2u32.to_le_bytes()); // stream cbytes
        frame.extend_from_slice(&[0u8, 0]);
        fs::write(dir.join("0"), frame).unwrap();
        let e = read_zarr_arrays(dir.to_str().unwrap()).err().unwrap();
        assert!(e.message().contains("zstd"), "got: {}", e.message());
        // Unknown top-level compressor id still rejects at metadata time.
        let lzma = tmp("lzma");
        fs::create_dir_all(&lzma).unwrap();
        let meta = serde_json::json!({
            "chunks": [1], "compressor": {"id": "lzma"},
            "dtype": "<f4", "fill_value": 0.0, "filters": null,
            "order": "C", "shape": [1], "zarr_format": 2,
        });
        fs::write(lzma.join(".zarray"), meta.to_string()).unwrap();
        assert!(read_zarr_arrays(lzma.to_str().unwrap()).is_err());
        let empty = tmp("empty");
        fs::create_dir_all(&empty).unwrap();
        assert!(read_zarr_arrays(empty.to_str().unwrap()).is_err());
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(&lzma);
        let _ = fs::remove_dir_all(&empty);
    }
}
