//! From-scratch TensorFlow checkpoint v2 reader (`ckpt.index` +
//! `ckpt.data-00000-of-00001`).
//!
//! The `.index` file is a LevelDB-style SSTable: prefix-compressed key/value
//! blocks with varint block handles, a 48-byte footer, and per-block
//! masked-CRC32C trailers. Values are `BundleEntryProto` protobuf messages
//! (dtype, shape, shard, offset, size, crc32c) pointing into the raw `.data`
//! shard files. No TensorFlow or LevelDB code is linked.

use half::{bf16, f16};
use mmn_core::MmnError;
use std::fs;

const TABLE_MAGIC: u64 = 0xdb47_7524_8b80_fb57;
const FOOTER_LEN: usize = 48;
const OBJECT_GRAPH_KEY: &str = "_CHECKPOINTABLE_OBJECT_GRAPH";
const VALUE_SUFFIX: &str = "/.ATTRIBUTES/VARIABLE_VALUE";

fn err(message: impl Into<String>) -> MmnError {
    MmnError::Other {
        message: message.into(),
    }
}

/// CRC32-C (Castagnoli, reflected 0x82F63B78) — used by LevelDB tables and
/// TensorFlow bundle entries, stored in "masked" form.
fn crc32c(data: &[u8]) -> u32 {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut table = [0u32; 256];
        for (i, slot) in table.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0x82F6_3B78 ^ (c >> 1) } else { c >> 1 };
            }
            *slot = c;
        }
        table
    });
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc = table[((crc ^ byte as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

fn masked_crc32c(data: &[u8]) -> u32 {
    crc32c(data).rotate_right(15).wrapping_add(0xa282_ead8)
}

fn varint(buf: &[u8], pos: &mut usize) -> Result<u64, MmnError> {
    let mut out = 0u64;
    let mut shift = 0u32;
    loop {
        let byte = *buf.get(*pos).ok_or_else(|| err("tf index varint truncated"))?;
        *pos += 1;
        out |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok(out);
        }
        shift += 7;
        if shift > 63 {
            return Err(err("tf index varint too long"));
        }
    }
}

/// Read a table block, verifying the type byte and masked-CRC32C trailer.
fn read_block(bytes: &[u8], offset: usize, size: usize) -> Result<&[u8], MmnError> {
    let block = bytes
        .get(offset..offset + size)
        .ok_or_else(|| err("tf index block out of bounds"))?;
    let trailer = bytes
        .get(offset + size..offset + size + 5)
        .ok_or_else(|| err("tf index block trailer truncated"))?;
    match trailer[0] {
        0 => {}
        1 => {
            return Err(err(
                "tf index block is snappy-compressed; MagicMindNet reads uncompressed bundles (TensorFlow's default)",
            ));
        }
        other => return Err(err(format!("tf index block compression {other} unknown"))),
    }
    let stored = u32::from_le_bytes([trailer[1], trailer[2], trailer[3], trailer[4]]);
    let mut with_type = Vec::with_capacity(size + 1);
    with_type.extend_from_slice(block);
    with_type.push(trailer[0]);
    if masked_crc32c(&with_type) != stored {
        return Err(err("tf index block CRC32C mismatch"));
    }
    Ok(block)
}

type BlockEntry = (Vec<u8>, Vec<u8>);

/// Iterate the prefix-compressed entries of one block.
fn block_entries(block: &[u8]) -> Result<Vec<BlockEntry>, MmnError> {
    if block.len() < 4 {
        return Err(err("tf index block too small"));
    }
    let num_restarts =
        u32::from_le_bytes(block[block.len() - 4..].try_into().expect("4 bytes")) as usize;
    let data_end = block
        .len()
        .checked_sub(4 + 4 * num_restarts)
        .ok_or_else(|| err("tf index restart array overruns block"))?;
    let mut entries = Vec::new();
    let mut pos = 0usize;
    let mut prev_key: Vec<u8> = Vec::new();
    while pos < data_end {
        let shared = varint(block, &mut pos)? as usize;
        let non_shared = varint(block, &mut pos)? as usize;
        let value_len = varint(block, &mut pos)? as usize;
        if shared > prev_key.len() {
            return Err(err("tf index shared key prefix longer than previous key"));
        }
        let mut key = prev_key[..shared].to_vec();
        key.extend_from_slice(
            block
                .get(pos..pos + non_shared)
                .ok_or_else(|| err("tf index key truncated"))?,
        );
        pos += non_shared;
        let value = block
            .get(pos..pos + value_len)
            .ok_or_else(|| err("tf index value truncated"))?
            .to_vec();
        pos += value_len;
        prev_key = key.clone();
        entries.push((key, value));
    }
    Ok(entries)
}

#[derive(Debug, Default)]
struct BundleEntry {
    dtype: u64,
    shape: Vec<usize>,
    shard_id: u64,
    offset: usize,
    size: usize,
    crc32c: u32,
}

/// Minimal protobuf walk of a `BundleEntryProto` (or `TensorShapeProto`).
fn parse_bundle_entry(buf: &[u8]) -> Result<BundleEntry, MmnError> {
    let mut entry = BundleEntry::default();
    let mut pos = 0usize;
    while pos < buf.len() {
        let tag = varint(buf, &mut pos)?;
        let field = tag >> 3;
        match tag & 7 {
            0 => {
                let v = varint(buf, &mut pos)?;
                match field {
                    1 => entry.dtype = v,
                    3 => entry.shard_id = v,
                    4 => entry.offset = v as usize,
                    5 => entry.size = v as usize,
                    _ => {}
                }
            }
            2 => {
                let len = varint(buf, &mut pos)? as usize;
                let body = buf
                    .get(pos..pos + len)
                    .ok_or_else(|| err("tf bundle proto field truncated"))?;
                if field == 2 {
                    entry.shape = parse_shape_proto(body)?;
                }
                pos += len;
            }
            5 => {
                let raw = buf
                    .get(pos..pos + 4)
                    .ok_or_else(|| err("tf bundle proto fixed32 truncated"))?;
                if field == 6 {
                    entry.crc32c = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
                }
                pos += 4;
            }
            1 => pos += 8,
            other => return Err(err(format!("tf bundle proto wire type {other} unexpected"))),
        }
    }
    Ok(entry)
}

fn parse_shape_proto(buf: &[u8]) -> Result<Vec<usize>, MmnError> {
    let mut dims = Vec::new();
    let mut pos = 0usize;
    while pos < buf.len() {
        let tag = varint(buf, &mut pos)?;
        match tag {
            // field 2 (dim), length-delimited
            0x12 => {
                let len = varint(buf, &mut pos)? as usize;
                let body = buf
                    .get(pos..pos + len)
                    .ok_or_else(|| err("tf shape proto truncated"))?;
                pos += len;
                let mut dpos = 0usize;
                let mut size = 0u64;
                while dpos < body.len() {
                    let dtag = varint(body, &mut dpos)?;
                    if dtag == 0x08 {
                        size = varint(body, &mut dpos)?;
                    } else if dtag & 7 == 2 {
                        let l = varint(body, &mut dpos)? as usize;
                        dpos += l;
                    } else {
                        varint(body, &mut dpos)?;
                    }
                }
                dims.push(size as usize);
            }
            _ => {
                if tag & 7 == 0 {
                    varint(buf, &mut pos)?;
                } else if tag & 7 == 2 {
                    let len = varint(buf, &mut pos)? as usize;
                    pos += len;
                }
            }
        }
    }
    Ok(dims)
}

/// Decode raw little-endian tensor bytes per TF `DataType` id.
fn decode_values(dtype: u64, raw: &[u8], name: &str) -> Result<Vec<f32>, MmnError> {
    let convert = |item: usize, f: &dyn Fn(&[u8]) -> f32| -> Result<Vec<f32>, MmnError> {
        if !raw.len().is_multiple_of(item) {
            return Err(err(format!("tf tensor {name}: byte length not a multiple of {item}")));
        }
        Ok(raw.chunks_exact(item).map(f).collect())
    };
    match dtype {
        1 => convert(4, &|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
        2 => convert(8, &|c| {
            f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
        }),
        3 => convert(4, &|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32),
        4 => convert(1, &|c| c[0] as f32),
        5 => convert(2, &|c| i16::from_le_bytes([c[0], c[1]]) as f32),
        6 => convert(1, &|c| (c[0] as i8) as f32),
        9 => convert(8, &|c| {
            i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
        }),
        10 => convert(1, &|c| if c[0] != 0 { 1.0 } else { 0.0 }),
        14 => convert(2, &|c| bf16::from_le_bytes([c[0], c[1]]).to_f32()),
        19 => convert(2, &|c| f16::from_le_bytes([c[0], c[1]]).to_f32()),
        22 => convert(4, &|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32),
        23 => convert(8, &|c| {
            u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
        }),
        other => Err(err(format!(
            "tf tensor {name}: DataType {other} not supported (numeric types only)"
        ))),
    }
}

fn checkpoint_prefix(path: &str) -> String {
    // SavedModel directory: use its `variables/variables` bundle.
    let as_dir = std::path::Path::new(path);
    if as_dir.is_dir() {
        let saved_model = as_dir.join("variables").join("variables.index");
        if saved_model.is_file() {
            return as_dir
                .join("variables")
                .join("variables")
                .to_string_lossy()
                .into_owned();
        }
        let direct = as_dir.join("variables.index");
        if direct.is_file() {
            return as_dir.join("variables").to_string_lossy().into_owned();
        }
    }
    path.strip_suffix(".index").unwrap_or(path).to_string()
}

/// Read every tensor in a TF checkpoint v2 (`prefix` or `prefix.index` path).
pub fn read_tf_checkpoint_arrays(path: &str) -> Result<Vec<super::NamedArray>, MmnError> {
    let prefix = checkpoint_prefix(path);
    let index_path = format!("{prefix}.index");
    let bytes = fs::read(&index_path)
        .map_err(|e| err(format!("cannot read tf checkpoint index {index_path}: {e}")))?;
    if bytes.len() < FOOTER_LEN {
        return Err(err("tf checkpoint index smaller than the table footer"));
    }
    let magic = u64::from_le_bytes(bytes[bytes.len() - 8..].try_into().expect("8 bytes"));
    if magic != TABLE_MAGIC {
        return Err(err(
            "not a tf checkpoint index (LevelDB table magic missing)",
        ));
    }
    let footer = &bytes[bytes.len() - FOOTER_LEN..bytes.len() - 8];
    let mut pos = 0usize;
    let _meta_off = varint(footer, &mut pos)?;
    let _meta_size = varint(footer, &mut pos)?;
    let index_off = varint(footer, &mut pos)? as usize;
    let index_size = varint(footer, &mut pos)? as usize;
    let index_block = read_block(&bytes, index_off, index_size)?;

    let mut num_shards = 1u64;
    let mut entries: Vec<(String, BundleEntry)> = Vec::new();
    for (_key, handle) in block_entries(index_block)? {
        let mut hpos = 0usize;
        let off = varint(&handle, &mut hpos)? as usize;
        let size = varint(&handle, &mut hpos)? as usize;
        let data_block = read_block(&bytes, off, size)?;
        for (key, value) in block_entries(data_block)? {
            let name = String::from_utf8_lossy(&key).into_owned();
            if name.is_empty() {
                // BundleHeaderProto: field 1 = num_shards.
                let mut hp = 0usize;
                while hp < value.len() {
                    let tag = varint(&value, &mut hp)?;
                    if tag == 0x08 {
                        num_shards = varint(&value, &mut hp)?;
                    } else if tag & 7 == 2 {
                        let len = varint(&value, &mut hp)? as usize;
                        hp += len;
                    } else if tag & 7 == 0 {
                        varint(&value, &mut hp)?;
                    } else {
                        break;
                    }
                }
                continue;
            }
            if name == OBJECT_GRAPH_KEY {
                continue;
            }
            entries.push((name, parse_bundle_entry(&value)?));
        }
    }
    // Load shard data files on demand.
    let mut shards: Vec<Option<Vec<u8>>> = vec![None; num_shards.max(1) as usize];
    let mut out = Vec::with_capacity(entries.len());
    for (name, entry) in entries {
        if entry.dtype == 7 {
            continue; // DT_STRING (metadata blobs)
        }
        let shard_idx = entry.shard_id as usize;
        if shard_idx >= shards.len() {
            return Err(err(format!("tf tensor {name}: shard {shard_idx} out of range")));
        }
        if shards[shard_idx].is_none() {
            let shard_path = format!("{prefix}.data-{shard_idx:05}-of-{:05}", shards.len());
            shards[shard_idx] = Some(fs::read(&shard_path).map_err(|e| {
                err(format!("cannot read tf checkpoint shard {shard_path}: {e}"))
            })?);
        }
        let shard = shards[shard_idx].as_ref().expect("just loaded");
        let raw = shard
            .get(entry.offset..entry.offset + entry.size)
            .ok_or_else(|| err(format!("tf tensor {name}: data out of shard bounds")))?;
        if entry.crc32c != 0 && masked_crc32c(raw) != entry.crc32c {
            return Err(err(format!("tf tensor {name}: data CRC32C mismatch")));
        }
        let values = decode_values(entry.dtype, raw, &name)?;
        let clean = name
            .strip_suffix(VALUE_SUFFIX)
            .unwrap_or(&name)
            .to_string();
        out.push((clean, entry.shape, values));
    }
    Ok(out)
}

/// Build one LevelDB table block (no key sharing, single restart) + trailer.
fn write_table_block(entries: &[(Vec<u8>, Vec<u8>)], out: &mut Vec<u8>) -> (usize, usize) {
    use super::proto::write_varint;
    let start = out.len();
    for (key, value) in entries {
        write_varint(0, out); // shared prefix length
        write_varint(key.len() as u64, out);
        write_varint(value.len() as u64, out);
        out.extend_from_slice(key);
        out.extend_from_slice(value);
    }
    out.extend_from_slice(&0u32.to_le_bytes()); // restart point 0
    out.extend_from_slice(&1u32.to_le_bytes()); // num restarts
    let size = out.len() - start;
    // Trailer: type byte + masked CRC32C(block + type).
    let mut with_type = out[start..].to_vec();
    with_type.push(0);
    out.push(0);
    out.extend_from_slice(&masked_crc32c(&with_type).to_le_bytes());
    (start, size)
}

fn encode_bundle_entry(shape: &[usize], offset: usize, size: usize, crc: u32) -> Vec<u8> {
    use super::proto::{write_field_bytes, write_field_fixed32, write_field_varint};
    let mut shape_proto = Vec::new();
    for &d in shape {
        let mut dim = Vec::new();
        write_field_varint(1, d as u64, &mut dim); // Dim.size
        write_field_bytes(2, &dim, &mut shape_proto); // Shape.dim
    }
    let mut entry = Vec::new();
    write_field_varint(1, 1, &mut entry); // dtype = DT_FLOAT
    write_field_bytes(2, &shape_proto, &mut entry);
    if offset > 0 {
        write_field_varint(4, offset as u64, &mut entry);
    }
    write_field_varint(5, size as u64, &mut entry);
    write_field_fixed32(6, crc, &mut entry);
    entry
}

/// Write a TF checkpoint v2 (`prefix.index` + `prefix.data-00000-of-00001`)
/// that `tf.train.load_checkpoint` reads.
pub fn write_tf_checkpoint_arrays(
    path: &str,
    arrays: &[super::NamedArray],
) -> Result<(), MmnError> {
    use super::proto::{write_field_bytes, write_field_varint, write_varint};
    let prefix = checkpoint_prefix(path);
    // Data shard: raw little-endian f32 payloads, entries sorted by name
    // (LevelDB tables require sorted keys).
    let mut sorted: Vec<&super::NamedArray> = arrays.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    let mut data = Vec::new();
    let mut entries: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    // "" key: BundleHeaderProto { num_shards = 1, version { producer = 1 } }.
    let mut header = Vec::new();
    write_field_varint(1, 1, &mut header);
    let mut version = Vec::new();
    write_field_varint(1, 1, &mut version);
    write_field_bytes(3, &version, &mut header);
    entries.push((Vec::new(), header));
    for (name, shape, values) in sorted {
        let numel: usize = shape.iter().product();
        if numel != values.len() {
            return Err(err(format!(
                "tf tensor {name}: shape {shape:?} needs {numel} values, got {}",
                values.len()
            )));
        }
        let offset = data.len();
        for v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let size = data.len() - offset;
        let crc = masked_crc32c(&data[offset..]);
        entries.push((
            name.as_bytes().to_vec(),
            encode_bundle_entry(shape, offset, size, crc),
        ));
    }
    // Index file: data block, empty metaindex block, index block, footer.
    let mut index = Vec::new();
    let (data_off, data_size) = write_table_block(&entries, &mut index);
    let (meta_off, meta_size) = write_table_block(&[], &mut index);
    let last_key = entries.last().map(|(k, _)| k.clone()).unwrap_or_default();
    let mut handle = Vec::new();
    write_varint(data_off as u64, &mut handle);
    write_varint(data_size as u64, &mut handle);
    let (index_off, index_size) = write_table_block(&[(last_key, handle)], &mut index);
    let mut footer = Vec::new();
    write_varint(meta_off as u64, &mut footer);
    write_varint(meta_size as u64, &mut footer);
    write_varint(index_off as u64, &mut footer);
    write_varint(index_size as u64, &mut footer);
    footer.resize(FOOTER_LEN - 8, 0);
    footer.extend_from_slice(&TABLE_MAGIC.to_le_bytes());
    index.extend_from_slice(&footer);
    crate::checkpoint_util::write_file_create_parents(&format!("{prefix}.index"), index)?;
    crate::checkpoint_util::write_file_create_parents(
        &format!("{prefix}.data-00000-of-00001"),
        data,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture_prefix() -> String {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tf/ckpt")
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn reads_real_tf_checkpoint_fixture() {
        let arrays = read_tf_checkpoint_arrays(&fixture_prefix()).unwrap();
        let names: Vec<&str> = arrays.iter().map(|(n, _, _)| n.as_str()).collect();
        assert!(names.contains(&"w"), "{names:?}");
        assert!(names.contains(&"b"));
        assert!(names.contains(&"steps"));
        let w = arrays.iter().find(|(n, _, _)| n == "w").unwrap();
        assert_eq!(w.1, vec![3, 4]);
        assert_eq!(w.2[0], 0.0);
        assert_eq!(w.2[11], 11.0);
        let b = arrays.iter().find(|(n, _, _)| n == "b").unwrap();
        assert_eq!(b.2, vec![0.5, -0.5]);
        let steps = arrays.iter().find(|(n, _, _)| n == "steps").unwrap();
        assert_eq!(steps.2, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn accepts_index_suffix_path() {
        let arrays =
            read_tf_checkpoint_arrays(&format!("{}.index", fixture_prefix())).unwrap();
        assert_eq!(arrays.len(), 3);
    }

    #[test]
    fn crc32c_reference_value() {
        // Standard CRC-32C check value for "123456789".
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
    }

    #[test]
    fn corrupted_data_detected() {
        let dir = std::env::temp_dir().join(format!("mmn_tf_corrupt_{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let prefix = fixture_prefix();
        fs::copy(format!("{prefix}.index"), dir.join("ckpt.index")).unwrap();
        let mut data = fs::read(format!("{prefix}.data-00000-of-00001")).unwrap();
        data[50] ^= 0xFF;
        fs::write(dir.join("ckpt.data-00000-of-00001"), data).unwrap();
        let e = read_tf_checkpoint_arrays(dir.join("ckpt").to_str().unwrap())
            .err()
            .unwrap();
        assert!(e.message().contains("CRC32C"), "{}", e.message());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn writer_reader_roundtrip() {
        let dir = std::env::temp_dir().join(format!("mmn_tf_write_{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let prefix = dir.join("out").to_string_lossy().into_owned();
        let arrays = vec![
            ("beta".to_string(), vec![2], vec![0.25, -0.75]),
            ("alpha/kernel".to_string(), vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ];
        write_tf_checkpoint_arrays(&prefix, &arrays).unwrap();
        let back = read_tf_checkpoint_arrays(&prefix).unwrap();
        let kernel = back.iter().find(|(n, _, _)| n == "alpha/kernel").unwrap();
        assert_eq!(kernel.1, vec![2, 3]);
        assert_eq!(kernel.2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let beta = back.iter().find(|(n, _, _)| n == "beta").unwrap();
        assert_eq!(beta.2, vec![0.25, -0.75]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn writer_shape_mismatch_errors() {
        let dir = std::env::temp_dir();
        let prefix = dir.join("mmn_tf_bad_write").to_string_lossy().into_owned();
        let arrays = vec![("x".to_string(), vec![4], vec![1.0])];
        assert!(write_tf_checkpoint_arrays(&prefix, &arrays).is_err());
    }

    #[test]
    fn missing_and_invalid_files_error() {
        assert!(read_tf_checkpoint_arrays("/nonexistent/ckpt").is_err());
        let dir = std::env::temp_dir();
        let bad = dir.join(format!("mmn_tf_bad_{}.index", std::process::id()));
        fs::write(&bad, vec![0u8; 100]).unwrap();
        let e = read_tf_checkpoint_arrays(bad.to_str().unwrap()).err().unwrap();
        assert!(e.message().contains("magic"), "{}", e.message());
        let _ = fs::remove_file(&bad);
    }
}
