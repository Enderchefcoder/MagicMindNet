//! From-scratch minimal HDF5 reader — the TensorFlow/Keras `.h5` bridge.
//!
//! Implements the subset the `h5py`/Keras writers produce ("earliest"
//! libver): superblock version 0/1, version-1 object headers (with
//! continuation blocks), symbol-table groups (B-tree v1 + local heap + SNOD
//! nodes), and compact, contiguous, or **chunked** datasets (B-tree v1 chunk
//! index) of fixed-point / IEEE float datatypes, with **gzip (deflate)** and
//! **shuffle** filters undone by the from-scratch inflate. No HDF5 library
//! is linked.

use super::inflate::inflate;
use super::zip::{is_zip_bytes, read_zip};
use half::f16;
use mmn_core::MmnError;
use std::collections::HashSet;
use std::fs;

const SIGNATURE: &[u8; 8] = b"\x89HDF\r\n\x1a\n";
const UNDEFINED_ADDR: u64 = u64::MAX;

const MSG_DATASPACE: u16 = 0x0001;
const MSG_DATATYPE: u16 = 0x0003;
const MSG_LAYOUT: u16 = 0x0008;
const MSG_FILTER_PIPELINE: u16 = 0x000B;
const MSG_CONTINUATION: u16 = 0x0010;
const MSG_SYMBOL_TABLE: u16 = 0x0011;

fn err(message: impl Into<String>) -> MmnError {
    MmnError::Other {
        message: message.into(),
    }
}

fn get(bytes: &[u8], pos: usize, n: usize) -> Result<&[u8], MmnError> {
    bytes
        .get(pos..pos.checked_add(n).ok_or_else(|| err("hdf5 offset overflow"))?)
        .ok_or_else(|| err("hdf5 file truncated"))
}

fn u16_at(bytes: &[u8], pos: usize) -> Result<u16, MmnError> {
    let b = get(bytes, pos, 2)?;
    Ok(u16::from_le_bytes([b[0], b[1]]))
}

fn u32_at(bytes: &[u8], pos: usize) -> Result<u32, MmnError> {
    let b = get(bytes, pos, 4)?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn u64_at(bytes: &[u8], pos: usize) -> Result<u64, MmnError> {
    let b = get(bytes, pos, 8)?;
    Ok(u64::from_le_bytes([
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
    ]))
}

/// Element type of a dataset, reduced to what we convert to `f32`.
#[derive(Clone, Copy, Debug)]
struct H5Dtype {
    class: u8, // 0 fixed-point, 1 float
    size: usize,
    big_endian: bool,
    signed: bool,
}

impl H5Dtype {
    fn decode(&self, chunk: &[u8]) -> Result<f32, MmnError> {
        let le: Vec<u8> = if self.big_endian {
            chunk.iter().rev().copied().collect()
        } else {
            chunk.to_vec()
        };
        Ok(match (self.class, self.size) {
            (1, 2) => f16::from_le_bytes([le[0], le[1]]).to_f32(),
            (1, 4) => f32::from_le_bytes([le[0], le[1], le[2], le[3]]),
            (1, 8) => f64::from_le_bytes([
                le[0], le[1], le[2], le[3], le[4], le[5], le[6], le[7],
            ]) as f32,
            (0, 1) => {
                if self.signed {
                    (le[0] as i8) as f32
                } else {
                    le[0] as f32
                }
            }
            (0, 2) => {
                if self.signed {
                    i16::from_le_bytes([le[0], le[1]]) as f32
                } else {
                    u16::from_le_bytes([le[0], le[1]]) as f32
                }
            }
            (0, 4) => {
                if self.signed {
                    i32::from_le_bytes([le[0], le[1], le[2], le[3]]) as f32
                } else {
                    u32::from_le_bytes([le[0], le[1], le[2], le[3]]) as f32
                }
            }
            (0, 8) => {
                if self.signed {
                    i64::from_le_bytes([
                        le[0], le[1], le[2], le[3], le[4], le[5], le[6], le[7],
                    ]) as f32
                } else {
                    u64::from_le_bytes([
                        le[0], le[1], le[2], le[3], le[4], le[5], le[6], le[7],
                    ]) as f32
                }
            }
            (class, size) => {
                return Err(err(format!(
                    "hdf5 datatype class {class} size {size} not supported (fixed-point and IEEE float only)"
                )));
            }
        })
    }
}

struct Message {
    msg_type: u16,
    body_start: usize,
}

/// Collect all messages of a version-1 object header, following continuations.
fn read_v1_messages(bytes: &[u8], addr: usize) -> Result<Vec<Message>, MmnError> {
    let version = get(bytes, addr, 1)?[0];
    if version != 1 {
        return Err(err(format!(
            "hdf5 object header version {version} not supported (h5py 'earliest' v1 headers only)"
        )));
    }
    let nmsgs = u16_at(bytes, addr + 2)? as usize;
    let header_size = u32_at(bytes, addr + 8)? as usize;
    let mut blocks: Vec<(usize, usize)> = vec![(addr + 16, header_size)];
    let mut messages = Vec::with_capacity(nmsgs);
    let mut block_idx = 0usize;
    let mut pos = blocks[0].0;
    let mut block_end = blocks[0].0 + blocks[0].1;
    while messages.len() < nmsgs {
        if pos + 8 > block_end {
            block_idx += 1;
            let Some(&(start, len)) = blocks.get(block_idx) else {
                return Err(err("hdf5 object header ran out of blocks before all messages"));
            };
            pos = start;
            block_end = start + len;
        }
        let msg_type = u16_at(bytes, pos)?;
        let size = u16_at(bytes, pos + 2)? as usize;
        let body_start = pos + 8;
        if body_start + size > block_end {
            return Err(err("hdf5 header message overruns its block"));
        }
        if msg_type == MSG_CONTINUATION {
            let cont_addr = u64_at(bytes, body_start)?;
            let cont_len = u64_at(bytes, body_start + 8)?;
            blocks.push((cont_addr as usize, cont_len as usize));
        }
        messages.push(Message {
            msg_type,
            body_start,
        });
        pos = body_start + size;
    }
    Ok(messages)
}

fn parse_dataspace(bytes: &[u8], msg: &Message) -> Result<Vec<usize>, MmnError> {
    let version = get(bytes, msg.body_start, 1)?[0];
    let rank = get(bytes, msg.body_start + 1, 1)?[0] as usize;
    let dims_start = match version {
        1 => msg.body_start + 8,
        2 => msg.body_start + 4,
        other => return Err(err(format!("hdf5 dataspace version {other} not supported"))),
    };
    let mut dims = Vec::with_capacity(rank);
    for i in 0..rank {
        dims.push(u64_at(bytes, dims_start + i * 8)? as usize);
    }
    Ok(dims)
}

fn parse_datatype(bytes: &[u8], msg: &Message) -> Result<H5Dtype, MmnError> {
    let class_and_version = get(bytes, msg.body_start, 1)?[0];
    let class = class_and_version & 0x0F;
    let bit_field0 = get(bytes, msg.body_start + 1, 1)?[0];
    let size = u32_at(bytes, msg.body_start + 4)? as usize;
    Ok(H5Dtype {
        class,
        size,
        big_endian: bit_field0 & 1 != 0,
        signed: class == 0 && bit_field0 & 0x08 != 0,
    })
}

enum Layout {
    Compact {
        start: usize,
        len: usize,
    },
    Contiguous {
        addr: u64,
        size: usize,
    },
    Chunked {
        btree_addr: u64,
        /// Chunk dims incl. the trailing element-size entry (spec layout).
        chunk_dims: Vec<usize>,
    },
}

fn parse_layout(bytes: &[u8], msg: &Message) -> Result<Layout, MmnError> {
    let version = get(bytes, msg.body_start, 1)?[0];
    if version != 3 {
        return Err(err(format!(
            "hdf5 data layout version {version} not supported (v3 only)"
        )));
    }
    let class = get(bytes, msg.body_start + 1, 1)?[0];
    match class {
        0 => {
            let len = u16_at(bytes, msg.body_start + 2)? as usize;
            Ok(Layout::Compact {
                start: msg.body_start + 4,
                len,
            })
        }
        1 => {
            let addr = u64_at(bytes, msg.body_start + 2)?;
            let size = u64_at(bytes, msg.body_start + 10)? as usize;
            Ok(Layout::Contiguous { addr, size })
        }
        2 => {
            let dimensionality = get(bytes, msg.body_start + 2, 1)?[0] as usize;
            let btree_addr = u64_at(bytes, msg.body_start + 3)?;
            let mut chunk_dims = Vec::with_capacity(dimensionality);
            for i in 0..dimensionality {
                chunk_dims.push(u32_at(bytes, msg.body_start + 11 + i * 4)? as usize);
            }
            Ok(Layout::Chunked {
                btree_addr,
                chunk_dims,
            })
        }
        other => Err(err(format!("hdf5 layout class {other} unknown"))),
    }
}

/// One entry of the filter pipeline message: (filter id, client values).
type Filter = (u16, Vec<u32>);

const FILTER_DEFLATE: u16 = 1;
const FILTER_SHUFFLE: u16 = 2;

fn parse_filter_pipeline(bytes: &[u8], msg: &Message) -> Result<Vec<Filter>, MmnError> {
    let version = get(bytes, msg.body_start, 1)?[0];
    let nfilters = get(bytes, msg.body_start + 1, 1)?[0] as usize;
    let mut pos = match version {
        1 => msg.body_start + 8, // 2 + 2 reserved + 4 reserved
        2 => msg.body_start + 2,
        other => return Err(err(format!("hdf5 filter pipeline version {other} unknown"))),
    };
    let mut filters = Vec::with_capacity(nfilters);
    for _ in 0..nfilters {
        let id = u16_at(bytes, pos)?;
        let has_name = version == 1 || id >= 256;
        let name_len = if has_name { u16_at(bytes, pos + 2)? as usize } else { 0 };
        let base = if has_name { pos + 4 } else { pos + 2 };
        let flags = u16_at(bytes, base)?;
        let n_values = u16_at(bytes, base + 2)? as usize;
        let mut vpos = base + 4 + name_len;
        let mut values = Vec::with_capacity(n_values);
        for _ in 0..n_values {
            values.push(u32_at(bytes, vpos)?);
            vpos += 4;
        }
        if version == 1 && n_values % 2 == 1 {
            vpos += 4; // pad to 8-byte multiple
        }
        let _ = flags;
        filters.push((id, values));
        pos = vpos;
    }
    Ok(filters)
}

/// Adler-32 (RFC 1950) for zlib stream verification.
pub(crate) fn adler32(data: &[u8]) -> u32 {
    const MOD: u32 = 65_521;
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for chunk in data.chunks(5_552) {
        for &byte in chunk {
            a += byte as u32;
            b += a;
        }
        a %= MOD;
        b %= MOD;
    }
    (b << 16) | a
}

/// Undo the HDF5 deflate filter: a zlib wrapper (header + deflate + adler32).
pub(crate) fn undo_deflate(raw: &[u8]) -> Result<Vec<u8>, MmnError> {
    if raw.len() < 6 || raw[0] & 0x0F != 8 {
        return Err(err("hdf5 gzip chunk is not a zlib stream"));
    }
    let body = &raw[2..raw.len() - 4];
    let out = inflate(body)?;
    let stored = u32::from_be_bytes([
        raw[raw.len() - 4],
        raw[raw.len() - 3],
        raw[raw.len() - 2],
        raw[raw.len() - 1],
    ]);
    if adler32(&out) != stored {
        return Err(err("hdf5 gzip chunk Adler-32 mismatch"));
    }
    Ok(out)
}

/// Undo the HDF5 shuffle filter (byte transpose by element size).
pub(crate) fn undo_shuffle(raw: &[u8], elem_size: usize) -> Vec<u8> {
    if elem_size <= 1 || !raw.len().is_multiple_of(elem_size) {
        return raw.to_vec();
    }
    let n = raw.len() / elem_size;
    let mut out = vec![0u8; raw.len()];
    for j in 0..elem_size {
        for i in 0..n {
            out[i * elem_size + j] = raw[j * n + i];
        }
    }
    out
}

/// Undo the filter pipeline in reverse write order, honoring the skip mask.
fn undo_filters(
    raw: &[u8],
    filters: &[Filter],
    mask: u32,
    elem_size: usize,
) -> Result<Vec<u8>, MmnError> {
    let mut data = raw.to_vec();
    for (i, (id, _values)) in filters.iter().enumerate().rev() {
        if mask & (1 << i) != 0 {
            continue; // filter skipped for this chunk
        }
        data = match *id {
            FILTER_DEFLATE => undo_deflate(&data)?,
            FILTER_SHUFFLE => undo_shuffle(&data, elem_size),
            other => {
                return Err(err(format!(
                    "hdf5 filter id {other} not supported (gzip and shuffle only)"
                )));
            }
        };
    }
    Ok(data)
}

/// A chunk located by the v1 B-tree: element offsets + data extent.
struct ChunkRef {
    offsets: Vec<usize>,
    filter_mask: u32,
    addr: usize,
    size: usize,
}

/// Walk a raw-data-chunk B-tree (v1, node type 1).
fn walk_chunk_btree(
    bytes: &[u8],
    addr: usize,
    dimensionality: usize,
    out: &mut Vec<ChunkRef>,
) -> Result<(), MmnError> {
    if get(bytes, addr, 4)? != b"TREE" {
        return Err(err("hdf5 chunk B-tree signature mismatch"));
    }
    let node_type = get(bytes, addr + 4, 1)?[0];
    let level = get(bytes, addr + 5, 1)?[0];
    let entries = u16_at(bytes, addr + 6)? as usize;
    if node_type != 1 {
        return Err(err("hdf5 chunk B-tree node type mismatch"));
    }
    let key_len = 8 + 8 * dimensionality;
    let mut pos = addr + 8 + 16; // skip siblings
    for _ in 0..entries {
        let size = u32_at(bytes, pos)? as usize;
        let filter_mask = u32_at(bytes, pos + 4)?;
        let mut offsets = Vec::with_capacity(dimensionality);
        for d in 0..dimensionality {
            offsets.push(u64_at(bytes, pos + 8 + d * 8)? as usize);
        }
        pos += key_len;
        let child = u64_at(bytes, pos)? as usize;
        pos += 8;
        if level > 0 {
            walk_chunk_btree(bytes, child, dimensionality, out)?;
        } else {
            out.push(ChunkRef {
                offsets,
                filter_mask,
                addr: child,
                size,
            });
        }
    }
    Ok(())
}

/// Assemble a chunked dataset into a contiguous row-major byte buffer.
fn assemble_chunked(
    bytes: &[u8],
    name: &str,
    dims: &[usize],
    elem_size: usize,
    btree_addr: u64,
    chunk_dims: &[usize],
    filters: &[Filter],
) -> Result<Vec<u8>, MmnError> {
    // The stored chunk dims carry a trailing element-size entry.
    let rank = dims.len();
    if chunk_dims.len() != rank + 1 {
        return Err(err(format!(
            "hdf5 dataset {name}: chunk rank {} does not match dataspace rank {rank}",
            chunk_dims.len().saturating_sub(1)
        )));
    }
    let chunk_shape = &chunk_dims[..rank];
    let numel: usize = dims.iter().product();
    let mut out = vec![0u8; numel * elem_size];
    if btree_addr == UNDEFINED_ADDR {
        return Ok(out); // never-written dataset
    }
    let mut chunks = Vec::new();
    walk_chunk_btree(bytes, btree_addr as usize, rank + 1, &mut chunks)?;
    // Row-major strides in elements.
    let mut strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    let chunk_numel: usize = chunk_shape.iter().product();
    // Decompress chunks in parallel (gzip inflation dominates load time).
    let decode_one = |chunk: &ChunkRef| -> Result<Vec<u8>, MmnError> {
        let raw = bytes
            .get(chunk.addr..chunk.addr + chunk.size)
            .ok_or_else(|| err(format!("hdf5 dataset {name}: chunk out of bounds")))?;
        undo_filters(raw, filters, chunk.filter_mask, elem_size)
    };
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(chunks.len().max(1));
    let decoded: Vec<Vec<u8>> = if workers <= 1 || chunks.len() <= 1 {
        chunks.iter().map(decode_one).collect::<Result<_, _>>()?
    } else {
        let chunk_size = chunks.len().div_ceil(workers);
        let results: Vec<Result<Vec<Vec<u8>>, MmnError>> = std::thread::scope(|scope| {
            let handles: Vec<_> = chunks
                .chunks(chunk_size)
                .map(|group| scope.spawn(move || group.iter().map(decode_one).collect()))
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("hdf5 chunk decode worker panicked"))
                .collect()
        });
        let mut all = Vec::with_capacity(chunks.len());
        for group in results {
            all.extend(group?);
        }
        all
    };
    for (chunk, data) in chunks.iter().zip(decoded) {
        if data.len() < chunk_numel * elem_size {
            return Err(err(format!(
                "hdf5 dataset {name}: chunk decoded to {} bytes, expected {}",
                data.len(),
                chunk_numel * elem_size
            )));
        }
        // Copy row-runs of the last dimension, clipping edge chunks.
        let last = rank - 1;
        let row_len = chunk_shape[last]
            .min(dims[last].saturating_sub(chunk.offsets[last]));
        if row_len == 0 {
            continue;
        }
        let outer: usize = chunk_shape[..last].iter().product();
        let mut index = vec![0usize; last];
        for _ in 0..outer.max(1) {
            let mut in_bounds = true;
            let mut target = chunk.offsets[last];
            for d in 0..last {
                let pos = chunk.offsets[d] + index[d];
                if pos >= dims[d] {
                    in_bounds = false;
                    break;
                }
                target += pos * strides[d];
            }
            if in_bounds {
                let mut src = 0usize;
                let mut mul = 1usize;
                for d in (0..last).rev() {
                    src += index[d] * mul * chunk_shape[last];
                    mul *= chunk_shape[d];
                }
                // src currently counts elements of the flattened chunk rows.
                let src_start = src * elem_size;
                let dst_start = target * elem_size;
                out[dst_start..dst_start + row_len * elem_size]
                    .copy_from_slice(&data[src_start..src_start + row_len * elem_size]);
            }
            // Odometer increment over the outer chunk dims.
            for d in (0..last).rev() {
                index[d] += 1;
                if index[d] < chunk_shape[d] {
                    break;
                }
                index[d] = 0;
            }
        }
    }
    Ok(out)
}

/// Read a NUL-terminated name from the local heap data segment.
fn heap_name(bytes: &[u8], heap_data_addr: usize, offset: usize) -> Result<String, MmnError> {
    let start = heap_data_addr + offset;
    let mut end = start;
    while end < bytes.len() && bytes[end] != 0 {
        end += 1;
    }
    Ok(String::from_utf8_lossy(get(bytes, start, end - start)?).into_owned())
}

/// Local heap: returns the data segment address.
fn parse_local_heap(bytes: &[u8], addr: usize) -> Result<usize, MmnError> {
    if get(bytes, addr, 4)? != b"HEAP" {
        return Err(err("hdf5 local heap signature mismatch"));
    }
    Ok(u64_at(bytes, addr + 24)? as usize)
}

/// Walk a group B-tree (v1), collecting `(name, object header addr)` links.
fn walk_btree(
    bytes: &[u8],
    btree_addr: usize,
    heap_data: usize,
    out: &mut Vec<(String, usize)>,
) -> Result<(), MmnError> {
    if get(bytes, btree_addr, 4)? != b"TREE" {
        return Err(err("hdf5 B-tree signature mismatch"));
    }
    let node_type = get(bytes, btree_addr + 4, 1)?[0];
    let level = get(bytes, btree_addr + 5, 1)?[0];
    let entries = u16_at(bytes, btree_addr + 6)? as usize;
    if node_type != 0 {
        return Err(err("hdf5 B-tree node type is not a group node"));
    }
    // Keys and children alternate after 2 sibling pointers (8 bytes each).
    let mut pos = btree_addr + 8 + 16;
    for _ in 0..entries {
        pos += 8; // key (heap offset) — names come from SNOD entries instead
        let child = u64_at(bytes, pos)? as usize;
        pos += 8;
        if level > 0 {
            walk_btree(bytes, child, heap_data, out)?;
        } else {
            read_snod(bytes, child, heap_data, out)?;
        }
    }
    Ok(())
}

fn read_snod(
    bytes: &[u8],
    addr: usize,
    heap_data: usize,
    out: &mut Vec<(String, usize)>,
) -> Result<(), MmnError> {
    if get(bytes, addr, 4)? != b"SNOD" {
        return Err(err("hdf5 symbol node signature mismatch"));
    }
    let count = u16_at(bytes, addr + 6)? as usize;
    let mut pos = addr + 8;
    for _ in 0..count {
        let name_offset = u64_at(bytes, pos)? as usize;
        let header_addr = u64_at(bytes, pos + 8)? as usize;
        let name = heap_name(bytes, heap_data, name_offset)?;
        out.push((name, header_addr));
        pos += 40; // entry: offsets(16) + cache type(4) + reserved(4) + scratch(16)
    }
    Ok(())
}

fn dataset_values(
    bytes: &[u8],
    name: &str,
    dims: &[usize],
    dtype: H5Dtype,
    layout: Layout,
    filters: &[Filter],
) -> Result<Vec<f32>, MmnError> {
    let numel: usize = dims.iter().product();
    let expected = numel * dtype.size;
    let assembled;
    let raw: &[u8] = match layout {
        Layout::Compact { start, len } => get(bytes, start, len)?,
        Layout::Contiguous { addr, size } => {
            if addr == UNDEFINED_ADDR {
                // Allocated-on-write dataset that was never written: zeros.
                return Ok(vec![0.0; numel]);
            }
            get(bytes, addr as usize, size)?
        }
        Layout::Chunked {
            btree_addr,
            chunk_dims,
        } => {
            if dims.is_empty() {
                return Err(err(format!("hdf5 dataset {name}: chunked scalar unsupported")));
            }
            assembled = assemble_chunked(
                bytes,
                name,
                dims,
                dtype.size,
                btree_addr,
                &chunk_dims,
                filters,
            )?;
            &assembled
        }
    };
    if raw.len() < expected {
        return Err(err(format!(
            "hdf5 dataset {name}: expected {expected} bytes, found {}",
            raw.len()
        )));
    }
    let mut values = Vec::with_capacity(numel);
    for chunk in raw[..expected].chunks_exact(dtype.size) {
        values.push(dtype.decode(chunk)?);
    }
    Ok(values)
}

fn visit_object(
    bytes: &[u8],
    path: &str,
    addr: usize,
    out: &mut Vec<super::NamedArray>,
    seen: &mut HashSet<usize>,
) -> Result<(), MmnError> {
    if !seen.insert(addr) {
        return Ok(()); // hard link cycle
    }
    let messages = read_v1_messages(bytes, addr)?;
    let mut dims = None;
    let mut dtype = None;
    let mut layout = None;
    let mut symbol_table = None;
    let mut filters: Vec<Filter> = Vec::new();
    for msg in &messages {
        match msg.msg_type {
            MSG_DATASPACE => dims = Some(parse_dataspace(bytes, msg)?),
            MSG_DATATYPE => dtype = Some(parse_datatype(bytes, msg)?),
            MSG_LAYOUT => layout = Some(parse_layout(bytes, msg)),
            MSG_FILTER_PIPELINE => filters = parse_filter_pipeline(bytes, msg)?,
            MSG_SYMBOL_TABLE => {
                let btree = u64_at(bytes, msg.body_start)? as usize;
                let heap = u64_at(bytes, msg.body_start + 8)? as usize;
                symbol_table = Some((btree, heap));
            }
            _ => {}
        }
    }
    if let Some((btree, heap)) = symbol_table {
        let heap_data = parse_local_heap(bytes, heap)?;
        let mut links = Vec::new();
        walk_btree(bytes, btree, heap_data, &mut links)?;
        links.sort_by(|a, b| a.0.cmp(&b.0));
        for (name, child_addr) in links {
            let child_path = if path.is_empty() {
                name
            } else {
                format!("{path}/{name}")
            };
            visit_object(bytes, &child_path, child_addr, out, seen)?;
        }
        return Ok(());
    }
    if let (Some(dims), Some(dtype), Some(layout)) = (dims, dtype, layout) {
        let values = dataset_values(bytes, path, &dims, dtype, layout?, &filters)?;
        out.push((path.to_string(), dims, values));
    }
    Ok(())
}

/// True when the buffer starts with the HDF5 signature.
pub fn is_hdf5_bytes(bytes: &[u8]) -> bool {
    bytes.len() >= SIGNATURE.len() && &bytes[..SIGNATURE.len()] == SIGNATURE
}

/// Read every dataset in an HDF5 byte buffer as `(path, shape, f32 data)`.
pub fn read_h5_arrays_bytes(bytes: &[u8]) -> Result<Vec<super::NamedArray>, MmnError> {
    if bytes.len() < 9 || &bytes[..8] != SIGNATURE {
        return Err(err("not an HDF5 file (missing \\x89HDF signature)"));
    }
    let version = bytes[8];
    if version > 1 {
        return Err(err(format!(
            "hdf5 superblock version {version} not supported (0/1 as written by h5py/Keras defaults)"
        )));
    }
    let size_of_offsets = get(bytes, 13, 1)?[0];
    let size_of_lengths = get(bytes, 14, 1)?[0];
    if size_of_offsets != 8 || size_of_lengths != 8 {
        return Err(err(format!(
            "hdf5 offset/length sizes {size_of_offsets}/{size_of_lengths} not supported (8/8 only)"
        )));
    }
    // Root group symbol table entry: superblock v0 at 24+32=56 (v1 adds 4 bytes).
    let entry = if version == 0 { 56 } else { 60 };
    let root_header = u64_at(bytes, entry + 8)? as usize;
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    visit_object(bytes, "", root_header, &mut out, &mut seen)?;
    Ok(out)
}

/// Read every dataset in an `.h5` file on disk.
pub fn read_h5_arrays(path: &str) -> Result<Vec<super::NamedArray>, MmnError> {
    let bytes = fs::read(path).map_err(|e| err(format!("cannot read h5 {path}: {e}")))?;
    read_h5_arrays_bytes(&bytes)
}

/// Read weights from a Keras archive: `.keras` zip (contains
/// `model.weights.h5`), `.weights.h5`, or a plain `.h5` file.
pub fn read_keras_arrays(path: &str) -> Result<Vec<super::NamedArray>, MmnError> {
    let bytes = fs::read(path).map_err(|e| err(format!("cannot read {path}: {e}")))?;
    if is_zip_bytes(&bytes) {
        let entries = read_zip(&bytes)?;
        let weights = entries
            .iter()
            .find(|e| e.name.ends_with(".weights.h5") || e.name.ends_with("model.weights.h5"))
            .ok_or_else(|| {
                err("keras archive has no model.weights.h5 entry (not a .keras v3 file)")
            })?;
        return read_h5_arrays_bytes(&weights.data);
    }
    read_h5_arrays_bytes(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/simple.h5")
    }

    #[test]
    fn reads_h5py_fixture_datasets() {
        let arrays = read_h5_arrays(fixture().to_str().unwrap()).unwrap();
        let names: Vec<&str> = arrays.iter().map(|(n, _, _)| n.as_str()).collect();
        assert!(names.contains(&"weights"), "{names:?}");
        assert!(names.contains(&"bias"));
        assert!(names.contains(&"layer1/kernel"));
        assert!(names.contains(&"layer1/ints"));
        let weights = arrays.iter().find(|(n, _, _)| n == "weights").unwrap();
        assert_eq!(weights.1, vec![3, 4]);
        assert_eq!(weights.2[5], 5.0);
        let bias = arrays.iter().find(|(n, _, _)| n == "bias").unwrap();
        assert_eq!(bias.2, vec![-1.0, 0.5]);
        let kernel = arrays.iter().find(|(n, _, _)| n == "layer1/kernel").unwrap();
        assert!(kernel.2.iter().all(|&v| v == 7.0));
        let ints = arrays.iter().find(|(n, _, _)| n == "layer1/ints").unwrap();
        assert_eq!(ints.2, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn keras_zip_wrapper_reads_weights() {
        let h5 = fs::read(fixture()).unwrap();
        let archive = crate::interop::zip::write_zip_stored(&[
            ("metadata.json".to_string(), b"{}".to_vec()),
            ("model.weights.h5".to_string(), h5),
        ])
        .unwrap();
        let dir = std::env::temp_dir();
        let path = dir.join(format!("mmn_keras_{}.keras", std::process::id()));
        fs::write(&path, archive).unwrap();
        let arrays = read_keras_arrays(path.to_str().unwrap()).unwrap();
        assert!(arrays.iter().any(|(n, _, _)| n == "layer1/kernel"));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn chunked_gzip_shuffle_fixture_reads() {
        // Written by h5py: gzip'd 64x64 arange, gzip+shuffle 32x32 arange,
        // plain-chunked 16x16 ones (chunks of (4,16)).
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/chunked.h5");
        let arrays = read_h5_arrays(path.to_str().unwrap()).unwrap();
        let gz = arrays.iter().find(|(n, _, _)| n == "gz").unwrap();
        assert_eq!(gz.1, vec![64, 64]);
        assert_eq!(gz.2[0], 0.0);
        assert_eq!(gz.2[65], 65.0);
        assert_eq!(gz.2[4095], 4095.0);
        let shuf = arrays.iter().find(|(n, _, _)| n == "shuf").unwrap();
        assert_eq!(shuf.1, vec![32, 32]);
        assert_eq!(shuf.2[100], 100.0);
        assert_eq!(shuf.2[1023], 1023.0);
        let plain = arrays.iter().find(|(n, _, _)| n == "plainchunk").unwrap();
        assert!(plain.2.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn adler32_reference_value() {
        // zlib.adler32(b"Wikipedia") == 0x11E60398.
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
        assert_eq!(adler32(b""), 1);
    }

    #[test]
    fn shuffle_filter_roundtrip_shape() {
        // shuffle stores all byte-0s, then byte-1s, ...; undo restores order.
        let original: Vec<u8> = (0..16).collect();
        let mut shuffled = vec![0u8; 16];
        for (i, chunk) in original.chunks(4).enumerate() {
            for (j, &b) in chunk.iter().enumerate() {
                shuffled[j * 4 + i] = b;
            }
        }
        assert_eq!(undo_shuffle(&shuffled, 4), original);
    }

    #[test]
    fn bad_signature_errors() {
        assert!(read_h5_arrays_bytes(b"not an hdf5 file").is_err());
        assert!(read_h5_arrays_bytes(b"").is_err());
    }

    #[test]
    fn truncated_file_errors() {
        let bytes = fs::read(fixture()).unwrap();
        assert!(read_h5_arrays_bytes(&bytes[..200]).is_err());
    }
}
