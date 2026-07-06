//! From-scratch minimal HDF5 writer: superblock v0, symbol-table groups
//! (B-tree v1 + local heap + SNOD), version-1 object headers, and contiguous
//! F32 datasets — the layout `h5py`/libhdf5 read natively.

use mmn_core::MmnError;
use std::collections::BTreeMap;

const SIGNATURE: &[u8; 8] = b"\x89HDF\r\n\x1a\n";
const UNDEF: u64 = u64::MAX;
const LEAF_K: usize = 4;
const INTERNAL_K: usize = 16;
const SNOD_CAP: usize = 2 * LEAF_K;
/// libhdf5 reads SNOD/B-tree nodes at their full fixed allocation size.
const SNOD_ALLOC: usize = 8 + SNOD_CAP * 40;
const BTREE_MAX_CHILDREN: usize = 2 * INTERNAL_K;
const BTREE_ALLOC: usize = 24 + (BTREE_MAX_CHILDREN + 1) * 8 + BTREE_MAX_CHILDREN * 8;

fn err(message: impl Into<String>) -> MmnError {
    MmnError::Other {
        message: message.into(),
    }
}

enum Node {
    Group(BTreeMap<String, Node>),
    Dataset(usize),
}

fn insert_path(root: &mut BTreeMap<String, Node>, path: &str, index: usize) -> Result<(), MmnError> {
    let mut parts = path.split('/').filter(|p| !p.is_empty()).peekable();
    let mut current = root;
    while let Some(part) = parts.next() {
        if parts.peek().is_none() {
            match current.insert(part.to_string(), Node::Dataset(index)) {
                None | Some(Node::Dataset(_)) => return Ok(()),
                Some(Node::Group(_)) => {
                    return Err(err(format!("h5 name {path} collides with a group")));
                }
            }
        }
        let entry = current
            .entry(part.to_string())
            .or_insert_with(|| Node::Group(BTreeMap::new()));
        current = match entry {
            Node::Group(children) => children,
            Node::Dataset(_) => {
                return Err(err(format!("h5 name {path} nests under a dataset")));
            }
        };
    }
    Err(err("h5 dataset name must be non-empty"))
}

fn align8(buf: &mut Vec<u8>) {
    while !buf.len().is_multiple_of(8) {
        buf.push(0);
    }
}

fn push_message(messages: &mut Vec<u8>, msg_type: u16, body: &[u8]) {
    let padded = body.len().div_ceil(8) * 8;
    messages.extend_from_slice(&msg_type.to_le_bytes());
    messages.extend_from_slice(&(padded as u16).to_le_bytes());
    messages.extend_from_slice(&[0u8; 4]); // flags + reserved
    messages.extend_from_slice(body);
    messages.extend(std::iter::repeat_n(0u8, padded - body.len()));
}

fn write_object_header(buf: &mut Vec<u8>, messages: &[(u16, Vec<u8>)]) -> u64 {
    let mut body = Vec::new();
    for (msg_type, msg_body) in messages {
        push_message(&mut body, *msg_type, msg_body);
    }
    align8(buf);
    let addr = buf.len() as u64;
    buf.push(1); // version
    buf.push(0);
    buf.extend_from_slice(&(messages.len() as u16).to_le_bytes());
    buf.extend_from_slice(&1u32.to_le_bytes()); // reference count
    buf.extend_from_slice(&(body.len() as u32).to_le_bytes());
    buf.extend_from_slice(&[0u8; 4]); // pad to 8
    buf.extend_from_slice(&body);
    addr
}

/// IEEE F32 little-endian datatype message (h5py bit-field conventions).
fn f32_datatype_body() -> Vec<u8> {
    let mut body = vec![0x11, 0x20, 0x1F, 0x00]; // v1|float, norm msb, sign loc 31
    body.extend_from_slice(&4u32.to_le_bytes()); // size
    body.extend_from_slice(&0u16.to_le_bytes()); // bit offset
    body.extend_from_slice(&32u16.to_le_bytes()); // precision
    body.push(23); // exponent location
    body.push(8); // exponent size
    body.push(0); // mantissa location
    body.push(23); // mantissa size
    body.extend_from_slice(&127u32.to_le_bytes()); // exponent bias
    body
}

fn write_dataset(buf: &mut Vec<u8>, shape: &[usize], values: &[f32]) -> u64 {
    align8(buf);
    let data_addr = buf.len() as u64;
    for v in values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    let mut dataspace = vec![1u8, shape.len() as u8, 0, 0, 0, 0, 0, 0];
    for &d in shape {
        dataspace.extend_from_slice(&(d as u64).to_le_bytes());
    }
    // Fill value v2: alloc time 2 (late), write time 2, defined 1, no data.
    let fill = vec![2u8, 2, 2, 1, 0, 0, 0, 0];
    let mut layout = vec![3u8, 1]; // v3, contiguous
    layout.extend_from_slice(&data_addr.to_le_bytes());
    layout.extend_from_slice(&((values.len() * 4) as u64).to_le_bytes());
    write_object_header(
        buf,
        &[
            (0x0001, dataspace),
            (0x0003, f32_datatype_body()),
            (0x0005, fill),
            (0x0008, layout),
        ],
    )
}

fn write_group(buf: &mut Vec<u8>, children: &[(String, u64)]) -> u64 {
    // Local heap data: 8 reserved bytes, NUL-terminated names (8-aligned),
    // then a 16-byte "no more free blocks" descriptor.
    let mut heap_data = vec![0u8; 8];
    let mut name_offsets = Vec::with_capacity(children.len());
    for (name, _) in children {
        name_offsets.push(heap_data.len() as u64);
        heap_data.extend_from_slice(name.as_bytes());
        heap_data.push(0);
        while !heap_data.len().is_multiple_of(8) {
            heap_data.push(0);
        }
    }
    let free_offset = heap_data.len() as u64;
    heap_data.extend_from_slice(&1u64.to_le_bytes()); // next free: none
    heap_data.extend_from_slice(&16u64.to_le_bytes()); // free block size
    align8(buf);
    let heap_data_addr = buf.len() as u64;
    buf.extend_from_slice(&heap_data);
    align8(buf);
    let heap_addr = buf.len() as u64;
    buf.extend_from_slice(b"HEAP");
    buf.extend_from_slice(&[0u8; 4]); // version + reserved
    buf.extend_from_slice(&(heap_data.len() as u64).to_le_bytes());
    buf.extend_from_slice(&free_offset.to_le_bytes());
    buf.extend_from_slice(&heap_data_addr.to_le_bytes());
    // SNOD leaves (children sorted by BTreeMap order = lexicographic),
    // each padded to the full fixed node allocation libhdf5 expects.
    let mut snod_addrs = Vec::new();
    let mut boundary_keys = Vec::new();
    for (chunk_idx, chunk) in children.chunks(SNOD_CAP).enumerate() {
        align8(buf);
        let snod_addr = buf.len() as u64;
        buf.extend_from_slice(b"SNOD");
        buf.push(1);
        buf.push(0);
        buf.extend_from_slice(&(chunk.len() as u16).to_le_bytes());
        for (i, (_, header_addr)) in chunk.iter().enumerate() {
            let offset = name_offsets[chunk_idx * SNOD_CAP + i];
            buf.extend_from_slice(&offset.to_le_bytes());
            buf.extend_from_slice(&header_addr.to_le_bytes());
            buf.extend_from_slice(&[0u8; 24]); // cache type 0 + reserved + scratch
        }
        let written = buf.len() - snod_addr as usize;
        buf.extend(std::iter::repeat_n(0u8, SNOD_ALLOC - written));
        snod_addrs.push(snod_addr);
        let last_index = chunk_idx * SNOD_CAP + chunk.len() - 1;
        boundary_keys.push(name_offsets[last_index]);
    }
    align8(buf);
    let btree_addr = buf.len() as u64;
    buf.extend_from_slice(b"TREE");
    buf.push(0); // node type: group
    buf.push(0); // level
    buf.extend_from_slice(&(snod_addrs.len() as u16).to_le_bytes());
    buf.extend_from_slice(&UNDEF.to_le_bytes()); // left sibling
    buf.extend_from_slice(&UNDEF.to_le_bytes()); // right sibling
    buf.extend_from_slice(&0u64.to_le_bytes()); // key 0: empty name (offset 0)
    for (snod_addr, boundary) in snod_addrs.iter().zip(&boundary_keys) {
        buf.extend_from_slice(&snod_addr.to_le_bytes());
        buf.extend_from_slice(&boundary.to_le_bytes());
    }
    let written = buf.len() - btree_addr as usize;
    buf.extend(std::iter::repeat_n(0u8, BTREE_ALLOC - written));
    let mut symbol_table = Vec::with_capacity(16);
    symbol_table.extend_from_slice(&btree_addr.to_le_bytes());
    symbol_table.extend_from_slice(&heap_addr.to_le_bytes());
    write_object_header(buf, &[(0x0011, symbol_table)])
}

/// Single-level group B-trees cap out at `SNOD_CAP * BTREE_MAX_CHILDREN`
/// children per group.
fn check_fanout(node: &Node) -> Result<(), MmnError> {
    if let Node::Group(children) = node {
        if children.len() > SNOD_CAP * BTREE_MAX_CHILDREN {
            return Err(err(format!(
                "h5 writer supports up to {} entries per group, got {}; nest names into subgroups",
                SNOD_CAP * BTREE_MAX_CHILDREN,
                children.len()
            )));
        }
        for child in children.values() {
            check_fanout(child)?;
        }
    }
    Ok(())
}

fn write_node(
    buf: &mut Vec<u8>,
    node: &Node,
    arrays: &[super::NamedArray],
) -> u64 {
    match node {
        Node::Dataset(index) => {
            let (_, shape, values) = &arrays[*index];
            write_dataset(buf, shape, values)
        }
        Node::Group(children) => {
            let resolved: Vec<(String, u64)> = children
                .iter()
                .map(|(name, child)| (name.clone(), write_node(buf, child, arrays)))
                .collect();
            write_group(buf, &resolved)
        }
    }
}

/// Serialize named arrays into an HDF5 byte buffer (`h5py`-readable).
pub fn write_h5_arrays_bytes(arrays: &[super::NamedArray]) -> Result<Vec<u8>, MmnError> {
    if arrays.is_empty() {
        return Err(err("h5 writer needs at least one array"));
    }
    let mut root = BTreeMap::new();
    for (index, (name, shape, values)) in arrays.iter().enumerate() {
        let numel: usize = shape.iter().product();
        if numel != values.len() {
            return Err(err(format!(
                "h5 array {name}: shape {shape:?} needs {numel} values, got {}",
                values.len()
            )));
        }
        insert_path(&mut root, name, index)?;
    }
    let root = Node::Group(root);
    check_fanout(&root)?;
    let mut buf = vec![0u8; 96]; // superblock placeholder
    let root_addr = write_node(&mut buf, &root, arrays);
    let eof = buf.len() as u64;
    // Superblock v0.
    buf[..8].copy_from_slice(SIGNATURE);
    // versions + sizes: sb 0, fs 0, root 0, reserved, shared 0, offsets 8, lengths 8, reserved
    buf[8..16].copy_from_slice(&[0, 0, 0, 0, 0, 8, 8, 0]);
    buf[16..18].copy_from_slice(&4u16.to_le_bytes()); // group leaf K
    buf[18..20].copy_from_slice(&16u16.to_le_bytes()); // group internal K
    buf[20..24].copy_from_slice(&0u32.to_le_bytes()); // consistency flags
    buf[24..32].copy_from_slice(&0u64.to_le_bytes()); // base address
    buf[32..40].copy_from_slice(&UNDEF.to_le_bytes()); // free space
    buf[40..48].copy_from_slice(&eof.to_le_bytes());
    buf[48..56].copy_from_slice(&UNDEF.to_le_bytes()); // driver info
    // Root symbol table entry: name offset 0, header addr, cache type 0.
    buf[56..64].copy_from_slice(&0u64.to_le_bytes());
    buf[64..72].copy_from_slice(&root_addr.to_le_bytes());
    // cache type + reserved + scratch already zero.
    Ok(buf)
}

/// Write named arrays to an `.h5` file readable by h5py/Keras tooling.
pub fn write_h5_arrays(path: &str, arrays: &[super::NamedArray]) -> Result<(), MmnError> {
    crate::checkpoint_util::write_file_create_parents(path, write_h5_arrays_bytes(arrays)?)
}

#[cfg(test)]
mod tests {
    use super::super::hdf5::read_h5_arrays_bytes;
    use super::*;

    #[test]
    fn writer_reader_roundtrip_flat_and_nested() {
        let arrays = vec![
            ("weights".to_string(), vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ("layer1/kernel".to_string(), vec![2, 2], vec![7.0, 8.0, 9.0, 10.0]),
            ("layer1/bias".to_string(), vec![2], vec![-1.0, 1.0]),
        ];
        let bytes = write_h5_arrays_bytes(&arrays).unwrap();
        let back = read_h5_arrays_bytes(&bytes).unwrap();
        let mut names: Vec<&str> = back.iter().map(|(n, _, _)| n.as_str()).collect();
        names.sort();
        assert_eq!(names, vec!["layer1/bias", "layer1/kernel", "weights"]);
        let w = back.iter().find(|(n, _, _)| n == "weights").unwrap();
        assert_eq!(w.1, vec![2, 3]);
        assert_eq!(w.2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = back.iter().find(|(n, _, _)| n == "layer1/bias").unwrap();
        assert_eq!(b.2, vec![-1.0, 1.0]);
    }

    #[test]
    fn many_children_use_multiple_snods() {
        let arrays: Vec<super::super::NamedArray> = (0..20)
            .map(|i| (format!("t{i:02}"), vec![2], vec![i as f32, -(i as f32)]))
            .collect();
        let bytes = write_h5_arrays_bytes(&arrays).unwrap();
        let back = read_h5_arrays_bytes(&bytes).unwrap();
        assert_eq!(back.len(), 20);
        let t13 = back.iter().find(|(n, _, _)| n == "t13").unwrap();
        assert_eq!(t13.2, vec![13.0, -13.0]);
    }

    #[test]
    fn invalid_inputs_error() {
        assert!(write_h5_arrays_bytes(&[]).is_err());
        let bad = vec![("x".to_string(), vec![3], vec![1.0])];
        assert!(write_h5_arrays_bytes(&bad).is_err());
        let clash = vec![
            ("a/b".to_string(), vec![1], vec![1.0]),
            ("a".to_string(), vec![1], vec![1.0]),
        ];
        assert!(write_h5_arrays_bytes(&clash).is_err());
    }
}
