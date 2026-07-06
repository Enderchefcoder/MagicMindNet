//! From-scratch Snappy (raw block format) decoder — Blosc's remaining
//! inner codec.
//!
//! Varint uncompressed-length preamble, then tagged elements: literals
//! (with 1-4 extra length bytes for long runs) and copies with 1-, 2-, or
//! 4-byte offsets, overlap-safe. No snappy library is linked.

use mmn_core::MmnError;

fn err(message: impl Into<String>) -> MmnError {
    MmnError::Other {
        message: message.into(),
    }
}

/// Decompress a raw Snappy block.
pub fn snappy_decompress(src: &[u8]) -> Result<Vec<u8>, MmnError> {
    let mut ip = 0usize;
    // Varint preamble: little-endian 7-bit groups.
    let mut expected = 0usize;
    let mut shift = 0u32;
    loop {
        let byte = *src.get(ip).ok_or_else(|| err("snappy length varint truncated"))?;
        ip += 1;
        expected |= ((byte & 0x7F) as usize) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift > 35 {
            return Err(err("snappy length varint too long"));
        }
    }
    let mut out: Vec<u8> = Vec::with_capacity(expected);
    while ip < src.len() {
        let tag = src[ip];
        ip += 1;
        match tag & 0x3 {
            0 => {
                // Literal; lengths 61-64 in the tag escape to extra bytes.
                let mut len = (tag >> 2) as usize + 1;
                if len > 60 {
                    let extra = len - 60;
                    let mut value = 0usize;
                    for i in 0..extra {
                        let byte = *src
                            .get(ip + i)
                            .ok_or_else(|| err("snappy literal length truncated"))?;
                        value |= (byte as usize) << (8 * i);
                    }
                    ip += extra;
                    len = value + 1;
                }
                let literals = src
                    .get(ip..ip + len)
                    .ok_or_else(|| err("snappy literals truncated"))?;
                out.extend_from_slice(literals);
                ip += len;
            }
            kind => {
                let (len, offset) = match kind {
                    1 => {
                        let byte = *src
                            .get(ip)
                            .ok_or_else(|| err("snappy copy offset truncated"))?;
                        ip += 1;
                        (
                            ((tag >> 2) & 0x7) as usize + 4,
                            (((tag >> 5) as usize) << 8) | byte as usize,
                        )
                    }
                    2 => {
                        let bytes = src
                            .get(ip..ip + 2)
                            .ok_or_else(|| err("snappy copy offset truncated"))?;
                        ip += 2;
                        (
                            (tag >> 2) as usize + 1,
                            u16::from_le_bytes([bytes[0], bytes[1]]) as usize,
                        )
                    }
                    _ => {
                        let bytes = src
                            .get(ip..ip + 4)
                            .ok_or_else(|| err("snappy copy offset truncated"))?;
                        ip += 4;
                        (
                            (tag >> 2) as usize + 1,
                            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                                as usize,
                        )
                    }
                };
                if offset == 0 || offset > out.len() {
                    return Err(err(format!(
                        "snappy copy offset {offset} invalid at output position {}",
                        out.len()
                    )));
                }
                let start = out.len() - offset;
                for k in 0..len {
                    let byte = out[start + k];
                    out.push(byte);
                }
            }
        }
        if out.len() > expected {
            return Err(err("snappy output exceeds its declared length"));
        }
    }
    if out.len() != expected {
        return Err(err(format!(
            "snappy output length {} != declared {expected}",
            out.len()
        )));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture(name: &str) -> Vec<u8> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/snappy")
            .join(name);
        std::fs::read(&path).unwrap_or_else(|e| panic!("fixture {name} missing: {e}"))
    }

    #[test]
    fn reference_streams_decode() {
        // Written by cramjam (rust-snappy) — repetitive and random cases.
        let out = snappy_decompress(&fixture("pattern.snappy")).unwrap();
        let expected: Vec<u8> = (0..50_000u32)
            .flat_map(|i| (((i % 97) as f32) / 7.0).to_le_bytes())
            .collect();
        assert_eq!(out, expected);
        let out = snappy_decompress(&fixture("text.snappy")).unwrap();
        assert_eq!(out, b"the quick brown fox jumps over the lazy dog. ".repeat(400));
    }

    #[test]
    fn hand_built_stream_decodes() {
        // len 9, literal "abc", copy len 6 offset 3.
        let src = [9u8, 0x08, b'a', b'b', b'c', 0x16, 0x03, 0x00];
        assert_eq!(snappy_decompress(&src).unwrap(), b"abcabcabc");
    }

    #[test]
    fn corrupt_streams_error() {
        assert!(snappy_decompress(&[]).is_err());
        assert!(snappy_decompress(&[9, 0x08, b'a']).is_err()); // literals cut
        assert!(snappy_decompress(&[9, 0x16, 0x03, 0x00]).is_err()); // copy first
        let src = [3u8, 0x08, b'a', b'b', b'c']; // declared 3, produces 3? ok
        assert_eq!(snappy_decompress(&src).unwrap(), b"abc");
    }
}
