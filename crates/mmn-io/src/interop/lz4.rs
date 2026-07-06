//! From-scratch LZ4 block-format decoder (the codec inside Blosc frames).
//!
//! Implements the documented block format: token byte with 4-bit literal
//! and match lengths (15 escapes to 255-run extension bytes), little-endian
//! 2-byte match offsets, 4-byte minimum matches, and overlap-safe copies.
//! No LZ4 library is linked.

use mmn_core::MmnError;

fn err(message: impl Into<String>) -> MmnError {
    MmnError::Other {
        message: message.into(),
    }
}

/// Decompress one LZ4 block into exactly `expected_len` bytes.
pub fn lz4_decompress_block(src: &[u8], expected_len: usize) -> Result<Vec<u8>, MmnError> {
    let mut out: Vec<u8> = Vec::with_capacity(expected_len);
    let mut ip = 0usize;
    loop {
        let token = *src.get(ip).ok_or_else(|| err("lz4 block truncated at token"))?;
        ip += 1;
        // Literal run.
        let mut literal_len = (token >> 4) as usize;
        if literal_len == 15 {
            loop {
                let byte = *src.get(ip).ok_or_else(|| err("lz4 literal length truncated"))?;
                ip += 1;
                literal_len += byte as usize;
                if byte != 255 {
                    break;
                }
            }
        }
        let literals = src
            .get(ip..ip + literal_len)
            .ok_or_else(|| err("lz4 literals truncated"))?;
        out.extend_from_slice(literals);
        ip += literal_len;
        if ip == src.len() {
            // Last sequence ends with literals only.
            break;
        }
        // Match.
        let offset_bytes = src
            .get(ip..ip + 2)
            .ok_or_else(|| err("lz4 match offset truncated"))?;
        let offset = u16::from_le_bytes([offset_bytes[0], offset_bytes[1]]) as usize;
        ip += 2;
        if offset == 0 || offset > out.len() {
            return Err(err(format!(
                "lz4 match offset {offset} invalid at output position {}",
                out.len()
            )));
        }
        let mut match_len = (token & 0x0F) as usize;
        if match_len == 15 {
            loop {
                let byte = *src.get(ip).ok_or_else(|| err("lz4 match length truncated"))?;
                ip += 1;
                match_len += byte as usize;
                if byte != 255 {
                    break;
                }
            }
        }
        match_len += 4; // minimum match
        // Overlap-safe byte copy (offsets smaller than the match length
        // repeat the just-written bytes).
        let start = out.len() - offset;
        for k in 0..match_len {
            let byte = out[start + k];
            out.push(byte);
        }
        if out.len() > expected_len {
            return Err(err("lz4 output exceeds expected length"));
        }
        if ip == src.len() {
            // Spec ends blocks with a literal-only sequence, but accept
            // match-terminated streams too.
            break;
        }
    }
    if out.len() != expected_len {
        return Err(err(format!(
            "lz4 output length {} != expected {expected_len}",
            out.len()
        )));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-built block: token 0x54 = 5 literals + (4+4)-byte match.
    #[test]
    fn literals_then_match_decodes() {
        // literals "abcde", match offset 5 len 8 -> repeats "abcde" cycling.
        let mut src = vec![0x54u8];
        src.extend_from_slice(b"abcde");
        src.extend_from_slice(&5u16.to_le_bytes());
        // trailing literal-only sequence: token 0x10, literal "Z".
        src.push(0x10);
        src.push(b'Z');
        let out = lz4_decompress_block(&src, 14).unwrap();
        assert_eq!(&out, b"abcdeabcdeabcZ");
    }

    #[test]
    fn run_length_extension_decodes() {
        // 20 literals: token 0xF0 + extension byte 5.
        let mut src = vec![0xF0u8, 5];
        src.extend_from_slice(&[7u8; 20]);
        let out = lz4_decompress_block(&src, 20).unwrap();
        assert_eq!(out, vec![7u8; 20]);
    }

    #[test]
    fn overlap_copy_repeats() {
        // 1 literal 'x', then match offset 1 len 9 -> "x" * 10.
        let mut src = vec![0x15u8, b'x'];
        src.extend_from_slice(&1u16.to_le_bytes());
        let out = lz4_decompress_block(&src, 10).unwrap();
        assert_eq!(out, vec![b'x'; 10]);
    }

    #[test]
    fn corrupt_blocks_error() {
        assert!(lz4_decompress_block(&[], 1).is_err());
        // Bad offset (0).
        let mut src = vec![0x14u8, b'x'];
        src.extend_from_slice(&0u16.to_le_bytes());
        assert!(lz4_decompress_block(&src, 6).is_err());
        // Wrong expected length.
        assert!(lz4_decompress_block(&[0x10, b'a'], 5).is_err());
    }
}
