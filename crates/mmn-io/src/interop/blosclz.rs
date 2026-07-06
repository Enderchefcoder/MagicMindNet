//! From-scratch BloscLZ decoder — numcodecs' default Blosc inner codec.
//!
//! FastLZ-family token stream: the first byte's low 5 bits open a literal
//! run (high bits carry the format level); control bytes >= 32 encode
//! matches with 3-bit lengths (7 escapes to 255-run extension bytes),
//! 13-bit distances (`(ctrl & 31) << 8 | code + 1`), and a far-distance
//! escape (`code == 255` at max offset adds a 16-bit distance beyond the
//! 8191-byte window). Verified byte-for-byte against c-blosc output.

use mmn_core::MmnError;

fn err(message: impl Into<String>) -> MmnError {
    MmnError::Other {
        message: message.into(),
    }
}

const MAX_DISTANCE: usize = 8191;

/// Decompress one BloscLZ stream into exactly `expected_len` bytes.
pub fn blosclz_decompress(src: &[u8], expected_len: usize) -> Result<Vec<u8>, MmnError> {
    let mut out: Vec<u8> = Vec::with_capacity(expected_len);
    let mut ip = 0usize;
    let take = |ip: &mut usize| -> Result<u8, MmnError> {
        let byte = *src.get(*ip).ok_or_else(|| err("blosclz stream truncated"))?;
        *ip += 1;
        Ok(byte)
    };
    if src.is_empty() {
        if expected_len == 0 {
            return Ok(out);
        }
        return Err(err("blosclz stream empty"));
    }
    // First control: low 5 bits are a literal run, high bits the level tag.
    let mut ctrl = (take(&mut ip)? & 31) as usize;
    loop {
        if ctrl >= 32 {
            // Match.
            let mut len = (ctrl >> 5) - 1;
            let ofs = (ctrl & 31) << 8;
            if (ctrl >> 5) == 7 {
                loop {
                    let code = take(&mut ip)?;
                    len += code as usize;
                    if code != 255 {
                        break;
                    }
                }
            }
            let code = take(&mut ip)?;
            len += 3;
            let mut distance = ofs + code as usize + 1;
            if code == 255 && ofs == (31 << 8) {
                // Far match: explicit 16-bit distance beyond the window.
                let high = take(&mut ip)? as usize;
                let low = take(&mut ip)? as usize;
                distance = (high << 8) + low + MAX_DISTANCE + 1;
            }
            if distance == 0 || distance > out.len() {
                return Err(err(format!(
                    "blosclz match distance {distance} invalid at output position {}",
                    out.len()
                )));
            }
            let start = out.len() - distance;
            for k in 0..len {
                let byte = out[start + k];
                out.push(byte);
            }
        } else {
            // Literal run of ctrl + 1 bytes.
            let literals = src
                .get(ip..ip + ctrl + 1)
                .ok_or_else(|| err("blosclz literals truncated"))?;
            out.extend_from_slice(literals);
            ip += ctrl + 1;
        }
        if out.len() > expected_len {
            return Err(err("blosclz output exceeds expected length"));
        }
        if ip == src.len() {
            break;
        }
        ctrl = take(&mut ip)? as usize;
    }
    if out.len() != expected_len {
        return Err(err(format!(
            "blosclz output length {} != expected {expected_len}",
            out.len()
        )));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stream captured from c-blosc (numcodecs) for
    /// `("abcabcabcabc" + "D"*50 + "abcabc") * 40`.
    #[test]
    fn reference_stream_decodes() {
        let unit: Vec<u8> = [b"abcabcabcabc".as_slice(), &[b'D'; 50], b"abcabc"].concat();
        let expected: Vec<u8> = unit.repeat(40);
        // First tokens hand-verified: 13 literals, distance-1 match of 48,
        // 4 literals, distance-3 match, then long far-period matches.
        let stream: Vec<u8> = vec![
            44, 97, 98, 99, 97, 98, 99, 97, 98, 99, 97, 98, 99, 68, 224, 39, 0, 3, 68, 97,
            98, 99, 224, 5, 2, 224, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 79,
            67, 2, 97, 98, 99,
        ];
        let out = blosclz_decompress(&stream, expected.len()).unwrap();
        assert_eq!(out, expected);
    }

    /// Long single-byte run exercising the 255-extension loop.
    #[test]
    fn long_run_decodes() {
        // Captured head for b"Q" * 5000: 4 literals then one huge match.
        let mut stream = vec![35u8, 81, 81, 81, 81, 224];
        // len = 6 + 255*k + rem + 3 = 4996  ->  255*19 = 4845, rem 142.
        stream.extend(std::iter::repeat_n(255u8, 19));
        stream.push(142);
        stream.push(0); // distance code -> distance 1
        let out = blosclz_decompress(&stream, 5000).unwrap();
        assert_eq!(out, vec![b'Q'; 5000]);
    }

    #[test]
    fn corrupt_streams_error() {
        assert!(blosclz_decompress(&[], 4).is_err());
        assert!(blosclz_decompress(&[3, 1, 2], 4).is_err()); // literals truncated
        // Match before any output.
        assert!(blosclz_decompress(&[0, 7, 224, 0, 0], 60).is_err());
    }
}
