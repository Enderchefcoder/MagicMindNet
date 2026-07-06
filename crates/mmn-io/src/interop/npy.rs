//! From-scratch NumPy `.npy` (format 1.0/2.0) encoder/decoder.
//!
//! Reads every common numeric dtype (little- and big-endian) into `f32`,
//! including Fortran-ordered arrays, and writes C-ordered `<f4` files that
//! `numpy.load` accepts.

use half::f16;
use mmn_core::MmnError;
use ndarray::{ArrayD, IxDyn};

const MAGIC: &[u8; 6] = b"\x93NUMPY";

fn err(message: impl Into<String>) -> MmnError {
    MmnError::Other {
        message: message.into(),
    }
}

/// A decoded array: C-ordered `f32` data plus its shape.
pub struct NpyArray {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Encode an `f32` array as NumPy format 1.0 (`<f4`, C order).
pub fn encode_npy_f32(shape: &[usize], data: &[f32]) -> Result<Vec<u8>, MmnError> {
    let numel: usize = shape.iter().product();
    if numel != data.len() {
        return Err(err(format!(
            "npy encode shape {:?} needs {} elements, got {}",
            shape,
            numel,
            data.len()
        )));
    }
    let shape_repr = match shape.len() {
        0 => "()".to_string(),
        1 => format!("({},)", shape[0]),
        _ => format!(
            "({})",
            shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_repr}, }}"
    );
    // Total header (magic + version + len field + dict + padding) must be a
    // multiple of 64, terminated by '\n'.
    let prefix_len = MAGIC.len() + 2 + 2;
    let unpadded = prefix_len + header.len() + 1;
    let padding = (64 - unpadded % 64) % 64;
    let header_len = header.len() + padding + 1;
    if header_len > u16::MAX as usize {
        return Err(err("npy header too large for format 1.0"));
    }
    let mut out = Vec::with_capacity(prefix_len + header_len + data.len() * 4);
    out.extend_from_slice(MAGIC);
    out.push(1); // major version
    out.push(0); // minor version
    out.extend_from_slice(&(header_len as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend(std::iter::repeat_n(b' ', padding));
    out.push(b'\n');
    for value in data {
        out.extend_from_slice(&value.to_le_bytes());
    }
    Ok(out)
}

struct NpyHeader {
    descr: String,
    fortran_order: bool,
    shape: Vec<usize>,
    data_start: usize,
}

fn parse_header(bytes: &[u8]) -> Result<NpyHeader, MmnError> {
    if bytes.len() < 10 || &bytes[..6] != MAGIC {
        return Err(err("not an npy file (missing \\x93NUMPY magic)"));
    }
    let major = bytes[6];
    let (header_len, header_start) = match major {
        1 => (
            u16::from_le_bytes([bytes[8], bytes[9]]) as usize,
            10usize,
        ),
        2 | 3 => {
            if bytes.len() < 12 {
                return Err(err("npy v2 header truncated"));
            }
            (
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                12usize,
            )
        }
        other => return Err(err(format!("unsupported npy major version {other}"))),
    };
    let header_bytes = bytes
        .get(header_start..header_start + header_len)
        .ok_or_else(|| err("npy header truncated"))?;
    let header = std::str::from_utf8(header_bytes).map_err(|e| err(format!("npy header not UTF-8: {e}")))?;
    let descr = extract_quoted(header, "descr")?;
    let fortran_order = header
        .split("'fortran_order':")
        .nth(1)
        .map(|rest| rest.trim_start().starts_with("True"))
        .ok_or_else(|| err("npy header missing fortran_order"))?;
    let shape = parse_shape(header)?;
    Ok(NpyHeader {
        descr,
        fortran_order,
        shape,
        data_start: header_start + header_len,
    })
}

fn extract_quoted(header: &str, key: &str) -> Result<String, MmnError> {
    let after = header
        .split(&format!("'{key}':"))
        .nth(1)
        .ok_or_else(|| err(format!("npy header missing {key}")))?;
    let start = after
        .find('\'')
        .ok_or_else(|| err(format!("npy header {key} not quoted")))?;
    let rest = &after[start + 1..];
    let end = rest
        .find('\'')
        .ok_or_else(|| err(format!("npy header {key} unterminated")))?;
    Ok(rest[..end].to_string())
}

fn parse_shape(header: &str) -> Result<Vec<usize>, MmnError> {
    let after = header
        .split("'shape':")
        .nth(1)
        .ok_or_else(|| err("npy header missing shape"))?;
    let open = after.find('(').ok_or_else(|| err("npy shape missing ("))?;
    let close = after[open..]
        .find(')')
        .ok_or_else(|| err("npy shape missing )"))?
        + open;
    after[open + 1..close]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .map_err(|e| err(format!("npy shape dim {s:?}: {e}")))
        })
        .collect()
}

fn decode_element(descr: &str, chunk: &[u8]) -> Result<f32, MmnError> {
    let big_endian = descr.starts_with('>');
    let code = &descr[1..];
    let val = |le: &[u8]| -> Vec<u8> {
        if big_endian {
            le.iter().rev().copied().collect()
        } else {
            le.to_vec()
        }
    };
    let b = val(chunk);
    Ok(match code {
        "f4" => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        "f8" => f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32,
        "f2" => f16::from_le_bytes([b[0], b[1]]).to_f32(),
        "i1" => b[0] as i8 as f32,
        "i2" => i16::from_le_bytes([b[0], b[1]]) as f32,
        "i4" => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32,
        "i8" => i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32,
        "u1" => b[0] as f32,
        "u2" => u16::from_le_bytes([b[0], b[1]]) as f32,
        "u4" => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32,
        "u8" => u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32,
        "b1" => {
            if b[0] != 0 {
                1.0
            } else {
                0.0
            }
        }
        other => return Err(err(format!("npy dtype {other:?} not supported"))),
    })
}

fn descr_item_size(descr: &str) -> Result<usize, MmnError> {
    let code = descr.get(1..).unwrap_or("");
    code.get(1..)
        .and_then(|n| n.parse::<usize>().ok())
        .filter(|&n| matches!(n, 1 | 2 | 4 | 8))
        .ok_or_else(|| err(format!("npy dtype {descr:?} not supported")))
}

/// Decode an `.npy` byte buffer into a C-ordered `f32` array.
pub fn decode_npy(bytes: &[u8]) -> Result<NpyArray, MmnError> {
    let header = parse_header(bytes)?;
    if !header.descr.starts_with('<') && !header.descr.starts_with('>') && !header.descr.starts_with('|') {
        return Err(err(format!("npy dtype {:?} not supported", header.descr)));
    }
    let item = descr_item_size(&header.descr)?;
    let numel: usize = header.shape.iter().product();
    let payload = bytes
        .get(header.data_start..header.data_start + numel * item)
        .ok_or_else(|| err("npy data truncated"))?;
    let mut data = Vec::with_capacity(numel);
    for chunk in payload.chunks_exact(item) {
        data.push(decode_element(&header.descr, chunk)?);
    }
    if header.fortran_order && header.shape.len() > 1 {
        // Fortran layout: read with reversed dims, then permute back to C order.
        let reversed: Vec<usize> = header.shape.iter().rev().copied().collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&reversed), data)
            .map_err(|e| err(format!("npy fortran reshape: {e}")))?;
        let axes: Vec<usize> = (0..header.shape.len()).rev().collect();
        let transposed = arr.permuted_axes(IxDyn(&axes));
        let standard = transposed.as_standard_layout().into_owned();
        data = standard.iter().copied().collect();
    }
    Ok(NpyArray {
        shape: header.shape,
        data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_roundtrip() {
        let shape = vec![2, 3];
        let data = vec![1.0f32, -2.5, 3.25, 0.0, 9.5, -7.0];
        let bytes = encode_npy_f32(&shape, &data).unwrap();
        assert_eq!(&bytes[..6], MAGIC);
        // Total header must be 64-byte aligned.
        let arr = decode_npy(&bytes).unwrap();
        assert_eq!(arr.shape, shape);
        assert_eq!(arr.data, data);
    }

    #[test]
    fn scalar_and_vector_shapes() {
        let scalar = encode_npy_f32(&[], &[42.0]).unwrap();
        let arr = decode_npy(&scalar).unwrap();
        assert!(arr.shape.is_empty());
        assert_eq!(arr.data, vec![42.0]);

        let vector = encode_npy_f32(&[3], &[1.0, 2.0, 3.0]).unwrap();
        let arr = decode_npy(&vector).unwrap();
        assert_eq!(arr.shape, vec![3]);
    }

    #[test]
    fn header_is_64_byte_aligned() {
        let bytes = encode_npy_f32(&[4], &[0.0; 4]).unwrap();
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        assert_eq!((10 + header_len) % 64, 0);
        assert_eq!(bytes[10 + header_len - 1], b'\n');
    }

    #[test]
    fn decodes_f64_and_int_dtypes() {
        // Handcrafted v1 header for <f8 shape (2,).
        let build = |descr: &str, payload: &[u8], shape: &str| -> Vec<u8> {
            let dict = format!(
                "{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}\n"
            );
            let mut v = MAGIC.to_vec();
            v.push(1);
            v.push(0);
            v.extend_from_slice(&(dict.len() as u16).to_le_bytes());
            v.extend_from_slice(dict.as_bytes());
            v.extend_from_slice(payload);
            v
        };
        let mut payload = Vec::new();
        payload.extend_from_slice(&1.5f64.to_le_bytes());
        payload.extend_from_slice(&(-2.0f64).to_le_bytes());
        let arr = decode_npy(&build("<f8", &payload, "(2,)")).unwrap();
        assert_eq!(arr.data, vec![1.5, -2.0]);

        let mut ints = Vec::new();
        ints.extend_from_slice(&7i64.to_le_bytes());
        ints.extend_from_slice(&(-3i64).to_le_bytes());
        let arr = decode_npy(&build("<i8", &ints, "(2,)")).unwrap();
        assert_eq!(arr.data, vec![7.0, -3.0]);

        let arr = decode_npy(&build("|u1", &[0, 128, 255], "(3,)")).unwrap();
        assert_eq!(arr.data, vec![0.0, 128.0, 255.0]);

        // Big-endian f4.
        let mut be = Vec::new();
        be.extend_from_slice(&2.5f32.to_be_bytes());
        let arr = decode_npy(&build(">f4", &be, "(1,)")).unwrap();
        assert_eq!(arr.data, vec![2.5]);
    }

    #[test]
    fn fortran_order_transposed_back() {
        // 2x3 array [[1,2,3],[4,5,6]] in Fortran order is stored 1,4,2,5,3,6.
        let dict = "{'descr': '<f4', 'fortran_order': True, 'shape': (2, 3), }\n";
        let mut v = MAGIC.to_vec();
        v.push(1);
        v.push(0);
        v.extend_from_slice(&(dict.len() as u16).to_le_bytes());
        v.extend_from_slice(dict.as_bytes());
        for x in [1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0] {
            v.extend_from_slice(&x.to_le_bytes());
        }
        let arr = decode_npy(&v).unwrap();
        assert_eq!(arr.shape, vec![2, 3]);
        assert_eq!(arr.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn bad_magic_and_truncation_error() {
        assert!(decode_npy(b"NOTNPY").is_err());
        let good = encode_npy_f32(&[4], &[1.0; 4]).unwrap();
        assert!(decode_npy(&good[..good.len() - 3]).is_err());
    }

    #[test]
    fn shape_element_count_mismatch_errors() {
        assert!(encode_npy_f32(&[2, 2], &[1.0]).is_err());
    }
}
