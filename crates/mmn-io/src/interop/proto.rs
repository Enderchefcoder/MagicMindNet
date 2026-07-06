//! Minimal protocol-buffers wire-format reader (varint, fixed, length-
//! delimited) shared by the ONNX and TensorFlow checkpoint parsers.

use mmn_core::MmnError;

fn err(message: impl Into<String>) -> MmnError {
    MmnError::Other {
        message: message.into(),
    }
}

/// One decoded field value.
pub enum ProtoValue<'a> {
    Varint(u64),
    Fixed64(u64),
    Fixed32(u32),
    Bytes(&'a [u8]),
}

impl<'a> ProtoValue<'a> {
    pub fn as_bytes(&self) -> Option<&'a [u8]> {
        match self {
            ProtoValue::Bytes(b) => Some(b),
            _ => None,
        }
    }
}

pub fn read_varint(buf: &[u8], pos: &mut usize) -> Result<u64, MmnError> {
    let mut out = 0u64;
    let mut shift = 0u32;
    loop {
        let byte = *buf.get(*pos).ok_or_else(|| err("protobuf varint truncated"))?;
        *pos += 1;
        out |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok(out);
        }
        shift += 7;
        if shift > 63 {
            return Err(err("protobuf varint too long"));
        }
    }
}

/// Walk every field of a message, calling `visit(field_number, value)`.
pub fn walk_message<'a>(
    buf: &'a [u8],
    mut visit: impl FnMut(u64, ProtoValue<'a>) -> Result<(), MmnError>,
) -> Result<(), MmnError> {
    let mut pos = 0usize;
    while pos < buf.len() {
        let tag = read_varint(buf, &mut pos)?;
        let field = tag >> 3;
        match tag & 7 {
            0 => {
                let v = read_varint(buf, &mut pos)?;
                visit(field, ProtoValue::Varint(v))?;
            }
            1 => {
                let raw = buf
                    .get(pos..pos + 8)
                    .ok_or_else(|| err("protobuf fixed64 truncated"))?;
                pos += 8;
                visit(
                    field,
                    ProtoValue::Fixed64(u64::from_le_bytes(raw.try_into().expect("8 bytes"))),
                )?;
            }
            2 => {
                let len = read_varint(buf, &mut pos)? as usize;
                let body = buf
                    .get(pos..pos + len)
                    .ok_or_else(|| err("protobuf length-delimited field truncated"))?;
                pos += len;
                visit(field, ProtoValue::Bytes(body))?;
            }
            5 => {
                let raw = buf
                    .get(pos..pos + 4)
                    .ok_or_else(|| err("protobuf fixed32 truncated"))?;
                pos += 4;
                visit(
                    field,
                    ProtoValue::Fixed32(u32::from_le_bytes(raw.try_into().expect("4 bytes"))),
                )?;
            }
            other => return Err(err(format!("protobuf wire type {other} unsupported"))),
        }
    }
    Ok(())
}

/// Decode a packed repeated varint field (e.g. `repeated int64` dims).
pub fn packed_varints(buf: &[u8]) -> Result<Vec<u64>, MmnError> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < buf.len() {
        out.push(read_varint(buf, &mut pos)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn walks_mixed_fields() {
        // field1 varint 150, field2 bytes "hi", field3 fixed32 1.0f
        let mut buf = vec![0x08, 0x96, 0x01, 0x12, 0x02, b'h', b'i', 0x1D];
        buf.extend_from_slice(&1.0f32.to_le_bytes());
        let mut seen = Vec::new();
        walk_message(&buf, |field, value| {
            match value {
                ProtoValue::Varint(v) => seen.push(format!("{field}:v{v}")),
                ProtoValue::Bytes(b) => seen.push(format!("{field}:b{}", b.len())),
                ProtoValue::Fixed32(v) => seen.push(format!("{field}:f{}", f32::from_bits(v))),
                ProtoValue::Fixed64(_) => seen.push(format!("{field}:x")),
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(seen, vec!["1:v150", "2:b2", "3:f1"]);
    }

    #[test]
    fn packed_varints_decode() {
        // [3, 270]
        let buf = vec![0x03, 0x8E, 0x02];
        assert_eq!(packed_varints(&buf).unwrap(), vec![3, 270]);
    }

    #[test]
    fn truncated_fields_error() {
        assert!(walk_message(&[0x12, 0x05, 0x01], |_, _| Ok(())).is_err());
        assert!(walk_message(&[0x08], |_, _| Ok(())).is_err());
    }
}
