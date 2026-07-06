//! From-scratch MessagePack codec — the JAX/Flax checkpoint substrate.
//!
//! Implements the msgpack wire format (nil, bool, all int widths, f32/f64,
//! str, bin, array, map, ext) as needed by `flax.serialization.to_bytes` /
//! `from_bytes`. No msgpack library is linked.

use mmn_core::MmnError;

fn err(message: impl Into<String>) -> MmnError {
    MmnError::Other {
        message: message.into(),
    }
}

/// One decoded MessagePack value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Nil,
    Bool(bool),
    Int(i64),
    UInt(u64),
    F32(f32),
    F64(f64),
    Str(String),
    Bin(Vec<u8>),
    Array(Vec<Value>),
    /// Key/value pairs in wire order (Flax maps use string keys).
    Map(Vec<(Value, Value)>),
    Ext(i8, Vec<u8>),
}

pub struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    pub fn done(&self) -> bool {
        self.pos == self.bytes.len()
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], MmnError> {
        let slice = self
            .bytes
            .get(self.pos..self.pos + n)
            .ok_or_else(|| err("msgpack truncated"))?;
        self.pos += n;
        Ok(slice)
    }

    fn byte(&mut self) -> Result<u8, MmnError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, MmnError> {
        let b = self.take(2)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32, MmnError> {
        let b = self.take(4)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64, MmnError> {
        let b = self.take(8)?;
        Ok(u64::from_be_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn str_body(&mut self, len: usize) -> Result<Value, MmnError> {
        let raw = self.take(len)?;
        Ok(Value::Str(
            std::str::from_utf8(raw)
                .map_err(|e| err(format!("msgpack string not UTF-8: {e}")))?
                .to_string(),
        ))
    }

    fn array_body(&mut self, len: usize) -> Result<Value, MmnError> {
        let mut out = Vec::with_capacity(len.min(1 << 16));
        for _ in 0..len {
            out.push(self.value()?);
        }
        Ok(Value::Array(out))
    }

    fn map_body(&mut self, len: usize) -> Result<Value, MmnError> {
        let mut out = Vec::with_capacity(len.min(1 << 16));
        for _ in 0..len {
            let key = self.value()?;
            let value = self.value()?;
            out.push((key, value));
        }
        Ok(Value::Map(out))
    }

    fn ext_body(&mut self, len: usize) -> Result<Value, MmnError> {
        let ext_type = self.byte()? as i8;
        Ok(Value::Ext(ext_type, self.take(len)?.to_vec()))
    }

    /// Decode the next value.
    pub fn value(&mut self) -> Result<Value, MmnError> {
        let tag = self.byte()?;
        match tag {
            0x00..=0x7F => Ok(Value::UInt(tag as u64)),
            0xE0..=0xFF => Ok(Value::Int(tag as i8 as i64)),
            0x80..=0x8F => self.map_body((tag & 0x0F) as usize),
            0x90..=0x9F => self.array_body((tag & 0x0F) as usize),
            0xA0..=0xBF => self.str_body((tag & 0x1F) as usize),
            0xC0 => Ok(Value::Nil),
            0xC2 => Ok(Value::Bool(false)),
            0xC3 => Ok(Value::Bool(true)),
            0xC4 => {
                let len = self.byte()? as usize;
                Ok(Value::Bin(self.take(len)?.to_vec()))
            }
            0xC5 => {
                let len = self.u16()? as usize;
                Ok(Value::Bin(self.take(len)?.to_vec()))
            }
            0xC6 => {
                let len = self.u32()? as usize;
                Ok(Value::Bin(self.take(len)?.to_vec()))
            }
            0xC7 => {
                let len = self.byte()? as usize;
                self.ext_body(len)
            }
            0xC8 => {
                let len = self.u16()? as usize;
                self.ext_body(len)
            }
            0xC9 => {
                let len = self.u32()? as usize;
                self.ext_body(len)
            }
            0xCA => {
                let b = self.take(4)?;
                Ok(Value::F32(f32::from_be_bytes([b[0], b[1], b[2], b[3]])))
            }
            0xCB => {
                let b = self.take(8)?;
                Ok(Value::F64(f64::from_be_bytes([
                    b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                ])))
            }
            0xCC => Ok(Value::UInt(self.byte()? as u64)),
            0xCD => Ok(Value::UInt(self.u16()? as u64)),
            0xCE => Ok(Value::UInt(self.u32()? as u64)),
            0xCF => Ok(Value::UInt(self.u64()?)),
            0xD0 => Ok(Value::Int(self.byte()? as i8 as i64)),
            0xD1 => Ok(Value::Int(self.u16()? as i16 as i64)),
            0xD2 => Ok(Value::Int(self.u32()? as i32 as i64)),
            0xD3 => Ok(Value::Int(self.u64()? as i64)),
            0xD4 => self.ext_body(1),
            0xD5 => self.ext_body(2),
            0xD6 => self.ext_body(4),
            0xD7 => self.ext_body(8),
            0xD8 => self.ext_body(16),
            0xD9 => {
                let len = self.byte()? as usize;
                self.str_body(len)
            }
            0xDA => {
                let len = self.u16()? as usize;
                self.str_body(len)
            }
            0xDB => {
                let len = self.u32()? as usize;
                self.str_body(len)
            }
            0xDC => {
                let len = self.u16()? as usize;
                self.array_body(len)
            }
            0xDD => {
                let len = self.u32()? as usize;
                self.array_body(len)
            }
            0xDE => {
                let len = self.u16()? as usize;
                self.map_body(len)
            }
            0xDF => {
                let len = self.u32()? as usize;
                self.map_body(len)
            }
            0xC1 => Err(err("msgpack 0xC1 is reserved")),
        }
    }
}

/// Encode one value (canonical shortest-form encodings, `use_bin_type` style:
/// strings as str, bytes as bin — matching msgpack-python defaults).
pub fn write_value(value: &Value, out: &mut Vec<u8>) {
    match value {
        Value::Nil => out.push(0xC0),
        Value::Bool(false) => out.push(0xC2),
        Value::Bool(true) => out.push(0xC3),
        Value::UInt(n) => write_uint(*n, out),
        Value::Int(n) => {
            if *n >= 0 {
                write_uint(*n as u64, out);
            } else if *n >= -32 {
                out.push(*n as i8 as u8);
            } else if *n >= i8::MIN as i64 {
                out.push(0xD0);
                out.push(*n as i8 as u8);
            } else if *n >= i16::MIN as i64 {
                out.push(0xD1);
                out.extend_from_slice(&(*n as i16).to_be_bytes());
            } else if *n >= i32::MIN as i64 {
                out.push(0xD2);
                out.extend_from_slice(&(*n as i32).to_be_bytes());
            } else {
                out.push(0xD3);
                out.extend_from_slice(&n.to_be_bytes());
            }
        }
        Value::F32(v) => {
            out.push(0xCA);
            out.extend_from_slice(&v.to_be_bytes());
        }
        Value::F64(v) => {
            out.push(0xCB);
            out.extend_from_slice(&v.to_be_bytes());
        }
        Value::Str(s) => {
            let bytes = s.as_bytes();
            match bytes.len() {
                0..=31 => out.push(0xA0 | bytes.len() as u8),
                32..=255 => {
                    out.push(0xD9);
                    out.push(bytes.len() as u8);
                }
                256..=65535 => {
                    out.push(0xDA);
                    out.extend_from_slice(&(bytes.len() as u16).to_be_bytes());
                }
                _ => {
                    out.push(0xDB);
                    out.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
                }
            }
            out.extend_from_slice(bytes);
        }
        Value::Bin(bytes) => {
            match bytes.len() {
                0..=255 => {
                    out.push(0xC4);
                    out.push(bytes.len() as u8);
                }
                256..=65535 => {
                    out.push(0xC5);
                    out.extend_from_slice(&(bytes.len() as u16).to_be_bytes());
                }
                _ => {
                    out.push(0xC6);
                    out.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
                }
            }
            out.extend_from_slice(bytes);
        }
        Value::Array(items) => {
            match items.len() {
                0..=15 => out.push(0x90 | items.len() as u8),
                16..=65535 => {
                    out.push(0xDC);
                    out.extend_from_slice(&(items.len() as u16).to_be_bytes());
                }
                _ => {
                    out.push(0xDD);
                    out.extend_from_slice(&(items.len() as u32).to_be_bytes());
                }
            }
            for item in items {
                write_value(item, out);
            }
        }
        Value::Map(pairs) => {
            match pairs.len() {
                0..=15 => out.push(0x80 | pairs.len() as u8),
                16..=65535 => {
                    out.push(0xDE);
                    out.extend_from_slice(&(pairs.len() as u16).to_be_bytes());
                }
                _ => {
                    out.push(0xDF);
                    out.extend_from_slice(&(pairs.len() as u32).to_be_bytes());
                }
            }
            for (key, value) in pairs {
                write_value(key, out);
                write_value(value, out);
            }
        }
        Value::Ext(ext_type, data) => {
            match data.len() {
                1 => out.push(0xD4),
                2 => out.push(0xD5),
                4 => out.push(0xD6),
                8 => out.push(0xD7),
                16 => out.push(0xD8),
                0..=255 => {
                    out.push(0xC7);
                    out.push(data.len() as u8);
                }
                256..=65535 => {
                    out.push(0xC8);
                    out.extend_from_slice(&(data.len() as u16).to_be_bytes());
                }
                _ => {
                    out.push(0xC9);
                    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
                }
            }
            out.push(*ext_type as u8);
            out.extend_from_slice(data);
        }
    }
}

fn write_uint(n: u64, out: &mut Vec<u8>) {
    if n <= 0x7F {
        out.push(n as u8);
    } else if n <= u8::MAX as u64 {
        out.push(0xCC);
        out.push(n as u8);
    } else if n <= u16::MAX as u64 {
        out.push(0xCD);
        out.extend_from_slice(&(n as u16).to_be_bytes());
    } else if n <= u32::MAX as u64 {
        out.push(0xCE);
        out.extend_from_slice(&(n as u32).to_be_bytes());
    } else {
        out.push(0xCF);
        out.extend_from_slice(&n.to_be_bytes());
    }
}

/// Decode one complete msgpack document (trailing bytes rejected).
pub fn decode(bytes: &[u8]) -> Result<Value, MmnError> {
    let mut reader = Reader::new(bytes);
    let value = reader.value()?;
    if !reader.done() {
        return Err(err("msgpack document has trailing bytes"));
    }
    Ok(value)
}

/// Encode one value into a fresh buffer.
pub fn encode(value: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    write_value(value, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(v: Value) {
        let bytes = encode(&v);
        assert_eq!(decode(&bytes).unwrap(), v, "roundtrip for {v:?}");
    }

    #[test]
    fn scalars_roundtrip() {
        roundtrip(Value::Nil);
        roundtrip(Value::Bool(true));
        roundtrip(Value::Bool(false));
        roundtrip(Value::UInt(0));
        roundtrip(Value::UInt(127));
        roundtrip(Value::UInt(200));
        roundtrip(Value::UInt(70000));
        roundtrip(Value::UInt(u64::MAX));
        roundtrip(Value::Int(-1));
        roundtrip(Value::Int(-32));
        roundtrip(Value::Int(-100));
        roundtrip(Value::Int(-40000));
        roundtrip(Value::Int(i64::MIN));
        roundtrip(Value::F32(1.5));
        roundtrip(Value::F64(-3.25));
    }

    #[test]
    fn containers_roundtrip() {
        roundtrip(Value::Str("hello".into()));
        roundtrip(Value::Str("x".repeat(300)));
        roundtrip(Value::Bin(vec![1, 2, 3]));
        roundtrip(Value::Bin(vec![7; 70000]));
        roundtrip(Value::Array(vec![
            Value::UInt(1),
            Value::Str("two".into()),
            Value::Array(vec![Value::Nil]),
        ]));
        roundtrip(Value::Map(vec![
            (Value::Str("a".into()), Value::UInt(1)),
            (
                Value::Str("b".into()),
                Value::Map(vec![(Value::Str("c".into()), Value::Bool(true))]),
            ),
        ]));
        roundtrip(Value::Array((0..20).map(Value::UInt).collect()));
    }

    #[test]
    fn ext_types_roundtrip() {
        roundtrip(Value::Ext(1, vec![9])); // fixext1
        roundtrip(Value::Ext(1, vec![1, 2, 3, 4])); // fixext4
        roundtrip(Value::Ext(1, vec![0; 34])); // ext8
        roundtrip(Value::Ext(1, vec![0; 300])); // ext16
        roundtrip(Value::Ext(-5, vec![1, 2, 3])); // negative type, ext8
    }

    /// Byte-exact vectors from the msgpack spec / msgpack-python output.
    #[test]
    fn matches_reference_encodings() {
        assert_eq!(encode(&Value::UInt(1)), vec![0x01]);
        assert_eq!(encode(&Value::Int(-1)), vec![0xFF]);
        assert_eq!(encode(&Value::Str("abc".into())), vec![0xA3, b'a', b'b', b'c']);
        assert_eq!(
            encode(&Value::Map(vec![(
                Value::Str("a".into()),
                Value::UInt(7)
            )])),
            vec![0x81, 0xA1, b'a', 0x07]
        );
        // msgpack.packb((2, 3)) == b'\x92\x02\x03'
        assert_eq!(
            encode(&Value::Array(vec![Value::UInt(2), Value::UInt(3)])),
            vec![0x92, 0x02, 0x03]
        );
    }

    #[test]
    fn truncated_and_reserved_error() {
        assert!(decode(&[0xA3, b'a']).is_err());
        assert!(decode(&[0xC1]).is_err());
        assert!(decode(&[0x01, 0x02]).is_err()); // trailing bytes
    }
}
