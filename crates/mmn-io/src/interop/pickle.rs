//! Minimal from-scratch Python pickle reader/writer.
//!
//! Implements exactly the opcode subset `torch.save` emits for state dicts
//! (protocols 2–4): containers, memoization, `GLOBAL`/`STACK_GLOBAL`,
//! `REDUCE`, and persistent IDs. No Python or pickle library is involved.

use mmn_core::MmnError;

fn err(message: impl Into<String>) -> MmnError {
    MmnError::Other {
        message: message.into(),
    }
}

/// A materialized pickle value (subset relevant to tensor archives).
#[derive(Clone, Debug, PartialEq)]
pub enum PickleValue {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
    Tuple(Vec<PickleValue>),
    List(Vec<PickleValue>),
    Dict(Vec<(PickleValue, PickleValue)>),
    /// `module.name` reference pushed by GLOBAL / STACK_GLOBAL.
    Global(String, String),
    /// Result of REDUCE: `callable(*args)` left symbolic.
    Reduce(Box<PickleValue>, Box<PickleValue>),
    /// Persistent-ID reference (torch storage descriptors).
    PersId(Box<PickleValue>),
    /// Internal stack marker; never appears in results.
    Mark,
}

impl PickleValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            PickleValue::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            PickleValue::Int(v) => Some(*v),
            PickleValue::Bool(b) => Some(*b as i64),
            _ => None,
        }
    }

    pub fn tuple_items(&self) -> Option<&[PickleValue]> {
        match self {
            PickleValue::Tuple(items) => Some(items),
            _ => None,
        }
    }
}

struct Vm<'a> {
    bytes: &'a [u8],
    pos: usize,
    stack: Vec<PickleValue>,
    memo: Vec<Option<PickleValue>>,
}

impl<'a> Vm<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], MmnError> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.bytes.len())
            .ok_or_else(|| err("pickle stream truncated"))?;
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8, MmnError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, MmnError> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32, MmnError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn line(&mut self) -> Result<String, MmnError> {
        let start = self.pos;
        while self.pos < self.bytes.len() && self.bytes[self.pos] != b'\n' {
            self.pos += 1;
        }
        if self.pos >= self.bytes.len() {
            return Err(err("pickle text line unterminated"));
        }
        let s = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|e| err(format!("pickle line not UTF-8: {e}")))?
            .to_string();
        self.pos += 1;
        Ok(s)
    }

    fn pop(&mut self) -> Result<PickleValue, MmnError> {
        self.stack.pop().ok_or_else(|| err("pickle stack underflow"))
    }

    fn pop_to_mark(&mut self) -> Result<Vec<PickleValue>, MmnError> {
        let mut items = Vec::new();
        loop {
            match self.pop()? {
                PickleValue::Mark => break,
                v => items.push(v),
            }
        }
        items.reverse();
        Ok(items)
    }

    fn memo_put(&mut self, idx: usize) -> Result<(), MmnError> {
        let value = self
            .stack
            .last()
            .cloned()
            .ok_or_else(|| err("pickle BINPUT on empty stack"))?;
        if idx >= self.memo.len() {
            if idx > 1 << 24 {
                return Err(err("pickle memo index unreasonably large"));
            }
            self.memo.resize(idx + 1, None);
        }
        self.memo[idx] = Some(value);
        Ok(())
    }

    fn memo_get(&mut self, idx: usize) -> Result<(), MmnError> {
        let value = self
            .memo
            .get(idx)
            .and_then(|v| v.clone())
            .ok_or_else(|| err(format!("pickle memo slot {idx} empty")))?;
        self.stack.push(value);
        Ok(())
    }

    fn set_items(&mut self, pairs: Vec<(PickleValue, PickleValue)>) -> Result<(), MmnError> {
        let target = self.pop()?;
        let updated = match target {
            PickleValue::Dict(mut existing) => {
                existing.extend(pairs);
                PickleValue::Dict(existing)
            }
            // OrderedDict comes through as Reduce(collections.OrderedDict, ...):
            // treat item assignment as building a plain dict.
            PickleValue::Reduce(..) => PickleValue::Dict(pairs),
            other => {
                return Err(err(format!(
                    "pickle SETITEMS on non-dict value {other:?}"
                )));
            }
        };
        self.stack.push(updated);
        Ok(())
    }

    fn run(&mut self) -> Result<PickleValue, MmnError> {
        loop {
            let op = self.u8()?;
            match op {
                0x80 => {
                    let proto = self.u8()?;
                    if proto > 5 {
                        return Err(err(format!("pickle protocol {proto} unsupported")));
                    }
                }
                0x95 => {
                    self.take(8)?; // FRAME length — framing is transparent here
                }
                b'(' => self.stack.push(PickleValue::Mark),
                b'.' => return self.pop(),
                b'0' => {
                    self.pop()?;
                }
                b'N' => self.stack.push(PickleValue::None),
                0x88 => self.stack.push(PickleValue::Bool(true)),
                0x89 => self.stack.push(PickleValue::Bool(false)),
                b'K' => {
                    let v = self.u8()?;
                    self.stack.push(PickleValue::Int(v as i64));
                }
                b'M' => {
                    let v = self.u16()?;
                    self.stack.push(PickleValue::Int(v as i64));
                }
                b'J' => {
                    let v = self.u32()? as i32;
                    self.stack.push(PickleValue::Int(v as i64));
                }
                0x8a => {
                    let n = self.u8()? as usize;
                    let raw = self.take(n)?;
                    if n > 8 {
                        // Big integers (e.g. torch's legacy magic number) keep
                        // their raw little-endian bytes for exact comparison.
                        self.stack.push(PickleValue::Bytes(raw.to_vec()));
                    } else {
                        let mut v: i64 = 0;
                        for (i, &b) in raw.iter().enumerate() {
                            v |= (b as i64) << (8 * i);
                        }
                        // Sign-extend.
                        if n > 0 && n < 8 && raw[n - 1] & 0x80 != 0 {
                            v |= -1i64 << (8 * n);
                        }
                        self.stack.push(PickleValue::Int(v));
                    }
                }
                b'L' => {
                    // LONG (text): decimal digits ending in 'L\n'.
                    let line = self.line()?;
                    let digits = line.trim_end_matches('L');
                    let v: i64 = digits.parse().map_err(|e| {
                        err(format!("pickle LONG {digits:?} not an i64: {e}"))
                    })?;
                    self.stack.push(PickleValue::Int(v));
                }
                b'G' => {
                    let b = self.take(8)?;
                    let v = f64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
                    self.stack.push(PickleValue::Float(v));
                }
                b'U' => {
                    let n = self.u8()? as usize;
                    let raw = self.take(n)?.to_vec();
                    self.stack.push(match String::from_utf8(raw.clone()) {
                        Ok(s) => PickleValue::Str(s),
                        Err(_) => PickleValue::Bytes(raw),
                    });
                }
                b'X' => {
                    let n = self.u32()? as usize;
                    let raw = self.take(n)?;
                    let s = std::str::from_utf8(raw)
                        .map_err(|e| err(format!("pickle BINUNICODE not UTF-8: {e}")))?;
                    self.stack.push(PickleValue::Str(s.to_string()));
                }
                0x8c => {
                    let n = self.u8()? as usize;
                    let raw = self.take(n)?;
                    let s = std::str::from_utf8(raw)
                        .map_err(|e| err(format!("pickle SHORT_BINUNICODE not UTF-8: {e}")))?;
                    self.stack.push(PickleValue::Str(s.to_string()));
                }
                b'C' => {
                    let n = self.u8()? as usize;
                    let raw = self.take(n)?.to_vec();
                    self.stack.push(PickleValue::Bytes(raw));
                }
                0x8e => {
                    // BINBYTES8 (u64 length).
                    let lo = self.u32()? as u64;
                    let hi = self.u32()? as u64;
                    let n = (hi << 32 | lo) as usize;
                    let raw = self.take(n)?.to_vec();
                    self.stack.push(PickleValue::Bytes(raw));
                }
                b'B' => {
                    let n = self.u32()? as usize;
                    let raw = self.take(n)?.to_vec();
                    self.stack.push(PickleValue::Bytes(raw));
                }
                b'}' => self.stack.push(PickleValue::Dict(Vec::new())),
                b']' => self.stack.push(PickleValue::List(Vec::new())),
                b')' => self.stack.push(PickleValue::Tuple(Vec::new())),
                0x85 => {
                    let a = self.pop()?;
                    self.stack.push(PickleValue::Tuple(vec![a]));
                }
                0x86 => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(PickleValue::Tuple(vec![a, b]));
                }
                0x87 => {
                    let c = self.pop()?;
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(PickleValue::Tuple(vec![a, b, c]));
                }
                b't' => {
                    let items = self.pop_to_mark()?;
                    self.stack.push(PickleValue::Tuple(items));
                }
                b'l' => {
                    let items = self.pop_to_mark()?;
                    self.stack.push(PickleValue::List(items));
                }
                b'a' => {
                    let item = self.pop()?;
                    match self.stack.last_mut() {
                        Some(PickleValue::List(items)) => items.push(item),
                        _ => return Err(err("pickle APPEND on non-list")),
                    }
                }
                b'e' => {
                    let items = self.pop_to_mark()?;
                    match self.stack.last_mut() {
                        Some(PickleValue::List(existing)) => existing.extend(items),
                        _ => return Err(err("pickle APPENDS on non-list")),
                    }
                }
                b's' => {
                    let value = self.pop()?;
                    let key = self.pop()?;
                    self.set_items(vec![(key, value)])?;
                }
                b'u' => {
                    let flat = self.pop_to_mark()?;
                    if !flat.len().is_multiple_of(2) {
                        return Err(err("pickle SETITEMS with odd item count"));
                    }
                    let mut pairs = Vec::with_capacity(flat.len() / 2);
                    let mut iter = flat.into_iter();
                    while let (Some(k), Some(v)) = (iter.next(), iter.next()) {
                        pairs.push((k, v));
                    }
                    self.set_items(pairs)?;
                }
                b'c' => {
                    let module = self.line()?;
                    let name = self.line()?;
                    self.stack.push(PickleValue::Global(module, name));
                }
                0x93 => {
                    let name = self.pop()?;
                    let module = self.pop()?;
                    match (module, name) {
                        (PickleValue::Str(m), PickleValue::Str(n)) => {
                            self.stack.push(PickleValue::Global(m, n));
                        }
                        _ => return Err(err("pickle STACK_GLOBAL needs two strings")),
                    }
                }
                b'R' => {
                    let args = self.pop()?;
                    let callable = self.pop()?;
                    self.stack
                        .push(reduce_value(callable, args));
                }
                b'b' => {
                    // BUILD: apply state to object — state is irrelevant for
                    // reading tensors, keep the object.
                    let _state = self.pop()?;
                }
                b'Q' => {
                    let pid = self.pop()?;
                    self.stack.push(PickleValue::PersId(Box::new(pid)));
                }
                b'P' => {
                    let text = self.line()?;
                    self.stack
                        .push(PickleValue::PersId(Box::new(PickleValue::Str(text))));
                }
                b'q' => {
                    let idx = self.u8()? as usize;
                    self.memo_put(idx)?;
                }
                b'r' => {
                    let idx = self.u32()? as usize;
                    self.memo_put(idx)?;
                }
                0x94 => {
                    let idx = self.memo.len();
                    self.memo_put(idx)?;
                }
                b'h' => {
                    let idx = self.u8()? as usize;
                    self.memo_get(idx)?;
                }
                b'j' => {
                    let idx = self.u32()? as usize;
                    self.memo_get(idx)?;
                }
                0x8f => self.stack.push(PickleValue::Dict(Vec::new())), // EMPTY_SET as dict-ish
                other => {
                    return Err(err(format!(
                        "pickle opcode 0x{other:02x} ({}) not supported",
                        other as char
                    )));
                }
            }
        }
    }
}

/// Normalize a REDUCE application (OrderedDict becomes a plain dict).
fn reduce_value(callable: PickleValue, args: PickleValue) -> PickleValue {
    if let PickleValue::Global(module, name) = &callable {
        if module == "collections" && name == "OrderedDict" {
            return PickleValue::Dict(Vec::new());
        }
    }
    PickleValue::Reduce(Box::new(callable), Box::new(args))
}

/// Parse a pickle byte stream into a value tree.
pub fn parse_pickle(bytes: &[u8]) -> Result<PickleValue, MmnError> {
    Ok(parse_pickle_prefix(bytes)?.0)
}

/// Parse one pickle from the front of `bytes`, returning the consumed length.
///
/// Legacy `torch.save` files concatenate several pickles followed by raw
/// storage data; this lets callers walk that layout.
pub fn parse_pickle_prefix(bytes: &[u8]) -> Result<(PickleValue, usize), MmnError> {
    let mut vm = Vm {
        bytes,
        pos: 0,
        stack: Vec::new(),
        memo: Vec::new(),
    };
    let value = vm.run()?;
    Ok((value, vm.pos))
}

/// Incremental pickle writer emitting a protocol-2 stream.
#[derive(Default)]
pub struct PickleWriter {
    out: Vec<u8>,
}

impl PickleWriter {
    pub fn new() -> Self {
        let mut out = Vec::new();
        out.extend_from_slice(&[0x80, 0x02]);
        Self { out }
    }

    pub fn finish(mut self) -> Vec<u8> {
        self.out.push(b'.');
        self.out
    }

    pub fn mark(&mut self) {
        self.out.push(b'(');
    }

    pub fn empty_dict(&mut self) {
        self.out.push(b'}');
    }

    pub fn set_items(&mut self) {
        self.out.push(b'u');
    }

    pub fn tuple_from_mark(&mut self) {
        self.out.push(b't');
    }

    pub fn list_from_mark(&mut self) {
        self.out.push(b'l');
    }

    pub fn empty_tuple(&mut self) {
        self.out.push(b')');
    }

    pub fn bool(&mut self, v: bool) {
        self.out.push(if v { 0x88 } else { 0x89 });
    }

    pub fn int(&mut self, v: i64) {
        if (0..=255).contains(&v) {
            self.out.push(b'K');
            self.out.push(v as u8);
        } else if (i32::MIN as i64..=i32::MAX as i64).contains(&v) {
            self.out.push(b'J');
            self.out.extend_from_slice(&(v as i32).to_le_bytes());
        } else {
            self.out.push(0x8a);
            self.out.push(8);
            self.out.extend_from_slice(&v.to_le_bytes());
        }
    }

    pub fn string(&mut self, s: &str) {
        self.out.push(b'X');
        self.out
            .extend_from_slice(&(s.len() as u32).to_le_bytes());
        self.out.extend_from_slice(s.as_bytes());
    }

    pub fn global(&mut self, module: &str, name: &str) {
        self.out.push(b'c');
        self.out.extend_from_slice(module.as_bytes());
        self.out.push(b'\n');
        self.out.extend_from_slice(name.as_bytes());
        self.out.push(b'\n');
    }

    pub fn reduce(&mut self) {
        self.out.push(b'R');
    }

    pub fn binpersid(&mut self) {
        self.out.push(b'Q');
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writer_reader_roundtrip_dict() {
        let mut w = PickleWriter::new();
        w.empty_dict();
        w.mark();
        w.string("alpha");
        w.int(7);
        w.string("beta");
        w.bool(true);
        w.set_items();
        let bytes = w.finish();
        let value = parse_pickle(&bytes).unwrap();
        match value {
            PickleValue::Dict(pairs) => {
                assert_eq!(pairs.len(), 2);
                assert_eq!(pairs[0].0.as_str(), Some("alpha"));
                assert_eq!(pairs[0].1.as_int(), Some(7));
                assert_eq!(pairs[1].1, PickleValue::Bool(true));
            }
            other => panic!("expected dict, got {other:?}"),
        }
    }

    #[test]
    fn none_opcode_parses() {
        assert_eq!(
            parse_pickle(&[0x80, 0x02, b'N', b'.']).unwrap(),
            PickleValue::None
        );
    }

    #[test]
    fn reduce_and_persid_shape() {
        let mut w = PickleWriter::new();
        w.global("torch._utils", "_rebuild_tensor_v2");
        w.mark();
        {
            // persistent id tuple ('storage', FloatStorage, '0', 'cpu', 4)
            w.mark();
            w.string("storage");
            w.global("torch", "FloatStorage");
            w.string("0");
            w.string("cpu");
            w.int(4);
            w.tuple_from_mark();
            w.binpersid();
        }
        w.int(0);
        w.mark();
        w.int(2);
        w.int(2);
        w.tuple_from_mark();
        w.mark();
        w.int(2);
        w.int(1);
        w.tuple_from_mark();
        w.bool(false);
        w.global("collections", "OrderedDict");
        w.empty_tuple();
        w.reduce();
        w.tuple_from_mark();
        w.reduce();
        let bytes = w.finish();
        let value = parse_pickle(&bytes).unwrap();
        let PickleValue::Reduce(callable, args) = value else {
            panic!("expected reduce");
        };
        assert_eq!(
            *callable,
            PickleValue::Global("torch._utils".into(), "_rebuild_tensor_v2".into())
        );
        let items = args.tuple_items().unwrap();
        let PickleValue::PersId(pid) = &items[0] else {
            panic!("expected persid");
        };
        let pid_items = pid.tuple_items().unwrap();
        assert_eq!(pid_items[0].as_str(), Some("storage"));
        assert_eq!(pid_items[4].as_int(), Some(4));
        assert_eq!(
            items[2],
            PickleValue::Tuple(vec![PickleValue::Int(2), PickleValue::Int(2)])
        );
    }

    #[test]
    fn ordered_dict_setitems_becomes_dict() {
        let mut w = PickleWriter::new();
        w.global("collections", "OrderedDict");
        w.empty_tuple();
        w.reduce();
        w.mark();
        w.string("k");
        w.int(1);
        w.set_items();
        let value = parse_pickle(&w.finish()).unwrap();
        assert_eq!(
            value,
            PickleValue::Dict(vec![(PickleValue::Str("k".into()), PickleValue::Int(1))])
        );
    }

    #[test]
    fn memo_put_get_roundtrip() {
        // 0x80 0x02, BINUNICODE "x", BINPUT 0, POP, BINGET 0, STOP
        let mut bytes: Vec<u8> = vec![0x80, 0x02];
        bytes.push(b'X');
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.push(b'x');
        bytes.extend_from_slice(&[b'q', 0, b'0', b'h', 0, b'.']);
        let value = parse_pickle(&bytes).unwrap();
        assert_eq!(value.as_str(), Some("x"));
    }

    #[test]
    fn protocol4_frame_and_memoize() {
        // PROTO 4, FRAME len, SHORT_BINUNICODE "hi", MEMOIZE, STOP
        let mut bytes: Vec<u8> = vec![0x80, 0x04, 0x95];
        bytes.extend_from_slice(&5u64.to_le_bytes());
        bytes.extend_from_slice(&[0x8c, 2, b'h', b'i', 0x94, b'.']);
        let value = parse_pickle(&bytes).unwrap();
        assert_eq!(value.as_str(), Some("hi"));
    }

    #[test]
    fn unknown_opcode_and_truncation_error() {
        assert!(parse_pickle(&[0x80, 0x02, 0xff]).is_err());
        assert!(parse_pickle(&[0x80]).is_err());
        assert!(parse_pickle(&[]).is_err());
    }

    #[test]
    fn long1_sign_extension() {
        // LONG1 with bytes [0xFF] == -1.
        let bytes = vec![0x80, 0x02, 0x8a, 1, 0xFF, b'.'];
        assert_eq!(parse_pickle(&bytes).unwrap().as_int(), Some(-1));
    }
}
