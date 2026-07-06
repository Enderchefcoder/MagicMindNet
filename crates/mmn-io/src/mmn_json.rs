//! Fast scanner for mmn JSON checkpoints (`mmn-safetensors-v1`,
//! `mmn-classifier-v1`, `mmn-diffusion-v1`).
//!
//! The default checkpoint format stores tensor bytes as JSON integer arrays;
//! generic serde parsing spends most of its time in number handling. This
//! hand-rolled scanner reads the fixed `{format, meta, tensors}` schema
//! directly (data bytes via a tight digit loop) and returns `None` on any
//! input it does not fully accept, so callers fall back to serde — the
//! scanner is a pure fast path and correctness never depends on it.

use crate::checkpoint_util::{TensorEntry, TensorMap};

/// Parsed `{format, meta, tensors}` checkpoint (shared by all JSON formats).
pub(crate) struct MmnJsonCheckpoint {
    pub format: Option<String>,
    pub meta: serde_json::Value,
    pub tensors: TensorMap,
}

/// Parse a checkpoint, using the fast scanner when it accepts the input and
/// generic serde otherwise (serde also produces the user-facing error).
pub(crate) fn parse_checkpoint(text: &str) -> Result<MmnJsonCheckpoint, serde_json::Error> {
    if let Some(ckpt) = parse_checkpoint_fast(text) {
        return Ok(ckpt);
    }
    #[derive(serde::Deserialize)]
    struct Raw {
        #[serde(default)]
        format: Option<String>,
        #[serde(default)]
        meta: serde_json::Value,
        #[serde(default)]
        tensors: TensorMap,
    }
    let raw: Raw = serde_json::from_str(text)?;
    Ok(MmnJsonCheckpoint {
        format: raw.format,
        meta: raw.meta,
        tensors: raw.tensors,
    })
}

struct Scanner<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Scanner<'a> {
    fn skip_ws(&mut self) {
        while let Some(&b) = self.bytes.get(self.pos) {
            if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn eat(&mut self, expected: u8) -> Option<()> {
        if self.peek() == Some(expected) {
            self.pos += 1;
            Some(())
        } else {
            None
        }
    }

    /// Consume a JSON string, returning the span including both quotes.
    fn string_span(&mut self) -> Option<(usize, usize)> {
        let start = self.pos;
        self.eat(b'"')?;
        loop {
            match self.bytes.get(self.pos)? {
                b'"' => {
                    self.pos += 1;
                    return Some((start, self.pos));
                }
                b'\\' => self.pos += 2,
                _ => self.pos += 1,
            }
        }
    }

    /// Consume a JSON string and decode it (escapes delegated to serde).
    fn string(&mut self) -> Option<String> {
        let (start, end) = self.string_span()?;
        let raw = &self.bytes[start..end];
        if !raw.contains(&b'\\') {
            return std::str::from_utf8(&raw[1..raw.len() - 1])
                .ok()
                .map(str::to_string);
        }
        serde_json::from_slice(raw).ok()
    }

    /// Skip any JSON value, returning its span.
    fn value_span(&mut self) -> Option<(usize, usize)> {
        self.skip_ws();
        let start = self.pos;
        match self.peek()? {
            b'"' => {
                self.string_span()?;
            }
            b'{' | b'[' => {
                let mut depth = 0usize;
                loop {
                    match self.bytes.get(self.pos)? {
                        b'{' | b'[' => {
                            depth += 1;
                            self.pos += 1;
                        }
                        b'}' | b']' => {
                            depth -= 1;
                            self.pos += 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        b'"' => {
                            self.string_span()?;
                        }
                        _ => {
                            // Bulk-skip scalars up to the next structural
                            // byte (numbers dominate checkpoint payloads).
                            let rest = &self.bytes[self.pos..];
                            let n = rest
                                .iter()
                                .position(|&b| {
                                    matches!(b, b'{' | b'[' | b'}' | b']' | b'"')
                                })
                                .unwrap_or(rest.len());
                            self.pos += n;
                        }
                    }
                }
            }
            _ => {
                while let Some(&b) = self.bytes.get(self.pos) {
                    if b == b',' || b == b'}' || b == b']' {
                        break;
                    }
                    self.pos += 1;
                }
                if self.pos == start {
                    return None;
                }
            }
        }
        Some((start, self.pos))
    }

    /// Parse `[b, b, ...]` where every element is an integer 0..=255.
    fn byte_array(&mut self) -> Option<Vec<u8>> {
        self.skip_ws();
        self.eat(b'[')?;
        // Rough capacity guess: ~4 chars per serialized byte.
        let mut out = Vec::with_capacity((self.bytes.len() - self.pos) / 4);
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Some(out);
        }
        loop {
            self.skip_ws();
            let mut value: u32 = 0;
            let digits_start = self.pos;
            while let Some(&b) = self.bytes.get(self.pos) {
                if b.is_ascii_digit() {
                    value = value * 10 + (b - b'0') as u32;
                    if value > 255 {
                        return None;
                    }
                    self.pos += 1;
                } else {
                    break;
                }
            }
            if self.pos == digits_start {
                return None;
            }
            out.push(value as u8);
            self.skip_ws();
            match self.peek()? {
                b',' => self.pos += 1,
                b']' => {
                    self.pos += 1;
                    return Some(out);
                }
                _ => return None,
            }
        }
    }

    /// Parse `[n, n, ...]` of non-negative integers (tensor shape).
    fn usize_array(&mut self) -> Option<Vec<usize>> {
        self.skip_ws();
        self.eat(b'[')?;
        let mut out = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Some(out);
        }
        loop {
            self.skip_ws();
            let mut value: usize = 0;
            let digits_start = self.pos;
            while let Some(&b) = self.bytes.get(self.pos) {
                if b.is_ascii_digit() {
                    value = value.checked_mul(10)?.checked_add((b - b'0') as usize)?;
                    self.pos += 1;
                } else {
                    break;
                }
            }
            if self.pos == digits_start {
                return None;
            }
            out.push(value);
            self.skip_ws();
            match self.peek()? {
                b',' => self.pos += 1,
                b']' => {
                    self.pos += 1;
                    return Some(out);
                }
                _ => return None,
            }
        }
    }

    /// Parse one `{"data": [...], "dtype": "...", "shape": [...]}` entry.
    fn tensor_entry(&mut self) -> Option<TensorEntry> {
        self.skip_ws();
        self.eat(b'{')?;
        let mut data: Option<Vec<u8>> = None;
        let mut dtype: Option<String> = None;
        let mut shape: Option<Vec<usize>> = None;
        loop {
            self.skip_ws();
            match self.peek()? {
                b'}' => {
                    self.pos += 1;
                    break;
                }
                b',' => {
                    self.pos += 1;
                    continue;
                }
                b'"' => {}
                _ => return None,
            }
            let key = self.string()?;
            self.skip_ws();
            self.eat(b':')?;
            match key.as_str() {
                "data" => data = Some(self.byte_array()?),
                "dtype" => {
                    self.skip_ws();
                    dtype = Some(self.string()?);
                }
                "shape" => shape = Some(self.usize_array()?),
                _ => return None,
            }
        }
        Some(TensorEntry {
            data: data?,
            dtype: dtype?,
            shape: shape?,
        })
    }

    /// Parse the `"tensors"` object.
    fn tensors(&mut self) -> Option<TensorMap> {
        self.skip_ws();
        self.eat(b'{')?;
        let mut map = TensorMap::new();
        loop {
            self.skip_ws();
            match self.peek()? {
                b'}' => {
                    self.pos += 1;
                    return Some(map);
                }
                b',' => {
                    self.pos += 1;
                    continue;
                }
                b'"' => {}
                _ => return None,
            }
            let key = self.string()?;
            self.skip_ws();
            self.eat(b':')?;
            let entry = self.tensor_entry()?;
            map.insert(key, entry);
        }
    }
}

/// Extract the top-level `"format"` string without materializing tensor
/// arrays (detection fast path). Outer `None` means "input not accepted,
/// fall back to a full serde parse".
pub(crate) fn top_level_format(text: &str) -> Option<Option<String>> {
    let mut s = Scanner {
        bytes: text.as_bytes(),
        pos: 0,
    };
    s.skip_ws();
    s.eat(b'{')?;
    let mut format: Option<String> = None;
    loop {
        s.skip_ws();
        match s.peek()? {
            b'}' => {
                s.pos += 1;
                break;
            }
            b',' => {
                s.pos += 1;
                continue;
            }
            b'"' => {}
            _ => return None,
        }
        let key = s.string()?;
        s.skip_ws();
        s.eat(b':')?;
        if key == "format" {
            s.skip_ws();
            format = Some(s.string()?);
        } else {
            s.value_span()?;
        }
    }
    s.skip_ws();
    if s.pos != s.bytes.len() {
        return None;
    }
    Some(format)
}

/// Check whether the top-level object has `key` mapped to a JSON object,
/// without materializing the document. Outer `None` means "input not
/// accepted, fall back to a full serde parse".
pub(crate) fn top_level_key_is_object(text: &str, key: &str) -> Option<bool> {
    let mut s = Scanner {
        bytes: text.as_bytes(),
        pos: 0,
    };
    s.skip_ws();
    s.eat(b'{')?;
    let mut found = false;
    loop {
        s.skip_ws();
        match s.peek()? {
            b'}' => {
                s.pos += 1;
                break;
            }
            b',' => {
                s.pos += 1;
                continue;
            }
            b'"' => {}
            _ => return None,
        }
        let this_key = s.string()?;
        s.skip_ws();
        s.eat(b':')?;
        s.skip_ws();
        if this_key == key && s.peek() == Some(b'{') {
            found = true;
        }
        s.value_span()?;
    }
    s.skip_ws();
    if s.pos != s.bytes.len() {
        return None;
    }
    Some(found)
}

/// Fast-path parse; `None` means "input not accepted, use serde".
fn parse_checkpoint_fast(text: &str) -> Option<MmnJsonCheckpoint> {
    let mut s = Scanner {
        bytes: text.as_bytes(),
        pos: 0,
    };
    s.skip_ws();
    s.eat(b'{')?;
    let mut format: Option<String> = None;
    let mut meta = serde_json::Value::Null;
    let mut tensors = TensorMap::new();
    loop {
        s.skip_ws();
        match s.peek()? {
            b'}' => {
                s.pos += 1;
                break;
            }
            b',' => {
                s.pos += 1;
                continue;
            }
            b'"' => {}
            _ => return None,
        }
        let key = s.string()?;
        s.skip_ws();
        s.eat(b':')?;
        match key.as_str() {
            "format" => {
                s.skip_ws();
                format = Some(s.string()?);
            }
            "meta" => {
                let (start, end) = s.value_span()?;
                meta = serde_json::from_slice(&s.bytes[start..end]).ok()?;
            }
            "tensors" => tensors = s.tensors()?,
            _ => {
                s.value_span()?;
            }
        }
    }
    s.skip_ws();
    if s.pos != s.bytes.len() {
        return None;
    }
    Some(MmnJsonCheckpoint {
        format,
        meta,
        tensors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_scanner_parses_canonical_checkpoint() {
        let text = r#"{"format":"mmn-safetensors-v1","meta":{"d_model":4,"n_layer":1,"vocab_size":8},"tensors":{"embed":{"data":[0,0,128,63,0,0,0,64],"dtype":"F32","shape":[2,1]}}}"#;
        let ckpt = parse_checkpoint_fast(text).expect("fast path should accept");
        assert_eq!(ckpt.format.as_deref(), Some("mmn-safetensors-v1"));
        assert_eq!(ckpt.meta["vocab_size"], 8);
        let entry = &ckpt.tensors["embed"];
        assert_eq!(entry.dtype, "F32");
        assert_eq!(entry.shape, vec![2, 1]);
        assert_eq!(entry.data, vec![0, 0, 128, 63, 0, 0, 0, 64]);
    }

    #[test]
    fn fast_scanner_handles_python_json_whitespace() {
        // json.dump default separators: ", " and ": ".
        let text = "{\"format\": \"mmn-classifier-v1\", \"meta\": {\"input_dim\": 3}, \"tensors\": {\"head\": {\"data\": [1, 2, 3, 4], \"dtype\": \"F32\", \"shape\": [1]}}}";
        let ckpt = parse_checkpoint_fast(text).expect("whitespace should parse");
        assert_eq!(ckpt.format.as_deref(), Some("mmn-classifier-v1"));
        assert_eq!(ckpt.tensors["head"].data, vec![1, 2, 3, 4]);
    }

    #[test]
    fn fast_scanner_rejects_byte_overflow_then_serde_reports() {
        let text = r#"{"format":"mmn-safetensors-v1","meta":{},"tensors":{"embed":{"data":[999],"dtype":"F32","shape":[1]}}}"#;
        assert!(parse_checkpoint_fast(text).is_none());
        let err = parse_checkpoint(text).err().expect("serde must also reject");
        let msg = err.to_string();
        assert!(msg.contains("invalid"), "expected invalid-value error: {msg}");
    }

    #[test]
    fn fast_scanner_rejects_trailing_garbage() {
        let text = r#"{"format":"mmn-safetensors-v1","meta":{},"tensors":{}} extra"#;
        assert!(parse_checkpoint_fast(text).is_none());
    }

    #[test]
    fn fast_and_serde_paths_agree_on_meta_and_unknown_keys() {
        let text = r#"{"extra":[1,{"x":"]"}],"format":"mmn-diffusion-v1","meta":{"latent_channels":4,"note":"a\"b"},"tensors":{}}"#;
        let fast = parse_checkpoint_fast(text).expect("unknown keys should be skipped");
        let via_serde = parse_checkpoint("{\"format\":\"mmn-diffusion-v1\",\"meta\":{\"latent_channels\":4,\"note\":\"a\\\"b\"},\"tensors\":{}}").unwrap();
        assert_eq!(fast.format, via_serde.format);
        assert_eq!(fast.meta["latent_channels"], via_serde.meta["latent_channels"]);
        assert_eq!(fast.meta["note"], via_serde.meta["note"]);
    }

    #[test]
    fn fast_scanner_handles_escaped_tensor_names() {
        let text = r#"{"format":"mmn-safetensors-v1","meta":{},"tensors":{"a\"b":{"data":[7],"dtype":"F32","shape":[]}}}"#;
        let ckpt = parse_checkpoint_fast(text).expect("escaped key should parse");
        assert_eq!(ckpt.tensors["a\"b"].data, vec![7]);
    }
}
