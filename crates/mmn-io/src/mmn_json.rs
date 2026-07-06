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

/// Append a JSON string literal (tensor names never need escaping in
/// practice, but escape control/quote/backslash to stay valid JSON).
fn push_json_string(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Append a byte as 1-3 decimal digits (the hot loop of serialization).
#[inline]
fn push_byte_decimal(out: &mut String, v: u8) {
    // SAFETY-free fast path: manual digit expansion beats fmt machinery.
    if v >= 100 {
        out.push((b'0' + v / 100) as char);
        out.push((b'0' + (v / 10) % 10) as char);
        out.push((b'0' + v % 10) as char);
    } else if v >= 10 {
        out.push((b'0' + v / 10) as char);
        out.push((b'0' + v % 10) as char);
    } else {
        out.push((b'0' + v) as char);
    }
}

/// Serialize one tensor entry as `{"data":[...],"dtype":"...","shape":[...]}`.
fn write_entry(out: &mut String, entry: &TensorEntry) {
    out.push_str("{\"data\":[");
    let mut first = true;
    for &b in &entry.data {
        if !first {
            out.push(',');
        }
        first = false;
        push_byte_decimal(out, b);
    }
    out.push_str("],\"dtype\":");
    push_json_string(out, &entry.dtype);
    out.push_str(",\"shape\":[");
    let mut first = true;
    for &d in &entry.shape {
        if !first {
            out.push(',');
        }
        first = false;
        out.push_str(&d.to_string());
    }
    out.push_str("]}");
}

/// Serialize a `{format, meta, tensors}` checkpoint — hand-rolled writer
/// with per-tensor parallelism (serde's generic number formatting was the
/// bottleneck on multi-megabyte byte arrays). Output is byte-identical to
/// the previous serde output: same field order, no whitespace.
pub(crate) fn write_checkpoint(
    format: &str,
    meta: &serde_json::Value,
    tensors: &TensorMap,
) -> String {
    let entries: Vec<(&String, &TensorEntry)> = tensors.iter().collect();
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(entries.len().max(1));
    let fragments: Vec<String> = if workers <= 1 || entries.len() <= 1 {
        entries
            .iter()
            .map(|(name, entry)| {
                let mut frag = String::with_capacity(entry.data.len() * 4 + 64);
                push_json_string(&mut frag, name);
                frag.push(':');
                write_entry(&mut frag, entry);
                frag
            })
            .collect()
    } else {
        let next = std::sync::atomic::AtomicUsize::new(0);
        let mut collected: Vec<(usize, String)> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..workers)
                .map(|_| {
                    scope.spawn(|| {
                        let mut done = Vec::new();
                        loop {
                            let idx = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let Some((name, entry)) = entries.get(idx) else {
                                break;
                            };
                            let mut frag =
                                String::with_capacity(entry.data.len() * 4 + 64);
                            push_json_string(&mut frag, name);
                            frag.push(':');
                            write_entry(&mut frag, entry);
                            done.push((idx, frag));
                        }
                        done
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().expect("checkpoint serialize worker panicked"))
                .collect()
        });
        collected.sort_by_key(|(idx, _)| *idx);
        collected.into_iter().map(|(_, frag)| frag).collect()
    };
    let total: usize = fragments.iter().map(|f| f.len()).sum();
    let mut out = String::with_capacity(total + 256);
    out.push_str("{\"format\":");
    push_json_string(&mut out, format);
    out.push_str(",\"meta\":");
    out.push_str(&meta.to_string());
    out.push_str(",\"tensors\":{");
    for (i, frag) in fragments.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(frag);
    }
    out.push_str("}}");
    out
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

    /// Collect `(name, entry span)` pairs from the `"tensors"` object
    /// without parsing entry bodies (those parse in parallel afterwards).
    fn tensor_entry_spans(&mut self) -> Option<Vec<(String, (usize, usize))>> {
        self.skip_ws();
        self.eat(b'{')?;
        let mut spans = Vec::new();
        loop {
            self.skip_ws();
            match self.peek()? {
                b'}' => {
                    self.pos += 1;
                    return Some(spans);
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
            let span = self.value_span()?;
            spans.push((key, span));
        }
    }
}

/// Parse one tensor entry from its span; the span must be fully consumed.
fn parse_entry_span(bytes: &[u8], span: (usize, usize)) -> Option<TensorEntry> {
    let mut s = Scanner {
        bytes: &bytes[..span.1],
        pos: span.0,
    };
    let entry = s.tensor_entry()?;
    s.skip_ws();
    if s.pos != span.1 {
        return None;
    }
    Some(entry)
}

/// Parse all tensor entries, fanning the digit-parsing work (which
/// dominates load time) across available cores.
fn parse_entries_parallel(
    bytes: &[u8],
    spans: Vec<(String, (usize, usize))>,
) -> Option<TensorMap> {
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(spans.len().max(1));
    if workers <= 1 || spans.len() <= 1 {
        let mut map = TensorMap::new();
        for (key, span) in spans {
            map.insert(key, parse_entry_span(bytes, span)?);
        }
        return Some(map);
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results: Vec<Vec<(usize, Option<TensorEntry>)>> = std::thread::scope(|scope| {
        let spans = &spans;
        let next = &next;
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                scope.spawn(move || {
                    let mut done = Vec::new();
                    loop {
                        let idx = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some((_, span)) = spans.get(idx) else {
                            break;
                        };
                        done.push((idx, parse_entry_span(bytes, *span)));
                    }
                    done
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("checkpoint parse worker panicked"))
            .collect()
    });
    let mut parsed: Vec<Option<TensorEntry>> = (0..spans.len()).map(|_| None).collect();
    for chunk in results {
        for (idx, entry) in chunk {
            parsed[idx] = Some(entry?);
        }
    }
    let mut map = TensorMap::new();
    for ((key, _), entry) in spans.into_iter().zip(parsed) {
        map.insert(key, entry.expect("all indexes visited"));
    }
    Some(map)
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
    let mut spans: Vec<(String, (usize, usize))> = Vec::new();
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
            "tensors" => spans = s.tensor_entry_spans()?,
            _ => {
                s.value_span()?;
            }
        }
    }
    s.skip_ws();
    if s.pos != s.bytes.len() {
        return None;
    }
    let tensors = parse_entries_parallel(text.as_bytes(), spans)?;
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

    /// The hand-rolled serializer must be byte-identical to serde's output
    /// for the same `{format, meta, tensors}` layout.
    #[test]
    fn writer_is_byte_identical_to_serde() {
        #[derive(serde::Serialize)]
        struct Wrapper<'a> {
            format: &'a str,
            meta: &'a serde_json::Value,
            tensors: &'a TensorMap,
        }
        let mut tensors = TensorMap::new();
        tensors.insert(
            "embed".to_string(),
            TensorEntry {
                data: (0u16..600).map(|i| (i % 256) as u8).collect(),
                dtype: "F32".to_string(),
                shape: vec![30, 5],
            },
        );
        tensors.insert(
            "b\"quoted".to_string(),
            TensorEntry {
                data: vec![],
                dtype: "F32".to_string(),
                shape: vec![0],
            },
        );
        let meta = serde_json::json!({"vocab_size": 8, "seed": 3, "use_rope": true});
        let ours = write_checkpoint("mmn-safetensors-v1", &meta, &tensors);
        let via_serde = serde_json::to_string(&Wrapper {
            format: "mmn-safetensors-v1",
            meta: &meta,
            tensors: &tensors,
        })
        .unwrap();
        assert_eq!(ours, via_serde);
        // And the fast parser reads its own writer's output.
        let back = parse_checkpoint(&ours).unwrap();
        assert_eq!(back.tensors["embed"].data.len(), 600);
        assert_eq!(back.tensors["embed"].shape, vec![30, 5]);
    }

    #[test]
    fn parallel_entry_parse_rejects_corrupt_entry() {
        // Second tensor has an overflowing byte: the whole fast path must
        // reject so serde produces the user-facing error.
        let text = r#"{"format":"mmn-safetensors-v1","meta":{},"tensors":{"a":{"data":[1,2,3,4],"dtype":"F32","shape":[1]},"b":{"data":[999],"dtype":"F32","shape":[1]}}}"#;
        assert!(parse_checkpoint_fast(text).is_none());
        assert!(parse_checkpoint(text).is_err());
    }
}
