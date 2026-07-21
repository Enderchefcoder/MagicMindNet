//! Minimal JSON / digit constrained decoding helpers for generation.

/// Close open quotes/braces/brackets and repair trailing junk so `json.loads` can parse.
pub fn finalize_json(text: &str) -> String {
    let trimmed = text.trim();
    let start = trimmed
        .find(['{', '['])
        .map(|i| &trimmed[i..])
        .unwrap_or("");
    if start.is_empty() {
        return "{}".to_string();
    }

    let mut out: Vec<u8> = start.as_bytes().to_vec();
    let (mut expect, _) = analyze_prefix(&out);

    // Complete incomplete literals / numbers / keys before closing containers.
    match expect {
        Expect::InString { .. } => {
            out.push(b'"');
            let (e2, _) = analyze_prefix(&out);
            expect = e2;
        }
        Expect::InLiteral { lit, idx } => {
            out.extend_from_slice(&lit[idx..]);
            expect = Expect::AfterValue;
        }
        Expect::InNumber { seen_digit: false } => {
            out.push(b'0');
            expect = Expect::AfterValue;
        }
        Expect::InNumber { seen_digit: true } => {
            expect = Expect::AfterValue;
        }
        Expect::AfterKey | Expect::Value | Expect::InObjectKeyOrEnd | Expect::ArrayValueOrEnd
        | Expect::Start | Expect::Done | Expect::AfterValue => {}
    }

    if matches!(expect, Expect::AfterKey) {
        out.extend_from_slice(b":null");
        expect = Expect::AfterValue;
    } else if matches!(expect, Expect::Value) {
        out.extend_from_slice(b"null");
        expect = Expect::AfterValue;
    }
    let _ = expect;

    // Drop a trailing comma before closers.
    while matches!(
        out.last(),
        Some(b',') | Some(b' ') | Some(b'\t') | Some(b'\n') | Some(b'\r')
    ) {
        if out.last() == Some(&b',') {
            out.pop();
            break;
        }
        out.pop();
    }
    // Re-sync stack after repairs.
    let (_, mut stack) = analyze_prefix(&out);

    while let Some(c) = stack.pop() {
        match c {
            Container::Object => out.push(b'}'),
            Container::Array => out.push(b']'),
        }
    }

    let s = String::from_utf8_lossy(&out).into_owned();
    let s = s.trim().to_string();
    if looks_like_closed_json(&s) {
        s
    } else {
        "{}".to_string()
    }
}

/// True when `prefix` is a fully closed JSON value (object or array).
pub fn json_is_complete(prefix: &str) -> bool {
    let (expect, stack) = analyze_prefix(prefix.as_bytes());
    matches!(expect, Expect::Done) && stack.is_empty()
}

/// Byte-level allow-mask for the next token under a digit-only grammar.
pub fn digit_allowed(byte: u8) -> bool {
    (b'0'..=b'9').contains(&byte)
}

/// Which ASCII bytes are legal next under a minimal JSON subset given `prefix`.
pub fn json_allowed_bytes(prefix: &str) -> [bool; 256] {
    let mut allow = [false; 256];
    let (expect, stack) = analyze_prefix(prefix.as_bytes());
    match expect {
        Expect::Start => {
            set_ws(&mut allow);
            allow[b'{' as usize] = true;
            allow[b'[' as usize] = true;
        }
        Expect::InObjectKeyOrEnd => {
            set_ws(&mut allow);
            allow[b'"' as usize] = true;
            allow[b'}' as usize] = true;
        }
        Expect::AfterKey => {
            set_ws(&mut allow);
            allow[b':' as usize] = true;
        }
        Expect::Value | Expect::ArrayValueOrEnd => {
            set_ws(&mut allow);
            allow[b'"' as usize] = true;
            allow[b'{' as usize] = true;
            allow[b'[' as usize] = true;
            allow[b'-' as usize] = true;
            for d in b'0'..=b'9' {
                allow[d as usize] = true;
            }
            allow[b't' as usize] = true;
            allow[b'f' as usize] = true;
            allow[b'n' as usize] = true;
            if matches!(expect, Expect::ArrayValueOrEnd) {
                allow[b']' as usize] = true;
            }
        }
        Expect::AfterValue => {
            set_ws(&mut allow);
            match stack.last() {
                Some(Container::Object) => {
                    allow[b',' as usize] = true;
                    allow[b'}' as usize] = true;
                }
                Some(Container::Array) => {
                    allow[b',' as usize] = true;
                    allow[b']' as usize] = true;
                }
                None => {}
            }
        }
        Expect::InString { escape } => {
            if escape {
                for &c in &[b'"', b'\\', b'/', b'b', b'f', b'n', b'r', b't', b'u'] {
                    allow[c as usize] = true;
                }
            } else {
                for b in 0x20u8..=0x7eu8 {
                    if b != b'\\' {
                        allow[b as usize] = true;
                    }
                }
                // quote and backslash still allowed (backslash enters escape)
                allow[b'"' as usize] = true;
                allow[b'\\' as usize] = true;
            }
        }
        Expect::InNumber { .. } => {
            for d in b'0'..=b'9' {
                allow[d as usize] = true;
            }
            allow[b'.' as usize] = true;
            allow[b'e' as usize] = true;
            allow[b'E' as usize] = true;
            allow[b'+' as usize] = true;
            allow[b'-' as usize] = true;
            // also allow tokens that end the number (structural / ws)
            set_ws(&mut allow);
            match stack.last() {
                Some(Container::Object) => {
                    allow[b',' as usize] = true;
                    allow[b'}' as usize] = true;
                }
                Some(Container::Array) => {
                    allow[b',' as usize] = true;
                    allow[b']' as usize] = true;
                }
                None => {}
            }
        }
        Expect::InLiteral { lit, idx } => {
            if idx < lit.len() {
                allow[lit[idx] as usize] = true;
            }
        }
        Expect::Done => {
            set_ws(&mut allow);
        }
    }
    // Always keep at least one structural option to avoid all -inf.
    if !allow.iter().any(|&a| a) {
        allow[b' ' as usize] = true;
    }
    allow
}

/// Mask logits in-place: illegal next bytes → `-inf` (byte vocab: `id % 256`).
pub fn apply_digit_mask(scores: &mut [f32]) {
    for (i, s) in scores.iter_mut().enumerate() {
        if !digit_allowed((i % 256) as u8) {
            *s = f32::NEG_INFINITY;
        }
    }
}

/// Mask logits for JSON mode given the decoded prefix of newly generated tokens.
pub fn apply_json_mask(scores: &mut [f32], prefix: &str) {
    let allow = json_allowed_bytes(prefix);
    for (i, s) in scores.iter_mut().enumerate() {
        let b = (i % 256) as u8;
        if !allow[b as usize] {
            *s = f32::NEG_INFINITY;
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Container {
    Object,
    Array,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Expect {
    Start,
    InObjectKeyOrEnd,
    AfterKey,
    Value,
    ArrayValueOrEnd,
    AfterValue,
    InString { escape: bool },
    InNumber { seen_digit: bool },
    InLiteral { lit: &'static [u8], idx: usize },
    Done,
}

fn set_ws(allow: &mut [bool; 256]) {
    allow[b' ' as usize] = true;
    allow[b'\t' as usize] = true;
    allow[b'\n' as usize] = true;
    allow[b'\r' as usize] = true;
}

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

fn analyze_prefix(bytes: &[u8]) -> (Expect, Vec<Container>) {
    let mut i = 0;
    let mut stack: Vec<Container> = Vec::new();
    let mut expect = Expect::Start;
    // Track whether the last completed string was an object key (awaiting colon).
    let mut string_was_key = false;

    while i < bytes.len() {
        let b = bytes[i];
        match expect {
            Expect::Start => {
                if is_ws(b) {
                    i += 1;
                    continue;
                }
                if b == b'{' {
                    stack.push(Container::Object);
                    expect = Expect::InObjectKeyOrEnd;
                    i += 1;
                } else if b == b'[' {
                    stack.push(Container::Array);
                    expect = Expect::ArrayValueOrEnd;
                    i += 1;
                } else {
                    break;
                }
            }
            Expect::InObjectKeyOrEnd => {
                if is_ws(b) {
                    i += 1;
                    continue;
                }
                if b == b'}' {
                    stack.pop();
                    expect = if stack.is_empty() {
                        Expect::Done
                    } else {
                        Expect::AfterValue
                    };
                    i += 1;
                } else if b == b'"' {
                    string_was_key = true;
                    expect = Expect::InString { escape: false };
                    i += 1;
                } else {
                    break;
                }
            }
            Expect::AfterKey => {
                if is_ws(b) {
                    i += 1;
                    continue;
                }
                if b == b':' {
                    expect = Expect::Value;
                    i += 1;
                } else {
                    break;
                }
            }
            Expect::Value | Expect::ArrayValueOrEnd => {
                if is_ws(b) {
                    i += 1;
                    continue;
                }
                if matches!(expect, Expect::ArrayValueOrEnd) && b == b']' {
                    stack.pop();
                    expect = if stack.is_empty() {
                        Expect::Done
                    } else {
                        Expect::AfterValue
                    };
                    i += 1;
                    continue;
                }
                if b == b'"' {
                    string_was_key = false;
                    expect = Expect::InString { escape: false };
                    i += 1;
                } else if b == b'{' {
                    stack.push(Container::Object);
                    expect = Expect::InObjectKeyOrEnd;
                    i += 1;
                } else if b == b'[' {
                    stack.push(Container::Array);
                    expect = Expect::ArrayValueOrEnd;
                    i += 1;
                } else if b == b'-' || (b'0'..=b'9').contains(&b) {
                    expect = Expect::InNumber {
                        seen_digit: (b'0'..=b'9').contains(&b),
                    };
                    i += 1;
                } else if b == b't' {
                    expect = Expect::InLiteral {
                        lit: b"true",
                        idx: 1,
                    };
                    i += 1;
                } else if b == b'f' {
                    expect = Expect::InLiteral {
                        lit: b"false",
                        idx: 1,
                    };
                    i += 1;
                } else if b == b'n' {
                    expect = Expect::InLiteral {
                        lit: b"null",
                        idx: 1,
                    };
                    i += 1;
                } else {
                    break;
                }
            }
            Expect::AfterValue => {
                if is_ws(b) {
                    i += 1;
                    continue;
                }
                match stack.last() {
                    Some(Container::Object) if b == b',' => {
                        expect = Expect::InObjectKeyOrEnd;
                        i += 1;
                    }
                    Some(Container::Object) if b == b'}' => {
                        stack.pop();
                        expect = if stack.is_empty() {
                            Expect::Done
                        } else {
                            Expect::AfterValue
                        };
                        i += 1;
                    }
                    Some(Container::Array) if b == b',' => {
                        expect = Expect::ArrayValueOrEnd;
                        i += 1;
                    }
                    Some(Container::Array) if b == b']' => {
                        stack.pop();
                        expect = if stack.is_empty() {
                            Expect::Done
                        } else {
                            Expect::AfterValue
                        };
                        i += 1;
                    }
                    _ => break,
                }
            }
            Expect::InString { escape } => {
                if escape {
                    expect = Expect::InString { escape: false };
                    i += 1;
                } else if b == b'\\' {
                    expect = Expect::InString { escape: true };
                    i += 1;
                } else if b == b'"' {
                    if string_was_key {
                        expect = Expect::AfterKey;
                        string_was_key = false;
                    } else {
                        expect = Expect::AfterValue;
                    }
                    i += 1;
                } else {
                    i += 1;
                }
            }
            Expect::InNumber { seen_digit } => {
                if (b'0'..=b'9').contains(&b)
                    || b == b'.'
                    || b == b'e'
                    || b == b'E'
                    || b == b'+'
                    || b == b'-'
                {
                    expect = Expect::InNumber {
                        seen_digit: seen_digit || (b'0'..=b'9').contains(&b),
                    };
                    i += 1;
                } else {
                    // Number ended; re-process this byte under AfterValue.
                    expect = Expect::AfterValue;
                }
            }
            Expect::InLiteral { lit, idx } => {
                if idx < lit.len() && b == lit[idx] {
                    let next = idx + 1;
                    if next >= lit.len() {
                        expect = Expect::AfterValue;
                    } else {
                        expect = Expect::InLiteral { lit, idx: next };
                    }
                    i += 1;
                } else {
                    break;
                }
            }
            Expect::Done => {
                if is_ws(b) {
                    i += 1;
                } else {
                    break;
                }
            }
        }
    }
    (expect, stack)
}

fn looks_like_closed_json(s: &str) -> bool {
    let t = s.trim();
    if t.is_empty() {
        return false;
    }
    let first = t.as_bytes()[0];
    let last = t.as_bytes()[t.len() - 1];
    (first == b'{' && last == b'}') || (first == b'[' && last == b']')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finalize_empty_becomes_object() {
        assert_eq!(finalize_json(""), "{}");
        assert_eq!(finalize_json("   "), "{}");
    }

    #[test]
    fn finalize_closes_open_object() {
        let out = finalize_json(r#"{"a":1"#);
        assert!(out.ends_with('}'), "{out}");
        assert!(out.starts_with('{'));
    }

    #[test]
    fn finalize_closes_string_and_braces() {
        let out = finalize_json(r#"{"k":"v"#);
        assert!(out.contains('"'));
        assert!(out.ends_with('}'));
    }

    #[test]
    fn json_start_allows_brace() {
        let a = json_allowed_bytes("");
        assert!(a[b'{' as usize]);
        assert!(a[b'[' as usize]);
        assert!(!a[b'a' as usize]);
    }

    #[test]
    fn digit_mask_keeps_ascii_digits() {
        let mut scores = vec![1.0f32; 256];
        apply_digit_mask(&mut scores);
        for i in 0..256 {
            let b = i as u8;
            if (b'0'..=b'9').contains(&b) {
                assert!(scores[i].is_finite());
            } else {
                assert!(scores[i].is_infinite() && scores[i].is_sign_negative());
            }
        }
    }
}
