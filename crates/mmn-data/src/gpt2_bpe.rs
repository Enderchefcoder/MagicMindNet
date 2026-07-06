//! GPT-2-style byte-level BPE decoder/encoder for external vocabularies
//! (GGUF `tokenizer.ggml.model == "gpt2"`, HF `vocab.json` + `merges.txt`).
//!
//! Token ids follow the supplied vocabulary order, so they align with the
//! source model's embedding rows. Pretokenization approximates the GPT-2
//! regex with a character-class scanner (letters / digits / punctuation runs
//! with an optional leading space); English contraction splitting is not
//! special-cased.

use mmn_core::{MmnError, Result};
use std::collections::HashMap;

/// GPT-2 printable byte ranges that map to themselves in the unicode table.
fn is_printable_gpt2_byte(b: u8) -> bool {
    (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b)
}

/// The GPT-2 bytes↔unicode bijection: printable bytes map to themselves,
/// everything else maps to U+0100.. in order.
fn byte_to_unicode_table() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut n = 0u32;
    for (b, slot) in table.iter_mut().enumerate() {
        let b = b as u8;
        *slot = if is_printable_gpt2_byte(b) {
            b as char
        } else {
            let c = char::from_u32(256 + n).expect("valid unicode offset");
            n += 1;
            c
        };
    }
    table
}

/// Byte-level BPE with external vocab + ranked merges.
#[derive(Clone, Debug)]
pub struct Gpt2BpeEncoder {
    tokens: Vec<String>,
    token_ids: HashMap<String, usize>,
    merge_ranks: HashMap<(String, String), usize>,
    byte_to_unicode: [char; 256],
    unicode_to_byte: HashMap<char, u8>,
}

impl Gpt2BpeEncoder {
    /// Build from a vocabulary (id order) and merge rules ("left right" lines).
    pub fn from_vocab(tokens: Vec<String>, merges: &[String]) -> Result<Self> {
        if tokens.is_empty() {
            return Err(MmnError::Other {
                message: "Gpt2BpeEncoder needs a non-empty vocabulary".into(),
            });
        }
        let mut token_ids = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            token_ids.entry(t.clone()).or_insert(i);
        }
        let mut merge_ranks = HashMap::with_capacity(merges.len());
        for (rank, line) in merges.iter().enumerate() {
            let Some((left, right)) = line.split_once(' ') else {
                return Err(MmnError::Other {
                    message: format!("BPE merge rule {line:?} is not \"left right\""),
                });
            };
            merge_ranks
                .entry((left.to_string(), right.to_string()))
                .or_insert(rank);
        }
        let byte_to_unicode = byte_to_unicode_table();
        let mut unicode_to_byte = HashMap::with_capacity(256);
        for (b, &c) in byte_to_unicode.iter().enumerate() {
            unicode_to_byte.insert(c, b as u8);
        }
        Ok(Self {
            tokens,
            token_ids,
            merge_ranks,
            byte_to_unicode,
            unicode_to_byte,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// Token string (unicode-mapped form) for an id, if in range.
    pub fn token(&self, id: usize) -> Option<&str> {
        self.tokens.get(id).map(String::as_str)
    }

    /// Simplified GPT-2 pretokenization on raw text: an optional single
    /// leading space attaches to the following run of letters, digits, or
    /// punctuation; remaining whitespace forms its own words.
    fn pretokenize(text: &str) -> Vec<String> {
        #[derive(PartialEq, Clone, Copy)]
        enum Class {
            Letter,
            Digit,
            Other,
        }
        fn classify(c: char) -> Class {
            if c.is_alphabetic() {
                Class::Letter
            } else if c.is_numeric() {
                Class::Digit
            } else {
                Class::Other
            }
        }
        let chars: Vec<char> = text.chars().collect();
        let mut words = Vec::new();
        let mut i = 0usize;
        while i < chars.len() {
            let start = i;
            if chars[i] == ' ' && i + 1 < chars.len() && !chars[i + 1].is_whitespace() {
                i += 1; // single space prefixes the next word
            }
            if chars[i].is_whitespace() {
                while i < chars.len() && chars[i].is_whitespace() {
                    i += 1;
                }
                // Leave a trailing space to prefix the following word.
                if i < chars.len() && chars[i - 1] == ' ' && i - start > 1 {
                    i -= 1;
                }
                words.push(chars[start..i].iter().collect());
                continue;
            }
            let class = classify(chars[i]);
            while i < chars.len() && !chars[i].is_whitespace() && classify(chars[i]) == class {
                i += 1;
            }
            words.push(chars[start..i].iter().collect());
        }
        words
    }

    /// Greedy lowest-rank merging of one pretokenized word.
    fn bpe_word(&self, word: &str) -> Vec<String> {
        let mut parts: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        while parts.len() > 1 {
            let mut best: Option<(usize, usize)> = None; // (rank, index)
            for i in 0..parts.len() - 1 {
                let key = (parts[i].clone(), parts[i + 1].clone());
                if let Some(&rank) = self.merge_ranks.get(&key) {
                    if best.map(|(r, _)| rank < r).unwrap_or(true) {
                        best = Some((rank, i));
                    }
                }
            }
            let Some((_, i)) = best else { break };
            let merged = format!("{}{}", parts[i], parts[i + 1]);
            parts.splice(i..=i + 1, [merged]);
        }
        parts
    }

    /// Encode text to token ids (unknown fragments fall back per-character).
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut ids = Vec::new();
        for word in Self::pretokenize(text) {
            let mapped: String = word
                .bytes()
                .map(|b| self.byte_to_unicode[b as usize])
                .collect();
            for piece in self.bpe_word(&mapped) {
                if let Some(&id) = self.token_ids.get(&piece) {
                    ids.push(id);
                } else {
                    for c in piece.chars() {
                        if let Some(&id) = self.token_ids.get(c.to_string().as_str()) {
                            ids.push(id);
                        }
                    }
                }
            }
        }
        ids
    }

    /// Decode token ids back to text (byte-level exact for known ids).
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let Some(token) = self.tokens.get(id) else {
                continue;
            };
            for c in token.chars() {
                match self.unicode_to_byte.get(&c) {
                    Some(&b) => bytes.push(b),
                    None => bytes.extend_from_slice(c.to_string().as_bytes()),
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny handmade vocab: byte singles + a few merges, GPT-2 style
    /// (`Ġ` = U+0120 marks a leading space, i.e. byte 0x20 mapped).
    fn sample_encoder() -> Gpt2BpeEncoder {
        let table = byte_to_unicode_table();
        let mut tokens: Vec<String> = (0..256).map(|b| table[b].to_string()).collect();
        tokens.push("he".into()); // 256
        tokens.push("hel".into()); // 257
        tokens.push("hello".into()); // 258
        tokens.push("Ġhello".into()); // 259 (" hello")
        let merges = vec![
            "h e".to_string(),
            "he l".to_string(),
            "hel lo".to_string(),
            "l o".to_string(),
            "Ġ hello".to_string(),
        ];
        Gpt2BpeEncoder::from_vocab(tokens, &merges).unwrap()
    }

    #[test]
    fn byte_table_is_a_bijection() {
        let table = byte_to_unicode_table();
        let unique: std::collections::HashSet<char> = table.iter().copied().collect();
        assert_eq!(unique.len(), 256);
        assert_eq!(table[b'a' as usize], 'a');
        assert_eq!(table[b' ' as usize], '\u{120}'); // Ġ
        assert_eq!(table[0x0A], '\u{10A}'); // newline remaps
    }

    #[test]
    fn encodes_with_merges_and_space_marker() {
        let enc = sample_encoder();
        let ids = enc.encode("hello hello");
        assert_eq!(ids[0], 258, "first word merges to 'hello'");
        assert!(ids.contains(&259), "second word uses the Ġhello token: {ids:?}");
    }

    #[test]
    fn decode_roundtrips_bytes_exactly() {
        let enc = sample_encoder();
        for text in ["hello hello", "hello world!", "tabs\tand\nnewlines", "héllo"] {
            let ids = enc.encode(text);
            assert_eq!(enc.decode(&ids), text, "{text:?}");
        }
    }

    #[test]
    fn unknown_vocab_falls_back_to_bytes() {
        let table = byte_to_unicode_table();
        let tokens: Vec<String> = (0..256).map(|b| table[b].to_string()).collect();
        let enc = Gpt2BpeEncoder::from_vocab(tokens, &[]).unwrap();
        let ids = enc.encode("xy z");
        assert_eq!(ids.len(), 4);
        assert_eq!(enc.decode(&ids), "xy z");
    }

    #[test]
    fn empty_vocab_and_bad_merge_error() {
        assert!(Gpt2BpeEncoder::from_vocab(vec![], &[]).is_err());
        let e = Gpt2BpeEncoder::from_vocab(vec!["a".into()], &["nospace".to_string()]);
        assert!(e.is_err());
    }

    #[test]
    fn digits_and_punctuation_split_from_letters() {
        let words = Gpt2BpeEncoder::pretokenize("ab12,cd");
        assert_eq!(words, vec!["ab", "12", ",", "cd"]);
        let words = Gpt2BpeEncoder::pretokenize(" leading space");
        assert_eq!(words, vec![" leading", " space"]);
        let words = Gpt2BpeEncoder::pretokenize("a  b");
        assert_eq!(words, vec!["a", " ", " b"]);
        let words = Gpt2BpeEncoder::pretokenize("trail ");
        assert_eq!(words, vec!["trail", " "]);
    }
}
