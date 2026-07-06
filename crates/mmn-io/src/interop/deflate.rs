//! From-scratch DEFLATE (RFC 1951) compressor: fixed-Huffman blocks with
//! greedy hash-chain LZ77 matching. Pairs with [`super::inflate`] and any
//! standard zlib/zip consumer.

const MIN_MATCH: usize = 3;
const MAX_MATCH: usize = 258;
const WINDOW: usize = 32 * 1024;
const MAX_CHAIN: usize = 32;

/// Length code table: (code, extra_bits, base) for lengths 3..=258.
const LENGTH_CODES: [(u16, u8, u16); 29] = [
    (257, 0, 3),
    (258, 0, 4),
    (259, 0, 5),
    (260, 0, 6),
    (261, 0, 7),
    (262, 0, 8),
    (263, 0, 9),
    (264, 0, 10),
    (265, 1, 11),
    (266, 1, 13),
    (267, 1, 15),
    (268, 1, 17),
    (269, 2, 19),
    (270, 2, 23),
    (271, 2, 27),
    (272, 2, 31),
    (273, 3, 35),
    (274, 3, 43),
    (275, 3, 51),
    (276, 3, 59),
    (277, 4, 67),
    (278, 4, 83),
    (279, 4, 99),
    (280, 4, 115),
    (281, 5, 131),
    (282, 5, 163),
    (283, 5, 195),
    (284, 5, 227),
    (285, 0, 258),
];

/// Distance code table: (code, extra_bits, base) for distances 1..=32768.
const DIST_CODES: [(u16, u8, u16); 30] = [
    (0, 0, 1),
    (1, 0, 2),
    (2, 0, 3),
    (3, 0, 4),
    (4, 1, 5),
    (5, 1, 7),
    (6, 2, 9),
    (7, 2, 13),
    (8, 3, 17),
    (9, 3, 25),
    (10, 4, 33),
    (11, 4, 49),
    (12, 5, 65),
    (13, 5, 97),
    (14, 6, 129),
    (15, 6, 193),
    (16, 7, 257),
    (17, 7, 385),
    (18, 8, 513),
    (19, 8, 769),
    (20, 9, 1025),
    (21, 9, 1537),
    (22, 10, 2049),
    (23, 10, 3073),
    (24, 11, 4097),
    (25, 11, 6145),
    (26, 12, 8193),
    (27, 12, 12289),
    (28, 13, 16385),
    (29, 13, 24577),
];

/// LSB-first bit writer (DEFLATE bit packing).
struct BitWriter {
    out: Vec<u8>,
    bit_buf: u32,
    bit_count: u32,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            out: Vec::new(),
            bit_buf: 0,
            bit_count: 0,
        }
    }

    /// Write `n` bits, LSB first (extra bits, block headers).
    fn bits(&mut self, value: u32, n: u32) {
        self.bit_buf |= value << self.bit_count;
        self.bit_count += n;
        while self.bit_count >= 8 {
            self.out.push((self.bit_buf & 0xFF) as u8);
            self.bit_buf >>= 8;
            self.bit_count -= 8;
        }
    }

    /// Write a Huffman code MSB-first (as DEFLATE requires).
    fn code(&mut self, code: u32, len: u32) {
        let mut reversed = 0u32;
        for i in 0..len {
            reversed |= ((code >> i) & 1) << (len - 1 - i);
        }
        self.bits(reversed, len);
    }

    fn finish(mut self) -> Vec<u8> {
        if self.bit_count > 0 {
            self.out.push((self.bit_buf & 0xFF) as u8);
        }
        self.out
    }
}

/// Fixed-table Huffman code for one literal/length symbol.
fn fixed_litlen_code(symbol: u16) -> (u32, u32) {
    match symbol {
        0..=143 => (0x30 + symbol as u32, 8),
        144..=255 => (0x190 + (symbol - 144) as u32, 9),
        256..=279 => ((symbol - 256) as u32, 7),
        _ => (0xC0 + (symbol - 280) as u32, 8),
    }
}

fn hash3(data: &[u8], pos: usize) -> usize {
    let a = data[pos] as usize;
    let b = data[pos + 1] as usize;
    let c = data[pos + 2] as usize;
    (a.wrapping_mul(506_832_829) ^ b.wrapping_mul(2_654_435_761) ^ c.wrapping_mul(40_503))
        & (HASH_SIZE - 1)
}

const HASH_SIZE: usize = 1 << 15;

fn emit_length(w: &mut BitWriter, length: usize) {
    let entry = LENGTH_CODES
        .iter()
        .rev()
        .find(|(_, _, base)| length >= *base as usize)
        .expect("length in range");
    let (code, extra, base) = *entry;
    let (huff, bits) = fixed_litlen_code(code);
    w.code(huff, bits);
    if extra > 0 {
        w.bits((length - base as usize) as u32, extra as u32);
    }
}

fn emit_distance(w: &mut BitWriter, distance: usize) {
    let entry = DIST_CODES
        .iter()
        .rev()
        .find(|(_, _, base)| distance >= *base as usize)
        .expect("distance in range");
    let (code, extra, base) = *entry;
    w.code(code as u32, 5);
    if extra > 0 {
        w.bits((distance - base as usize) as u32, extra as u32);
    }
}

/// Compress `data` into a single fixed-Huffman DEFLATE stream.
pub fn deflate(data: &[u8]) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.bits(1, 1); // BFINAL
    w.bits(1, 2); // BTYPE = 01 fixed Huffman
    let mut head = vec![usize::MAX; HASH_SIZE];
    let mut prev = vec![usize::MAX; data.len()];
    let mut pos = 0usize;
    while pos < data.len() {
        let mut best_len = 0usize;
        let mut best_dist = 0usize;
        if pos + MIN_MATCH <= data.len() {
            let h = hash3(data, pos);
            let old_head = head[h];
            let mut candidate = old_head;
            let mut chain = 0usize;
            while candidate != usize::MAX
                && candidate + WINDOW > pos
                && chain < MAX_CHAIN
            {
                let limit = (data.len() - pos).min(MAX_MATCH);
                let mut len = 0usize;
                while len < limit && data[candidate + len] == data[pos + len] {
                    len += 1;
                }
                if len > best_len {
                    best_len = len;
                    best_dist = pos - candidate;
                    if len == limit {
                        break;
                    }
                }
                candidate = prev[candidate];
                chain += 1;
            }
            prev[pos] = old_head;
            head[h] = pos;
        }
        if best_len >= MIN_MATCH {
            emit_length(&mut w, best_len);
            emit_distance(&mut w, best_dist);
            // Insert hash entries for skipped positions so later matches see them.
            let end = pos + best_len;
            let mut p = pos + 1;
            while p < end && p + MIN_MATCH <= data.len() {
                let h = hash3(data, p);
                prev[p] = head[h];
                head[h] = p;
                p += 1;
            }
            pos = end;
        } else {
            let (huff, bits) = fixed_litlen_code(data[pos] as u16);
            w.code(huff, bits);
            pos += 1;
        }
    }
    let (eob, eob_bits) = fixed_litlen_code(256);
    w.code(eob, eob_bits);
    w.finish()
}

#[cfg(test)]
mod tests {
    use super::super::inflate::inflate;
    use super::*;

    #[test]
    fn roundtrip_through_own_inflate() {
        for data in [
            b"".to_vec(),
            b"a".to_vec(),
            b"hello hello hello hello hello".to_vec(),
            (0..2000).map(|i| (i * 7 % 256) as u8).collect(),
            vec![0u8; 100_000],
            b"the quick brown fox jumps over the lazy dog. ".repeat(100).to_vec(),
        ] {
            let packed = deflate(&data);
            let back = inflate(&packed).unwrap();
            assert_eq!(back, data, "len {}", data.len());
        }
    }

    #[test]
    fn repetitive_data_compresses_well() {
        let data = b"abcdefgh".repeat(1000);
        let packed = deflate(&data);
        assert!(
            packed.len() < data.len() / 10,
            "8000 bytes -> {} bytes",
            packed.len()
        );
    }

    #[test]
    fn random_like_data_still_roundtrips() {
        // Pseudo-random bytes (LCG) — incompressible but must stay correct.
        let mut state = 12345u64;
        let data: Vec<u8> = (0..50_000)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (state >> 33) as u8
            })
            .collect();
        let packed = deflate(&data);
        assert_eq!(inflate(&packed).unwrap(), data);
    }

    #[test]
    fn long_matches_cap_at_258() {
        let data = vec![7u8; 1000];
        let packed = deflate(&data);
        assert_eq!(inflate(&packed).unwrap(), data);
        assert!(packed.len() < 40);
    }
}
