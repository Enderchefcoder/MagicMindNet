//! From-scratch Zstandard decoder (RFC 8878) — the final compression gap:
//! zarr-python 3.x's default codec for both v2 and v3 stores, and Blosc's
//! zstd mode.
//!
//! Implements frame headers, raw/RLE/compressed blocks, Huffman-coded
//! literals (direct or FSE-compressed weight tables, 1- and 4-stream),
//! FSE-coded sequences (predefined / RLE / custom tables), the interleaved
//! backward bitstream, repeat-offset history, and sequence execution.
//! Content checksums are skipped, dictionaries are rejected. No zstd
//! library is linked.

use mmn_core::MmnError;

fn err(message: impl Into<String>) -> MmnError {
    MmnError::Other {
        message: message.into(),
    }
}

const MAGIC: u32 = 0xFD2F_B528;

// ---------------------------------------------------------------- bit IO

/// Forward little-endian bit reader (FSE table descriptions).
struct ForwardBits<'a> {
    bytes: &'a [u8],
    pos: usize, // in bits
}

impl<'a> ForwardBits<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn read(&mut self, n: usize) -> Result<u64, MmnError> {
        let mut value = 0u64;
        for i in 0..n {
            let bit_pos = self.pos + i;
            let byte = *self
                .bytes
                .get(bit_pos / 8)
                .ok_or_else(|| err("zstd FSE header truncated"))?;
            if byte & (1 << (bit_pos % 8)) != 0 {
                value |= 1 << i;
            }
        }
        self.pos += n;
        Ok(value)
    }

    fn rewind(&mut self, n: usize) {
        self.pos -= n;
    }

    fn bytes_consumed(&self) -> usize {
        self.pos.div_ceil(8)
    }
}

/// Backward bit reader: bits start below the sentinel bit of the last byte
/// and proceed toward the first byte, MSB-first within each byte.
struct BackwardBits<'a> {
    bytes: &'a [u8],
    /// Bits still unread (bit index counted from the stream start).
    remaining: usize,
}

impl<'a> BackwardBits<'a> {
    fn new(bytes: &'a [u8]) -> Result<Self, MmnError> {
        let last = *bytes.last().ok_or_else(|| err("zstd bitstream empty"))?;
        if last == 0 {
            return Err(err("zstd bitstream has no sentinel bit"));
        }
        let sentinel = 7 - last.leading_zeros() as usize;
        Ok(Self {
            bytes,
            remaining: (bytes.len() - 1) * 8 + sentinel,
        })
    }

    fn bit(&mut self) -> u64 {
        // Callers check `remaining` (reads past the start return zeros,
        // matching zstd's defined behavior for the final state updates).
        if self.remaining == 0 {
            return 0;
        }
        self.remaining -= 1;
        let byte = self.bytes[self.remaining / 8];
        ((byte >> (self.remaining % 8)) & 1) as u64
    }

    /// Read `n` bits; the first bit read is the most significant.
    fn read(&mut self, n: usize) -> u64 {
        let mut value = 0u64;
        for _ in 0..n {
            value = (value << 1) | self.bit();
        }
        value
    }
}

// ---------------------------------------------------------------- FSE

#[derive(Clone, Copy, Default)]
struct FseEntry {
    symbol: u16,
    nb_bits: u8,
    baseline: u16,
}

struct FseTable {
    accuracy_log: usize,
    entries: Vec<FseEntry>,
}

fn highest_bit(v: usize) -> usize {
    usize::BITS as usize - 1 - v.leading_zeros() as usize
}

/// Build a decoding table from normalized counts (RFC 8878 spread + fill).
fn build_fse_table(counts: &[i32], accuracy_log: usize) -> Result<FseTable, MmnError> {
    let table_size = 1usize << accuracy_log;
    let mut symbols = vec![0u16; table_size];
    let mut high_threshold = table_size - 1;
    // "Less than one" symbols occupy the table's tail slots.
    for (s, &c) in counts.iter().enumerate() {
        if c == -1 {
            symbols[high_threshold] = s as u16;
            high_threshold = high_threshold.wrapping_sub(1);
        }
    }
    let step = (table_size >> 1) + (table_size >> 3) + 3;
    let mask = table_size - 1;
    let mut position = 0usize;
    for (s, &c) in counts.iter().enumerate() {
        if c <= 0 {
            continue;
        }
        for _ in 0..c {
            symbols[position] = s as u16;
            position = (position + step) & mask;
            while position > high_threshold {
                position = (position + step) & mask;
            }
        }
    }
    if position != 0 {
        return Err(err("zstd FSE table spread did not return to zero"));
    }
    let mut next_state: Vec<u32> = counts
        .iter()
        .map(|&c| if c == -1 { 1 } else { c.max(0) as u32 })
        .collect();
    let mut entries = vec![FseEntry::default(); table_size];
    for (i, entry) in entries.iter_mut().enumerate() {
        let s = symbols[i] as usize;
        let state = next_state[s];
        next_state[s] += 1;
        let nb_bits = accuracy_log as u32 - highest_bit(state as usize) as u32;
        entry.symbol = s as u16;
        entry.nb_bits = nb_bits as u8;
        entry.baseline = ((state << nb_bits) as usize - table_size) as u16;
    }
    Ok(FseTable {
        accuracy_log,
        entries,
    })
}

/// Parse an FSE table description (forward bitstream) into counts.
fn parse_fse_description(
    bytes: &[u8],
    max_accuracy: usize,
    max_symbols: usize,
) -> Result<(FseTable, usize), MmnError> {
    let mut bits = ForwardBits::new(bytes);
    let accuracy_log = bits.read(4)? as usize + 5;
    if accuracy_log > max_accuracy {
        return Err(err(format!(
            "zstd FSE accuracy {accuracy_log} exceeds limit {max_accuracy}"
        )));
    }
    let table_size = 1i64 << accuracy_log;
    let mut remaining = table_size + 1;
    let mut counts: Vec<i32> = Vec::new();
    while remaining > 1 && counts.len() <= max_symbols {
        let nb = highest_bit(remaining as usize) + 1;
        let value = bits.read(nb)?;
        let lower_mask = (1u64 << (nb - 1)) - 1;
        let threshold = (1i64 << nb) - 1 - remaining;
        let decoded = if (value & lower_mask) < threshold as u64 {
            bits.rewind(1);
            value & lower_mask
        } else if value > lower_mask {
            value - threshold as u64
        } else {
            value
        };
        let proba = decoded as i64 - 1;
        counts.push(proba as i32);
        remaining -= proba.abs().max(if proba == -1 { 1 } else { proba.abs() });
        if proba == -1 {
            remaining -= 0; // already subtracted 1 via abs
        }
        if proba == 0 {
            loop {
                let repeat = bits.read(2)?;
                counts.extend(std::iter::repeat_n(0, repeat as usize));
                if repeat != 3 {
                    break;
                }
            }
        }
    }
    if counts.len() > max_symbols + 1 {
        return Err(err("zstd FSE description has too many symbols"));
    }
    let consumed = bits.bytes_consumed();
    Ok((build_fse_table(&counts, accuracy_log)?, consumed))
}

/// A one-symbol "RLE table" (nb_bits 0, always the same symbol).
fn rle_table(symbol: u16) -> FseTable {
    FseTable {
        accuracy_log: 0,
        entries: vec![FseEntry {
            symbol,
            nb_bits: 0,
            baseline: 0,
        }],
    }
}

// ---------------------------------------------------------------- Huffman

struct HuffmanTable {
    table_log: usize,
    /// 2^table_log entries of (symbol, code length).
    entries: Vec<(u8, u8)>,
}

/// Build the literals decoder from symbol weights.
fn huffman_from_weights(weights: &[u8]) -> Result<HuffmanTable, MmnError> {
    // The final symbol's weight is implicit: it completes the sum to a
    // power of two.
    let sum: u64 = weights
        .iter()
        .map(|&w| if w == 0 { 0 } else { 1u64 << (w - 1) })
        .sum();
    if sum == 0 {
        return Err(err("zstd Huffman weights sum to zero"));
    }
    let table_log = (highest_bit(sum as usize) + 1).max(1);
    if table_log > 11 {
        return Err(err("zstd Huffman table log exceeds 11"));
    }
    let total = 1u64 << table_log;
    let rest = total - sum;
    if rest == 0 || rest & (rest - 1) != 0 {
        return Err(err("zstd Huffman weights leave an invalid remainder"));
    }
    let last_weight = highest_bit(rest as usize) as u8 + 1;
    let mut all: Vec<u8> = weights.to_vec();
    all.push(last_weight);
    // Fill table: symbols in weight order (1 = longest code first),
    // each occupying 2^(weight-1) consecutive entries.
    let mut entries = vec![(0u8, 0u8); total as usize];
    let mut position = 0usize;
    for weight in 1..=table_log as u8 + 1 {
        for (symbol, &w) in all.iter().enumerate() {
            if w != weight {
                continue;
            }
            let nb_bits = table_log as u8 + 1 - w;
            let span = 1usize << (weight - 1);
            for entry in entries.iter_mut().skip(position).take(span) {
                *entry = (symbol as u8, nb_bits);
            }
            position += span;
        }
    }
    if position != total as usize {
        return Err(err("zstd Huffman table underfilled"));
    }
    Ok(HuffmanTable {
        table_log,
        entries,
    })
}

/// Parse a Huffman tree description; returns (table, bytes consumed).
fn parse_huffman(bytes: &[u8]) -> Result<(HuffmanTable, usize), MmnError> {
    let header = *bytes.first().ok_or_else(|| err("zstd Huffman header missing"))?;
    if header >= 128 {
        // Direct representation: n weights, 4 bits each.
        let n = header as usize - 127;
        let packed = n.div_ceil(2);
        let raw = bytes
            .get(1..1 + packed)
            .ok_or_else(|| err("zstd Huffman weights truncated"))?;
        let mut weights = Vec::with_capacity(n);
        for i in 0..n {
            let byte = raw[i / 2];
            weights.push(if i % 2 == 0 { byte >> 4 } else { byte & 0x0F });
        }
        Ok((huffman_from_weights(&weights)?, 1 + packed))
    } else {
        // FSE-compressed weights.
        let compressed = bytes
            .get(1..1 + header as usize)
            .ok_or_else(|| err("zstd Huffman FSE weights truncated"))?;
        let (table, desc_len) = parse_fse_description(compressed, 6, 255)?;
        let stream = &compressed[desc_len..];
        let mut bits = BackwardBits::new(stream)?;
        let mut states = [
            bits.read(table.accuracy_log) as usize,
            bits.read(table.accuracy_log) as usize,
        ];
        let mut weights: Vec<u8> = Vec::new();
        let mut current = 0usize;
        loop {
            let entry = table.entries[states[current]];
            weights.push(entry.symbol as u8);
            if weights.len() > 255 {
                return Err(err("zstd Huffman has too many weights"));
            }
            if (entry.nb_bits as usize) > bits.remaining {
                // Stream exhausted: flush the other state and stop.
                let other = table.entries[states[current ^ 1]];
                weights.push(other.symbol as u8);
                break;
            }
            states[current] = entry.baseline as usize + bits.read(entry.nb_bits as usize) as usize;
            current ^= 1;
        }
        Ok((huffman_from_weights(&weights)?, 1 + header as usize))
    }
}

/// Decode one Huffman-coded literal stream of known output size.
fn huffman_decode_stream(
    table: &HuffmanTable,
    stream: &[u8],
    out_len: usize,
) -> Result<Vec<u8>, MmnError> {
    let mut bits = BackwardBits::new(stream)?;
    let mut out = Vec::with_capacity(out_len);
    // Peek table_log bits (short final codes borrow zero padding).
    let mut window = bits.read(table.table_log) as usize;
    loop {
        let (symbol, nb_bits) = table.entries[window];
        out.push(symbol);
        if out.len() == out_len {
            break;
        }
        let consume = nb_bits as usize;
        window = ((window << consume) | bits.read(consume) as usize)
            & ((1 << table.table_log) - 1);
    }
    Ok(out)
}

// ---------------------------------------------------------------- codes

/// Literal-length code -> (baseline, extra bits).
fn ll_code_value(code: u16) -> Result<(usize, usize), MmnError> {
    Ok(match code {
        0..=15 => (code as usize, 0),
        16 => (16, 1),
        17 => (18, 1),
        18 => (20, 1),
        19 => (22, 1),
        20 => (24, 2),
        21 => (28, 2),
        22 => (32, 3),
        23 => (40, 3),
        24 => (48, 4),
        25 => (64, 6),
        26 => (128, 7),
        27 => (256, 8),
        28 => (512, 9),
        29 => (1024, 10),
        30 => (2048, 11),
        31 => (4096, 12),
        32 => (8192, 13),
        33 => (16384, 14),
        34 => (32768, 15),
        35 => (65536, 16),
        other => return Err(err(format!("zstd literal-length code {other} invalid"))),
    })
}

/// Match-length code -> (baseline, extra bits).
fn ml_code_value(code: u16) -> Result<(usize, usize), MmnError> {
    Ok(match code {
        0..=31 => (code as usize + 3, 0),
        32 => (35, 1),
        33 => (37, 1),
        34 => (39, 1),
        35 => (41, 1),
        36 => (43, 2),
        37 => (47, 2),
        38 => (51, 3),
        39 => (59, 3),
        40 => (67, 4),
        41 => (83, 4),
        42 => (99, 5),
        43 => (131, 7),
        44 => (259, 8),
        45 => (515, 9),
        46 => (1027, 10),
        47 => (2051, 11),
        48 => (4099, 12),
        49 => (8195, 13),
        50 => (16387, 14),
        51 => (32771, 15),
        52 => (65539, 16),
        other => return Err(err(format!("zstd match-length code {other} invalid"))),
    })
}

const LL_DEFAULT: [i32; 36] = [
    4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1,
    1, 1, 1, -1, -1, -1, -1,
];
const OF_DEFAULT: [i32; 29] = [
    1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1,
    -1,
];
const ML_DEFAULT: [i32; 53] = [
    1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1,
];

// ---------------------------------------------------------------- frame

struct SequenceTables {
    ll: FseTable,
    of: FseTable,
    ml: FseTable,
}

struct DecodeContext {
    rep: [usize; 3],
    tables: Option<SequenceTables>,
}

/// Parse the literals section; returns (literals, bytes consumed).
fn decode_literals(
    block: &[u8],
    huffman: &mut Option<HuffmanTable>,
) -> Result<(Vec<u8>, usize), MmnError> {
    let byte0 = *block.first().ok_or_else(|| err("zstd literals header missing"))?;
    let literals_type = byte0 & 0x3;
    let size_format = (byte0 >> 2) & 0x3;
    match literals_type {
        0 | 1 => {
            // Raw / RLE.
            let (regen, header) = match size_format {
                0 | 2 => ((byte0 >> 3) as usize, 1usize),
                1 => {
                    let b1 = *block.get(1).ok_or_else(|| err("zstd literals header truncated"))?;
                    (((byte0 >> 4) as usize) | ((b1 as usize) << 4), 2)
                }
                _ => {
                    let b1 = *block.get(1).ok_or_else(|| err("zstd literals header truncated"))?;
                    let b2 = *block.get(2).ok_or_else(|| err("zstd literals header truncated"))?;
                    (
                        ((byte0 >> 4) as usize) | ((b1 as usize) << 4) | ((b2 as usize) << 12),
                        3,
                    )
                }
            };
            if literals_type == 0 {
                let data = block
                    .get(header..header + regen)
                    .ok_or_else(|| err("zstd raw literals truncated"))?;
                Ok((data.to_vec(), header + regen))
            } else {
                let byte = *block
                    .get(header)
                    .ok_or_else(|| err("zstd RLE literal missing"))?;
                Ok((vec![byte; regen], header + 1))
            }
        }
        _ => {
            // Compressed (2) / treeless (3).
            let b1 = *block.get(1).ok_or_else(|| err("zstd literals header truncated"))?;
            let b2 = *block.get(2).ok_or_else(|| err("zstd literals header truncated"))?;
            let (regen, compressed, header, four_streams) = match size_format {
                0 | 1 => {
                    let regen = ((byte0 >> 4) as usize) | (((b1 & 0x3F) as usize) << 4);
                    let comp = ((b1 >> 6) as usize) | ((b2 as usize) << 2);
                    (regen, comp, 3usize, size_format == 1)
                }
                2 => {
                    let b3 = *block.get(3).ok_or_else(|| err("zstd literals header truncated"))?;
                    let regen = ((byte0 >> 4) as usize)
                        | ((b1 as usize) << 4)
                        | (((b2 & 0x3) as usize) << 12);
                    let comp = ((b2 >> 2) as usize) | ((b3 as usize) << 6);
                    (regen, comp, 4, true)
                }
                _ => {
                    let b3 = *block.get(3).ok_or_else(|| err("zstd literals header truncated"))?;
                    let b4 = *block.get(4).ok_or_else(|| err("zstd literals header truncated"))?;
                    let regen = ((byte0 >> 4) as usize)
                        | ((b1 as usize) << 4)
                        | (((b2 & 0x3F) as usize) << 12);
                    let comp =
                        ((b2 >> 6) as usize) | ((b3 as usize) << 2) | ((b4 as usize) << 10);
                    (regen, comp, 5, true)
                }
            };
            let payload = block
                .get(header..header + compressed)
                .ok_or_else(|| err("zstd compressed literals truncated"))?;
            let (table_ref, tree_bytes) = if literals_type == 2 {
                let (table, consumed) = parse_huffman(payload)?;
                *huffman = Some(table);
                (huffman.as_ref().unwrap(), consumed)
            } else {
                (
                    huffman
                        .as_ref()
                        .ok_or_else(|| err("zstd treeless literals without a previous tree"))?,
                    0,
                )
            };
            let streams = &payload[tree_bytes..];
            let literals = if !four_streams {
                huffman_decode_stream(table_ref, streams, regen)?
            } else {
                if streams.len() < 6 {
                    return Err(err("zstd 4-stream jump table truncated"));
                }
                let s1 = u16::from_le_bytes([streams[0], streams[1]]) as usize;
                let s2 = u16::from_le_bytes([streams[2], streams[3]]) as usize;
                let s3 = u16::from_le_bytes([streams[4], streams[5]]) as usize;
                let body = &streams[6..];
                let part = regen.div_ceil(4);
                let sizes = [part, part, part, regen - 3 * part];
                let mut literals = Vec::with_capacity(regen);
                let mut offset = 0usize;
                for (i, &stream_len) in [s1, s2, s3, body.len() - s1 - s2 - s3]
                    .iter()
                    .enumerate()
                {
                    let stream = body
                        .get(offset..offset + stream_len)
                        .ok_or_else(|| err("zstd literal stream truncated"))?;
                    literals.extend(huffman_decode_stream(table_ref, stream, sizes[i])?);
                    offset += stream_len;
                }
                literals
            };
            Ok((literals, header + compressed))
        }
    }
}

/// Decode a compressed block's sequences + execute them.
fn decode_compressed_block(
    block: &[u8],
    out: &mut Vec<u8>,
    ctx: &mut DecodeContext,
    huffman: &mut Option<HuffmanTable>,
) -> Result<(), MmnError> {
    let (literals, consumed) = decode_literals(block, huffman)?;
    let rest = &block[consumed..];
    let byte0 = *rest.first().ok_or_else(|| err("zstd sequences header missing"))?;
    let (num_sequences, mut pos) = if byte0 == 0 {
        (0usize, 1usize)
    } else if byte0 < 128 {
        (byte0 as usize, 1)
    } else if byte0 < 255 {
        let b1 = *rest.get(1).ok_or_else(|| err("zstd sequence count truncated"))?;
        ((((byte0 as usize) - 128) << 8) + b1 as usize, 2)
    } else {
        let b1 = *rest.get(1).ok_or_else(|| err("zstd sequence count truncated"))?;
        let b2 = *rest.get(2).ok_or_else(|| err("zstd sequence count truncated"))?;
        (b1 as usize + ((b2 as usize) << 8) + 0x7F00, 3)
    };
    if num_sequences == 0 {
        out.extend_from_slice(&literals);
        return Ok(());
    }
    let modes = *rest.get(pos).ok_or_else(|| err("zstd sequence modes missing"))?;
    pos += 1;
    let load_table = |mode: u8,
                          pos: &mut usize,
                          default: &[i32],
                          default_log: usize,
                          max_log: usize,
                          max_symbols: usize,
                          previous: Option<FseTable>|
     -> Result<FseTable, MmnError> {
        match mode {
            0 => build_fse_table(default, default_log),
            1 => {
                let symbol = *rest.get(*pos).ok_or_else(|| err("zstd RLE symbol missing"))?;
                *pos += 1;
                Ok(rle_table(symbol as u16))
            }
            2 => {
                let (table, consumed) =
                    parse_fse_description(&rest[*pos..], max_log, max_symbols)?;
                *pos += consumed;
                Ok(table)
            }
            _ => previous.ok_or_else(|| err("zstd repeat mode without a previous table")),
        }
    };
    let previous = ctx.tables.take();
    let (prev_ll, prev_of, prev_ml) = match previous {
        Some(t) => (Some(t.ll), Some(t.of), Some(t.ml)),
        None => (None, None, None),
    };
    let ll = load_table((modes >> 6) & 3, &mut pos, &LL_DEFAULT, 6, 9, 35, prev_ll)?;
    let of = load_table((modes >> 4) & 3, &mut pos, &OF_DEFAULT, 5, 8, 31, prev_of)?;
    let ml = load_table((modes >> 2) & 3, &mut pos, &ML_DEFAULT, 6, 9, 52, prev_ml)?;
    let mut bits = BackwardBits::new(&rest[pos..])?;
    let mut ll_state = bits.read(ll.accuracy_log) as usize;
    let mut of_state = bits.read(of.accuracy_log) as usize;
    let mut ml_state = bits.read(ml.accuracy_log) as usize;
    let mut literal_pos = 0usize;
    for seq in 0..num_sequences {
        let of_code = of.entries[of_state].symbol as usize;
        if of_code > 31 {
            return Err(err("zstd offset code too large"));
        }
        let of_value = (1usize << of_code) + bits.read(of_code) as usize;
        let (ml_base, ml_extra) = ml_code_value(ml.entries[ml_state].symbol)?;
        let match_len = ml_base + bits.read(ml_extra) as usize;
        let (ll_base, ll_extra) = ll_code_value(ll.entries[ll_state].symbol)?;
        let lit_len = ll_base + bits.read(ll_extra) as usize;
        // Repeat-offset resolution.
        let offset = if of_value > 3 {
            let offset = of_value - 3;
            ctx.rep = [offset, ctx.rep[0], ctx.rep[1]];
            offset
        } else {
            let idx = of_value - 1 + usize::from(lit_len == 0);
            let offset = match idx {
                0 => ctx.rep[0],
                1 => ctx.rep[1],
                2 => ctx.rep[2],
                _ => ctx.rep[0].checked_sub(1).ok_or_else(|| err("zstd repeat offset underflow"))?,
            };
            if offset == 0 {
                return Err(err("zstd repeat offset is zero"));
            }
            if idx == 1 {
                ctx.rep = [offset, ctx.rep[0], ctx.rep[2]];
            } else if idx >= 2 {
                ctx.rep = [offset, ctx.rep[0], ctx.rep[1]];
            }
            offset
        };
        let lits = literals
            .get(literal_pos..literal_pos + lit_len)
            .ok_or_else(|| err("zstd sequence literals overrun"))?;
        out.extend_from_slice(lits);
        literal_pos += lit_len;
        if offset > out.len() {
            return Err(err("zstd match offset beyond output"));
        }
        let start = out.len() - offset;
        for k in 0..match_len {
            let byte = out[start + k];
            out.push(byte);
        }
        if seq + 1 < num_sequences {
            // State updates in LL, ML, OF order.
            let e = ll.entries[ll_state];
            ll_state = e.baseline as usize + bits.read(e.nb_bits as usize) as usize;
            let e = ml.entries[ml_state];
            ml_state = e.baseline as usize + bits.read(e.nb_bits as usize) as usize;
            let e = of.entries[of_state];
            of_state = e.baseline as usize + bits.read(e.nb_bits as usize) as usize;
        }
    }
    out.extend_from_slice(&literals[literal_pos..]);
    ctx.tables = Some(SequenceTables { ll, of, ml });
    Ok(())
}

/// Decompress one zstd frame (skippable frames skipped, checksum ignored).
pub fn zstd_decompress(bytes: &[u8]) -> Result<Vec<u8>, MmnError> {
    let mut pos = 0usize;
    let mut out: Vec<u8> = Vec::new();
    while pos < bytes.len() {
        let magic = u32::from_le_bytes(
            bytes
                .get(pos..pos + 4)
                .ok_or_else(|| err("zstd magic truncated"))?
                .try_into()
                .unwrap(),
        );
        pos += 4;
        if (0x184D_2A50..=0x184D_2A5F).contains(&magic) {
            // Skippable frame.
            let size = u32::from_le_bytes(
                bytes
                    .get(pos..pos + 4)
                    .ok_or_else(|| err("zstd skippable frame truncated"))?
                    .try_into()
                    .unwrap(),
            ) as usize;
            pos += 4 + size;
            continue;
        }
        if magic != MAGIC {
            return Err(err("not a zstd frame (bad magic)"));
        }
        let descriptor = *bytes.get(pos).ok_or_else(|| err("zstd frame header missing"))?;
        pos += 1;
        let fcs_flag = descriptor >> 6;
        let single_segment = descriptor & 0x20 != 0;
        let checksum = descriptor & 0x04 != 0;
        let dict_flag = descriptor & 0x03;
        if descriptor & 0x08 != 0 {
            return Err(err("zstd reserved frame-header bit set"));
        }
        if !single_segment {
            pos += 1; // window descriptor
        }
        pos += match dict_flag {
            0 => 0,
            1 => 1,
            2 => 2,
            _ => 4,
        };
        if dict_flag != 0 {
            return Err(err("zstd dictionaries are not supported"));
        }
        pos += match fcs_flag {
            0 => usize::from(single_segment),
            1 => 2,
            2 => 4,
            _ => 8,
        };
        let mut ctx = DecodeContext {
            rep: [1, 4, 8],
            tables: None,
        };
        let mut huffman: Option<HuffmanTable> = None;
        loop {
            let header = bytes
                .get(pos..pos + 3)
                .ok_or_else(|| err("zstd block header truncated"))?;
            let word = header[0] as usize | (header[1] as usize) << 8 | (header[2] as usize) << 16;
            pos += 3;
            let last = word & 1 != 0;
            let block_type = (word >> 1) & 3;
            let block_size = word >> 3;
            match block_type {
                0 => {
                    let data = bytes
                        .get(pos..pos + block_size)
                        .ok_or_else(|| err("zstd raw block truncated"))?;
                    out.extend_from_slice(data);
                    pos += block_size;
                }
                1 => {
                    let byte = *bytes.get(pos).ok_or_else(|| err("zstd RLE block truncated"))?;
                    out.extend(std::iter::repeat_n(byte, block_size));
                    pos += 1;
                }
                2 => {
                    let data = bytes
                        .get(pos..pos + block_size)
                        .ok_or_else(|| err("zstd compressed block truncated"))?;
                    decode_compressed_block(data, &mut out, &mut ctx, &mut huffman)?;
                    pos += block_size;
                }
                _ => return Err(err("zstd reserved block type")),
            }
            if last {
                break;
            }
        }
        if checksum {
            pos += 4; // xxhash64 low bits — not verified
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture(name: &str) -> Vec<u8> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/zstd")
            .join(name);
        std::fs::read(&path).unwrap_or_else(|e| panic!("fixture {name} missing: {e}"))
    }

    fn expected_pattern(n: usize) -> Vec<u8> {
        (0..n)
            .flat_map(|i| (((i % 97) as f32) / 7.0).to_le_bytes())
            .collect()
    }

    #[test]
    fn repetitive_f32_frame_decodes() {
        let out = zstd_decompress(&fixture("pattern_f32.zst")).unwrap();
        assert_eq!(out, expected_pattern(10_000));
    }

    #[test]
    fn large_multiblock_frame_decodes() {
        let out = zstd_decompress(&fixture("large_f32.zst")).unwrap();
        assert_eq!(out, expected_pattern(200_000));
    }

    #[test]
    fn text_frame_decodes() {
        let out = zstd_decompress(&fixture("text.zst")).unwrap();
        let expected: Vec<u8> = b"the quick brown fox jumps over the lazy dog. "
            .repeat(400);
        assert_eq!(out, expected);
    }

    #[test]
    fn random_bytes_frame_decodes() {
        // Mostly-incompressible input exercises raw blocks/literals.
        let out = zstd_decompress(&fixture("random.zst")).unwrap();
        assert_eq!(out.len(), 4096);
        assert_eq!(out[0], 43);
        assert_eq!(out[4095], 227);
    }

    #[test]
    fn high_level_frame_decodes() {
        let out = zstd_decompress(&fixture("pattern_f32_lvl19.zst")).unwrap();
        assert_eq!(out, expected_pattern(10_000));
    }

    #[test]
    fn corrupt_frames_error() {
        assert!(zstd_decompress(b"nope").is_err());
        assert!(zstd_decompress(&[0x28, 0xB5, 0x2F, 0xFD]).is_err());
        let mut frame = fixture("pattern_f32.zst");
        let len = frame.len();
        frame.truncate(len - 8);
        assert!(zstd_decompress(&frame).is_err());
    }
}
