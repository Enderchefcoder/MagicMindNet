//! From-scratch DEFLATE (RFC 1951) decompressor — no external compression crates.
//!
//! Supports all three block types (stored, fixed Huffman, dynamic Huffman) so
//! compressed ZIP entries (`np.savez_compressed`, compressed `.pt` archives)
//! can be read without linking zlib.

use mmn_core::MmnError;

const MAX_BITS: usize = 15;

/// Length-code base values for symbols 257..=285.
const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
/// Distance-code base values for symbols 0..=29.
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];
/// Order in which code-length code lengths are stored in dynamic blocks.
const CLEN_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

fn err(message: impl Into<String>) -> MmnError {
    MmnError::Other {
        message: message.into(),
    }
}

/// LSB-first bit reader over a byte slice.
struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u32,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    fn take_bit(&mut self) -> Result<u32, MmnError> {
        let byte = *self
            .data
            .get(self.byte_pos)
            .ok_or_else(|| err("deflate stream truncated"))?;
        let bit = (byte >> self.bit_pos) & 1;
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
        Ok(bit as u32)
    }

    fn take_bits(&mut self, n: u32) -> Result<u32, MmnError> {
        let mut out = 0u32;
        for i in 0..n {
            out |= self.take_bit()? << i;
        }
        Ok(out)
    }

    fn align_to_byte(&mut self) {
        if self.bit_pos != 0 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
    }

    fn take_bytes(&mut self, n: usize) -> Result<&'a [u8], MmnError> {
        let end = self
            .byte_pos
            .checked_add(n)
            .filter(|&e| e <= self.data.len())
            .ok_or_else(|| err("deflate stored block truncated"))?;
        let slice = &self.data[self.byte_pos..end];
        self.byte_pos = end;
        Ok(slice)
    }
}

/// Canonical Huffman decoding table (counts-per-length + length-ordered symbols).
struct Huffman {
    counts: [u16; MAX_BITS + 1],
    symbols: Vec<u16>,
}

impl Huffman {
    fn from_lengths(lengths: &[u8]) -> Result<Self, MmnError> {
        let mut counts = [0u16; MAX_BITS + 1];
        for &len in lengths {
            if len as usize > MAX_BITS {
                return Err(err("deflate code length exceeds 15 bits"));
            }
            counts[len as usize] += 1;
        }
        counts[0] = 0;
        let mut offsets = [0u16; MAX_BITS + 2];
        for len in 1..=MAX_BITS {
            offsets[len + 1] = offsets[len] + counts[len];
        }
        let mut symbols = vec![0u16; offsets[MAX_BITS + 1] as usize];
        for (symbol, &len) in lengths.iter().enumerate() {
            if len != 0 {
                symbols[offsets[len as usize] as usize] = symbol as u16;
                offsets[len as usize] += 1;
            }
        }
        Ok(Self { counts, symbols })
    }

    fn decode(&self, reader: &mut BitReader<'_>) -> Result<u16, MmnError> {
        let mut code = 0i32;
        let mut first = 0i32;
        let mut index = 0i32;
        for len in 1..=MAX_BITS {
            code |= reader.take_bit()? as i32;
            let count = self.counts[len] as i32;
            if code - first < count {
                return Ok(self.symbols[(index + (code - first)) as usize]);
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err(err("invalid deflate Huffman code"))
    }
}

fn fixed_tables() -> Result<(Huffman, Huffman), MmnError> {
    let mut litlen = [0u8; 288];
    for (i, slot) in litlen.iter_mut().enumerate() {
        *slot = match i {
            0..=143 => 8,
            144..=255 => 9,
            256..=279 => 7,
            _ => 8,
        };
    }
    let dist = [5u8; 30];
    Ok((Huffman::from_lengths(&litlen)?, Huffman::from_lengths(&dist)?))
}

fn dynamic_tables(reader: &mut BitReader<'_>) -> Result<(Huffman, Huffman), MmnError> {
    let hlit = reader.take_bits(5)? as usize + 257;
    let hdist = reader.take_bits(5)? as usize + 1;
    let hclen = reader.take_bits(4)? as usize + 4;
    if hlit > 286 || hdist > 30 {
        return Err(err("invalid deflate dynamic header counts"));
    }
    let mut clen_lengths = [0u8; 19];
    for &pos in CLEN_ORDER.iter().take(hclen) {
        clen_lengths[pos] = reader.take_bits(3)? as u8;
    }
    let clen_table = Huffman::from_lengths(&clen_lengths)?;
    let mut lengths = vec![0u8; hlit + hdist];
    let mut i = 0;
    while i < lengths.len() {
        let symbol = clen_table.decode(reader)?;
        match symbol {
            0..=15 => {
                lengths[i] = symbol as u8;
                i += 1;
            }
            16 => {
                let prev = *lengths
                    .get(i.wrapping_sub(1))
                    .filter(|_| i > 0)
                    .ok_or_else(|| err("deflate repeat with no previous length"))?;
                let repeat = reader.take_bits(2)? as usize + 3;
                for _ in 0..repeat {
                    if i >= lengths.len() {
                        return Err(err("deflate length repeat overflow"));
                    }
                    lengths[i] = prev;
                    i += 1;
                }
            }
            17 | 18 => {
                let repeat = if symbol == 17 {
                    reader.take_bits(3)? as usize + 3
                } else {
                    reader.take_bits(7)? as usize + 11
                };
                if i + repeat > lengths.len() {
                    return Err(err("deflate zero-run overflow"));
                }
                i += repeat;
            }
            _ => return Err(err("invalid deflate code-length symbol")),
        }
    }
    let litlen = Huffman::from_lengths(&lengths[..hlit])?;
    let dist = Huffman::from_lengths(&lengths[hlit..])?;
    Ok((litlen, dist))
}

fn inflate_block(
    reader: &mut BitReader<'_>,
    out: &mut Vec<u8>,
    litlen: &Huffman,
    dist: &Huffman,
) -> Result<(), MmnError> {
    loop {
        let symbol = litlen.decode(reader)?;
        match symbol {
            0..=255 => out.push(symbol as u8),
            256 => return Ok(()),
            257..=285 => {
                let idx = symbol as usize - 257;
                let length =
                    LENGTH_BASE[idx] as usize + reader.take_bits(LENGTH_EXTRA[idx] as u32)? as usize;
                let dist_symbol = dist.decode(reader)? as usize;
                if dist_symbol >= 30 {
                    return Err(err("invalid deflate distance symbol"));
                }
                let distance = DIST_BASE[dist_symbol] as usize
                    + reader.take_bits(DIST_EXTRA[dist_symbol] as u32)? as usize;
                if distance > out.len() {
                    return Err(err("deflate back-reference before output start"));
                }
                let start = out.len() - distance;
                for j in 0..length {
                    let byte = out[start + j];
                    out.push(byte);
                }
            }
            _ => return Err(err("invalid deflate literal/length symbol")),
        }
    }
}

/// Decompress a raw DEFLATE stream (RFC 1951, no zlib/gzip wrapper).
pub fn inflate(data: &[u8]) -> Result<Vec<u8>, MmnError> {
    let mut reader = BitReader::new(data);
    let mut out = Vec::with_capacity(data.len() * 3);
    loop {
        let is_final = reader.take_bit()? == 1;
        let block_type = reader.take_bits(2)?;
        match block_type {
            0 => {
                reader.align_to_byte();
                let header = reader.take_bytes(4)?;
                let len = u16::from_le_bytes([header[0], header[1]]);
                let nlen = u16::from_le_bytes([header[2], header[3]]);
                if len != !nlen {
                    return Err(err("deflate stored block LEN/NLEN mismatch"));
                }
                out.extend_from_slice(reader.take_bytes(len as usize)?);
            }
            1 => {
                let (litlen, dist) = fixed_tables()?;
                inflate_block(&mut reader, &mut out, &litlen, &dist)?;
            }
            2 => {
                let (litlen, dist) = dynamic_tables(&mut reader)?;
                inflate_block(&mut reader, &mut out, &litlen, &dist)?;
            }
            _ => return Err(err("invalid deflate block type 3")),
        }
        if is_final {
            return Ok(out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stored_block_roundtrip() {
        // Hand-built stored block: final=1, type=00, then LEN/NLEN + payload.
        let payload = b"magicmindnet";
        let mut stream = vec![0x01]; // BFINAL=1, BTYPE=00 (bits fill the rest of the byte)
        stream.extend_from_slice(&(payload.len() as u16).to_le_bytes());
        stream.extend_from_slice(&(!(payload.len() as u16)).to_le_bytes());
        stream.extend_from_slice(payload);
        assert_eq!(inflate(&stream).unwrap(), payload);
    }

    #[test]
    fn fixed_huffman_literals() {
        // `zlib.compressobj(1, wbits=-15)` output for b"abc" uses a fixed block.
        // 0x4b 0x4c 0x4a 0x06 0x00 == fixed-huffman "abc".
        let stream = [0x4b, 0x4c, 0x4a, 0x06, 0x00];
        assert_eq!(inflate(&stream).unwrap(), b"abc");
    }

    #[test]
    fn back_reference_repeats() {
        // zlib raw-deflate stream for b"aaaaaaaaaa" (literal 'a' + back-reference).
        let stream = [75, 76, 132, 1, 0];
        assert_eq!(inflate(&stream).unwrap(), b"a".repeat(10));
    }

    #[test]
    fn dynamic_huffman_block_decodes() {
        // zlib level-9 raw-deflate of a 1200-byte pseudo-random text repeated
        // three times; first byte confirms BTYPE=2 (dynamic Huffman).
        let stream: [u8; 302] = [
            237, 209, 9, 177, 228, 32, 0, 0, 81, 43, 40, 160, 194, 13, 114, 194, 112, 19, 32, 23,
            33, 131, 250, 253, 6, 214, 193, 180, 130, 174, 122, 61, 77, 99, 133, 127, 161, 97,
            245, 227, 22, 96, 119, 39, 23, 163, 66, 131, 70, 193, 105, 218, 71, 198, 11, 36, 17,
            212, 45, 55, 15, 85, 121, 189, 180, 202, 84, 42, 150, 142, 33, 126, 239, 125, 219,
            157, 186, 57, 125, 208, 101, 3, 3, 249, 73, 20, 124, 172, 84, 253, 25, 20, 98, 235,
            78, 98, 205, 173, 208, 245, 29, 26, 143, 28, 168, 169, 87, 220, 231, 164, 46, 163, 41,
            207, 184, 200, 19, 140, 111, 75, 110, 75, 173, 173, 20, 110, 199, 181, 38, 32, 94,
            213, 35, 51, 88, 206, 191, 60, 153, 166, 216, 138, 114, 120, 140, 95, 85, 250, 155,
            213, 182, 126, 211, 49, 94, 18, 2, 197, 132, 220, 46, 249, 231, 32, 153, 235, 202,
            223, 36, 52, 191, 221, 193, 223, 60, 154, 16, 236, 105, 101, 159, 173, 112, 58, 180,
            62, 201, 81, 6, 26, 175, 107, 190, 145, 242, 84, 178, 146, 225, 194, 183, 144, 109,
            121, 220, 196, 211, 229, 28, 117, 130, 56, 145, 145, 164, 140, 122, 245, 60, 46, 165,
            234, 163, 94, 108, 135, 253, 16, 32, 154, 129, 33, 7, 44, 138, 196, 153, 70, 219, 154,
            182, 68, 130, 52, 157, 115, 73, 188, 52, 123, 57, 63, 158, 33, 169, 45, 234, 140, 149,
            19, 49, 65, 216, 206, 15, 89, 80, 4, 97, 162, 110, 247, 197, 214, 59, 164, 55, 29, 17,
            55, 63, 105, 110, 121, 97, 243, 1, 101, 116, 247, 234, 71, 246, 159, 199, 207, 227,
            231, 241, 95, 143, 127,
        ];
        assert_eq!((stream[0] >> 1) & 3, 2, "vector must be a dynamic block");
        let out = inflate(&stream).unwrap();
        assert_eq!(out.len(), 1200);
        assert_eq!(&out[..400], &out[400..800]);
        assert_eq!(&out[..16], b"ujzde7gx.d5ncf0 ");
        assert_eq!(crate::interop::zip::crc32(&out), 0xe10f_8ee9);
    }

    #[test]
    fn truncated_stream_errors() {
        assert!(inflate(&[0x01, 0x05]).is_err());
        assert!(inflate(&[]).is_err());
    }

    #[test]
    fn invalid_block_type_errors() {
        // BFINAL=1, BTYPE=11 (invalid).
        assert!(inflate(&[0x07]).is_err());
    }
}
