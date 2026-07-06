//! From-scratch Blosc1 frame decoder — Zarr's default compressor container.
//!
//! Parses the 16-byte header (flags carry byte-shuffle / memcpy bits and
//! the codec id), per-block start offsets, split streams (byte-shuffled
//! LZ4/BloscLZ blocks store one compressed stream per byte lane), raw
//! stream passthrough, and undoes the byte shuffle. LZ4 decodes through
//! the from-scratch [`super::lz4`] block decoder and zlib streams through
//! the HDF5 zlib path. Bit-shuffle / zstd / snappy / blosclz report clear
//! errors with a re-encode hint.

use super::hdf5::{undo_deflate, undo_shuffle};
use super::lz4::lz4_decompress_block;
use mmn_core::MmnError;

fn err(message: impl Into<String>) -> MmnError {
    MmnError::Other {
        message: message.into(),
    }
}

const FLAG_BYTE_SHUFFLE: u8 = 0x1;
const FLAG_MEMCPY: u8 = 0x2;
const FLAG_BIT_SHUFFLE: u8 = 0x4;

fn u32_at(bytes: &[u8], pos: usize) -> Result<u32, MmnError> {
    bytes
        .get(pos..pos + 4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .ok_or_else(|| err("blosc frame truncated"))
}

/// Decode one compressed stream (`cbytes == expected` means stored raw).
fn decode_stream(
    codec: u8,
    src: &[u8],
    expected: usize,
) -> Result<Vec<u8>, MmnError> {
    if src.len() == expected {
        return Ok(src.to_vec());
    }
    match codec {
        1 => lz4_decompress_block(src, expected), // lz4 / lz4hc
        3 => {
            let out = undo_deflate(src)?;
            if out.len() != expected {
                return Err(err("blosc zlib stream length mismatch"));
            }
            Ok(out)
        }
        0 => Err(err(
            "blosc blosclz codec not supported (re-encode with cname='lz4' or 'zlib')",
        )),
        2 => Err(err("blosc snappy codec not supported")),
        4 => Err(err(
            "blosc zstd codec not supported (re-encode with cname='lz4' or 'zlib')",
        )),
        other => Err(err(format!("blosc codec id {other} unknown"))),
    }
}

/// Decompress a Blosc1 frame into its original bytes.
pub fn blosc_decompress(frame: &[u8]) -> Result<Vec<u8>, MmnError> {
    if frame.len() < 16 {
        return Err(err("blosc frame shorter than its 16-byte header"));
    }
    let flags = frame[2];
    let typesize = frame[3] as usize;
    let nbytes = u32_at(frame, 4)? as usize;
    let blocksize = u32_at(frame, 8)? as usize;
    let cbytes = u32_at(frame, 12)? as usize;
    if cbytes != frame.len() {
        return Err(err(format!(
            "blosc frame length {} != header cbytes {cbytes}",
            frame.len()
        )));
    }
    if flags & FLAG_BIT_SHUFFLE != 0 {
        return Err(err(
            "blosc bit-shuffle is not supported (re-encode with shuffle=Blosc.SHUFFLE)",
        ));
    }
    if nbytes == 0 {
        return Ok(Vec::new());
    }
    if flags & FLAG_MEMCPY != 0 {
        let data = frame
            .get(16..16 + nbytes)
            .ok_or_else(|| err("blosc memcpy payload truncated"))?;
        return Ok(data.to_vec());
    }
    if blocksize == 0 {
        return Err(err("blosc blocksize is zero"));
    }
    let codec = (flags >> 5) & 0x7;
    let byte_shuffle = flags & FLAG_BYTE_SHUFFLE != 0 && typesize > 1;
    let nblocks = nbytes.div_ceil(blocksize);
    // Decode one block assuming `nstreams` split streams.
    let decode_block = |bstart: usize, bsize: usize, nstreams: usize| -> Result<Vec<u8>, MmnError> {
        let neblock = bsize / nstreams;
        let mut pos = bstart;
        let mut block: Vec<u8> = Vec::with_capacity(bsize);
        for _ in 0..nstreams {
            let stream_cbytes = u32_at(frame, pos)? as usize;
            pos += 4;
            let src = frame
                .get(pos..pos + stream_cbytes)
                .ok_or_else(|| err("blosc stream truncated"))?;
            pos += stream_cbytes;
            block.extend_from_slice(&decode_stream(codec, src, neblock)?);
        }
        Ok(block)
    };
    let mut out = Vec::with_capacity(nbytes);
    for i in 0..nblocks {
        let bstart = u32_at(frame, 16 + 4 * i)? as usize;
        let bsize = blocksize.min(nbytes - i * blocksize);
        // c-blosc builds differ on when blocks split into one stream per
        // byte lane (split modes are a compile/runtime option), so try the
        // split layout first and fall back to a single stream — stream
        // length validation makes misdetection fail loudly, not silently.
        let mut block = if typesize > 1 && bsize.is_multiple_of(typesize) {
            decode_block(bstart, bsize, typesize)
                .or_else(|_| decode_block(bstart, bsize, 1))?
        } else {
            decode_block(bstart, bsize, 1)?
        };
        if byte_shuffle {
            block = undo_shuffle(&block, typesize);
        }
        out.extend_from_slice(&block);
    }
    if out.len() != nbytes {
        return Err(err(format!(
            "blosc output length {} != header nbytes {nbytes}",
            out.len()
        )));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture(name: &str) -> Vec<u8> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/blosc")
            .join(name);
        std::fs::read(&path).unwrap_or_else(|e| panic!("fixture {name} missing: {e}"))
    }

    fn expected_f32(n: usize) -> Vec<u8> {
        // Matches the generator: (arange(n) % 97) / 7 as f32.
        (0..n)
            .flat_map(|i| (((i % 97) as f32) / 7.0).to_le_bytes())
            .collect::<Vec<u8>>()
    }

    #[test]
    fn lz4_shuffle_frame_decodes() {
        let out = blosc_decompress(&fixture("lz4_shuffle_f32.blosc")).unwrap();
        assert_eq!(out, expected_f32(10_000));
    }

    #[test]
    fn lz4_noshuffle_frame_decodes() {
        let out = blosc_decompress(&fixture("lz4_noshuffle_f32.blosc")).unwrap();
        assert_eq!(out, expected_f32(10_000));
    }

    #[test]
    fn zlib_shuffle_frame_decodes() {
        let out = blosc_decompress(&fixture("zlib_shuffle_f32.blosc")).unwrap();
        assert_eq!(out, expected_f32(10_000));
    }

    #[test]
    fn multiblock_frame_decodes() {
        let out = blosc_decompress(&fixture("lz4_shuffle_multiblock_f32.blosc")).unwrap();
        assert_eq!(out, expected_f32(600_000));
    }

    #[test]
    fn memcpy_frame_decodes() {
        let out = blosc_decompress(&fixture("memcpy_f32.blosc")).unwrap();
        // Generator stores arange(64) (incompressible after typesize-1 view).
        let expected: Vec<u8> = (0..64u32)
            .flat_map(|i| (i as f32).to_le_bytes())
            .collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn truncated_and_unsupported_frames_error() {
        assert!(blosc_decompress(&[0u8; 8]).is_err());
        let mut frame = fixture("lz4_shuffle_f32.blosc");
        frame[2] |= 0x4; // claim bit-shuffle
        let e = blosc_decompress(&frame).err().unwrap();
        assert!(e.message().contains("bit-shuffle"));
    }
}
