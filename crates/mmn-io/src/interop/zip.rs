//! Minimal from-scratch ZIP archive reader/writer (PKWARE APPNOTE subset).
//!
//! Reads stored and DEFLATE-compressed entries (the latter via the
//! from-scratch [`super::inflate`] decompressor) and writes stored archives.
//! Covers `numpy` `.npz` files and PyTorch `.pt` archives without external
//! compression crates.

use super::inflate::inflate;
use mmn_core::MmnError;

const LOCAL_HEADER_SIG: u32 = 0x0403_4b50;
const CENTRAL_HEADER_SIG: u32 = 0x0201_4b50;
const EOCD_SIG: u32 = 0x0605_4b50;
const METHOD_STORED: u16 = 0;
const METHOD_DEFLATE: u16 = 8;

fn err(message: impl Into<String>) -> MmnError {
    MmnError::Other {
        message: message.into(),
    }
}

/// CRC-32 (IEEE 802.3, reflected polynomial 0xEDB88320).
///
/// The lookup table is built once and cached; the hot loop is a plain
/// table-driven byte scan.
pub fn crc32(data: &[u8]) -> u32 {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut table = [0u32; 256];
        for (i, slot) in table.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            }
            *slot = c;
        }
        table
    });
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc = table[((crc ^ byte as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

/// One file inside a ZIP archive.
#[derive(Debug)]
pub struct ZipEntry {
    pub name: String,
    pub data: Vec<u8>,
}

fn read_u16(bytes: &[u8], pos: usize) -> Result<u16, MmnError> {
    bytes
        .get(pos..pos + 2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .ok_or_else(|| err("zip archive truncated"))
}

fn read_u32(bytes: &[u8], pos: usize) -> Result<u32, MmnError> {
    bytes
        .get(pos..pos + 4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .ok_or_else(|| err("zip archive truncated"))
}

fn find_eocd(bytes: &[u8]) -> Result<usize, MmnError> {
    // EOCD is at least 22 bytes and sits within the trailing 64 KiB + 22.
    if bytes.len() < 22 {
        return Err(err("zip archive too small for end-of-central-directory"));
    }
    let search_start = bytes.len().saturating_sub(22 + 65_535);
    for pos in (search_start..=bytes.len() - 22).rev() {
        if read_u32(bytes, pos)? == EOCD_SIG {
            return Ok(pos);
        }
    }
    Err(err("zip end-of-central-directory signature not found"))
}

struct CentralEntry {
    name: String,
    method: u16,
    crc: u32,
    compressed_size: usize,
    uncompressed_size: usize,
    local_offset: usize,
}

fn parse_central_directory(bytes: &[u8]) -> Result<Vec<CentralEntry>, MmnError> {
    let eocd = find_eocd(bytes)?;
    let entry_count = read_u16(bytes, eocd + 10)? as usize;
    let cd_offset = read_u32(bytes, eocd + 16)? as usize;
    if entry_count == 0xFFFF || cd_offset == 0xFFFF_FFFF {
        return Err(err("zip64 archives are not supported (archive > 4 GiB)"));
    }
    let mut entries = Vec::with_capacity(entry_count);
    let mut pos = cd_offset;
    for _ in 0..entry_count {
        if read_u32(bytes, pos)? != CENTRAL_HEADER_SIG {
            return Err(err("zip central directory entry signature mismatch"));
        }
        let method = read_u16(bytes, pos + 10)?;
        let crc = read_u32(bytes, pos + 16)?;
        let compressed_size = read_u32(bytes, pos + 20)? as usize;
        let uncompressed_size = read_u32(bytes, pos + 24)? as usize;
        let name_len = read_u16(bytes, pos + 28)? as usize;
        let extra_len = read_u16(bytes, pos + 30)? as usize;
        let comment_len = read_u16(bytes, pos + 32)? as usize;
        let local_offset = read_u32(bytes, pos + 42)? as usize;
        let name_bytes = bytes
            .get(pos + 46..pos + 46 + name_len)
            .ok_or_else(|| err("zip central directory name truncated"))?;
        let name = String::from_utf8_lossy(name_bytes).into_owned();
        entries.push(CentralEntry {
            name,
            method,
            crc,
            compressed_size,
            uncompressed_size,
            local_offset,
        });
        pos += 46 + name_len + extra_len + comment_len;
    }
    Ok(entries)
}

fn entry_data(bytes: &[u8], entry: &CentralEntry) -> Result<Vec<u8>, MmnError> {
    let pos = entry.local_offset;
    if read_u32(bytes, pos)? != LOCAL_HEADER_SIG {
        return Err(err(format!(
            "zip local header signature mismatch for {}",
            entry.name
        )));
    }
    let name_len = read_u16(bytes, pos + 26)? as usize;
    let extra_len = read_u16(bytes, pos + 28)? as usize;
    let data_start = pos + 30 + name_len + extra_len;
    let raw = bytes
        .get(data_start..data_start + entry.compressed_size)
        .ok_or_else(|| err(format!("zip entry {} data truncated", entry.name)))?;
    let data = match entry.method {
        METHOD_STORED => raw.to_vec(),
        METHOD_DEFLATE => inflate(raw)?,
        other => {
            return Err(err(format!(
                "zip entry {} uses unsupported compression method {other}",
                entry.name
            )));
        }
    };
    if data.len() != entry.uncompressed_size {
        return Err(err(format!(
            "zip entry {} size mismatch: expected {}, got {}",
            entry.name,
            entry.uncompressed_size,
            data.len()
        )));
    }
    if crc32(&data) != entry.crc {
        return Err(err(format!("zip entry {} CRC-32 mismatch", entry.name)));
    }
    Ok(data)
}

/// List entry names without decompressing any data.
pub fn zip_entry_names(bytes: &[u8]) -> Result<Vec<String>, MmnError> {
    Ok(parse_central_directory(bytes)?
        .into_iter()
        .map(|e| e.name)
        .collect())
}

/// Read and decompress every entry in the archive.
pub fn read_zip(bytes: &[u8]) -> Result<Vec<ZipEntry>, MmnError> {
    let central = parse_central_directory(bytes)?;
    let mut out = Vec::with_capacity(central.len());
    for entry in central.iter() {
        out.push(ZipEntry {
            name: entry.name.clone(),
            data: entry_data(bytes, entry)?,
        });
    }
    Ok(out)
}

/// Read a single entry by exact name.
pub fn read_zip_entry(bytes: &[u8], name: &str) -> Result<Vec<u8>, MmnError> {
    let central = parse_central_directory(bytes)?;
    let entry = central
        .iter()
        .find(|e| e.name == name)
        .ok_or_else(|| err(format!("zip archive has no entry named {name}")))?;
    entry_data(bytes, entry)
}

/// Serialize entries into a stored (uncompressed) ZIP archive.
pub fn write_zip_stored(entries: &[(String, Vec<u8>)]) -> Result<Vec<u8>, MmnError> {
    let mut out = Vec::new();
    let mut central = Vec::new();
    for (name, data) in entries {
        if data.len() > u32::MAX as usize || out.len() > u32::MAX as usize {
            return Err(err("zip64 writing not supported (entry or archive > 4 GiB)"));
        }
        let offset = out.len() as u32;
        let crc = crc32(data);
        let name_bytes = name.as_bytes();
        // Local file header.
        out.extend_from_slice(&LOCAL_HEADER_SIG.to_le_bytes());
        out.extend_from_slice(&20u16.to_le_bytes()); // version needed
        out.extend_from_slice(&0u16.to_le_bytes()); // flags
        out.extend_from_slice(&METHOD_STORED.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // mod time
        out.extend_from_slice(&0u16.to_le_bytes()); // mod date
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes()); // compressed
        out.extend_from_slice(&(data.len() as u32).to_le_bytes()); // uncompressed
        out.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // extra len
        out.extend_from_slice(name_bytes);
        out.extend_from_slice(data);
        central.push((name.clone(), crc, data.len() as u32, offset));
    }
    let cd_start = out.len() as u32;
    for (name, crc, size, offset) in central.iter() {
        let name_bytes = name.as_bytes();
        out.extend_from_slice(&CENTRAL_HEADER_SIG.to_le_bytes());
        out.extend_from_slice(&20u16.to_le_bytes()); // version made by
        out.extend_from_slice(&20u16.to_le_bytes()); // version needed
        out.extend_from_slice(&0u16.to_le_bytes()); // flags
        out.extend_from_slice(&METHOD_STORED.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // mod time
        out.extend_from_slice(&0u16.to_le_bytes()); // mod date
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&size.to_le_bytes()); // compressed
        out.extend_from_slice(&size.to_le_bytes()); // uncompressed
        out.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // extra len
        out.extend_from_slice(&0u16.to_le_bytes()); // comment len
        out.extend_from_slice(&0u16.to_le_bytes()); // disk number
        out.extend_from_slice(&0u16.to_le_bytes()); // internal attrs
        out.extend_from_slice(&0u32.to_le_bytes()); // external attrs
        out.extend_from_slice(&offset.to_le_bytes());
        out.extend_from_slice(name_bytes);
    }
    let cd_size = out.len() as u32 - cd_start;
    out.extend_from_slice(&EOCD_SIG.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // disk number
    out.extend_from_slice(&0u16.to_le_bytes()); // cd start disk
    out.extend_from_slice(&(entries.len() as u16).to_le_bytes()); // entries on disk
    out.extend_from_slice(&(entries.len() as u16).to_le_bytes()); // total entries
    out.extend_from_slice(&cd_size.to_le_bytes());
    out.extend_from_slice(&cd_start.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // comment len
    Ok(out)
}

/// True when `bytes` begin with a ZIP local-header or EOCD magic.
pub fn is_zip_bytes(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && bytes[0] == b'P' && bytes[1] == b'K' && matches!(bytes[2], 3 | 5) && bytes[3] == bytes[2] + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_reference() {
        // zlib.crc32(b"123456789") == 0xCBF43926 (standard check value).
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn stored_zip_roundtrip() {
        let entries = vec![
            ("hello.txt".to_string(), b"hello world".to_vec()),
            ("dir/data.bin".to_string(), vec![0u8, 1, 2, 250, 255]),
        ];
        let bytes = write_zip_stored(&entries).unwrap();
        assert!(is_zip_bytes(&bytes));
        let names = zip_entry_names(&bytes).unwrap();
        assert_eq!(names, vec!["hello.txt", "dir/data.bin"]);
        let read = read_zip(&bytes).unwrap();
        assert_eq!(read[0].data, b"hello world");
        assert_eq!(read[1].data, vec![0u8, 1, 2, 250, 255]);
        assert_eq!(
            read_zip_entry(&bytes, "dir/data.bin").unwrap(),
            vec![0u8, 1, 2, 250, 255]
        );
    }

    #[test]
    fn missing_entry_errors() {
        let bytes = write_zip_stored(&[("a".to_string(), vec![1])]).unwrap();
        let e = read_zip_entry(&bytes, "b").unwrap_err();
        assert!(e.message().contains("no entry named b"));
    }

    #[test]
    fn corrupt_crc_detected() {
        let mut bytes = write_zip_stored(&[("a.txt".to_string(), b"payload".to_vec())]).unwrap();
        // Local header is 30 bytes + 5-byte name; flip a payload byte.
        bytes[36] ^= 0xFF;
        let e = read_zip(&bytes).unwrap_err();
        assert!(e.message().contains("CRC-32 mismatch"));
    }

    #[test]
    fn truncated_archive_errors() {
        assert!(read_zip(b"PK\x03\x04short").is_err());
        assert!(read_zip(b"").is_err());
    }

    #[test]
    fn is_zip_bytes_detects_magic() {
        assert!(is_zip_bytes(b"PK\x03\x04rest"));
        assert!(is_zip_bytes(b"PK\x05\x06rest"));
        assert!(!is_zip_bytes(b"GGUF"));
        assert!(!is_zip_bytes(b"{"));
    }
}
