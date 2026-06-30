use std::path::PathBuf;

pub fn temp_file(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("mmn_io_{name}_{}", std::process::id()))
}
