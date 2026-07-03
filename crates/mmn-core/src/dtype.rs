#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum DType {
    #[default]
    F32,
    BF16,
}

impl DType {
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::BF16 => 2,
        }
    }
}

