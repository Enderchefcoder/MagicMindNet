use crate::error::{MmnError, Result};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Cuda,
}

impl Device {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "cpu" => Ok(Device::Cpu),
            "cuda" | "gpu" => Ok(Device::Cuda),
            _ => Err(MmnError::Other {
                message: format!("Unknown device: {s}"),
            }),
        }
    }

    pub fn require_cuda_available(cuda_requested: bool) -> Result<()> {
        if cuda_requested {
            return Err(MmnError::cuda_missing());
        }
        Ok(())
    }

    /// Called from `mmn-cuda` when built with CUDA support.
    pub fn require_cuda_available_checked(
        cuda_requested: bool,
        cuda_ok: bool,
    ) -> Result<()> {
        if cuda_requested && !cuda_ok {
            return Err(MmnError::cuda_missing());
        }
        Ok(())
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Cuda => "cuda",
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl Default for Device {
    fn default() -> Self {
        Device::Cpu
    }
}
