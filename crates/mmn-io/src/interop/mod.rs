//! Global model-format interop, implemented from scratch.
//!
//! - [`gguf`] / [`gguf_chatbot`] — GGUF container + llama.cpp-name bridge
//! - [`torch_pt`] / [`pickle`] — PyTorch `.pt` archives + pickle VM
//! - [`npy`] / [`npz_chatbot`] — NumPy `.npy` codec + `.npz` checkpoints
//! - [`zip`] / [`inflate`] — ZIP container + DEFLATE decompressor

/// A named array: `(name, shape, C-ordered f32 data)`.
pub type NamedArray = (String, Vec<usize>, Vec<f32>);

pub mod deflate;
pub mod gguf;
pub mod gguf_chatbot;
pub mod gguf_info;
pub mod gguf_iq_grids;
pub mod gguf_quant;
pub mod gguf_quant_iq;
pub mod gguf_quant_k_encode;
pub mod hdf5;
pub mod inflate;
pub mod npy;
pub mod npz_chatbot;
pub mod onnx;
pub mod pickle;
pub mod proto;
pub mod sharded;
pub mod tf_checkpoint;
pub mod torch_pt;
pub mod zip;
