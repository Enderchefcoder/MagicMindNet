mod block_tensors;
mod chatbot_io;
mod checkpoint_util;
mod classifier_io;
mod detect;
mod diffusion_io;
mod hf_adapt;
mod hf_classifier_safetensors;
mod hf_safetensors;
mod hf_tensor_codec;
mod interop;
mod tensor_merge;

pub use chatbot_io::{
    export_bin, export_safetensors, import_bin, import_safetensors, merge_models, quantize_model,
    TokenizerSidecarRefs,
};
pub use detect::{detect_checkpoint_kind, CheckpointKind};
pub use interop::gguf::{is_gguf_bytes, read_gguf, write_gguf, GgufValue, GgufWriteTensor};
pub use interop::gguf_chatbot::{export_gguf, gguf_name_to_mmn, import_gguf, mmn_name_to_gguf};
pub use interop::gguf_quant::GgmlType;
pub use interop::inflate::inflate;
pub use interop::npy::{decode_npy, encode_npy_f32, NpyArray};
pub use interop::npz_chatbot::{
    export_npz, import_npz, read_npz_arrays, write_npz_arrays,
};
pub use interop::torch_pt::{
    export_torch_pt, import_torch_pt, read_torch_arrays, write_torch_arrays,
};
pub use interop::zip::{
    crc32, is_zip_bytes, read_zip, read_zip_entry, write_zip_stored, zip_entry_names,
};
pub use interop::NamedArray;
pub use hf_classifier_safetensors::{
    export_hf_classifier_safetensors, hf_classifier_name_to_mmn, import_hf_classifier_safetensors,
    import_hf_classifier_safetensors_bytes, HF_CLASSIFIER_FORMAT,
};
pub use hf_safetensors::{
    export_hf_safetensors, hf_name_to_mmn, import_hf_safetensors, import_hf_safetensors_bytes,
    is_hf_safetensors_bytes, HF_FORMAT,
};
pub use hf_tensor_codec::HF_CHATBOT_FORMAT;
pub use classifier_io::{export_classifier, import_classifier, merge_classifiers, quantize_classifier};
pub use diffusion_io::{export_diffusion, import_diffusion, merge_diffusion, quantize_diffusion};

#[cfg(test)]
mod io_tests;
