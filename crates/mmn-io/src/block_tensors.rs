//! Chatbot transformer block checkpoint export/import.

use crate::checkpoint_util::{
    expect_tensor_shape, require_tensor_entry, tensor_from_entry, tensor_to_entry, TensorMap,
};
use mmn_core::{MmnError, Tensor};
use mmn_models::{Chatbot, TransformerBlock};

/// Export a list of blocks with a custom prefix (e.g. "prelude" or "coda").
pub(crate) fn export_named_block_list(blocks: &[TransformerBlock], prefix: &str, map: &mut TensorMap) {
    for (i, block) in blocks.iter().enumerate() {
        let p = format!("{prefix}.{i}");
        map.insert(format!("{p}.attn.q"), tensor_to_entry(&block.attn.q_proj.weight));
        map.insert(format!("{p}.attn.k"), tensor_to_entry(&block.attn.k_proj.weight));
        map.insert(format!("{p}.attn.v"), tensor_to_entry(&block.attn.v_proj.weight));
        map.insert(format!("{p}.attn.out"), tensor_to_entry(&block.attn.out_proj.weight));
        map.insert(format!("{p}.ffn"), tensor_to_entry(&block.ffn.weight));
        map.insert(format!("{p}.ffn2"), tensor_to_entry(&block.ffn2.weight));
        if let Some(gate) = &block.ffn_gate {
            map.insert(format!("{p}.ffn_gate"), tensor_to_entry(&gate.weight));
        }
        map.insert(format!("{p}.ln1.gamma"), tensor_to_entry(&block.ln1.gamma));
        map.insert(format!("{p}.ln1.beta"), tensor_to_entry(&block.ln1.beta));
        map.insert(format!("{p}.ln2.gamma"), tensor_to_entry(&block.ln2.gamma));
        map.insert(format!("{p}.ln2.beta"), tensor_to_entry(&block.ln2.beta));
    }
}

/// Import a list of blocks with a custom prefix (e.g. "prelude" or "coda").
pub(crate) fn import_named_block_list(
    blocks: &mut [TransformerBlock],
    prefix: &str,
    tensors: &TensorMap,
    d_model: usize,
    ffn_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    use_rms: bool,
) -> Result<(), MmnError> {
    for (i, block) in blocks.iter_mut().enumerate() {
        let p = format!("{prefix}.{i}");
        let keys: [(&str, &mut Tensor); 10] = [
            ("attn.q", &mut block.attn.q_proj.weight),
            ("attn.k", &mut block.attn.k_proj.weight),
            ("attn.v", &mut block.attn.v_proj.weight),
            ("attn.out", &mut block.attn.out_proj.weight),
            ("ffn", &mut block.ffn.weight),
            ("ffn2", &mut block.ffn2.weight),
            ("ln1.gamma", &mut block.ln1.gamma),
            ("ln1.beta", &mut block.ln1.beta),
            ("ln2.gamma", &mut block.ln2.gamma),
            ("ln2.beta", &mut block.ln2.beta),
        ];
        for (suffix, dest) in keys {
            let key = format!("{p}.{suffix}");
            *dest = tensor_from_entry(require_tensor_entry(tensors, &key)?)?;
        }
        if let Some(gate) = block.ffn_gate.as_mut() {
            let key = format!("{p}.ffn_gate");
            gate.weight = tensor_from_entry(require_tensor_entry(tensors, &key)?)?;
            expect_tensor_shape(&gate.weight, &[ffn_dim, d_model], &key)?;
        }
        expect_tensor_shape(&block.attn.q_proj.weight, &[q_dim, d_model], &format!("{p}.attn.q"))?;
        expect_tensor_shape(&block.attn.k_proj.weight, &[kv_dim, d_model], &format!("{p}.attn.k"))?;
        expect_tensor_shape(&block.attn.v_proj.weight, &[kv_dim, d_model], &format!("{p}.attn.v"))?;
        expect_tensor_shape(&block.attn.out_proj.weight, &[d_model, q_dim], &format!("{p}.attn.out"))?;
        expect_tensor_shape(&block.ffn.weight, &[ffn_dim, d_model], &format!("{p}.ffn"))?;
        expect_tensor_shape(&block.ffn2.weight, &[d_model, ffn_dim], &format!("{p}.ffn2"))?;
        expect_tensor_shape(&block.ln1.gamma, &[d_model], &format!("{p}.ln1.gamma"))?;
        expect_tensor_shape(&block.ln1.beta, &[d_model], &format!("{p}.ln1.beta"))?;
        expect_tensor_shape(&block.ln2.gamma, &[d_model], &format!("{p}.ln2.gamma"))?;
        expect_tensor_shape(&block.ln2.beta, &[d_model], &format!("{p}.ln2.beta"))?;
        block.ln1.use_rms = use_rms;
        block.ln2.use_rms = use_rms;
    }
    Ok(())
}

pub(crate) fn export_block_tensors(model: &Chatbot, map: &mut TensorMap) {
    for (i, block) in model.blocks.iter().enumerate() {
        let p = format!("blocks.{i}");
        map.insert(
            format!("{p}.attn.q"),
            tensor_to_entry(&block.attn.q_proj.weight),
        );
        map.insert(
            format!("{p}.attn.k"),
            tensor_to_entry(&block.attn.k_proj.weight),
        );
        map.insert(
            format!("{p}.attn.v"),
            tensor_to_entry(&block.attn.v_proj.weight),
        );
        map.insert(
            format!("{p}.attn.out"),
            tensor_to_entry(&block.attn.out_proj.weight),
        );
        map.insert(format!("{p}.ffn"), tensor_to_entry(&block.ffn.weight));
        map.insert(format!("{p}.ffn2"), tensor_to_entry(&block.ffn2.weight));
        if let Some(gate) = &block.ffn_gate {
            map.insert(format!("{p}.ffn_gate"), tensor_to_entry(&gate.weight));
        }
        map.insert(format!("{p}.ln1.gamma"), tensor_to_entry(&block.ln1.gamma));
        map.insert(format!("{p}.ln1.beta"), tensor_to_entry(&block.ln1.beta));
        map.insert(format!("{p}.ln2.gamma"), tensor_to_entry(&block.ln2.gamma));
        map.insert(format!("{p}.ln2.beta"), tensor_to_entry(&block.ln2.beta));
    }
}

pub(crate) fn import_block_tensors(model: &mut Chatbot, tensors: &TensorMap) -> Result<(), MmnError> {
    let d_model = model.shape.d_model;
    let ffn_dim = model.shape.ffn_dim;
    let q_dim = model.shape.q_dim();
    let kv_dim = model.shape.kv_dim();
    for (i, block) in model.blocks.iter_mut().enumerate() {
        let p = format!("blocks.{i}");
        let prefix = p.clone();
        let keys: [(&str, &mut Tensor); 10] = [
            ("attn.q", &mut block.attn.q_proj.weight),
            ("attn.k", &mut block.attn.k_proj.weight),
            ("attn.v", &mut block.attn.v_proj.weight),
            ("attn.out", &mut block.attn.out_proj.weight),
            ("ffn", &mut block.ffn.weight),
            ("ffn2", &mut block.ffn2.weight),
            ("ln1.gamma", &mut block.ln1.gamma),
            ("ln1.beta", &mut block.ln1.beta),
            ("ln2.gamma", &mut block.ln2.gamma),
            ("ln2.beta", &mut block.ln2.beta),
        ];
        for (suffix, dest) in keys {
            let key = format!("{p}.{suffix}");
            *dest = tensor_from_entry(require_tensor_entry(tensors, &key)?)?;
        }
        if let Some(gate) = block.ffn_gate.as_mut() {
            let key = format!("{p}.ffn_gate");
            gate.weight = tensor_from_entry(require_tensor_entry(tensors, &key)?)?;
            expect_tensor_shape(&gate.weight, &[ffn_dim, d_model], &key)?;
        }
        expect_tensor_shape(
            &block.attn.q_proj.weight,
            &[q_dim, d_model],
            &format!("{prefix}.attn.q"),
        )?;
        expect_tensor_shape(
            &block.attn.k_proj.weight,
            &[kv_dim, d_model],
            &format!("{prefix}.attn.k"),
        )?;
        expect_tensor_shape(
            &block.attn.v_proj.weight,
            &[kv_dim, d_model],
            &format!("{prefix}.attn.v"),
        )?;
        expect_tensor_shape(
            &block.attn.out_proj.weight,
            &[d_model, q_dim],
            &format!("{prefix}.attn.out"),
        )?;
        expect_tensor_shape(
            &block.ffn.weight,
            &[ffn_dim, d_model],
            &format!("{prefix}.ffn"),
        )?;
        expect_tensor_shape(
            &block.ffn2.weight,
            &[d_model, ffn_dim],
            &format!("{prefix}.ffn2"),
        )?;
        expect_tensor_shape(
            &block.ln1.gamma,
            &[d_model],
            &format!("{prefix}.ln1.gamma"),
        )?;
        expect_tensor_shape(
            &block.ln1.beta,
            &[d_model],
            &format!("{prefix}.ln1.beta"),
        )?;
        expect_tensor_shape(
            &block.ln2.gamma,
            &[d_model],
            &format!("{prefix}.ln2.gamma"),
        )?;
        expect_tensor_shape(
            &block.ln2.beta,
            &[d_model],
            &format!("{prefix}.ln2.beta"),
        )?;
        block.ln1.use_rms = model.norm_kind == "rms";
        block.ln2.use_rms = model.norm_kind == "rms";
    }
    Ok(())
}
