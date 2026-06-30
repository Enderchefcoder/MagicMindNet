//! Chatbot transformer block checkpoint export/import.

use crate::checkpoint_util::{expect_tensor_shape, require_tensor_entry, tensor_from_entry, tensor_to_entry};
use mmn_core::{MmnError, Tensor};
use mmn_models::Chatbot;
use std::collections::HashMap;

pub(crate) fn export_block_tensors(model: &Chatbot, map: &mut HashMap<String, serde_json::Value>) {
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
        map.insert(format!("{p}.ln1.gamma"), tensor_to_entry(&block.ln1.gamma));
        map.insert(format!("{p}.ln1.beta"), tensor_to_entry(&block.ln1.beta));
        map.insert(format!("{p}.ln2.gamma"), tensor_to_entry(&block.ln2.gamma));
        map.insert(format!("{p}.ln2.beta"), tensor_to_entry(&block.ln2.beta));
    }
}

pub(crate) fn import_block_tensors(model: &mut Chatbot, tensors: &serde_json::Value) -> Result<(), MmnError> {
    let d_model = model.shape.d_model;
    let ffn_dim = model.shape.ffn_dim;
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
        expect_tensor_shape(
            &block.attn.q_proj.weight,
            &[d_model, d_model],
            &format!("{prefix}.attn.q"),
        )?;
        expect_tensor_shape(
            &block.attn.k_proj.weight,
            &[d_model, d_model],
            &format!("{prefix}.attn.k"),
        )?;
        expect_tensor_shape(
            &block.attn.v_proj.weight,
            &[d_model, d_model],
            &format!("{prefix}.attn.v"),
        )?;
        expect_tensor_shape(
            &block.attn.out_proj.weight,
            &[d_model, d_model],
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
    }
    Ok(())
}
