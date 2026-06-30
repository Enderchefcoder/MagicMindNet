//! Adapt external Hugging Face checkpoint layouts to MMN tensor keys/shapes.

use mmn_core::{MmnError, Tensor};
use ndarray::{s, Ix2};
use std::collections::{HashMap, HashSet};

/// Split fused QKV, fuse SwiGLU gate×up into `ffn`, tie `lm_head` to `embed`, infer `ffn_dim`.
pub fn adapt_external_hf_tensors(
    tensors: &mut HashMap<String, Tensor>,
    meta: &mut serde_json::Value,
) -> Result<(), MmnError> {
    split_fused_qkv_tensors(tensors)?;
    fuse_swiglu_gate_up_tensors(tensors);
    expand_gqa_kv_tensors(tensors, meta)?;
    tie_missing_lm_head(tensors);
    infer_ffn_dim_meta(tensors, meta);
    Ok(())
}

/// HF RMSNorm / fused checkpoints may omit LayerNorm β; default γ=1, β=0.
pub fn fill_missing_block_layernorm_defaults(
    tensors: &mut HashMap<String, Tensor>,
    n_layer: usize,
    d_model: usize,
) {
    for i in 0..n_layer {
        for (suffix, val) in [
            ("ln1.gamma", 1.0f32),
            ("ln2.gamma", 1.0),
            ("ln1.beta", 0.0),
            ("ln2.beta", 0.0),
        ] {
            let key = format!("blocks.{i}.{suffix}");
            if !tensors.contains_key(&key) {
                tensors.insert(key, vector_tensor(d_model, val));
            }
        }
    }
}

fn vector_tensor(len: usize, val: f32) -> Tensor {
    Tensor::from_array(ndarray::ArrayD::from_elem(ndarray::IxDyn(&[len]), val), true)
}

fn split_fused_qkv_tensors(tensors: &mut HashMap<String, Tensor>) -> Result<(), MmnError> {
    let keys: Vec<String> = tensors
        .keys()
        .filter(|k| k.ends_with(".attn.qkv"))
        .cloned()
        .collect();
    for key in keys {
        let fused = tensors.remove(&key).unwrap();
        let prefix = key
            .strip_suffix(".attn.qkv")
            .ok_or_else(|| MmnError::Other {
                message: format!("invalid fused qkv key {key}"),
            })?;
        let (q, k, v) = split_qkv_weight(&fused, &key)?;
        tensors.insert(format!("{prefix}.attn.q"), q);
        tensors.insert(format!("{prefix}.attn.k"), k);
        tensors.insert(format!("{prefix}.attn.v"), v);
    }
    Ok(())
}

fn split_qkv_weight(fused: &Tensor, name: &str) -> Result<(Tensor, Tensor, Tensor), MmnError> {
    let shape: Vec<usize> = fused.data.shape().iter().copied().collect();
    if shape.len() != 2 {
        return Err(MmnError::Other {
            message: format!("tensor {name}: fused qkv must be rank-2, got {shape:?}"),
        });
    }
    let view = fused
        .data
        .view()
        .into_dimensionality::<Ix2>()
        .map_err(|e| MmnError::Other {
            message: format!("tensor {name}: {e}"),
        })?;
    let (d0, d1) = (shape[0], shape[1]);
    if d1 == d0 * 3 {
        let d = d0;
        let q = mmn_chunk_from_hf_in_out(&view.slice(s![.., 0..d]), name, "q")?;
        let k = mmn_chunk_from_hf_in_out(&view.slice(s![.., d..2 * d]), name, "k")?;
        let v = mmn_chunk_from_hf_in_out(&view.slice(s![.., 2 * d..3 * d]), name, "v")?;
        Ok((q, k, v))
    } else if d0 == d1 * 3 {
        let d = d1;
        let q = tensor_from_out_in_chunk(&view.slice(s![0..d, ..]), name, "q")?;
        let k = tensor_from_out_in_chunk(&view.slice(s![d..2 * d, ..]), name, "k")?;
        let v = tensor_from_out_in_chunk(&view.slice(s![2 * d..3 * d, ..]), name, "v")?;
        Ok((q, k, v))
    } else {
        Err(MmnError::Other {
            message: format!(
                "tensor {name}: fused qkv shape [{d0}, {d1}] is not [d, 3d] or [3d, d]"
            ),
        })
    }
}

/// HF Conv1d / GPT-2 stores `[in_features, out_features]`; MMN Linear uses `[out, in]`.
fn mmn_chunk_from_hf_in_out(
    chunk: &ndarray::ArrayView2<f32>,
    name: &str,
    which: &str,
) -> Result<Tensor, MmnError> {
    let t = chunk.t().to_owned();
    if t.nrows() != t.ncols() {
        return Err(MmnError::Other {
            message: format!("tensor {name} {which}: expected square projection, got {:?}", t.dim()),
        });
    }
    Ok(Tensor::from_array(t.into_dyn(), true))
}

fn tensor_from_out_in_chunk(
    chunk: &ndarray::ArrayView2<f32>,
    name: &str,
    which: &str,
) -> Result<Tensor, MmnError> {
    let t = chunk.to_owned();
    if t.nrows() != t.ncols() {
        return Err(MmnError::Other {
            message: format!("tensor {name} {which}: expected square projection, got {:?}", t.dim()),
        });
    }
    Ok(Tensor::from_array(t.into_dyn(), true))
}

fn fuse_swiglu_gate_up_tensors(tensors: &mut HashMap<String, Tensor>) {
    let indices: HashSet<usize> = tensors
        .keys()
        .filter_map(|k| {
            k.strip_prefix("blocks.")?
                .split('.')
                .next()?
                .parse()
                .ok()
        })
        .collect();
    for i in indices {
        let gate_key = format!("blocks.{i}.ffn");
        let up_key = format!("blocks.{i}.ffn.up");
        let Some(up) = tensors.remove(&up_key) else {
            continue;
        };
        let Some(gate) = tensors.get(&gate_key) else {
            tensors.insert(up_key, up);
            continue;
        };
        if let Ok(fused) = elementwise_mul(gate, &up) {
            tensors.insert(gate_key, fused);
        } else {
            tensors.insert(up_key, up);
        }
    }
}

fn elementwise_mul(a: &Tensor, b: &Tensor) -> Result<Tensor, MmnError> {
    if a.data.shape() != b.data.shape() {
        return Err(MmnError::Other {
            message: format!(
                "SwiGLU fuse shape mismatch {:?} vs {:?}",
                a.data.shape(),
                b.data.shape()
            ),
        });
    }
    let out = a.data.as_ref() * b.data.as_ref();
    Ok(Tensor::from_array(out, true))
}

fn tie_missing_lm_head(tensors: &mut HashMap<String, Tensor>) {
    if !tensors.contains_key("lm_head") {
        if let Some(embed) = tensors.get("embed") {
            tensors.insert("lm_head".to_string(), embed.clone());
        }
    }
}

fn infer_ffn_dim_meta(tensors: &HashMap<String, Tensor>, meta: &mut serde_json::Value) {
    if meta.get("ffn_dim").is_some() {
        return;
    }
    if let Some(ffn) = tensors.get("blocks.0.ffn") {
        let shape: Vec<usize> = ffn.data.shape().iter().copied().collect();
        if shape.len() == 2 {
            meta["ffn_dim"] = serde_json::json!(shape[0]);
        }
    }
}

/// Expand grouped-query K/V projections `[n_kv_heads * head_dim, d_model]` to MMN MHA shape.
fn expand_gqa_kv_tensors(
    tensors: &mut HashMap<String, Tensor>,
    meta: &mut serde_json::Value,
) -> Result<(), MmnError> {
    let indices: HashSet<usize> = tensors
        .keys()
        .filter_map(|k| {
            k.strip_prefix("blocks.")?
                .split('.')
                .next()?
                .parse()
                .ok()
        })
        .collect();
    for i in indices {
        let q_key = format!("blocks.{i}.attn.q");
        let k_key = format!("blocks.{i}.attn.k");
        let v_key = format!("blocks.{i}.attn.v");
        let (Some(q), Some(k), Some(v)) = (
            tensors.get(&q_key),
            tensors.get(&k_key),
            tensors.get(&v_key),
        ) else {
            continue;
        };
        let q_shape: Vec<usize> = q.data.shape().iter().copied().collect();
        let k_shape: Vec<usize> = k.data.shape().iter().copied().collect();
        let v_shape: Vec<usize> = v.data.shape().iter().copied().collect();
        if q_shape.len() != 2 || k_shape.len() != 2 || k_shape != v_shape {
            continue;
        }
        let d_model = q_shape[0];
        if q_shape[1] != d_model || k_shape[1] != d_model {
            continue;
        }
        let kv_dim = k_shape[0];
        if kv_dim >= d_model || d_model % kv_dim != 0 {
            continue;
        }
        let Some((head_dim, n_heads, n_kv_heads)) = gqa_dims_from_meta_or_guess(meta, d_model, kv_dim)
        else {
            continue;
        };
        if n_kv_heads == n_heads {
            continue;
        }
        let k_exp = expand_gqa_proj(k, d_model, head_dim, n_heads, n_kv_heads, &k_key)?;
        let v_exp = expand_gqa_proj(v, d_model, head_dim, n_heads, n_kv_heads, &v_key)?;
        tensors.insert(k_key, k_exp);
        tensors.insert(v_key, v_exp);
    }
    Ok(())
}

fn gqa_dims_from_meta_or_guess(
    meta: &serde_json::Value,
    d_model: usize,
    kv_dim: usize,
) -> Option<(usize, usize, usize)> {
    if let (Some(n_heads), Some(n_kv)) = (
        meta.get("num_attention_heads").and_then(|v| v.as_u64()),
        meta.get("num_key_value_heads").and_then(|v| v.as_u64()),
    ) {
        let n_heads = n_heads as usize;
        let n_kv_heads = n_kv as usize;
        if n_heads > 0 && n_kv_heads > 0 && d_model % n_heads == 0 {
            let head_dim = d_model / n_heads;
            if kv_dim == n_kv_heads * head_dim {
                return Some((head_dim, n_heads, n_kv_heads));
            }
        }
    }
    guess_gqa_dims(d_model, kv_dim)
}

fn guess_gqa_dims(d_model: usize, kv_dim: usize) -> Option<(usize, usize, usize)> {
    if kv_dim >= d_model || d_model % kv_dim != 0 {
        return None;
    }
    const CANDIDATES: [usize; 9] = [128, 64, 256, 32, 96, 48, 80, 72, 112];
    for &head_dim in &CANDIDATES {
        if d_model % head_dim != 0 || kv_dim % head_dim != 0 {
            continue;
        }
        let n_heads = d_model / head_dim;
        let n_kv_heads = kv_dim / head_dim;
        if n_heads % n_kv_heads == 0 && n_heads >= n_kv_heads {
            return Some((head_dim, n_heads, n_kv_heads));
        }
    }
    Some((kv_dim, d_model / kv_dim, 1))
}

fn expand_gqa_proj(
    kv: &Tensor,
    d_model: usize,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    name: &str,
) -> Result<Tensor, MmnError> {
    let kv_dim = n_kv_heads * head_dim;
    let shape: Vec<usize> = kv.data.shape().iter().copied().collect();
    if shape != [kv_dim, d_model] {
        return Err(MmnError::Other {
            message: format!("tensor {name}: expected GQA shape [{kv_dim}, {d_model}], got {shape:?}"),
        });
    }
    let view = kv
        .data
        .view()
        .into_dimensionality::<Ix2>()
        .map_err(|e| MmnError::Other {
            message: format!("tensor {name}: {e}"),
        })?;
    let mut out = ndarray::Array2::<f32>::zeros((d_model, d_model));
    for h in 0..n_heads {
        let kv_h = h * n_kv_heads / n_heads;
        out.slice_mut(s![h * head_dim..(h + 1) * head_dim, ..])
            .assign(&view.slice(s![kv_h * head_dim..(kv_h + 1) * head_dim, ..]));
    }
    Ok(Tensor::from_array(out.into_dyn(), true))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    fn tensor2x2(data: [[f32; 2]; 2]) -> Tensor {
        Tensor::from_array(arr2(&data).into_dyn(), true)
    }

    #[test]
    fn split_gpt2_style_qkv_transposes_chunks() {
        let fused = arr2(&[
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        ]);
        let t = Tensor::from_array(fused.into_dyn(), true);
        let (q, k, v) = split_qkv_weight(&t, "blocks.0.attn.qkv").unwrap();
        assert_eq!(q.data.shape(), &[2, 2]);
        assert_eq!(q.data[[0, 0]], 1.0);
        assert_eq!(q.data[[0, 1]], 7.0);
        assert_eq!(k.data[[0, 1]], 9.0);
        assert_eq!(v.data[[1, 1]], 12.0);
    }

    #[test]
    fn fuse_swiglu_multiplies_gate_and_up() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "blocks.0.ffn".into(),
            tensor2x2([[2.0, 3.0], [4.0, 5.0]]),
        );
        tensors.insert(
            "blocks.0.ffn.up".into(),
            tensor2x2([[10.0, 1.0], [2.0, 3.0]]),
        );
        fuse_swiglu_gate_up_tensors(&mut tensors);
        let ffn = tensors.get("blocks.0.ffn").unwrap();
        assert_eq!(ffn.data[[0, 0]], 20.0);
        assert_eq!(ffn.data[[1, 1]], 15.0);
        assert!(!tensors.contains_key("blocks.0.ffn.up"));
    }

    #[test]
    fn adapt_ties_lm_head_to_embed() {
        let mut tensors = HashMap::new();
        let embed = tensor2x2([[1.0, 0.0], [0.0, 1.0]]);
        tensors.insert("embed".into(), embed);
        let mut meta = serde_json::json!({});
        adapt_external_hf_tensors(&mut tensors, &mut meta).unwrap();
        assert!(tensors.contains_key("lm_head"));
    }

    #[test]
    fn expand_gqa_repeats_kv_heads_across_query_heads() {
        let d_model = 8usize;
        let head_dim = 2usize;
        let n_heads = 4usize;
        let n_kv_heads = 2usize;
        let mut k_data = ndarray::Array2::<f32>::zeros((n_kv_heads * head_dim, d_model));
        k_data[[0, 0]] = 1.0;
        k_data[[1, 0]] = 2.0;
        k_data[[2, 0]] = 3.0;
        k_data[[3, 0]] = 4.0;
        let k = Tensor::from_array(k_data.into_dyn(), true);
        let expanded = expand_gqa_proj(&k, d_model, head_dim, n_heads, n_kv_heads, "blocks.0.attn.k")
            .unwrap();
        assert_eq!(expanded.data.shape(), &[d_model, d_model]);
        assert_eq!(expanded.data[[0, 0]], 1.0);
        assert_eq!(expanded.data[[2, 0]], 1.0);
        assert_eq!(expanded.data[[4, 0]], 3.0);
        assert_eq!(expanded.data[[6, 0]], 3.0);
    }

    #[test]
    fn gqa_dims_from_meta_prefers_config() {
        let meta = serde_json::json!({
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
        });
        let (hd, nh, nkv) = gqa_dims_from_meta_or_guess(&meta, 512, 128).unwrap();
        assert_eq!((hd, nh, nkv), (64, 8, 2));
    }
}
