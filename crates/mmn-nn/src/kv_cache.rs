//! Inference KV cache for autoregressive generation.

use mmn_core::{MmnError, Result, Tensor};
use ndarray::{ArrayD, IxDyn};

use crate::{concat_sequence_rows, gelu, slice_sequence_rows, MultiHeadAttention, TransformerBlock};

/// Per-layer cached K/V projections (after RoPE when enabled).
#[derive(Clone, Default)]
pub struct LayerKvCache {
    pub k: Option<Tensor>,
    pub v: Option<Tensor>,
}

impl LayerKvCache {
    pub fn len(&self) -> usize {
        self.k.as_ref().map(|t| t.shape[0]).unwrap_or(0)
    }

    pub fn clear(&mut self) {
        self.k = None;
        self.v = None;
    }

    /// Drop the first `n` cached sequence rows from K/V (sliding-window eviction).
    pub fn truncate_front(&mut self, n: usize) -> Result<()> {
        for _ in 0..n {
            self.truncate_at(0)?;
        }
        Ok(())
    }

    /// Drop row `index` from cached K/V (used for sliding text past a vision prefix).
    pub fn truncate_at(&mut self, index: usize) -> Result<()> {
        let len = self.len();
        if index >= len {
            return Err(MmnError::Shape {
                message: format!("truncate_at index {index} invalid for KV cache len {len}"),
            });
        }
        if len == 1 {
            self.clear();
            return Ok(());
        }
        if let (Some(k), Some(v)) = (&self.k, &self.v) {
            let head = slice_sequence_rows(k, 0, index)?;
            let tail = slice_sequence_rows(k, index + 1, len)?;
            self.k = Some(concat_sequence_rows(&head, &tail)?);
            let head_v = slice_sequence_rows(v, 0, index)?;
            let tail_v = slice_sequence_rows(v, index + 1, len)?;
            self.v = Some(concat_sequence_rows(&head_v, &tail_v)?);
        }
        Ok(())
    }
}

/// Re-index RoPE-encoded K rows from `row_start` after deleting one row at that index.
pub fn rerope_k_cache_shift_down_from(
    k: &mut Tensor,
    n_kv_heads: usize,
    theta: f32,
    row_start: usize,
) -> Result<()> {
    let rows = k.shape[0];
    if row_start >= rows {
        return Ok(());
    }
    let kv_dim = k.shape[1];
    let head_dim = kv_dim / n_kv_heads;
    let half = head_dim / 2;
    let mut data = (*k.data).clone();
    for r in row_start..rows {
        let from_pos = r + 1;
        let to_pos = r;
        for h in 0..n_kv_heads {
            let base = h * head_dim;
            for j in 0..half {
                let idx0 = base + 2 * j;
                let idx1 = base + 2 * j + 1;
                let freq = 1.0 / theta.powf(2.0 * j as f32 / head_dim as f32);
                let delta = (to_pos as f32 - from_pos as f32) * freq;
                let c = delta.cos();
                let s = delta.sin();
                let x0 = data[[r, idx0]];
                let x1 = data[[r, idx1]];
                data[[r, idx0]] = x0 * c - x1 * s;
                data[[r, idx1]] = x0 * s + x1 * c;
            }
        }
    }
    *k = Tensor::from_array(data, k.requires_grad);
    Ok(())
}

/// Re-index RoPE-encoded K rows after dropping the front cache row (absolute positions shift down by 1).
pub fn rerope_k_cache_after_front_drop(
    k: &mut Tensor,
    n_kv_heads: usize,
    theta: f32,
) -> Result<()> {
    rerope_k_cache_shift_down_from(k, n_kv_heads, theta, 0)
}

/// Drop one cached row at `index` and fix RoPE K positions from that row onward.
pub fn slide_rope_kv_window_at(
    cache: &mut LayerKvCache,
    index: usize,
    n_kv_heads: usize,
    theta: f32,
) -> Result<()> {
    cache.truncate_at(index)?;
    if let Some(k) = cache.k.as_mut() {
        rerope_k_cache_shift_down_from(k, n_kv_heads, theta, index)?;
    }
    Ok(())
}

/// Drop one cached token and fix RoPE positions on surviving K rows (V unchanged).
pub fn slide_rope_kv_window_one(
    cache: &mut LayerKvCache,
    n_kv_heads: usize,
    theta: f32,
) -> Result<()> {
    cache.truncate_front(1)?;
    if let Some(k) = cache.k.as_mut() {
        rerope_k_cache_after_front_drop(k, n_kv_heads, theta)?;
    }
    Ok(())
}

/// One KV slot per transformer block.
#[derive(Clone, Default)]
pub struct TransformerKvCache {
    pub layers: Vec<LayerKvCache>,
}

impl TransformerKvCache {
    pub fn new(n_layers: usize) -> Self {
        Self {
            layers: (0..n_layers).map(|_| LayerKvCache::default()).collect(),
        }
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }
}

fn rope_cos_sin_offset(
    position_offset: usize,
    seq_len: usize,
    head_dim: usize,
    theta: f32,
) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = vec![0.0f32; seq_len * half];
    let mut sin = vec![0.0f32; seq_len * half];
    for i in 0..seq_len {
        let pos = position_offset + i;
        for j in 0..half {
            let freq = 1.0 / theta.powf(2.0 * j as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            cos[i * half + j] = angle.cos();
            sin[i * half + j] = angle.sin();
        }
    }
    (cos, sin)
}

/// RoPE with absolute positions `position_offset..position_offset+seq`.
pub fn apply_rope_with_position_offset(
    q: &Tensor,
    k: &Tensor,
    n_heads: usize,
    n_kv_heads: usize,
    theta: f32,
    position_offset: usize,
) -> Result<(Tensor, Tensor)> {
    if q.shape.len() != 2 || k.shape.len() != 2 || q.shape[0] != k.shape[0] {
        return Err(MmnError::Shape {
            message: "apply_rope_with_position_offset expects matching [seq, features]".into(),
        });
    }
    let seq = q.shape[0];
    let d_model = q.shape[1];
    if d_model % n_heads != 0 {
        return Err(MmnError::Shape {
            message: format!("d_model {d_model} not divisible by n_heads {n_heads}"),
        });
    }
    let head_dim = d_model / n_heads;
    let kv_dim = n_kv_heads * head_dim;
    if k.shape[1] != kv_dim {
        return Err(MmnError::Shape {
            message: format!("k width {kv_dim} expected, got {}", k.shape[1]),
        });
    }
    if head_dim % 2 != 0 {
        return Err(MmnError::Shape {
            message: format!("head_dim {head_dim} must be even for RoPE"),
        });
    }
    let half = head_dim / 2;
    let (cos, sin) = rope_cos_sin_offset(position_offset, seq, head_dim, theta);
    let mut q_out = q.data.as_ref().clone();
    let mut k_out = k.data.as_ref().clone();
    for h in 0..n_heads {
        let base = h * head_dim;
        for s in 0..seq {
            for i in 0..half {
                let idx0 = base + 2 * i;
                let idx1 = base + 2 * i + 1;
                let c = cos[s * half + i];
                let sn = sin[s * half + i];
                let x0 = q.data[[s, idx0]];
                let x1 = q.data[[s, idx1]];
                q_out[[s, idx0]] = x0 * c - x1 * sn;
                q_out[[s, idx1]] = x0 * sn + x1 * c;
            }
        }
    }
    for h in 0..n_kv_heads {
        let base = h * head_dim;
        for s in 0..seq {
            for i in 0..half {
                let idx0 = base + 2 * i;
                let idx1 = base + 2 * i + 1;
                let c = cos[s * half + i];
                let sn = sin[s * half + i];
                let x0 = k.data[[s, idx0]];
                let x1 = k.data[[s, idx1]];
                k_out[[s, idx0]] = x0 * c - x1 * sn;
                k_out[[s, idx1]] = x0 * sn + x1 * c;
            }
        }
    }
    Ok((
        Tensor::from_array(q_out, q.requires_grad),
        Tensor::from_array(k_out, k.requires_grad),
    ))
}

fn append_kv(cache: &mut LayerKvCache, k_new: &Tensor, v_new: &Tensor) -> Result<()> {
    match (&cache.k, &cache.v) {
        (None, None) => {
            cache.k = Some(k_new.clone());
            cache.v = Some(v_new.clone());
        }
        (Some(k), Some(v)) => {
            cache.k = Some(concat_sequence_rows(k, k_new)?);
            cache.v = Some(concat_sequence_rows(v, v_new)?);
        }
        _ => {
            return Err(MmnError::Shape {
                message: "KV cache k/v length mismatch".into(),
            });
        }
    }
    Ok(())
}

/// Self-attention where `q` has `q_len` rows at absolute positions `query_start_pos..`.
pub fn scaled_dot_product_attention_with_kv(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_heads: usize,
    n_kv_heads: usize,
    causal: bool,
    query_start_pos: usize,
) -> Result<Tensor> {
    if q.shape.len() != 2 || k.shape.len() != 2 || v.shape.len() != 2 {
        return Err(MmnError::Shape {
            message: "attention_with_kv expects q,k,v [rows, features]".into(),
        });
    }
    let q_len = q.shape[0];
    let kv_len = k.shape[0];
    if v.shape[0] != kv_len {
        return Err(MmnError::Shape {
            message: "attention_with_kv k/v seq_len mismatch".into(),
        });
    }
    let d_model = q.shape[1];
    if d_model % n_heads != 0 {
        return Err(MmnError::Shape {
            message: format!("d_model {d_model} not divisible by n_heads {n_heads}"),
        });
    }
    let head_dim = d_model / n_heads;
    let kv_dim = n_kv_heads * head_dim;
    if k.shape[1] != kv_dim || v.shape[1] != kv_dim {
        return Err(MmnError::Shape {
            message: format!("k,v width {kv_dim} expected"),
        });
    }
    if n_heads % n_kv_heads != 0 {
        return Err(MmnError::Shape {
            message: format!("n_heads {n_heads} must be divisible by n_kv_heads {n_kv_heads}"),
        });
    }
    let scale = (head_dim as f32).sqrt().recip();
    let mut out = vec![0.0f32; q_len * d_model];

    for h in 0..n_heads {
        let q_base = h * head_dim;
        let kv_h = h * n_kv_heads / n_heads;
        let k_base = kv_h * head_dim;
        for s in 0..q_len {
            let abs_pos = query_start_pos + s;
            let mut scores = vec![0.0f32; kv_len];
            for t in 0..kv_len {
                if causal && t > abs_pos {
                    scores[t] = f32::NEG_INFINITY;
                    continue;
                }
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q.data[[s, q_base + d]] * k.data[[t, k_base + d]];
                }
                scores[t] = dot * scale;
            }
            let max = scores
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut row_weights: Vec<f32> = scores.iter().map(|&x| (x - max).exp()).collect();
            let sum: f32 = row_weights.iter().sum();
            if sum > 0.0 {
                for w in &mut row_weights {
                    *w /= sum;
                }
            }
            for d in 0..head_dim {
                let mut ctx = 0.0f32;
                for t in 0..kv_len {
                    ctx += row_weights[t] * v.data[[t, k_base + d]];
                }
                out[s * d_model + q_base + d] = ctx;
            }
        }
    }

    Ok(Tensor::from_array(
        ArrayD::from_shape_vec(IxDyn(&[q_len, d_model]), out).unwrap(),
        q.requires_grad || k.requires_grad || v.requires_grad,
    ))
}

/// Attention forward that appends projected K/V into `cache`.
pub fn mha_forward_with_kv_cache(
    attn: &MultiHeadAttention,
    x: &Tensor,
    cache: &mut LayerKvCache,
    start_pos: usize,
) -> Result<Tensor> {
    let q_lin = attn.q_proj.forward(x)?;
    let k_lin = attn.k_proj.forward(x)?;
    let v = attn.v_proj.forward(x)?;
    let (q, k) = if let Some(theta) = attn.rope_theta {
        apply_rope_with_position_offset(
            &q_lin,
            &k_lin,
            attn.n_heads,
            attn.n_kv_heads,
            theta,
            start_pos,
        )?
    } else {
        (q_lin, k_lin)
    };
    append_kv(cache, &k, &v)?;
    let k_all = cache.k.as_ref().unwrap();
    let v_all = cache.v.as_ref().unwrap();
    let merged = scaled_dot_product_attention_with_kv(
        &q,
        k_all,
        v_all,
        attn.n_heads,
        attn.n_kv_heads,
        attn.causal,
        start_pos,
    )?;
    attn.out_proj.forward(&merged)
}

/// Transformer block forward with per-layer KV cache (inference only).
pub fn block_forward_with_kv_cache(
    block: &TransformerBlock,
    x: &Tensor,
    cache: &mut LayerKvCache,
    start_pos: usize,
) -> Result<Tensor> {
    let x_in = x.clone();
    let h_ln1 = block.ln1.forward(x)?;
    let a = mha_forward_with_kv_cache(&block.attn, &h_ln1, cache, start_pos)?;
    let x2 = x_in.add(&a)?;
    let h2 = block.ln2.forward(&x2)?;
    let f_lin = block.ffn.forward(&h2)?;
    let f_post = gelu(&f_lin);
    let ffn_out = block.ffn2.forward(&f_post)?;
    x2.add(&ffn_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TransformerBlock;
    use rand::SeedableRng;

    #[test]
    fn kv_cache_incremental_matches_full_block_forward() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(77);
        let d_model = 32;
        let block = TransformerBlock::new_rng_rope_gqa(d_model, 4, 2, 64, Some(10_000.0), &mut rng);
        let tokens = 5usize;
        let mut x_full = vec![0.0f32; tokens * d_model];
        for (i, v) in x_full.iter_mut().enumerate() {
            *v = ((i % 17) as f32) * 0.03 - 0.1;
        }
        let x = Tensor::from_array(
            ArrayD::from_shape_vec(IxDyn(&[tokens, d_model]), x_full).unwrap(),
            false,
        );
        let full = block.forward(&x).unwrap();

        let mut cache = LayerKvCache::default();
        let mut outs = Vec::new();
        for pos in 0..tokens {
            let row = Tensor::from_array(
                ArrayD::from_shape_vec(
                    IxDyn(&[1, d_model]),
                    x.data
                        .slice(ndarray::s![pos..pos + 1, ..])
                        .iter()
                        .copied()
                        .collect(),
                )
                .unwrap(),
                false,
            );
            let out = block_forward_with_kv_cache(&block, &row, &mut cache, pos).unwrap();
            outs.push(out);
        }
        let incremental_last = outs.last().unwrap();
        let full_last = Tensor::from_array(
            ArrayD::from_shape_vec(
                IxDyn(&[1, d_model]),
                full.data
                    .slice(ndarray::s![tokens - 1..tokens, ..])
                    .iter()
                    .copied()
                    .collect(),
            )
            .unwrap(),
            false,
        );
        for i in 0..d_model {
            let a = incremental_last.data[[0, i]];
            let b = full_last.data[[0, i]];
            assert!(
                (a - b).abs() < 1e-4,
                "mismatch at dim {i}: incremental={a} full={b}"
            );
        }
        assert_eq!(cache.len(), tokens);
    }

    #[test]
    fn rope_kv_slide_matches_windowed_block_forward() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(101);
        let d_model = 32;
        let block = TransformerBlock::new_rng_rope_gqa(d_model, 4, 2, 64, Some(10_000.0), &mut rng);
        let total = 6usize;
        let mut rows = vec![0.0f32; total * d_model];
        for (i, v) in rows.iter_mut().enumerate() {
            *v = ((i % 19) as f32) * 0.04 - 0.2;
        }
        let row_tensor = |start: usize| {
            Tensor::from_array(
                ArrayD::from_shape_vec(
                    IxDyn(&[1, d_model]),
                    rows[start * d_model..(start + 1) * d_model].to_vec(),
                )
                .unwrap(),
                false,
            )
        };
        let window = Tensor::from_array(
            ArrayD::from_shape_vec(
                IxDyn(&[5, d_model]),
                rows[d_model..6 * d_model].to_vec(),
            )
            .unwrap(),
            false,
        );
        let full = block.forward(&window).unwrap();

        let mut cache = LayerKvCache::default();
        for pos in 0..5 {
            block_forward_with_kv_cache(&block, &row_tensor(pos), &mut cache, pos).unwrap();
        }
        slide_rope_kv_window_one(&mut cache, block.attn.n_kv_heads, 10_000.0).unwrap();
        let slide = block_forward_with_kv_cache(&block, &row_tensor(5), &mut cache, 4).unwrap();

        for i in 0..d_model {
            let a = slide.data[[0, i]];
            let b = full.data[[4, i]];
            assert!((a - b).abs() < 1e-4, "dim {i}: slide={a} full={b}");
        }
    }

    #[test]
    fn rope_kv_slide_at_prefix_matches_windowed_block_forward() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(202);
        let d_model = 32;
        let block = TransformerBlock::new_rng_rope_gqa(d_model, 4, 2, 64, Some(10_000.0), &mut rng);
        let total = 7usize;
        let mut rows = vec![0.0f32; total * d_model];
        for (i, v) in rows.iter_mut().enumerate() {
            *v = ((i % 23) as f32) * 0.03 - 0.15;
        }
        let row_tensor = |start: usize| {
            Tensor::from_array(
                ArrayD::from_shape_vec(
                    IxDyn(&[1, d_model]),
                    rows[start * d_model..(start + 1) * d_model].to_vec(),
                )
                .unwrap(),
                false,
            )
        };
        let mut win_rows = Vec::with_capacity(5 * d_model);
        for idx in [0usize, 2, 3, 4, 5] {
            win_rows.extend_from_slice(&rows[idx * d_model..(idx + 1) * d_model]);
        }
        let window = Tensor::from_array(
            ArrayD::from_shape_vec(IxDyn(&[5, d_model]), win_rows).unwrap(),
            false,
        );
        let full = block.forward(&window).unwrap();

        let mut cache = LayerKvCache::default();
        for pos in 0..5 {
            block_forward_with_kv_cache(&block, &row_tensor(pos), &mut cache, pos).unwrap();
        }
        slide_rope_kv_window_at(&mut cache, 1, block.attn.n_kv_heads, 10_000.0).unwrap();
        let slide = block_forward_with_kv_cache(&block, &row_tensor(5), &mut cache, 4).unwrap();

        for i in 0..d_model {
            let a = slide.data[[0, i]];
            let b = full.data[[4, i]];
            assert!((a - b).abs() < 1e-4, "dim {i}: prefix-slide={a} full={b}");
        }
    }
}
