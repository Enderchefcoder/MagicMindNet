use crate::autoset::{autoset, ModelShape};
use mmn_core::{cross_entropy_grad, embedding_backward, linear_backward, Device, Result, Tensor};
use mmn_data::DatasetType;
use mmn_nn::{
    gelu, gelu_backward, vision_cross_attn_residual, vision_cross_attn_residual_backward,
    BlockForwardCache, Conv2d, CrossAttention, CrossAttentionForwardCache, Embedding, Linear,
    TransformerBlock,
};
use ndarray::ArrayD;
use std::collections::HashMap;

/// Default maximum sequence length for learned position embeddings.
pub const DEFAULT_MAX_SEQ_LEN: usize = 512;

pub use mmn_nn::DEFAULT_ROPE_THETA;

/// Flat 8×8 grayscale patch size for the vision prefix projector.
pub const VISION_PATCH_DIM: usize = 64;
/// Flat 8×8×3 RGB patch size (`NCHW` planes) for the vision conv encoder.
pub const VISION_RGB_DIM: usize = VISION_PATCH_DIM * 3;
pub const VISION_RGB_SPATIAL: usize = 8;
pub const VISION_RGB_CHANNELS: usize = 3;

/// Build a normalized demo patch from UTF-8 bytes (pads with zeros).
pub fn vision_patch_from_text(text: &str) -> Vec<f32> {
    let mut v = vec![0.0f32; VISION_PATCH_DIM];
    for (i, b) in text.bytes().enumerate().take(VISION_PATCH_DIM) {
        v[i] = b as f32 / 255.0;
    }
    v
}

/// Build a normalized 8×8×3 RGB patch (`NCHW` layout) from UTF-8 bytes.
pub fn vision_rgb_patch_from_text(text: &str) -> Vec<f32> {
    let bytes: Vec<u8> = text.bytes().collect();
    let mut v = vec![0.0f32; VISION_RGB_DIM];
    if bytes.is_empty() {
        return v;
    }
    for idx in 0..VISION_PATCH_DIM {
        let b0 = bytes[idx % bytes.len()];
        let b1 = bytes[(idx + 1) % bytes.len()];
        let b2 = bytes[(idx + 2) % bytes.len()];
        v[idx] = b0 as f32 / 255.0;
        v[VISION_PATCH_DIM + idx] = b1 as f32 / 255.0;
        v[2 * VISION_PATCH_DIM + idx] = b2 as f32 / 255.0;
    }
    v
}

/// Load an on-disk image into a normalized 8×8×3 RGB vision patch (`NCHW`).
pub fn vision_rgb_patch_from_image_path(path: &std::path::Path) -> Result<Vec<f32>> {
    mmn_data::rgb_patch_from_image_path(path)
}

/// Load an image as `grid×grid` tiled RGB patches (each 8×8×3 NCHW).
pub fn vision_rgb_patches_from_image_path(
    path: &std::path::Path,
    grid: usize,
) -> Result<Vec<Vec<f32>>> {
    mmn_data::rgb_patches_from_image_path(path, grid)
}

/// Prepend ignored CE targets for `n_patches` vision prefix rows (`target == vocab_size` skips loss).
pub fn targets_with_vision_prefix(
    targets: &[usize],
    n_patches: usize,
    vocab_size: usize,
) -> Vec<usize> {
    let mut out = vec![vocab_size; n_patches];
    out.extend_from_slice(targets);
    out
}

pub struct Chatbot {
    pub shape: ModelShape,
    pub embed: Embedding,
    pub blocks: Vec<TransformerBlock>,
    pub lm_head: Linear,
    pub tokenizer: String,
    pub vision: bool,
    /// Linear patch projector (`vision=true`): `[VISION_PATCH_DIM] → d_model`.
    pub vision_patch_proj: Option<Linear>,
    /// RGB conv patch encoder (`vision=true`): `3×8×8 → 1×8×8` before linear proj.
    pub vision_patch_conv: Option<Conv2d>,
    /// Text→image cross-attention after block 0 when vision patches are present.
    pub vision_cross_attn: Option<CrossAttention>,
    pub device: Device,
    /// RNG seed used at construction (`None` = non-deterministic init).
    pub init_seed: Option<u64>,
    /// When true, use trainable `pos_embed` instead of fixed sinusoidal PE.
    pub use_learned_pos_embed: bool,
    /// When true, apply rotary position embedding in attention (no additive PE).
    pub use_rope: bool,
    pub rope_theta: f32,
    pub max_seq_len: usize,
    pub pos_embed: Option<Embedding>,
}

struct BlockFfnCache {
    block: BlockForwardCache,
}

fn apply_block_lm_grads(
    block: &mut TransformerBlock,
    grads: &[ArrayD<f32>; 10],
    hybrid: &mut Option<&mut mmn_optim::HybridOptimizer>,
    adamw: &mut mmn_optim::AdamW,
    use_hybrid: bool,
    param_id: &mut usize,
) {
    optim_step_weight(
        hybrid,
        adamw,
        use_hybrid,
        param_id,
        &mut block.ffn2.weight,
        &grads[0],
    );
    optim_step_weight(
        hybrid,
        adamw,
        use_hybrid,
        param_id,
        &mut block.ffn.weight,
        &grads[1],
    );
    optim_step_weight(
        hybrid,
        adamw,
        use_hybrid,
        param_id,
        &mut block.attn.out_proj.weight,
        &grads[2],
    );
    optim_step_weight(
        hybrid,
        adamw,
        use_hybrid,
        param_id,
        &mut block.attn.q_proj.weight,
        &grads[3],
    );
    optim_step_weight(
        hybrid,
        adamw,
        use_hybrid,
        param_id,
        &mut block.attn.k_proj.weight,
        &grads[4],
    );
    optim_step_weight(
        hybrid,
        adamw,
        use_hybrid,
        param_id,
        &mut block.attn.v_proj.weight,
        &grads[5],
    );
    optim_step_weight(
        hybrid,
        adamw,
        use_hybrid,
        param_id,
        &mut block.ln2.gamma,
        &grads[6],
    );
    optim_step_weight(
        hybrid,
        adamw,
        use_hybrid,
        param_id,
        &mut block.ln2.beta,
        &grads[7],
    );
    optim_step_weight(
        hybrid,
        adamw,
        use_hybrid,
        param_id,
        &mut block.ln1.gamma,
        &grads[8],
    );
    optim_step_weight(
        hybrid,
        adamw,
        use_hybrid,
        param_id,
        &mut block.ln1.beta,
        &grads[9],
    );
}

fn optim_step_weight(
    hybrid: &mut Option<&mut mmn_optim::HybridOptimizer>,
    adamw: &mut mmn_optim::AdamW,
    use_hybrid: bool,
    param_id: &mut usize,
    weight: &mut Tensor,
    grad_w: &ArrayD<f32>,
) {
    let mut w = weight.data.as_ref().clone();
    if use_hybrid {
        hybrid
            .as_mut()
            .expect("hybrid optimizer required")
            .step(*param_id, &mut w, grad_w);
    } else {
        adamw.step_param(*param_id, &mut w, grad_w);
    }
    *param_id += 1;
    *weight = Tensor::from_array(w, true);
}

fn apply_cross_attn_lm_grads(
    cross: &mut CrossAttention,
    grads: &[ArrayD<f32>; 4],
    hybrid: &mut Option<&mut mmn_optim::HybridOptimizer>,
    adamw: &mut mmn_optim::AdamW,
    use_hybrid: bool,
    param_id: &mut usize,
) {
    optim_step_weight(
        hybrid,
        adamw,
        use_hybrid,
        param_id,
        &mut cross.out_proj.weight,
        &grads[0],
    );
    optim_step_weight(
        hybrid,
        adamw,
        use_hybrid,
        param_id,
        &mut cross.q_proj.weight,
        &grads[1],
    );
    optim_step_weight(
        hybrid,
        adamw,
        use_hybrid,
        param_id,
        &mut cross.k_proj.weight,
        &grads[2],
    );
    optim_step_weight(
        hybrid,
        adamw,
        use_hybrid,
        param_id,
        &mut cross.v_proj.weight,
        &grads[3],
    );
}

impl Chatbot {
    pub fn new(
        vision: bool,
        autoset_budget: Option<&str>,
        vocab_size: usize,
        n_layer: Option<usize>,
        d_model: Option<usize>,
    ) -> Self {
        Self::new_with_seed(vision, autoset_budget, vocab_size, n_layer, d_model, None)
    }

    pub fn new_with_seed(
        vision: bool,
        autoset_budget: Option<&str>,
        vocab_size: usize,
        n_layer: Option<usize>,
        d_model: Option<usize>,
        seed: Option<u64>,
    ) -> Self {
        Self::new_with_pe_options(
            vision,
            autoset_budget,
            vocab_size,
            n_layer,
            d_model,
            seed,
            false,
            DEFAULT_MAX_SEQ_LEN,
        )
    }

    pub fn new_with_pe_options(
        vision: bool,
        autoset_budget: Option<&str>,
        vocab_size: usize,
        n_layer: Option<usize>,
        d_model: Option<usize>,
        seed: Option<u64>,
        use_learned_pos_embed: bool,
        max_seq_len: usize,
    ) -> Self {
        Self::new_with_position_options(
            vision,
            autoset_budget,
            vocab_size,
            n_layer,
            d_model,
            seed,
            use_learned_pos_embed,
            max_seq_len,
            false,
            DEFAULT_ROPE_THETA,
        )
    }

    pub fn new_with_position_options(
        vision: bool,
        autoset_budget: Option<&str>,
        vocab_size: usize,
        n_layer: Option<usize>,
        d_model: Option<usize>,
        seed: Option<u64>,
        use_learned_pos_embed: bool,
        max_seq_len: usize,
        use_rope: bool,
        rope_theta: f32,
    ) -> Self {
        if use_learned_pos_embed && use_rope {
            panic!("Chatbot cannot use both use_learned_pos_embed and use_rope");
        }
        let mut rng = mmn_nn::rng_from_seed(seed);
        let shape = if let Some(b) = autoset_budget {
            autoset(b, vocab_size)
        } else {
            ModelShape {
                n_layer: n_layer.unwrap_or(4),
                d_model: d_model.unwrap_or(128),
                n_heads: 4,
                ffn_dim: d_model.unwrap_or(128) * 4,
                vocab_size,
                estimated_params: 0,
            }
        };
        let mut blocks = Vec::new();
        let rope = if use_rope { Some(rope_theta) } else { None };
        for _ in 0..shape.n_layer {
            blocks.push(TransformerBlock::new_rng_rope(
                shape.d_model,
                shape.n_heads,
                shape.ffn_dim,
                rope,
                &mut rng,
            ));
        }
        let pos_embed = if use_learned_pos_embed {
            Some(Embedding::new_rng(max_seq_len, shape.d_model, &mut rng))
        } else {
            None
        };
        let vision_patch_proj = if vision {
            Some(Linear::new_rng(
                VISION_PATCH_DIM,
                shape.d_model,
                &mut rng,
            ))
        } else {
            None
        };
        let vision_patch_conv = if vision {
            Some(Conv2d::new(VISION_RGB_CHANNELS, 1, 3))
        } else {
            None
        };
        let vision_cross_attn = if vision {
            Some(CrossAttention::new_rng(
                shape.d_model,
                shape.n_heads,
                &mut rng,
            ))
        } else {
            None
        };
        Self {
            embed: Embedding::new_rng(shape.vocab_size, shape.d_model, &mut rng),
            blocks,
            lm_head: Linear::new_rng(shape.d_model, shape.vocab_size, &mut rng),
            shape,
            tokenizer: "ChatXML".into(),
            vision,
            vision_patch_proj,
            vision_patch_conv,
            vision_cross_attn,
            device: Device::Cpu,
            init_seed: seed,
            use_learned_pos_embed,
            use_rope,
            rope_theta,
            max_seq_len,
            pos_embed,
        }
    }

    fn apply_position_encoding(&self, h: Tensor) -> Result<Tensor> {
        if self.use_rope {
            return Ok(h);
        }
        if let Some(pe) = &self.pos_embed {
            let seq = h.shape[0];
            if seq > self.max_seq_len {
                return Err(mmn_core::MmnError::Shape {
                    message: format!(
                        "sequence length {seq} exceeds max_seq_len {}",
                        self.max_seq_len
                    ),
                });
            }
            let pos_ids: Vec<usize> = (0..seq).collect();
            let pos = pe.forward(&pos_ids)?;
            h.add(&pos)
        } else {
            mmn_nn::add_sinusoidal_position_encoding(&h)
        }
    }

    pub fn parameters(&self) -> usize {
        let base = if self.shape.estimated_params > 0 {
            self.shape.estimated_params
        } else {
            crate::autoset::estimate_params(
                self.shape.n_layer,
                self.shape.d_model,
                self.shape.ffn_dim,
                self.shape.vocab_size,
                self.shape.n_heads,
            )
        };
        let pe = if self.use_learned_pos_embed {
            self.max_seq_len * self.shape.d_model
        } else {
            0
        };
        let vision = self
            .vision_patch_proj
            .as_ref()
            .map(|p| p.in_features * p.out_features + p.out_features)
            .unwrap_or(0)
            + self
                .vision_patch_conv
                .as_ref()
                .map(|c| c.out_ch * c.in_ch * c.kernel * c.kernel)
                .unwrap_or(0)
            + self
                .vision_cross_attn
                .as_ref()
                .map(|c| 4 * c.d_model * c.d_model)
                .unwrap_or(0);
        base + pe + vision
    }

    pub fn vision_patch_dim(&self) -> usize {
        VISION_PATCH_DIM
    }

    pub fn vision_rgb_dim(&self) -> usize {
        VISION_RGB_DIM
    }

    pub fn has_vision_patch_encoder(&self) -> bool {
        self.vision_patch_proj.is_some()
    }

    pub fn has_vision_rgb_conv(&self) -> bool {
        self.vision_patch_conv.is_some()
    }

    pub fn has_vision_cross_attn(&self) -> bool {
        self.vision_cross_attn.is_some()
    }

    pub fn layer_size(&self) -> usize {
        self.shape.n_layer
    }

    pub fn has_vision(&self) -> bool {
        self.vision
    }

    fn rgb_patch_to_nchw(patch: &[f32]) -> Result<Tensor> {
        if patch.len() != VISION_RGB_DIM {
            return Err(mmn_core::MmnError::Shape {
                message: format!(
                    "RGB vision patch has length {}; expected {VISION_RGB_DIM}",
                    patch.len()
                ),
            });
        }
        Ok(Tensor::from_array(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[1, VISION_RGB_CHANNELS, VISION_RGB_SPATIAL, VISION_RGB_SPATIAL]),
                patch.to_vec(),
            )
            .unwrap(),
            true,
        ))
    }

    fn conv_flatten_to_rows(conv_out: &Tensor) -> Result<Vec<f32>> {
        let spatial = VISION_RGB_SPATIAL * VISION_RGB_SPATIAL;
        let view = conv_out.data.view();
        let n = conv_out.shape[0];
        let mut flat = vec![0.0f32; n * VISION_PATCH_DIM];
        for i in 0..n {
            for j in 0..spatial {
                flat[i * VISION_PATCH_DIM + j] = view[[i, 0, j / VISION_RGB_SPATIAL, j % VISION_RGB_SPATIAL]];
            }
        }
        Ok(flat)
    }

    fn patches_to_linear_input(
        &self,
        patches: &[Vec<f32>],
    ) -> Result<(Tensor, Option<Tensor>)> {
        let proj = self
            .vision_patch_proj
            .as_ref()
            .ok_or_else(|| mmn_core::MmnError::Other {
                message: "vision patches require Chatbot(vision=True)".into(),
            })?;
        if patches.is_empty() {
            return Err(mmn_core::MmnError::Shape {
                message: "vision patches list is empty".into(),
            });
        }
        let n = patches.len();
        let mut linear_rows = vec![0.0f32; n * VISION_PATCH_DIM];
        let mut conv_inputs: Option<Vec<f32>> = None;
        for (i, patch) in patches.iter().enumerate() {
            if patch.len() == VISION_RGB_DIM {
                let conv = self.vision_patch_conv.as_ref().ok_or_else(|| {
                    mmn_core::MmnError::Shape {
                        message: format!(
                            "RGB patch length {VISION_RGB_DIM} requires vision_patch_conv; reload a checkpoint that includes vision_patch_conv or pass a {VISION_PATCH_DIM}-float grayscale patch"
                        ),
                    }
                })?;
                let x = Self::rgb_patch_to_nchw(patch)?;
                let conv_out = conv.forward(&x)?;
                let row = Self::conv_flatten_to_rows(&conv_out)?;
                linear_rows[i * VISION_PATCH_DIM..(i + 1) * VISION_PATCH_DIM]
                    .copy_from_slice(&row);
                let mut stacked = conv_inputs.take().unwrap_or_default();
                stacked.extend_from_slice(patch);
                conv_inputs = Some(stacked);
            } else if patch.len() == VISION_PATCH_DIM {
                linear_rows[i * VISION_PATCH_DIM..(i + 1) * VISION_PATCH_DIM]
                    .copy_from_slice(patch);
            } else {
                return Err(mmn_core::MmnError::Shape {
                    message: format!(
                        "vision patch {i} has length {}; expected {VISION_PATCH_DIM} or {VISION_RGB_DIM}",
                        patch.len()
                    ),
                });
            }
        }
        let linear_input = Tensor::from_array(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[n, VISION_PATCH_DIM]),
                linear_rows,
            )
            .unwrap(),
            true,
        );
        let conv_input = conv_inputs.map(|stacked| {
            Tensor::from_array(
                ndarray::ArrayD::from_shape_vec(
                    ndarray::IxDyn(&[
                        n,
                        VISION_RGB_CHANNELS,
                        VISION_RGB_SPATIAL,
                        VISION_RGB_SPATIAL,
                    ]),
                    stacked,
                )
                .unwrap(),
                true,
            )
        });
        let _ = proj;
        Ok((linear_input, conv_input))
    }

    fn embed_with_optional_patches(
        &self,
        token_ids: &[usize],
        patches: Option<&[Vec<f32>]>,
    ) -> Result<(Tensor, usize, Option<Tensor>, Option<Tensor>)> {
        let mut h = self.embed.forward(token_ids)?;
        let n_patch = patches.map(|p| p.len()).unwrap_or(0);
        let (patch_input, conv_input) = if n_patch > 0 {
            let patches = patches.unwrap();
            let (linear_input, conv_input) = self.patches_to_linear_input(patches)?;
            let h_patch = self
                .vision_patch_proj
                .as_ref()
                .unwrap()
                .forward(&linear_input)?;
            h = mmn_nn::concat_sequence_rows(&h_patch, &h)?;
            (Some(linear_input), conv_input)
        } else {
            (None, None)
        };
        Ok((h, n_patch, patch_input, conv_input))
    }

    pub fn uses_causal_attention(&self) -> bool {
        self.blocks
            .first()
            .map(|b| b.attn.causal)
            .unwrap_or(true)
    }

    pub fn uses_rope(&self) -> bool {
        self.use_rope
    }

    pub fn forward_hidden(&self, token_ids: &[usize]) -> Result<Tensor> {
        self.forward_hidden_with_patches(token_ids, None)
    }

    pub fn forward_hidden_with_patches(
        &self,
        token_ids: &[usize],
        patches: Option<&[Vec<f32>]>,
    ) -> Result<Tensor> {
        let (mut h, n_patch, _, _) = self.embed_with_optional_patches(token_ids, patches)?;
        h = self.apply_position_encoding(h)?;
        for (i, block) in self.blocks.iter().enumerate() {
            h = block.forward(&h)?;
            if i == 0
                && n_patch > 0
                && self.vision_cross_attn.is_some()
            {
                h = vision_cross_attn_residual(
                    self.vision_cross_attn.as_ref().unwrap(),
                    &h,
                    n_patch,
                )?
                .0;
            }
        }
        Ok(h)
    }

    pub fn forward_logits(&self, token_ids: &[usize]) -> Result<Tensor> {
        self.forward_logits_with_patches(token_ids, None)
    }

    pub fn forward_logits_with_patches(
        &self,
        token_ids: &[usize],
        patches: Option<&[Vec<f32>]>,
    ) -> Result<Tensor> {
        self.lm_head
            .forward(&self.forward_hidden_with_patches(token_ids, patches)?)
    }

    pub fn loss_on_batch(&self, token_ids: &[usize], targets: &[usize]) -> Result<f32> {
        self.loss_on_batch_with_patches(token_ids, targets, None)
    }

    pub fn loss_on_batch_with_patches(
        &self,
        token_ids: &[usize],
        targets: &[usize],
        patches: Option<&[Vec<f32>]>,
    ) -> Result<f32> {
        let logits = self.forward_logits_with_patches(token_ids, patches)?;
        let loss_t = logits.cross_entropy_loss(targets)?;
        Ok(loss_t.data.as_slice().unwrap()[0])
    }

    /// Backward pass; either apply optimizer immediately or accumulate grads for `batch_size` > 1.
    #[allow(clippy::too_many_arguments)]
    pub fn train_step_lm(
        &mut self,
        token_ids: &[usize],
        targets: &[usize],
        hybrid: &mut mmn_optim::HybridOptimizer,
        adamw: &mut mmn_optim::AdamW,
        use_hybrid: bool,
        param_id_base: &mut usize,
        mut accum: Option<&mut mmn_optim::GradAccumulator>,
        patches: Option<&[Vec<f32>]>,
    ) -> Result<f32> {
        if let Some(acc) = accum.as_mut() {
            acc.begin_micro_batch();
            let loss_val = self.backward_lm_accumulate(token_ids, targets, patches, acc)?;
            acc.finish_micro_batch();
            return Ok(loss_val);
        }
        self.backward_lm(
            token_ids,
            targets,
            patches,
            hybrid,
            adamw,
            use_hybrid,
            param_id_base,
        )
    }

    /// Apply averaged accumulated gradients after `batch_size` micro-batches.
    pub fn apply_accumulated_lm_grads(
        &mut self,
        accum: &mmn_optim::GradAccumulator,
        hybrid: &mut mmn_optim::HybridOptimizer,
        adamw: &mut mmn_optim::AdamW,
        use_hybrid: bool,
        param_id_base: &mut usize,
    ) -> Result<()> {
        let mut i = 0usize;
        let mut hybrid_opt = Some(hybrid);
        let grad = accum.averaged_grad(i);
        optim_step_weight(
            &mut hybrid_opt,
            adamw,
            use_hybrid,
            param_id_base,
            &mut self.lm_head.weight,
            &grad,
        );
        i += 1;

        let n_blocks = self.blocks.len();
        for (rev_pos, block) in self.blocks.iter_mut().rev().enumerate() {
            if rev_pos == n_blocks - 1 {
                if let Some(cross) = self.vision_cross_attn.as_mut() {
                    let cg = [
                        accum.averaged_grad(i),
                        accum.averaged_grad(i + 1),
                        accum.averaged_grad(i + 2),
                        accum.averaged_grad(i + 3),
                    ];
                    apply_cross_attn_lm_grads(
                        cross,
                        &cg,
                        &mut hybrid_opt,
                        adamw,
                        use_hybrid,
                        param_id_base,
                    );
                    i += 4;
                }
            }
            let g = [
                accum.averaged_grad(i),
                accum.averaged_grad(i + 1),
                accum.averaged_grad(i + 2),
                accum.averaged_grad(i + 3),
                accum.averaged_grad(i + 4),
                accum.averaged_grad(i + 5),
                accum.averaged_grad(i + 6),
                accum.averaged_grad(i + 7),
                accum.averaged_grad(i + 8),
                accum.averaged_grad(i + 9),
            ];
            apply_block_lm_grads(
                block,
                &g,
                &mut hybrid_opt,
                adamw,
                use_hybrid,
                param_id_base,
            );
            i += 10;
        }

        let grad = accum.averaged_grad(i);
        optim_step_weight(
            &mut hybrid_opt,
            adamw,
            use_hybrid,
            param_id_base,
            &mut self.embed.weight,
            &grad,
        );
        if self.use_learned_pos_embed {
            i += 1;
            let grad = accum.averaged_grad(i);
            optim_step_weight(
                &mut hybrid_opt,
                adamw,
                use_hybrid,
                param_id_base,
                &mut self.pos_embed.as_mut().unwrap().weight,
                &grad,
            );
        }
        if self.vision_patch_proj.is_some() {
            i += 1;
            let grad = accum.averaged_grad(i);
            optim_step_weight(
                &mut hybrid_opt,
                adamw,
                use_hybrid,
                param_id_base,
                &mut self.vision_patch_proj.as_mut().unwrap().weight,
                &grad,
            );
        }
        if self.vision_patch_conv.is_some() {
            i += 1;
            let grad = accum.averaged_grad(i);
            optim_step_weight(
                &mut hybrid_opt,
                adamw,
                use_hybrid,
                param_id_base,
                &mut self.vision_patch_conv.as_mut().unwrap().weight,
                &grad,
            );
        }
        Ok(())
    }

    fn backward_lm_accumulate(
        &mut self,
        token_ids: &[usize],
        targets: &[usize],
        patches: Option<&[Vec<f32>]>,
        acc: &mut mmn_optim::GradAccumulator,
    ) -> Result<f32> {
        let (loss_val, grads) = self.backward_lm_grads(token_ids, targets, patches)?;
        for g in grads {
            acc.add_param_grad(&g);
        }
        Ok(loss_val)
    }

    fn backward_lm(
        &mut self,
        token_ids: &[usize],
        targets: &[usize],
        patches: Option<&[Vec<f32>]>,
        hybrid: &mut mmn_optim::HybridOptimizer,
        adamw: &mut mmn_optim::AdamW,
        use_hybrid: bool,
        param_id_base: &mut usize,
    ) -> Result<f32> {
        let (loss_val, grads) = self.backward_lm_grads(token_ids, targets, patches)?;
        let mut i = 0usize;
        let mut hybrid_opt = Some(hybrid);
        optim_step_weight(
            &mut hybrid_opt,
            adamw,
            use_hybrid,
            param_id_base,
            &mut self.lm_head.weight,
            &grads[i],
        );
        i += 1;
        let n_blocks = self.blocks.len();
        for (rev_pos, block) in self.blocks.iter_mut().rev().enumerate() {
            if rev_pos == n_blocks - 1 {
                if let Some(cross) = self.vision_cross_attn.as_mut() {
                    let cg = [
                        grads[i].clone(),
                        grads[i + 1].clone(),
                        grads[i + 2].clone(),
                        grads[i + 3].clone(),
                    ];
                    apply_cross_attn_lm_grads(
                        cross,
                        &cg,
                        &mut hybrid_opt,
                        adamw,
                        use_hybrid,
                        param_id_base,
                    );
                    i += 4;
                }
            }
            let g = [
                grads[i].clone(),
                grads[i + 1].clone(),
                grads[i + 2].clone(),
                grads[i + 3].clone(),
                grads[i + 4].clone(),
                grads[i + 5].clone(),
                grads[i + 6].clone(),
                grads[i + 7].clone(),
                grads[i + 8].clone(),
                grads[i + 9].clone(),
            ];
            apply_block_lm_grads(
                block,
                &g,
                &mut hybrid_opt,
                adamw,
                use_hybrid,
                param_id_base,
            );
            i += 10;
        }
        optim_step_weight(
            &mut hybrid_opt,
            adamw,
            use_hybrid,
            param_id_base,
            &mut self.embed.weight,
            &grads[i],
        );
        if self.use_learned_pos_embed {
            i += 1;
            optim_step_weight(
                &mut hybrid_opt,
                adamw,
                use_hybrid,
                param_id_base,
                &mut self.pos_embed.as_mut().unwrap().weight,
                &grads[i],
            );
        }
        if self.vision_patch_proj.is_some() {
            i += 1;
            optim_step_weight(
                &mut hybrid_opt,
                adamw,
                use_hybrid,
                param_id_base,
                &mut self.vision_patch_proj.as_mut().unwrap().weight,
                &grads[i],
            );
        }
        if self.vision_patch_conv.is_some() {
            i += 1;
            optim_step_weight(
                &mut hybrid_opt,
                adamw,
                use_hybrid,
                param_id_base,
                &mut self.vision_patch_conv.as_mut().unwrap().weight,
                &grads[i],
            );
        }
        Ok(loss_val)
    }

    fn backward_lm_grads(
        &mut self,
        token_ids: &[usize],
        targets: &[usize],
        patches: Option<&[Vec<f32>]>,
    ) -> Result<(f32, Vec<ArrayD<f32>>)> {
        let (mut h, n_patch, patch_input, conv_input) =
            self.embed_with_optional_patches(token_ids, patches)?;
        h = self.apply_position_encoding(h)?;
        let seq = token_ids.len();
        let mut caches = Vec::with_capacity(self.blocks.len());
        let mut cross_cache: Option<CrossAttentionForwardCache> = None;
        for (i, block) in self.blocks.iter().enumerate() {
            let (out, cache) = block.forward_with_cache(&h)?;
            caches.push(BlockFfnCache { block: cache });
            h = out;
            if i == 0
                && n_patch > 0
                && self.vision_cross_attn.is_some()
            {
                let (h_new, xc) = vision_cross_attn_residual(
                    self.vision_cross_attn.as_ref().unwrap(),
                    &h,
                    n_patch,
                )?;
                h = h_new;
                cross_cache = Some(xc);
            }
        }

        let logits = self.lm_head.forward(&h)?;
        let loss = logits.cross_entropy_loss(targets)?;
        let loss_val = loss.data.as_slice().unwrap()[0];

        let grad_logits = cross_entropy_grad(&logits, targets)?;
        let (grad_lm_w, mut grad_h) = linear_backward(
            h.data.as_ref(),
            self.lm_head.weight.data.as_ref(),
            &grad_logits,
        )?;
        let mut grads = vec![grad_lm_w];

        let n_blocks = self.blocks.len();
        for bi in (0..n_blocks).rev() {
            let block = &self.blocks[bi];
            let cache = &caches[bi];
            let mut cross_grads: Option<[ArrayD<f32>; 4]> = None;
            if bi == 0 {
                if let (Some(cross), Some(xc)) =
                    (self.vision_cross_attn.as_ref(), cross_cache.as_ref())
                {
                    if n_patch > 0 {
                        let (grad_h_new, cg) = vision_cross_attn_residual_backward(
                            cross,
                            xc,
                            &grad_h,
                            n_patch,
                        )?;
                        grad_h = grad_h_new;
                        cross_grads = Some(cg);
                    }
                }
            }
            let (grad_h_block, block_grads) =
                block.backward_attn_ffn(&cache.block, &grad_h)?;
            if bi == 0 && self.vision_cross_attn.is_some() {
                let cg = cross_grads.unwrap_or_else(|| {
                    let z = ArrayD::zeros(
                        self.vision_cross_attn.as_ref().unwrap().out_proj.weight.data.shape(),
                    );
                    [z.clone(), z.clone(), z.clone(), z]
                });
                for g in cg {
                    grads.push(g);
                }
            }
            for g in block_grads {
                grads.push(g);
            }
            grad_h = grad_h_block;
        }

        let grad_h2 = grad_h
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| mmn_core::MmnError::Shape {
                message: e.to_string(),
            })?;
        let d_model = self.shape.d_model;
        let grad_suffix = if n_patch > 0 {
            let mut suffix = ndarray::Array2::<f32>::zeros((seq, d_model));
            for r in 0..seq {
                for c in 0..d_model {
                    suffix[[r, c]] = grad_h2[[n_patch + r, c]];
                }
            }
            suffix.into_dyn()
        } else {
            grad_h.clone()
        };

        let grad_embed_w = embedding_backward(
            token_ids,
            &grad_suffix,
            self.shape.vocab_size,
            self.shape.d_model,
        );
        grads.push(grad_embed_w);
        if self.use_learned_pos_embed {
            let pos_ids: Vec<usize> = (0..n_patch + seq).collect();
            let grad_pos = embedding_backward(
                &pos_ids,
                &grad_h,
                self.max_seq_len,
                self.shape.d_model,
            );
            grads.push(grad_pos);
        }
        if let Some(proj) = &self.vision_patch_proj {
            let (grad_proj, grad_conv) = if n_patch > 0 {
                let mut prefix = ndarray::Array2::<f32>::zeros((n_patch, d_model));
                for r in 0..n_patch {
                    for c in 0..d_model {
                        prefix[[r, c]] = grad_h2[[r, c]];
                    }
                }
                let (grad_w, grad_linear_in) = linear_backward(
                    patch_input.as_ref().unwrap().data.as_ref(),
                    proj.weight.data.as_ref(),
                    &prefix.into_dyn(),
                )?;
                let grad_conv = if let (Some(conv), Some(conv_in)) =
                    (self.vision_patch_conv.as_ref(), conv_input.as_ref())
                {
                    let gli = grad_linear_in
                        .view()
                        .into_dimensionality::<ndarray::Ix2>()
                        .map_err(|e| mmn_core::MmnError::Shape {
                            message: e.to_string(),
                        })?;
                    let mut grad_conv_out =
                        ndarray::Array4::<f32>::zeros((n_patch, 1, VISION_RGB_SPATIAL, VISION_RGB_SPATIAL));
                    for i in 0..n_patch {
                        for j in 0..VISION_PATCH_DIM {
                            grad_conv_out[[i, 0, j / VISION_RGB_SPATIAL, j % VISION_RGB_SPATIAL]] =
                                gli[[i, j]];
                        }
                    }
                    let (_, grad_conv_w) =
                        conv.backward(conv_in, &grad_conv_out.into_dyn())?;
                    Some(grad_conv_w)
                } else {
                    None
                };
                (grad_w, grad_conv)
            } else {
                (
                    ArrayD::zeros(proj.weight.data.shape()),
                    None,
                )
            };
            grads.push(grad_proj);
            if self.vision_patch_conv.is_some() {
                grads.push(grad_conv.unwrap_or_else(|| {
                    ArrayD::zeros(
                        self.vision_patch_conv.as_ref().unwrap().weight.data.shape(),
                    )
                }));
            }
        }
        Ok((loss_val, grads))
    }
}

pub struct Classifier {
    pub backbone: Linear,
    pub head: Linear,
    pub labels: Vec<String>,
    pub input_dim: usize,
    pub init_seed: Option<u64>,
}

impl Classifier {
    pub fn new(num_labels: usize, input_dim: usize) -> Self {
        let labels = (0..num_labels).map(|i| format!("class_{i}")).collect();
        Self::with_labels(labels, input_dim)
    }

    pub fn with_labels(labels: Vec<String>, input_dim: usize) -> Self {
        Self::with_labels_seed(labels, input_dim, None)
    }

    pub fn with_labels_seed(mut labels: Vec<String>, input_dim: usize, seed: Option<u64>) -> Self {
        if labels.is_empty() {
            labels.push("class_0".into());
        }
        let n = labels.len();
        let mut rng = mmn_nn::rng_from_seed(seed);
        Self {
            backbone: Linear::new_rng(input_dim, 128, &mut rng),
            head: Linear::new_rng(128, n, &mut rng),
            labels,
            input_dim,
            init_seed: seed,
        }
    }

    pub fn from_classification_dataset(
        ds: &mmn_data::DatasetClassification,
        input_dim: usize,
    ) -> Self {
        Self::with_labels_seed(ds.unique_labels(), input_dim, None)
    }

    pub fn from_classification_dataset_seed(
        ds: &mmn_data::DatasetClassification,
        input_dim: usize,
        seed: Option<u64>,
    ) -> Self {
        Self::with_labels_seed(ds.unique_labels(), input_dim, seed)
    }

    pub fn encode_text(&self, text: &str) -> Tensor {
        let mut v = vec![0f32; self.input_dim];
        for (i, b) in text.bytes().enumerate().take(self.input_dim) {
            v[i] = (b as f32) / 255.0;
        }
        Tensor::from_array(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, self.input_dim]), v).unwrap(),
            false,
        )
    }

    pub fn label_index(&self, tag: &str) -> Option<usize> {
        self.labels.iter().position(|l| l == tag)
    }

    fn forward_cached(&self, text: &str) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let x = self.encode_text(text);
        let h_lin = self.backbone.forward(&x)?;
        let h = gelu(&h_lin);
        let logits = self.head.forward(&h)?;
        Ok((x, h_lin, h, logits))
    }

    pub fn forward_logits(&self, x: &Tensor) -> Result<Tensor> {
        let h = gelu(&self.backbone.forward(x)?);
        self.head.forward(&h)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_logits(x)?.softmax(1)
    }

    pub fn loss_on_label(&self, text: &str, label_idx: usize) -> Result<f32> {
        let (_, _, _, logits) = self.forward_cached(text)?;
        let loss = logits.cross_entropy_loss(&[label_idx])?;
        Ok(loss.data.as_slice().unwrap()[0])
    }

    fn backward_classifier_grads(
        &self,
        text: &str,
        label_idx: usize,
    ) -> Result<(f32, ArrayD<f32>, ArrayD<f32>)> {
        let (x, h_lin, h, logits) = self.forward_cached(text)?;
        let loss_val = logits.cross_entropy_loss(&[label_idx])?.data.as_slice().unwrap()[0];
        let grad_logits = cross_entropy_grad(&logits, &[label_idx])?;
        let (grad_head_w, grad_h) = linear_backward(
            h.data.as_ref(),
            self.head.weight.data.as_ref(),
            &grad_logits,
        )?;
        let grad_h_lin = gelu_backward(&h_lin, &grad_h);
        let (grad_back_w, _) = linear_backward(
            x.data.as_ref(),
            self.backbone.weight.data.as_ref(),
            &grad_h_lin,
        )?;
        Ok((loss_val, grad_head_w, grad_back_w))
    }

    /// CE backward; apply AdamW immediately or accumulate for `TrainConfig.batch_size` > 1.
    pub fn train_step(
        &mut self,
        text: &str,
        label_idx: usize,
        adamw: &mut mmn_optim::AdamW,
        param_id: &mut usize,
        mut accum: Option<&mut mmn_optim::GradAccumulator>,
    ) -> Result<f32> {
        if let Some(acc) = accum.as_mut() {
            acc.begin_micro_batch();
            let (loss_val, grad_head_w, grad_back_w) =
                self.backward_classifier_grads(text, label_idx)?;
            acc.add_param_grad(&grad_head_w);
            acc.add_param_grad(&grad_back_w);
            acc.finish_micro_batch();
            return Ok(loss_val);
        }
        let (loss_val, grad_head_w, grad_back_w) = self.backward_classifier_grads(text, label_idx)?;
        let mut hybrid_opt = None;
        optim_step_weight(
            &mut hybrid_opt,
            adamw,
            false,
            param_id,
            &mut self.head.weight,
            &grad_head_w,
        );
        optim_step_weight(
            &mut hybrid_opt,
            adamw,
            false,
            param_id,
            &mut self.backbone.weight,
            &grad_back_w,
        );
        Ok(loss_val)
    }

    /// Apply averaged accumulated classifier gradients after `batch_size` micro-batches.
    pub fn apply_accumulated_classifier_grads(
        &mut self,
        accum: &mmn_optim::GradAccumulator,
        adamw: &mut mmn_optim::AdamW,
        param_id: &mut usize,
    ) -> Result<()> {
        let grad_head = accum.averaged_grad(0);
        let mut hybrid_opt = None;
        optim_step_weight(
            &mut hybrid_opt,
            adamw,
            false,
            param_id,
            &mut self.head.weight,
            &grad_head,
        );
        let grad_back = accum.averaged_grad(1);
        optim_step_weight(
            &mut hybrid_opt,
            adamw,
            false,
            param_id,
            &mut self.backbone.weight,
            &grad_back,
        );
        Ok(())
    }

    pub fn predict_text(&self, text: &str) -> Result<HashMap<String, f32>> {
        let x = self.encode_text(text);
        let probs = self.forward(&x)?;
        let mut m = HashMap::new();
        let n = self.labels.len();
        for (i, label) in self.labels.iter().enumerate().take(n) {
            let p = probs.data[[0, i.min(probs.shape[1].saturating_sub(1))]];
            m.insert(label.clone(), p);
        }
        if m.is_empty() {
            for l in &self.labels {
                m.insert(l.clone(), 1.0 / self.labels.len() as f32);
            }
        }
        Ok(m)
    }
}

#[cfg(test)]
mod chatbot_tests {
    use super::*;

    #[test]
    fn train_step_updates_embed_and_ffn2() {
        use mmn_optim::{AdamW, AdamWConfig, HybridOptimizer, MuonConfig};
        let mut model = Chatbot::new(false, None, 64, Some(2), Some(16));
        let tokens: Vec<usize> = (0..4).collect();
        let targets: Vec<usize> = (1..5).collect();
        let embed_before: Vec<f32> = model.embed.weight.data.iter().copied().collect();
        let ffn_before: Vec<f32> = model.blocks[1].ffn2.weight.data.iter().copied().collect();
        let mut hybrid = HybridOptimizer::new(MuonConfig::default(), AdamWConfig::default());
        let mut adamw = AdamW::new(AdamWConfig { lr: 0.01, ..Default::default() });
        model
            .train_step_lm(&tokens, &targets, &mut hybrid, &mut adamw, false, &mut 0, None, None)
            .unwrap();
        let embed_after: Vec<f32> = model.embed.weight.data.iter().copied().collect();
        let ffn_after: Vec<f32> = model.blocks[1].ffn2.weight.data.iter().copied().collect();
        assert_ne!(embed_before, embed_after, "embed should get gradients");
        assert_ne!(ffn_before, ffn_after, "ffn2 should get gradients");
    }

    #[test]
    fn train_step_updates_all_blocks_ffn2() {
        use mmn_optim::{AdamW, AdamWConfig, HybridOptimizer, MuonConfig};
        let mut model = Chatbot::new(false, None, 64, Some(2), Some(16));
        let tokens: Vec<usize> = (0..4).collect();
        let targets: Vec<usize> = (1..5).collect();
        let w0_before: Vec<f32> = model.blocks[0].ffn2.weight.data.iter().copied().collect();
        let w1_before: Vec<f32> = model.blocks[1].ffn2.weight.data.iter().copied().collect();
        let mut hybrid = HybridOptimizer::new(MuonConfig::default(), AdamWConfig::default());
        let mut adamw = AdamW::new(AdamWConfig { lr: 0.01, ..Default::default() });
        model
            .train_step_lm(&tokens, &targets, &mut hybrid, &mut adamw, false, &mut 0, None, None)
            .unwrap();
        let w0_after: Vec<f32> = model.blocks[0].ffn2.weight.data.iter().copied().collect();
        let w1_after: Vec<f32> = model.blocks[1].ffn2.weight.data.iter().copied().collect();
        assert_ne!(w0_before, w0_after, "first block ffn2 should get gradients");
        assert_ne!(w1_before, w1_after, "last block ffn2 should get gradients");
    }

    #[test]
    fn train_step_updates_attn_q_weights() {
        use mmn_optim::{AdamW, AdamWConfig, HybridOptimizer, MuonConfig};
        let mut model = Chatbot::new(false, None, 64, Some(2), Some(16));
        let tokens: Vec<usize> = (0..4).collect();
        let targets: Vec<usize> = (1..5).collect();
        let q_before: Vec<f32> = model.blocks[0].attn.q_proj.weight.data.iter().copied().collect();
        let k_before: Vec<f32> = model.blocks[1].attn.k_proj.weight.data.iter().copied().collect();
        let mut hybrid = HybridOptimizer::new(MuonConfig::default(), AdamWConfig::default());
        let mut adamw = AdamW::new(AdamWConfig { lr: 0.05, ..Default::default() });
        model
            .train_step_lm(&tokens, &targets, &mut hybrid, &mut adamw, true, &mut 0, None, None)
            .unwrap();
        let q_after: Vec<f32> = model.blocks[0].attn.q_proj.weight.data.iter().copied().collect();
        let k_after: Vec<f32> = model.blocks[1].attn.k_proj.weight.data.iter().copied().collect();
        assert_ne!(q_before, q_after, "attn q_proj should get gradients");
        assert_ne!(k_before, k_after, "attn k_proj should get gradients");
    }

    #[test]
    fn train_step_updates_layernorm_gamma() {
        use mmn_optim::{AdamW, AdamWConfig, HybridOptimizer, MuonConfig};
        let mut model = Chatbot::new(false, None, 64, Some(2), Some(16));
        let tokens: Vec<usize> = (0..4).collect();
        let targets: Vec<usize> = (1..5).collect();
        let g_before: Vec<f32> = model.blocks[0].ln1.gamma.data.iter().copied().collect();
        let b_before: Vec<f32> = model.blocks[1].ln2.beta.data.iter().copied().collect();
        let mut hybrid = HybridOptimizer::new(MuonConfig::default(), AdamWConfig::default());
        let mut adamw = AdamW::new(AdamWConfig { lr: 0.05, ..Default::default() });
        model
            .train_step_lm(&tokens, &targets, &mut hybrid, &mut adamw, true, &mut 0, None, None)
            .unwrap();
        let g_after: Vec<f32> = model.blocks[0].ln1.gamma.data.iter().copied().collect();
        let b_after: Vec<f32> = model.blocks[1].ln2.beta.data.iter().copied().collect();
        assert_ne!(g_before, g_after, "ln1 gamma should get gradients");
        assert_ne!(b_before, b_after, "ln2 beta should get gradients");
    }

    #[test]
    fn train_step_hybrid_updates_ffn2() {
        use mmn_optim::{AdamW, AdamWConfig, HybridOptimizer, MuonConfig};
        let mut model = Chatbot::new(false, None, 64, Some(2), Some(16));
        let tokens: Vec<usize> = (0..4).collect();
        let targets: Vec<usize> = (1..5).collect();
        let ffn_before: Vec<f32> = model.blocks[1].ffn2.weight.data.iter().copied().collect();
        let mut hybrid = HybridOptimizer::new(
            MuonConfig {
                lr: 0.1,
                ..Default::default()
            },
            AdamWConfig {
                lr: 0.01,
                ..Default::default()
            },
        );
        let mut adamw = AdamW::new(AdamWConfig { lr: 0.01, ..Default::default() });
        model
            .train_step_lm(&tokens, &targets, &mut hybrid, &mut adamw, true, &mut 0, None, None)
            .unwrap();
        let ffn_after: Vec<f32> = model.blocks[1].ffn2.weight.data.iter().copied().collect();
        assert_ne!(ffn_before, ffn_after, "hybrid Muon path should update ffn2");
    }

    #[test]
    fn chatbot_uses_causal_attention_by_default() {
        let model = Chatbot::new(false, None, 64, Some(2), Some(16));
        assert!(model.uses_causal_attention());
    }

    #[test]
    fn position_encoding_affects_forward_hidden() {
        let model = Chatbot::new(false, None, 64, Some(1), Some(8));
        let h_one = model.forward_hidden(&[10]).unwrap();
        let h_two = model.forward_hidden(&[10, 10]).unwrap();
        assert_ne!(h_one.data[[0, 0]], h_two.data[[1, 0]]);
    }

    #[test]
    fn train_step_updates_learned_pos_embed() {
        use mmn_optim::{AdamW, AdamWConfig, HybridOptimizer, MuonConfig};
        let mut model = Chatbot::new_with_pe_options(
            false, None, 64, Some(1), Some(16), Some(7), true, 32,
        );
        let tokens: Vec<usize> = (0..4).collect();
        let targets: Vec<usize> = (1..5).collect();
        let before: Vec<f32> = model
            .pos_embed
            .as_ref()
            .unwrap()
            .weight
            .data
            .iter()
            .copied()
            .collect();
        let mut hybrid = HybridOptimizer::new(MuonConfig::default(), AdamWConfig::default());
        let mut adamw = AdamW::new(AdamWConfig { lr: 0.05, ..Default::default() });
        model
            .train_step_lm(&tokens, &targets, &mut hybrid, &mut adamw, true, &mut 0, None, None)
            .unwrap();
        let after: Vec<f32> = model
            .pos_embed
            .as_ref()
            .unwrap()
            .weight
            .data
            .iter()
            .copied()
            .collect();
        assert_ne!(before, after);
    }

    #[test]
    fn learned_pos_embed_rejects_long_sequence() {
        let model = Chatbot::new_with_pe_options(
            false, None, 64, Some(1), Some(8), None, true, 4,
        );
        let tokens: Vec<usize> = (0..8).collect();
        let result = model.forward_hidden(&tokens);
        let msg = result.as_ref().err().expect("forward should fail").message();
        assert!(
            msg.contains("max_seq_len") || msg.contains("sequence"),
            "expected max_seq_len error, got: {msg}"
        );
    }

    #[test]
    fn rope_enabled_on_attention_blocks() {
        let rope_model = Chatbot::new_with_position_options(
            false, None, 64, Some(1), Some(16), Some(1), false, 512, true, DEFAULT_ROPE_THETA,
        );
        assert!(rope_model.use_rope);
        assert_eq!(
            rope_model.blocks[0].attn.rope_theta,
            Some(DEFAULT_ROPE_THETA)
        );
        let plain = Chatbot::new_with_seed(false, None, 64, Some(1), Some(16), Some(1));
        assert!(plain.blocks[0].attn.rope_theta.is_none());
    }

    #[test]
    fn rope_attention_differs_from_no_rope() {
        let plain = Chatbot::new_with_seed(false, None, 64, Some(1), Some(16), Some(2));
        let rope_model = Chatbot::new_with_position_options(
            false, None, 64, Some(1), Some(16), Some(2), false, 512, true, DEFAULT_ROPE_THETA,
        );
        let tokens = vec![3, 4, 5, 6];
        let l_plain = plain.loss_on_batch(&tokens, &tokens).unwrap();
        let l_rope = rope_model.loss_on_batch(&tokens, &tokens).unwrap();
        assert_ne!(l_plain, l_rope);
    }

    #[test]
    fn vision_patch_prefix_changes_loss() {
        let model = Chatbot::new(true, None, 256, Some(1), Some(32));
        assert!(model.has_vision_patch_encoder());
        let tokens = vec![10, 20, 30];
        let targets = vec![20, 30, 40];
        let loss_plain = model.loss_on_batch(&tokens, &targets).unwrap();
        let patch = vision_patch_from_text("image bytes");
        let padded = targets_with_vision_prefix(&targets, 1, 256);
        let loss_patch = model
            .loss_on_batch_with_patches(&tokens, &padded, Some(&[patch]))
            .unwrap();
        assert_ne!(loss_plain, loss_patch);
    }

    #[test]
    fn vision_patch_proj_updates_on_train_step() {
        use mmn_optim::{AdamW, AdamWConfig, HybridOptimizer, MuonConfig};
        let mut model = Chatbot::new(true, None, 256, Some(1), Some(16));
        let tokens = vec![1, 2, 3];
        let targets = targets_with_vision_prefix(&[2, 3, 4], 1, 256);
        let patch = vec![vision_patch_from_text("prompt")];
        let w_before = model.vision_patch_proj.as_ref().unwrap().weight.data[[0, 0]];
        let mut hybrid = HybridOptimizer::new(MuonConfig::default(), AdamWConfig::default());
        let mut adamw = AdamW::new(AdamWConfig { lr: 0.1, ..Default::default() });
        model
            .train_step_lm(
                &tokens,
                &targets,
                &mut hybrid,
                &mut adamw,
                false,
                &mut 0,
                None,
                Some(&patch),
            )
            .unwrap();
        let w_after = model.vision_patch_proj.as_ref().unwrap().weight.data[[0, 0]];
        assert_ne!(w_before, w_after);
    }

    #[test]
    fn vision_rgb_patch_differs_from_grayscale() {
        let model = Chatbot::new(true, None, 256, Some(1), Some(32));
        assert!(model.has_vision_rgb_conv());
        let tokens = vec![10, 20, 30];
        let padded = targets_with_vision_prefix(&[20, 30, 40], 1, 256);
        let gray = vision_patch_from_text("photo");
        let rgb = vision_rgb_patch_from_text("photo");
        let loss_gray = model
            .loss_on_batch_with_patches(&tokens, &padded, Some(&[gray]))
            .unwrap();
        let loss_rgb = model
            .loss_on_batch_with_patches(&tokens, &padded, Some(&[rgb]))
            .unwrap();
        assert_ne!(loss_gray, loss_rgb);
    }

    #[test]
    fn vision_rgb_conv_updates_on_train_step() {
        use mmn_optim::{AdamW, AdamWConfig, HybridOptimizer, MuonConfig};
        let mut model = Chatbot::new(true, None, 256, Some(1), Some(16));
        let tokens = vec![1, 2, 3];
        let targets = targets_with_vision_prefix(&[2, 3, 4], 1, 256);
        let patch = vec![vision_rgb_patch_from_text("rgb prompt")];
        let w_before = model.vision_patch_conv.as_ref().unwrap().weight.data[[0, 0, 0, 0]];
        let mut hybrid = HybridOptimizer::new(MuonConfig::default(), AdamWConfig::default());
        let mut adamw = AdamW::new(AdamWConfig { lr: 0.1, ..Default::default() });
        model
            .train_step_lm(
                &tokens,
                &targets,
                &mut hybrid,
                &mut adamw,
                false,
                &mut 0,
                None,
                Some(&patch),
            )
            .unwrap();
        let w_after = model.vision_patch_conv.as_ref().unwrap().weight.data[[0, 0, 0, 0]];
        assert_ne!(w_before, w_after);
    }

    #[test]
    fn vision_cross_attn_changes_loss_with_patch() {
        let model = Chatbot::new(true, None, 256, Some(2), Some(32));
        assert!(model.has_vision_cross_attn());
        let tokens = vec![10, 20, 30];
        let padded = targets_with_vision_prefix(&[20, 30, 40], 1, 256);
        let patch = vision_rgb_patch_from_text("scene");
        let loss_with = model
            .loss_on_batch_with_patches(&tokens, &padded, Some(&[patch.clone()]))
            .unwrap();
        let mut no_cross = model;
        no_cross.vision_cross_attn = None;
        let loss_without = no_cross
            .loss_on_batch_with_patches(&tokens, &padded, Some(&[patch]))
            .unwrap();
        assert_ne!(loss_with, loss_without);
    }

    #[test]
    fn multi_patch_prefix_changes_loss() {
        let model = Chatbot::new(true, None, 256, Some(1), Some(16));
        let tokens = vec![5, 6, 7];
        let p1 = vision_rgb_patch_from_text("tile-a");
        let p2 = vision_rgb_patch_from_text("tile-b");
        let loss_one = model
            .loss_on_batch_with_patches(
                &tokens,
                &targets_with_vision_prefix(&[6, 7, 8], 1, 256),
                Some(&[p1.clone()]),
            )
            .unwrap();
        let loss_two = model
            .loss_on_batch_with_patches(
                &tokens,
                &targets_with_vision_prefix(&[6, 7, 8], 2, 256),
                Some(&[p1, p2]),
            )
            .unwrap();
        assert_ne!(loss_one, loss_two);
    }

    #[test]
    fn vision_cross_attn_updates_on_train_step() {
        use mmn_optim::{AdamW, AdamWConfig, HybridOptimizer, MuonConfig};
        let mut model = Chatbot::new(true, None, 256, Some(2), Some(16));
        let tokens = vec![1, 2, 3];
        let targets = targets_with_vision_prefix(&[2, 3, 4], 1, 256);
        let patch = vec![vision_rgb_patch_from_text("cross prompt")];
        let w_before = model
            .vision_cross_attn
            .as_ref()
            .unwrap()
            .q_proj
            .weight
            .data[[0, 0]];
        let mut hybrid = HybridOptimizer::new(MuonConfig::default(), AdamWConfig::default());
        let mut adamw = AdamW::new(AdamWConfig { lr: 0.1, ..Default::default() });
        model
            .train_step_lm(
                &tokens,
                &targets,
                &mut hybrid,
                &mut adamw,
                false,
                &mut 0,
                None,
                Some(&patch),
            )
            .unwrap();
        let w_after = model
            .vision_cross_attn
            .as_ref()
            .unwrap()
            .q_proj
            .weight
            .data[[0, 0]];
        assert_ne!(w_before, w_after);
    }
}

#[cfg(test)]
mod classifier_tests {
    use super::*;

    #[test]
    fn encode_text_normalizes_bytes() {
        let c = Classifier::new(2, 4);
        let hi = c.encode_text("Z");
        assert!((hi.data[[0, 0]] - 90.0 / 255.0).abs() < 1e-5);
        let empty = c.encode_text("");
        assert_eq!(empty.data[[0, 0]], 0.0);
        assert_eq!(empty.data[[0, 1]], 0.0);
    }

    #[test]
    fn same_seed_same_classifier_loss() {
        let a = Classifier::with_labels_seed(vec!["A".into(), "B".into()], 16, Some(99));
        let b = Classifier::with_labels_seed(vec!["A".into(), "B".into()], 16, Some(99));
        let la = a.loss_on_label("test", 0).unwrap();
        let lb = b.loss_on_label("test", 0).unwrap();
        assert!((la - lb).abs() < 1e-6);
    }

    #[test]
    fn from_dataset_uses_tag_labels() {
        use mmn_data::{DatasetClassification, DatasetMeta, DatasetType};
        let ds = DatasetClassification {
            meta: DatasetMeta {
                rows: 2,
                format: "test".into(),
                dataset_type: DatasetType::Classification,
            },
            samples: vec![
                ("good".into(), "Happy".into()),
                ("bad".into(), "Sad".into()),
            ],
        };
        let c = Classifier::from_classification_dataset(&ds, 16);
        assert_eq!(c.labels, vec!["Happy", "Sad"]);
        let m = c.predict_text("test").unwrap();
        assert!(m.contains_key("Happy"));
        assert!(m.contains_key("Sad"));
    }

    #[test]
    fn train_step_reduces_loss_on_label() {
        use mmn_optim::{AdamW, AdamWConfig};
        let mut c = Classifier::with_labels(vec!["A".into(), "B".into()], 16);
        let before = c.loss_on_label("hello", 0).unwrap();
        let mut adamw = AdamW::new(AdamWConfig {
            lr: 0.05,
            ..Default::default()
        });
        let mut pid = 0usize;
        for _ in 0..30 {
            c.train_step("hello", 0, &mut adamw, &mut pid, None).unwrap();
        }
        let after = c.loss_on_label("hello", 0).unwrap();
        assert!(
            after < before,
            "loss should decrease: before={before} after={after}"
        );
    }

    #[test]
    fn predict_probs_sum_to_one() {
        let c = Classifier::new(3, 16);
        let x = c.encode_text("hello");
        let probs = c.forward(&x).unwrap();
        assert_eq!(probs.shape, vec![1, 3], "unexpected logits shape {:?}", probs.shape);
        let m = c.predict_text("hello").unwrap();
        let s: f32 = m.values().sum();
        assert!(
            (s - 1.0).abs() < 1e-4,
            "probabilities should sum to 1, got {s} {:?}",
            m
        );
    }
}

pub struct Diffusion {
    pub vae: mmn_nn::VaeEncoder,
    pub unet: mmn_nn::UNet2D,
    pub latent_channels: usize,
}

impl Diffusion {
    pub fn new() -> Self {
        Self {
            vae: mmn_nn::VaeEncoder::new(),
            unet: mmn_nn::UNet2D::new(),
            latent_channels: 4,
        }
    }

    pub fn training_step(&self, x: &Tensor, t: usize) -> Result<Tensor> {
        let t_emb = Tensor::from_array(
            ndarray::ArrayD::from_elem(ndarray::IxDyn(&[1]), t as f32),
            false,
        );
        let latent = self.vae.encode(x)?;
        let noise = Tensor::randn(&latent.shape, false);
        let noisy = latent.add(&noise)?;
        self.unet.forward(&noisy, &t_emb)
    }

    pub fn sample_step(&self, x: &Tensor, t: usize) -> Result<Tensor> {
        let t_emb = Tensor::from_array(
            ndarray::ArrayD::from_elem(ndarray::IxDyn(&[1]), t as f32),
            false,
        );
        self.unet.forward(x, &t_emb)
    }
}

#[cfg(test)]
mod diffusion_tests {
    use super::*;
    use mmn_core::Tensor;

    #[test]
    fn training_step_output_finite() {
        let d = Diffusion::new();
        let x = Tensor::randn(&[1, 3, 8, 8], false);
        let out = d.training_step(&x, 3).unwrap();
        assert!(out.data.iter().all(|v| v.is_finite()));
    }
}

#[cfg(test)]
mod dataset_validation_tests {
    use super::*;
    use mmn_data::DatasetType;

    #[test]
    fn diffusion_rejects_corpus_dataset_type() {
        let err = validate_dataset_for_diffusion(&DatasetType::Corpus).unwrap_err();
        assert!(err.to_string().contains("Corpus"));
    }

    #[test]
    fn diffusion_accepts_image_gen_dataset_type() {
        validate_dataset_for_diffusion(&DatasetType::ImageGen).unwrap();
    }
}

pub fn validate_dataset_for_chatbot(ds_type: &DatasetType) -> Result<()> {
    mmn_data::DatasetQA::validate_for_model(match ds_type {
        DatasetType::Qa => "chatbot",
        DatasetType::Corpus => "chatbot",
        _ => "chatbot",
    })
}

pub fn validate_dataset_for_classifier(ds_type: &DatasetType) -> Result<()> {
    if *ds_type != DatasetType::Classification {
        return Err(mmn_core::MmnError::DataMismatch {
            message: format!("Expected classification dataset, got {ds_type:?}"),
            fix: "Use DatasetClassification with text and tag columns.".into(),
            explanation: "Classifier training requires labeled classification rows.".into(),
        });
    }
    Ok(())
}

pub fn validate_dataset_for_diffusion(ds_type: &DatasetType) -> Result<()> {
    if *ds_type == DatasetType::Corpus {
        return Err(mmn_core::MmnError::DataMismatch {
            message: "Corpus dataset on diffusion model".into(),
            fix: "Use DatasetImageGen or DatasetImageEdit.".into(),
            explanation: "Diffusion models require image datasets.".into(),
        });
    }
    Ok(())
}
