use crate::autoset::{autoset, ModelShape};
use mmn_core::{cross_entropy_grad, embedding_backward, linear_backward, Device, Result, Tensor};
use mmn_data::DatasetType;
use mmn_nn::{gelu, gelu_backward, BlockForwardCache, Embedding, Linear, TransformerBlock};
use ndarray::ArrayD;
use std::collections::HashMap;

/// Default maximum sequence length for learned position embeddings.
pub const DEFAULT_MAX_SEQ_LEN: usize = 512;

pub struct Chatbot {
    pub shape: ModelShape,
    pub embed: Embedding,
    pub blocks: Vec<TransformerBlock>,
    pub lm_head: Linear,
    pub tokenizer: String,
    pub vision: bool,
    pub device: Device,
    /// RNG seed used at construction (`None` = non-deterministic init).
    pub init_seed: Option<u64>,
    /// When true, use trainable `pos_embed` instead of fixed sinusoidal PE.
    pub use_learned_pos_embed: bool,
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
        for _ in 0..shape.n_layer {
            blocks.push(TransformerBlock::new_rng(
                shape.d_model,
                shape.n_heads,
                shape.ffn_dim,
                &mut rng,
            ));
        }
        let pos_embed = if use_learned_pos_embed {
            Some(Embedding::new_rng(max_seq_len, shape.d_model, &mut rng))
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
            device: Device::Cpu,
            init_seed: seed,
            use_learned_pos_embed,
            max_seq_len,
            pos_embed,
        }
    }

    fn apply_position_encoding(&self, h: Tensor) -> Result<Tensor> {
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
        if self.use_learned_pos_embed {
            base + self.max_seq_len * self.shape.d_model
        } else {
            base
        }
    }

    pub fn layer_size(&self) -> usize {
        self.shape.n_layer
    }

    pub fn has_vision(&self) -> bool {
        self.vision
    }

    pub fn uses_causal_attention(&self) -> bool {
        self.blocks
            .first()
            .map(|b| b.attn.causal)
            .unwrap_or(true)
    }

    pub fn forward_hidden(&self, token_ids: &[usize]) -> Result<Tensor> {
        let mut h = self.apply_position_encoding(self.embed.forward(token_ids)?)?;
        for block in &self.blocks {
            h = block.forward(&h)?;
        }
        Ok(h)
    }

    pub fn forward_logits(&self, token_ids: &[usize]) -> Result<Tensor> {
        self.lm_head.forward(&self.forward_hidden(token_ids)?)
    }

    pub fn loss_on_batch(&self, token_ids: &[usize], targets: &[usize]) -> Result<f32> {
        let logits = self.forward_logits(token_ids)?;
        let loss_t = logits.cross_entropy_loss(targets)?;
        Ok(loss_t.data.as_slice().unwrap()[0])
    }

    /// Backward pass; either apply optimizer immediately or accumulate grads for `batch_size` > 1.
    pub fn train_step_lm(
        &mut self,
        token_ids: &[usize],
        targets: &[usize],
        hybrid: &mut mmn_optim::HybridOptimizer,
        adamw: &mut mmn_optim::AdamW,
        use_hybrid: bool,
        param_id_base: &mut usize,
        mut accum: Option<&mut mmn_optim::GradAccumulator>,
    ) -> Result<f32> {
        if let Some(acc) = accum.as_mut() {
            acc.begin_micro_batch();
            let loss_val = self.backward_lm_accumulate(token_ids, targets, acc)?;
            acc.finish_micro_batch();
            return Ok(loss_val);
        }
        self.backward_lm(
            token_ids,
            targets,
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

        for block in self.blocks.iter_mut().rev() {
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
        Ok(())
    }

    fn backward_lm_accumulate(
        &mut self,
        token_ids: &[usize],
        targets: &[usize],
        acc: &mut mmn_optim::GradAccumulator,
    ) -> Result<f32> {
        let (loss_val, grads) = self.backward_lm_grads(token_ids, targets)?;
        for g in grads {
            acc.add_param_grad(&g);
        }
        Ok(loss_val)
    }

    fn backward_lm(
        &mut self,
        token_ids: &[usize],
        targets: &[usize],
        hybrid: &mut mmn_optim::HybridOptimizer,
        adamw: &mut mmn_optim::AdamW,
        use_hybrid: bool,
        param_id_base: &mut usize,
    ) -> Result<f32> {
        let (loss_val, grads) = self.backward_lm_grads(token_ids, targets)?;
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
        for block in self.blocks.iter_mut().rev() {
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
        Ok(loss_val)
    }

    fn backward_lm_grads(
        &mut self,
        token_ids: &[usize],
        targets: &[usize],
    ) -> Result<(f32, Vec<ArrayD<f32>>)> {
        let mut h = self.embed.forward(token_ids)?;
        h = self.apply_position_encoding(h)?;
        let mut caches = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let (out, cache) = block.forward_with_cache(&h)?;
            caches.push(BlockFfnCache { block: cache });
            h = out;
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

        for (block, cache) in self.blocks.iter().zip(caches.iter()).rev() {
            let (grad_h_block, block_grads) =
                block.backward_attn_ffn(&cache.block, &grad_h)?;
            for g in block_grads {
                grads.push(g);
            }
            grad_h = grad_h_block;
        }

        let grad_embed_w = embedding_backward(
            token_ids,
            &grad_h,
            self.shape.vocab_size,
            self.shape.d_model,
        );
        grads.push(grad_embed_w);
        if self.use_learned_pos_embed {
            let pos_ids: Vec<usize> = (0..token_ids.len()).collect();
            let grad_pos = embedding_backward(
                &pos_ids,
                &grad_h,
                self.max_seq_len,
                self.shape.d_model,
            );
            grads.push(grad_pos);
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
            .train_step_lm(&tokens, &targets, &mut hybrid, &mut adamw, false, &mut 0, None)
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
            .train_step_lm(&tokens, &targets, &mut hybrid, &mut adamw, false, &mut 0, None)
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
            .train_step_lm(&tokens, &targets, &mut hybrid, &mut adamw, true, &mut 0, None)
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
            .train_step_lm(&tokens, &targets, &mut hybrid, &mut adamw, true, &mut 0, None)
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
            .train_step_lm(&tokens, &targets, &mut hybrid, &mut adamw, true, &mut 0, None)
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
            .train_step_lm(&tokens, &targets, &mut hybrid, &mut adamw, true, &mut 0, None)
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
