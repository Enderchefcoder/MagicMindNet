use mmn_models::{Chatbot, ChatbotArchExtras};
use mmn_train::{
    align_qa_token_pairs, mean_corpus_loss_with_encoder, mean_qa_loss_with_encoder, tokenize_lm,
};
use mmn_models::{targets_with_vision_prefix, vision_patch_from_text, vision_rgb_patch_from_text};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::datasets::{PyDatasetCorpus, PyDatasetQA};
use crate::encoder_util::resolve_text_encoder;
use crate::errors::{mmn_err_to_py, DataMismatchError};
use crate::io::{expect_checkpoint_family, export_chatbot_to_path, import_chatbot_from_path};
use crate::tokenizer::{PyBytePairEncoder, PyGpt2BpeEncoder, PyUnigramEncoder};
use crate::train::train_chatbot_dispatch;
use crate::train_config::{resolve_train_config, PyTrainConfig};

fn resolve_generate_vision_patches(
    bot: &Chatbot,
    prompt: &str,
    image_patch: Option<Vec<f32>>,
    image_patches: Option<Vec<Vec<f32>>>,
) -> PyResult<Option<Vec<Vec<f32>>>> {
    if !bot.has_vision_patch_encoder() {
        if image_patch.is_some() || image_patches.is_some() {
            return Err(PyErr::new::<DataMismatchError, _>(
                "image_patch/image_patches require Chatbot(vision=True).\nFix: Construct Chatbot with vision=True or omit image patches.".to_string(),
            ));
        }
        return Ok(None);
    }
    let gray = bot.vision_patch_dim();
    let rgb = bot.vision_rgb_dim();
    let validate_patch = |p: &Vec<f32>| -> PyResult<()> {
        if p.len() != gray && p.len() != rgb {
            return Err(PyErr::new::<DataMismatchError, _>(format!(
                "image patch length {} != vision_patch_dim ({gray}) or vision_rgb_dim ({rgb}).\nFix: Pass flat {gray}-float grayscale or {rgb}-float RGB patches.\nExplanation: Vision Chatbot accepts 8×8 grayscale or 8×8×3 RGB patches.",
                p.len(),
            )));
        }
        if p.len() == rgb && !bot.has_vision_rgb_conv() {
            return Err(PyErr::new::<DataMismatchError, _>(format!(
                "RGB patch length {rgb} requires vision_rgb_conv.\nFix: Pass {gray}-float grayscale patches or load a vision checkpoint with vision_patch_conv."
            )));
        }
        Ok(())
    };
    let patch_list: Vec<Vec<f32>> = if let Some(pl) = image_patches {
        for p in &pl {
            validate_patch(p)?;
        }
        pl
    } else if let Some(p) = image_patch {
        validate_patch(&p)?;
        vec![p]
    } else if bot.has_vision_rgb_conv() {
        vec![vision_rgb_patch_from_text(prompt)]
    } else {
        vec![vision_patch_from_text(prompt)]
    };
    Ok(Some(patch_list))
}

#[allow(clippy::too_many_arguments)]
fn build_generate_config(
    max_new_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    typical_p: f32,
    mirostat: u32,
    mirostat_tau: f32,
    mirostat_eta: f32,
    repetition_penalty: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
    use_kv_cache: bool,
    vision_patches: Option<Vec<Vec<f32>>>,
    stop_token_ids: Option<Vec<usize>>,
    stop_strings: Option<Vec<String>>,
    json_mode: bool,
    grammar: Option<String>,
) -> mmn_train::GenerateConfig {
    let mut cfg = mmn_train::GenerateConfig {
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        min_p,
        typical_p,
        mirostat,
        mirostat_tau,
        mirostat_eta,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        use_kv_cache,
        vision_patches,
        stop_token_ids: stop_token_ids.unwrap_or_default(),
        stop_strings: stop_strings.unwrap_or_default(),
        json_mode,
        grammar,
        ..Default::default()
    };
    // Keep Mirostat mu consistent with tau when using defaults.
    if mirostat != 0 {
        cfg.mirostat_mu = 2.0 * mirostat_tau;
    }
    cfg
}

/// A small transformer language model you can train, chat with, and save.
#[pyclass(name = "Chatbot")]
pub struct PyChatbot {
    pub(crate) inner: Chatbot,
}

#[pymethods]
impl PyChatbot {
    #[new]
    #[pyo3(signature = (
        vision=false,
        autoset=None,
        vocab_size=32000,
        n_layer=None,
        d_model=None,
        seed=None,
        use_learned_pos_embed=false,
        max_seq_len=512,
        use_rope=false,
        rope_theta=10000.0,
        n_heads=None,
        n_kv_heads=None,
        head_dim=None,
        ffn_dim=None,
        n_loops=1,
        tie_embeddings=false,
        norm="layer",
        ffn="gelu",
        loop_embed=false,
        final_norm=false,
        lora_rank=0,
        coda_layers=0,
        prelude_layers=0,
        max_loops=None,
        attention_window=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vision: bool,
        autoset: Option<String>,
        vocab_size: usize,
        n_layer: Option<usize>,
        d_model: Option<usize>,
        seed: Option<u64>,
        use_learned_pos_embed: bool,
        max_seq_len: usize,
        use_rope: bool,
        rope_theta: f32,
        n_heads: Option<usize>,
        n_kv_heads: Option<usize>,
        head_dim: Option<usize>,
        ffn_dim: Option<usize>,
        n_loops: usize,
        tie_embeddings: bool,
        norm: &str,
        ffn: &str,
        loop_embed: bool,
        final_norm: bool,
        lora_rank: isize,
        coda_layers: usize,
        prelude_layers: usize,
        max_loops: Option<usize>,
        attention_window: Option<usize>,
    ) -> PyResult<Self> {
        if use_learned_pos_embed && use_rope {
            return Err(PyValueError::new_err(
                "Chatbot cannot use both use_learned_pos_embed=True and use_rope=True.\nFix: Pick one position-encoding mode (learned table or rotary).",
            ));
        }
        if let Some(budget) = autoset.as_deref() {
            if !mmn_models::is_valid_autoset_budget(budget) {
                return Err(PyValueError::new_err(format!(
                    "Unknown autoset preset {budget:?}. Valid presets: \"sub-1M\", \"sub-10M\", \"sub-50M\", \"sub-100M\", \"sub-1B\", \"sub-10B\".",
                )));
            }
        }
        if vocab_size == 0 {
            return Err(PyValueError::new_err(
                "vocab_size must be at least 1.\nFix: Use vocab_size=512 for byte-level toy models or 32000 for BPE-scale vocabularies.",
            ));
        }
        if lora_rank < 0 {
            return Err(PyValueError::new_err(
                "lora_rank must be >= 0.\nFix: Use lora_rank=0 to disable LoopLoRA or a positive rank (e.g. 4).",
            ));
        }
        let use_rms_norm = match norm {
            "layer" => false,
            "rms" => true,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown norm={other:?}. Valid values: \"layer\", \"rms\"."
                )));
            }
        };
        let use_swiglu = match ffn {
            "gelu" => false,
            "swiglu" => true,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown ffn={other:?}. Valid values: \"gelu\", \"swiglu\"."
                )));
            }
        };
        if n_loops == 0 {
            return Err(PyValueError::new_err(
                "n_loops must be at least 1.\nFix: Use n_loops=1 for a standard transformer or n_loops>1 to reuse blocks.",
            ));
        }
        Ok(Self {
            inner: Chatbot::new_with_arch(
                vision,
                autoset.as_deref(),
                vocab_size,
                n_layer,
                d_model,
                ffn_dim,
                n_heads,
                n_kv_heads,
                head_dim,
                seed,
                use_learned_pos_embed,
                max_seq_len,
                use_rope,
                rope_theta,
                ChatbotArchExtras {
                    n_loops,
                    tie_embeddings,
                    use_rms_norm,
                    use_swiglu,
                    loop_embed,
                    final_norm,
                    lora_rank: lora_rank as usize,
                    coda_layers,
                    prelude_layers,
                    max_loops,
                    attention_window,
                },
            ),
        })
    }

    /// Save this chatbot to `path`. Formats: "safetensors" (JSON, default),
    /// "hf-safetensors" (binary), "bin" (architecture stub), "gguf" /
    /// "gguf-q8_0" (GGUF), "npz" (NumPy), or "pt" (PyTorch state dict).
    #[pyo3(signature = (path, format="safetensors", bpe_encoder=None, unigram_encoder=None))]
    fn save(
        &self,
        path: &str,
        format: &str,
        bpe_encoder: Option<&PyBytePairEncoder>,
        unigram_encoder: Option<&PyUnigramEncoder>,
    ) -> PyResult<()> {
        export_chatbot_to_path(&self.inner, format, path, bpe_encoder, unigram_encoder)
    }

    /// Load a chatbot checkpoint; the file format is detected automatically.
    #[staticmethod]
    #[pyo3(signature = (path, format=None))]
    fn load(path: &str, format: Option<&str>) -> PyResult<Self> {
        let format = match format {
            Some(f) => f.to_string(),
            None => {
                match expect_checkpoint_family(
                    path,
                    "Chatbot",
                    "Use Classifier.load() / Diffusion.load() or the universal ai.load().",
                )? {
                    mmn_io::CheckpointKind::ChatbotBin => "bin".to_string(),
                    mmn_io::CheckpointKind::ChatbotGguf => "gguf".to_string(),
                    mmn_io::CheckpointKind::ChatbotNpz => "npz".to_string(),
                    mmn_io::CheckpointKind::ChatbotTorch => "pt".to_string(),
                    mmn_io::CheckpointKind::ChatbotSharded => "sharded".to_string(),
                    mmn_io::CheckpointKind::ChatbotGgmlLegacy => "ggml-legacy".to_string(),
                    _ => "safetensors".to_string(),
                }
            }
        };
        Ok(Self {
            inner: import_chatbot_from_path(&format, path)?,
        })
    }

    /// Train on a `DatasetQA` or `DatasetCorpus`; returns one mean loss per epoch.
    ///
    /// All settings are optional: `bot.train(data)` uses sensible defaults, or
    /// pass `epochs=`, `learning_rate=`, ... to override (a full `TrainConfig`
    /// in `config=` also works).
    #[pyo3(signature = (dataset, config=None, *, epochs=None, batch_size=None, learning_rate=None, optimizer=None, cuda=None, verbose=None, bpe_encoder=None, unigram_encoder=None, gpt2_encoder=None))]
    #[allow(clippy::too_many_arguments)]
    fn train(
        &mut self,
        dataset: &Bound<'_, PyAny>,
        config: Option<&PyTrainConfig>,
        epochs: Option<usize>,
        batch_size: Option<usize>,
        learning_rate: Option<f32>,
        optimizer: Option<&str>,
        cuda: Option<bool>,
        verbose: Option<bool>,
        bpe_encoder: Option<&PyBytePairEncoder>,
        unigram_encoder: Option<&PyUnigramEncoder>,
        gpt2_encoder: Option<&PyGpt2BpeEncoder>,
    ) -> PyResult<Vec<f32>> {
        let cfg = resolve_train_config(
            config,
            epochs,
            batch_size,
            learning_rate,
            optimizer,
            cuda,
            verbose,
        )?;
        let enc = resolve_text_encoder(bpe_encoder, unigram_encoder, gpt2_encoder)?;
        train_chatbot_dispatch(self, dataset, &cfg, enc)
    }

    /// Generate a reply with beginner-friendly sampling defaults
    /// (temperature 0.8, top-p 0.95, light repetition penalty).
    #[pyo3(signature = (prompt, *, max_new_tokens=64, temperature=0.8, top_p=0.95, top_k=0, repetition_penalty=1.1, stop_strings=None, bpe_encoder=None, unigram_encoder=None, gpt2_encoder=None))]
    #[allow(clippy::too_many_arguments)]
    fn chat(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        repetition_penalty: f32,
        stop_strings: Option<Vec<String>>,
        bpe_encoder: Option<&PyBytePairEncoder>,
        unigram_encoder: Option<&PyUnigramEncoder>,
        gpt2_encoder: Option<&PyGpt2BpeEncoder>,
    ) -> PyResult<String> {
        let enc = resolve_text_encoder(bpe_encoder, unigram_encoder, gpt2_encoder)?;
        let cfg = mmn_train::GenerateConfig {
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            stop_strings: stop_strings.unwrap_or_default(),
            ..Default::default()
        };
        mmn_train::generate_text(&self.inner, prompt, enc, &cfg).map_err(mmn_err_to_py)
    }

    #[getter]
    fn parameters(&self) -> usize {
        self.inner.parameters()
    }

    #[getter]
    fn layer_size(&self) -> usize {
        self.inner.layer_size()
    }

    #[getter]
    fn tokenizer(&self) -> String {
        self.inner.tokenizer.clone()
    }

    #[getter]
    fn has_vision(&self) -> bool {
        self.inner.has_vision()
    }

    #[getter]
    fn has_vision_patch_encoder(&self) -> bool {
        self.inner.has_vision_patch_encoder()
    }

    #[getter]
    fn vision_patch_dim(&self) -> usize {
        self.inner.vision_patch_dim()
    }

    #[getter]
    fn has_vision_rgb_conv(&self) -> bool {
        self.inner.has_vision_rgb_conv()
    }

    #[getter]
    fn has_vision_cross_attn(&self) -> bool {
        self.inner.has_vision_cross_attn()
    }

    #[getter]
    fn vision_rgb_dim(&self) -> usize {
        self.inner.vision_rgb_dim()
    }

    #[getter]
    fn uses_causal_attention(&self) -> bool {
        self.inner.uses_causal_attention()
    }

    #[getter]
    fn use_learned_pos_embed(&self) -> bool {
        self.inner.use_learned_pos_embed
    }

    #[getter]
    fn use_rope(&self) -> bool {
        self.inner.use_rope
    }

    #[getter]
    fn rope_theta(&self) -> f32 {
        self.inner.rope_theta
    }

    #[getter]
    fn max_seq_len(&self) -> usize {
        self.inner.max_seq_len
    }

    #[getter]
    fn init_seed(&self) -> Option<u64> {
        self.inner.init_seed
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.shape.vocab_size
    }

    #[getter]
    fn n_layer(&self) -> usize {
        self.inner.shape.n_layer
    }

    #[getter]
    fn d_model(&self) -> usize {
        self.inner.shape.d_model
    }

    #[getter]
    fn n_heads(&self) -> usize {
        self.inner.shape.n_heads
    }

    #[getter]
    fn n_kv_heads(&self) -> usize {
        self.inner.shape.n_kv_heads
    }

    #[getter]
    fn head_dim(&self) -> usize {
        self.inner.shape.effective_head_dim()
    }

    #[getter]
    fn ffn_dim(&self) -> usize {
        self.inner.shape.ffn_dim
    }

    #[getter]
    fn n_loops(&self) -> usize {
        self.inner.n_loops
    }

    #[getter]
    fn tie_embeddings(&self) -> bool {
        self.inner.tie_embeddings
    }

    #[getter]
    fn norm(&self) -> String {
        self.inner.norm_kind.clone()
    }

    #[getter]
    fn ffn(&self) -> String {
        self.inner.ffn_kind.clone()
    }

    #[getter]
    fn loop_embed(&self) -> bool {
        self.inner.loop_embed.is_some()
    }

    #[getter]
    fn final_norm(&self) -> bool {
        self.inner.final_norm.is_some()
    }

    #[getter]
    fn lora_rank(&self) -> usize {
        self.inner
            .loop_lora
            .as_ref()
            .map(|l| l.rank)
            .unwrap_or(0)
    }

    #[getter]
    fn coda_layers(&self) -> usize {
        self.inner.coda_blocks.len()
    }

    #[getter]
    fn prelude_layers(&self) -> usize {
        self.inner.prelude_blocks.len()
    }

    #[getter]
    fn max_loops(&self) -> usize {
        self.inner.max_loops
    }

    #[getter]
    fn attention_window(&self) -> Option<usize> {
        self.inner.attention_window
    }

    fn __repr__(&self) -> String {
        let s = &self.inner.shape;
        let vision = if self.inner.vision { "True" } else { "False" };
        match self.inner.init_seed {
            Some(seed) => format!(
                "Chatbot(vocab_size={}, n_layer={}, d_model={}, vision={vision}, parameters={}, init_seed={seed})",
                s.vocab_size,
                s.n_layer,
                s.d_model,
                self.inner.parameters()
            ),
            None => format!(
                "Chatbot(vocab_size={}, n_layer={}, d_model={}, vision={vision}, parameters={})",
                s.vocab_size,
                s.n_layer,
                s.d_model,
                self.inner.parameters()
            ),
        }
    }

    /// Mean cross-entropy for tokenized `input` → `target` (same tokenization as `Train`).
    #[pyo3(signature = (input, target, bpe_encoder=None, unigram_encoder=None, gpt2_encoder=None, image_patch=None, image_patches=None))]
    fn compute_loss(
        &self,
        input: &str,
        target: &str,
        bpe_encoder: Option<&PyBytePairEncoder>,
        unigram_encoder: Option<&PyUnigramEncoder>,
        gpt2_encoder: Option<&PyGpt2BpeEncoder>,
        image_patch: Option<Vec<f32>>,
        image_patches: Option<Vec<Vec<f32>>>,
    ) -> PyResult<f32> {
        let vocab = self.inner.shape.vocab_size;
        let enc = resolve_text_encoder(bpe_encoder, unigram_encoder, gpt2_encoder)?;
        let mut tokens = tokenize_lm(input, vocab, enc);
        let mut targets = tokenize_lm(target, vocab, enc);
        align_qa_token_pairs(&mut tokens, &mut targets);
        if self.inner.has_vision_patch_encoder() {
            let gray = self.inner.vision_patch_dim();
            let rgb = self.inner.vision_rgb_dim();
            let validate_patch = |p: &Vec<f32>| -> PyResult<()> {
                if p.len() != gray && p.len() != rgb {
                    return Err(PyErr::new::<DataMismatchError, _>(format!(
                        "image patch length {} != vision_patch_dim ({gray}) or vision_rgb_dim ({rgb}).\nFix: Pass flat {gray}-float grayscale or {rgb}-float RGB patches.\nExplanation: Vision Chatbot accepts 8×8 grayscale or 8×8×3 RGB patches.",
                        p.len(),
                    )));
                }
                if p.len() == rgb && !self.inner.has_vision_rgb_conv() {
                    return Err(PyErr::new::<DataMismatchError, _>(format!(
                        "RGB patch length {rgb} requires vision_rgb_conv.\nFix: Pass {gray}-float grayscale patches or load a vision checkpoint with vision_patch_conv."
                    )));
                }
                Ok(())
            };
            let patch_list: Vec<Vec<f32>> = if let Some(pl) = image_patches {
                for p in &pl {
                    validate_patch(p)?;
                }
                pl
            } else if let Some(p) = image_patch {
                validate_patch(&p)?;
                vec![p]
            } else if self.inner.has_vision_rgb_conv() {
                vec![vision_rgb_patch_from_text(input)]
            } else {
                vec![vision_patch_from_text(input)]
            };
            let padded = targets_with_vision_prefix(&targets, patch_list.len(), vocab);
            return self
                .inner
                .loss_on_batch_with_patches(&tokens, &padded, Some(&patch_list))
                .map_err(mmn_err_to_py);
        }
        if image_patch.is_some() || image_patches.is_some() {
            return Err(PyErr::new::<DataMismatchError, _>(
                "image_patch/image_patches require Chatbot(vision=True).\nFix: Construct Chatbot with vision=True or omit image patches.".to_string(),
            ));
        }
        self.inner
            .loss_on_batch(&tokens, &targets)
            .map_err(mmn_err_to_py)
    }

    /// Mean CE over all rows in a `DatasetQA` or `DatasetCorpus`.
    #[pyo3(signature = (dataset, bpe_encoder=None, unigram_encoder=None, gpt2_encoder=None))]
    fn compute_mean_loss(
        &self,
        dataset: &Bound<'_, PyAny>,
        bpe_encoder: Option<&PyBytePairEncoder>,
        unigram_encoder: Option<&PyUnigramEncoder>,
        gpt2_encoder: Option<&PyGpt2BpeEncoder>,
    ) -> PyResult<f32> {
        let enc = resolve_text_encoder(bpe_encoder, unigram_encoder, gpt2_encoder)?;
        if let Ok(ds) = dataset.downcast::<PyDatasetQA>() {
            return mean_qa_loss_with_encoder(&self.inner, &ds.borrow().inner, enc)
                .map_err(mmn_err_to_py);
        }
        if let Ok(ds) = dataset.downcast::<PyDatasetCorpus>() {
            return mean_corpus_loss_with_encoder(&self.inner, &ds.borrow().inner, enc)
                .map_err(mmn_err_to_py);
        }
        Err(PyErr::new::<DataMismatchError, _>(
            "compute_mean_loss on Chatbot requires DatasetQA or DatasetCorpus.\nFix: Use DatasetQA or DatasetCorpus.\nExplanation: Classification datasets use Classifier.compute_mean_loss.".to_string(),
        ))
    }

    /// Autoregressive continuation from `prompt` (greedy when `temperature=0`).
    #[pyo3(signature = (
        prompt,
        *,
        max_new_tokens=32,
        temperature=0.0,
        top_k=0,
        top_p=0.0,
        min_p=0.0,
        typical_p=0.0,
        mirostat=0,
        mirostat_tau=5.0,
        mirostat_eta=0.1,
        repetition_penalty=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        use_kv_cache=true,
        bpe_encoder=None,
        unigram_encoder=None,
        gpt2_encoder=None,
        image_patch=None,
        image_patches=None,
        stop_token_ids=None,
        stop_strings=None,
        json_mode=false,
        grammar=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn generate(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        min_p: f32,
        typical_p: f32,
        mirostat: u32,
        mirostat_tau: f32,
        mirostat_eta: f32,
        repetition_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
        use_kv_cache: bool,
        bpe_encoder: Option<&PyBytePairEncoder>,
        unigram_encoder: Option<&PyUnigramEncoder>,
        gpt2_encoder: Option<&PyGpt2BpeEncoder>,
        image_patch: Option<Vec<f32>>,
        image_patches: Option<Vec<Vec<f32>>>,
        stop_token_ids: Option<Vec<usize>>,
        stop_strings: Option<Vec<String>>,
        json_mode: bool,
        grammar: Option<String>,
    ) -> PyResult<String> {
        let enc = resolve_text_encoder(bpe_encoder, unigram_encoder, gpt2_encoder)?;
        let vision_patches =
            resolve_generate_vision_patches(&self.inner, prompt, image_patch, image_patches)?;
        let cfg = build_generate_config(
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            min_p,
            typical_p,
            mirostat,
            mirostat_tau,
            mirostat_eta,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            use_kv_cache,
            vision_patches,
            stop_token_ids,
            stop_strings,
            json_mode,
            grammar,
        );
        mmn_train::generate_text(&self.inner, prompt, enc, &cfg).map_err(mmn_err_to_py)
    }

    /// Like `generate`, but returns one decoded string piece per new token.
    #[pyo3(signature = (
        prompt,
        *,
        max_new_tokens=32,
        temperature=0.0,
        top_k=0,
        top_p=0.0,
        min_p=0.0,
        typical_p=0.0,
        mirostat=0,
        mirostat_tau=5.0,
        mirostat_eta=0.1,
        repetition_penalty=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        use_kv_cache=true,
        bpe_encoder=None,
        unigram_encoder=None,
        gpt2_encoder=None,
        image_patch=None,
        image_patches=None,
        stop_token_ids=None,
        stop_strings=None,
        json_mode=false,
        grammar=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn generate_stream(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        min_p: f32,
        typical_p: f32,
        mirostat: u32,
        mirostat_tau: f32,
        mirostat_eta: f32,
        repetition_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
        use_kv_cache: bool,
        bpe_encoder: Option<&PyBytePairEncoder>,
        unigram_encoder: Option<&PyUnigramEncoder>,
        gpt2_encoder: Option<&PyGpt2BpeEncoder>,
        image_patch: Option<Vec<f32>>,
        image_patches: Option<Vec<Vec<f32>>>,
        stop_token_ids: Option<Vec<usize>>,
        stop_strings: Option<Vec<String>>,
        json_mode: bool,
        grammar: Option<String>,
    ) -> PyResult<Vec<String>> {
        let enc = resolve_text_encoder(bpe_encoder, unigram_encoder, gpt2_encoder)?;
        let vision_patches =
            resolve_generate_vision_patches(&self.inner, prompt, image_patch, image_patches)?;
        let cfg = build_generate_config(
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            min_p,
            typical_p,
            mirostat,
            mirostat_tau,
            mirostat_eta,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            use_kv_cache,
            vision_patches,
            stop_token_ids,
            stop_strings,
            json_mode,
            grammar,
        );
        mmn_train::generate_text_stream(&self.inner, prompt, enc, &cfg).map_err(mmn_err_to_py)
    }

    /// Sample new token ids after `prompt` (excludes prompt tokens).
    #[pyo3(signature = (
        prompt,
        *,
        max_new_tokens=32,
        temperature=0.0,
        top_k=0,
        top_p=0.0,
        min_p=0.0,
        typical_p=0.0,
        mirostat=0,
        mirostat_tau=5.0,
        mirostat_eta=0.1,
        repetition_penalty=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        use_kv_cache=true,
        bpe_encoder=None,
        unigram_encoder=None,
        gpt2_encoder=None,
        image_patch=None,
        image_patches=None,
        stop_token_ids=None,
        stop_strings=None,
        json_mode=false,
        grammar=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn generate_tokens(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        min_p: f32,
        typical_p: f32,
        mirostat: u32,
        mirostat_tau: f32,
        mirostat_eta: f32,
        repetition_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
        use_kv_cache: bool,
        bpe_encoder: Option<&PyBytePairEncoder>,
        unigram_encoder: Option<&PyUnigramEncoder>,
        gpt2_encoder: Option<&PyGpt2BpeEncoder>,
        image_patch: Option<Vec<f32>>,
        image_patches: Option<Vec<Vec<f32>>>,
        stop_token_ids: Option<Vec<usize>>,
        stop_strings: Option<Vec<String>>,
        json_mode: bool,
        grammar: Option<String>,
    ) -> PyResult<Vec<usize>> {
        let enc = resolve_text_encoder(bpe_encoder, unigram_encoder, gpt2_encoder)?;
        let vision_patches =
            resolve_generate_vision_patches(&self.inner, prompt, image_patch, image_patches)?;
        let cfg = build_generate_config(
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            min_p,
            typical_p,
            mirostat,
            mirostat_tau,
            mirostat_eta,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            use_kv_cache,
            vision_patches,
            stop_token_ids,
            stop_strings,
            json_mode,
            grammar,
        );
        mmn_train::generate_token_ids(&self.inner, prompt, enc, &cfg).map_err(mmn_err_to_py)
    }

    /// Mean-pool hidden states for one string or a list of strings.
    #[pyo3(signature = (texts, *, bpe_encoder=None, unigram_encoder=None, gpt2_encoder=None))]
    fn embed(
        &self,
        py: Python<'_>,
        texts: &Bound<'_, PyAny>,
        bpe_encoder: Option<&PyBytePairEncoder>,
        unigram_encoder: Option<&PyUnigramEncoder>,
        gpt2_encoder: Option<&PyGpt2BpeEncoder>,
    ) -> PyResult<PyObject> {
        let enc = resolve_text_encoder(bpe_encoder, unigram_encoder, gpt2_encoder)?;
        let max_ctx = if self.inner.use_learned_pos_embed || self.inner.uses_rope() {
            self.inner.max_seq_len
        } else {
            512
        };
        let vocab = self.inner.shape.vocab_size;
        if let Ok(s) = texts.extract::<&str>() {
            let ids = mmn_train::tokenize_for_generate(s, vocab, enc, max_ctx);
            let v = mmn_train::embed_mean_pool(&self.inner, &ids).map_err(mmn_err_to_py)?;
            return Ok(v.into_pyobject(py)?.into_any().unbind());
        }
        let batch: Vec<String> = texts.extract()?;
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(batch.len());
        for s in &batch {
            let ids = mmn_train::tokenize_for_generate(s, vocab, enc, max_ctx);
            out.push(mmn_train::embed_mean_pool(&self.inner, &ids).map_err(mmn_err_to_py)?);
        }
        Ok(out.into_pyobject(py)?.into_any().unbind())
    }

    /// Format ChatML messages and generate an assistant reply.
    #[pyo3(signature = (
        messages,
        *,
        max_new_tokens=32,
        temperature=0.0,
        top_k=0,
        top_p=0.0,
        min_p=0.0,
        typical_p=0.0,
        mirostat=0,
        mirostat_tau=5.0,
        mirostat_eta=0.1,
        repetition_penalty=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        use_kv_cache=true,
        bpe_encoder=None,
        unigram_encoder=None,
        gpt2_encoder=None,
        stop_token_ids=None,
        stop_strings=None,
        json_mode=false,
        grammar=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn chat_messages(
        &self,
        messages: &Bound<'_, PyAny>,
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        min_p: f32,
        typical_p: f32,
        mirostat: u32,
        mirostat_tau: f32,
        mirostat_eta: f32,
        repetition_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
        use_kv_cache: bool,
        bpe_encoder: Option<&PyBytePairEncoder>,
        unigram_encoder: Option<&PyUnigramEncoder>,
        gpt2_encoder: Option<&PyGpt2BpeEncoder>,
        stop_token_ids: Option<Vec<usize>>,
        stop_strings: Option<Vec<String>>,
        json_mode: bool,
        grammar: Option<String>,
    ) -> PyResult<String> {
        let list = messages.downcast::<pyo3::types::PyList>()?;
        let mut pairs: Vec<(String, String)> = Vec::with_capacity(list.len());
        for item in list.iter() {
            let dict = item.downcast::<pyo3::types::PyDict>()?;
            let role: String = dict
                .get_item("role")?
                .ok_or_else(|| PyValueError::new_err("chat message missing 'role'"))?
                .extract()?;
            let content: String = dict
                .get_item("content")?
                .ok_or_else(|| PyValueError::new_err("chat message missing 'content'"))?
                .extract()?;
            pairs.push((role, content));
        }
        let prompt = mmn_train::format_chat_messages(&pairs, true);
        let enc = resolve_text_encoder(bpe_encoder, unigram_encoder, gpt2_encoder)?;
        let cfg = build_generate_config(
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            min_p,
            typical_p,
            mirostat,
            mirostat_tau,
            mirostat_eta,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            use_kv_cache,
            None,
            stop_token_ids,
            stop_strings,
            json_mode,
            grammar,
        );
        mmn_train::generate_text(&self.inner, &prompt, enc, &cfg).map_err(mmn_err_to_py)
    }
}

/// Format OpenAI/Ollama-style chat messages as ChatML.
#[pyfunction]
#[pyo3(name = "format_chat_messages", signature = (messages, *, add_generation_prompt=true))]
pub fn format_chat_messages_py(
    messages: &Bound<'_, PyAny>,
    add_generation_prompt: bool,
) -> PyResult<String> {
    let list = messages.downcast::<pyo3::types::PyList>()?;
    let mut pairs: Vec<(String, String)> = Vec::with_capacity(list.len());
    for item in list.iter() {
        let dict = item.downcast::<pyo3::types::PyDict>()?;
        let role: String = dict
            .get_item("role")?
            .ok_or_else(|| PyValueError::new_err("chat message missing 'role'"))?
            .extract()?;
        let content: String = dict
            .get_item("content")?
            .ok_or_else(|| PyValueError::new_err("chat message missing 'content'"))?
            .extract()?;
        pairs.push((role, content));
    }
    Ok(mmn_train::format_chat_messages(&pairs, add_generation_prompt))
}
