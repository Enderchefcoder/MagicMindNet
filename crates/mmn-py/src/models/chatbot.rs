use mmn_models::Chatbot;
use mmn_train::{
    align_qa_token_pairs, mean_corpus_loss_with_bpe, mean_qa_loss_with_bpe, tokenize_lm,
};
use mmn_models::{targets_with_vision_prefix, vision_patch_from_text, vision_rgb_patch_from_text};
use pyo3::prelude::*;

use crate::datasets::{PyDatasetCorpus, PyDatasetQA};
use crate::errors::{mmn_err_to_py, DataMismatchError};
use crate::tokenizer::PyBytePairEncoder;

#[pyclass(name = "Chatbot")]
pub struct PyChatbot {
    pub(crate) inner: Chatbot,
}

#[pymethods]
impl PyChatbot {
    #[new]
    #[pyo3(signature = (vision=false, autoset=None, vocab_size=32000, n_layer=None, d_model=None, seed=None, use_learned_pos_embed=false, max_seq_len=512, use_rope=false, rope_theta=10000.0))]
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
    ) -> Self {
        if use_learned_pos_embed && use_rope {
            panic!("Chatbot cannot use both use_learned_pos_embed and use_rope");
        }
        Self {
            inner: Chatbot::new_with_position_options(
                vision,
                autoset.as_deref(),
                vocab_size,
                n_layer,
                d_model,
                seed,
                use_learned_pos_embed,
                max_seq_len,
                use_rope,
                rope_theta,
            ),
        }
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

    fn __repr__(&self) -> String {
        let s = &self.inner.shape;
        match self.inner.init_seed {
            Some(seed) => format!(
                "Chatbot(vocab_size={}, n_layer={}, d_model={}, vision={}, parameters={}, init_seed={seed})",
                s.vocab_size,
                s.n_layer,
                s.d_model,
                self.inner.vision,
                self.inner.parameters()
            ),
            None => format!(
                "Chatbot(vocab_size={}, n_layer={}, d_model={}, vision={}, parameters={})",
                s.vocab_size,
                s.n_layer,
                s.d_model,
                self.inner.vision,
                self.inner.parameters()
            ),
        }
    }

    /// Mean cross-entropy for tokenized `input` → `target` (same tokenization as `Train`).
    #[pyo3(signature = (input, target, bpe_encoder=None, image_patch=None))]
    fn compute_loss(
        &self,
        input: &str,
        target: &str,
        bpe_encoder: Option<&PyBytePairEncoder>,
        image_patch: Option<Vec<f32>>,
    ) -> PyResult<f32> {
        let vocab = self.inner.shape.vocab_size;
        let bpe = bpe_encoder.map(|e| &e.inner);
        let mut tokens = tokenize_lm(input, vocab, bpe);
        let mut targets = tokenize_lm(target, vocab, bpe);
        align_qa_token_pairs(&mut tokens, &mut targets);
        if self.inner.has_vision_patch_encoder() {
            let patch = if let Some(p) = image_patch {
                let gray = self.inner.vision_patch_dim();
                let rgb = self.inner.vision_rgb_dim();
                if p.len() != gray && p.len() != rgb {
                    return Err(PyErr::new::<DataMismatchError, _>(format!(
                        "image_patch length {} != vision_patch_dim ({gray}) or vision_rgb_dim ({rgb}).\nFix: Pass a flat {gray}-float grayscale patch or {rgb}-float RGB patch.\nExplanation: Vision Chatbot accepts 8×8 grayscale or 8×8×3 RGB patches.",
                        p.len(),
                    )));
                }
                if p.len() == rgb && !self.inner.has_vision_rgb_conv() {
                    return Err(PyErr::new::<DataMismatchError, _>(format!(
                        "image_patch length {rgb} requires vision_rgb_conv in the checkpoint.\nFix: Pass a {gray}-float grayscale patch or export/import a vision checkpoint with vision_patch_conv.\nExplanation: RGB patches need the conv encoder loaded from checkpoint."
                    )));
                }
                p
            } else if self.inner.has_vision_rgb_conv() {
                vision_rgb_patch_from_text(input)
            } else {
                vision_patch_from_text(input)
            };
            let padded = targets_with_vision_prefix(&targets, 1, vocab);
            return self
                .inner
                .loss_on_batch_with_patches(&tokens, &padded, Some(&[patch]))
                .map_err(mmn_err_to_py);
        }
        if image_patch.is_some() {
            return Err(PyErr::new::<DataMismatchError, _>(
                "image_patch requires Chatbot(vision=True).\nFix: Construct Chatbot with vision=True or omit image_patch.\nExplanation: Only vision chatbots accept image patches.".to_string(),
            ));
        }
        self.inner
            .loss_on_batch(&tokens, &targets)
            .map_err(mmn_err_to_py)
    }

    /// Mean CE over all rows in a `DatasetQA` or `DatasetCorpus`.
    #[pyo3(signature = (dataset, bpe_encoder=None))]
    fn compute_mean_loss(
        &self,
        dataset: &Bound<'_, PyAny>,
        bpe_encoder: Option<&PyBytePairEncoder>,
    ) -> PyResult<f32> {
        let bpe = bpe_encoder.map(|e| &e.inner);
        if let Ok(ds) = dataset.downcast::<PyDatasetQA>() {
            return mean_qa_loss_with_bpe(&self.inner, &ds.borrow().inner, bpe).map_err(mmn_err_to_py);
        }
        if let Ok(ds) = dataset.downcast::<PyDatasetCorpus>() {
            return mean_corpus_loss_with_bpe(&self.inner, &ds.borrow().inner, bpe)
                .map_err(mmn_err_to_py);
        }
        Err(PyErr::new::<DataMismatchError, _>(
            "compute_mean_loss on Chatbot requires DatasetQA or DatasetCorpus.\nFix: Use DatasetQA or DatasetCorpus.\nExplanation: Classification datasets use Classifier.compute_mean_loss.".to_string(),
        ))
    }
}
