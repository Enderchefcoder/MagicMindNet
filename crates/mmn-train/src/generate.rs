//! Autoregressive text generation for `Chatbot`.

use mmn_core::{MmnError, Result};
use mmn_data::BytePairEncoder;
use mmn_models::Chatbot;
use rand::Rng;

use crate::tokenize_lm;

/// Sampling options for `generate_text`.
#[derive(Clone, Debug)]
pub struct GenerateConfig {
    pub max_new_tokens: usize,
    /// `0.0` = greedy argmax; values `> 0` apply temperature-scaled sampling.
    pub temperature: f32,
    /// Keep only the top-k logits before sampling (`0` = disabled).
    pub top_k: usize,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 32,
            temperature: 0.0,
            top_k: 0,
        }
    }
}

/// Decode token ids to UTF-8 (byte fallback or BPE expansion).
pub fn decode_tokens(ids: &[usize], bpe: Option<&BytePairEncoder>) -> String {
    match bpe {
        Some(enc) => enc.decode(ids),
        None => {
            let bytes: Vec<u8> = ids.iter().map(|&id| (id % 256) as u8).collect();
            String::from_utf8_lossy(&bytes).into_owned()
        }
    }
}

fn max_context_len(model: &Chatbot) -> usize {
    if model.use_learned_pos_embed {
        model.max_seq_len
    } else {
        512
    }
}

fn sample_next_token(scores: &mut [f32], temperature: f32, top_k: usize, rng: &mut impl Rng) -> usize {
    if top_k > 0 && top_k < scores.len() {
        let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = indexed[top_k.saturating_sub(1)].1;
        for s in scores.iter_mut() {
            if *s < threshold {
                *s = f32::NEG_INFINITY;
            }
        }
    }
    let inv_t = 1.0 / temperature.max(1e-6);
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = scores.iter().map(|&s| ((s - max) * inv_t).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 {
        return scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
    }
    for p in &mut probs {
        *p /= sum;
    }
    let r: f32 = rng.gen();
    let mut acc = 0.0f32;
    for (i, p) in probs.iter().enumerate() {
        acc += p;
        if r <= acc {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

/// Greedy or temperature-sampled continuation from `prompt`.
pub fn generate_text(
    model: &Chatbot,
    prompt: &str,
    bpe: Option<&BytePairEncoder>,
    config: &GenerateConfig,
) -> Result<String> {
    let vocab = model.shape.vocab_size;
    let prompt_tokens = tokenize_lm(prompt, vocab, bpe);
    let prompt_len = prompt_tokens.len();
    let mut tokens = prompt_tokens;
    let max_ctx = max_context_len(model);
    let mut rng = rand::thread_rng();

    for _ in 0..config.max_new_tokens {
        if tokens.len() >= max_ctx {
            break;
        }
        let start = tokens.len().saturating_sub(max_ctx);
        let ctx = &tokens[start..];
        let logits = model.forward_logits(ctx)?;
        if logits.shape.len() != 2 {
            return Err(MmnError::Shape {
                message: "generate expected logits [seq, vocab]".into(),
            });
        }
        let last = logits.shape[0] - 1;
        let vocab_size = logits.shape[1].min(vocab);
        let mut scores: Vec<f32> = (0..vocab_size).map(|i| logits.data[[last, i]]).collect();
        let next = if config.temperature <= 0.0 {
            scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0)
        } else {
            sample_next_token(&mut scores, config.temperature, config.top_k, &mut rng)
        };
        tokens.push(next);
    }

    Ok(decode_tokens(&tokens[prompt_len..], bpe))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mmn_models::Chatbot;

    #[test]
    fn greedy_generate_is_deterministic() {
        let model = Chatbot::new_with_seed(false, None, 128, Some(1), Some(16), Some(42));
        let cfg = GenerateConfig {
            max_new_tokens: 8,
            temperature: 0.0,
            top_k: 0,
        };
        let a = generate_text(&model, "hi", None, &cfg).unwrap();
        let b = generate_text(&model, "hi", None, &cfg).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn generate_respects_max_new_tokens() {
        let model = Chatbot::new_with_seed(false, None, 128, Some(1), Some(16), Some(1));
        let cfg = GenerateConfig {
            max_new_tokens: 4,
            ..Default::default()
        };
        let out = generate_text(&model, "x", None, &cfg).unwrap();
        assert!(out.len() <= 4);
    }
}
