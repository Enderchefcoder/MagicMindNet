//! Autoregressive text generation for `Chatbot`.

use mmn_core::{MmnError, Result};
use mmn_data::TextEncoderRef;
use mmn_models::{Chatbot, ChatbotKvCache};
use rand::Rng;

/// Sampling options for `generate_text` / `generate_token_ids`.
#[derive(Clone, Debug)]
pub struct GenerateConfig {
    pub max_new_tokens: usize,
    /// `0.0` = greedy argmax; values `> 0` apply temperature-scaled sampling.
    pub temperature: f32,
    /// Keep only the top-k logits before sampling (`0` = disabled).
    pub top_k: usize,
    /// Nucleus sampling: keep smallest set with cumulative prob >= `top_p` (`0` = disabled).
    pub top_p: f32,
    /// Drop tokens with probability below `min_p` after softmax (`0` = disabled).
    pub min_p: f32,
    /// Penalize tokens already in the generation context (`1.0` = off, `>1` discourages repeats).
    pub repetition_penalty: f32,
    /// Subtract `frequency_penalty * count(token)` from logits (`0` = off).
    pub frequency_penalty: f32,
    /// Subtract `presence_penalty` once per token type seen in context (`0` = off).
    pub presence_penalty: f32,
    /// Stop when a sampled token id is in this set (token is excluded from output).
    pub stop_token_ids: Vec<usize>,
    /// Stop when decoded new text contains any of these substrings (suffix removed).
    pub stop_strings: Vec<String>,
    /// Reuse per-layer K/V cache during generation (faster; text-only Chatbot).
    pub use_kv_cache: bool,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 32,
            temperature: 0.0,
            top_k: 0,
            top_p: 0.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            stop_token_ids: Vec::new(),
            stop_strings: Vec::new(),
            use_kv_cache: true,
        }
    }
}

/// Tokenize a prompt for generation (no training 32-token cap; bounded by `max_tokens`).
pub fn tokenize_for_generate(
    text: &str,
    vocab_size: usize,
    encoder: Option<TextEncoderRef<'_>>,
    max_tokens: usize,
) -> Vec<usize> {
    match encoder {
        Some(enc) => {
            let mut ids = enc.encode(text);
            ids.truncate(max_tokens);
            ids
        }
        None => text
            .bytes()
            .map(|b| (b as usize) % vocab_size)
            .take(max_tokens)
            .collect(),
    }
}

/// Decode token ids to UTF-8 (byte fallback or trained encoder).
pub fn decode_tokens(ids: &[usize], encoder: Option<TextEncoderRef<'_>>) -> String {
    match encoder {
        Some(enc) => enc.decode(ids),
        None => {
            let bytes: Vec<u8> = ids.iter().map(|&id| (id % 256) as u8).collect();
            String::from_utf8_lossy(&bytes).into_owned()
        }
    }
}

/// Trim decoded text at the earliest `stop_strings` match.
pub fn truncate_at_stop_strings(text: &str, stop_strings: &[String]) -> String {
    let mut end = text.len();
    for stop in stop_strings {
        if stop.is_empty() {
            continue;
        }
        if let Some(pos) = text.find(stop) {
            end = end.min(pos);
        }
    }
    text[..end].to_string()
}

fn max_context_len(model: &Chatbot) -> usize {
    if model.use_learned_pos_embed {
        model.max_seq_len
    } else {
        512
    }
}

/// Down-weight logits for tokens already present in `context` (HF-style repetition penalty).
pub fn apply_repetition_penalty(scores: &mut [f32], context: &[usize], penalty: f32) {
    if (penalty - 1.0).abs() < 1e-6 {
        return;
    }
    for &t in context {
        if t >= scores.len() {
            continue;
        }
        if scores[t] > 0.0 {
            scores[t] /= penalty;
        } else {
            scores[t] *= penalty;
        }
    }
}

/// Subtract `penalty * occurrence_count` from logits (OpenAI-style frequency penalty).
pub fn apply_frequency_penalty(scores: &mut [f32], context: &[usize], penalty: f32) {
    if penalty.abs() < 1e-6 {
        return;
    }
    let mut counts = std::collections::HashMap::new();
    for &t in context {
        *counts.entry(t).or_insert(0usize) += 1;
    }
    for (t, c) in counts {
        if t < scores.len() {
            scores[t] -= penalty * c as f32;
        }
    }
}

/// Subtract `penalty` once per distinct token in context (OpenAI-style presence penalty).
pub fn apply_presence_penalty(scores: &mut [f32], context: &[usize], penalty: f32) {
    if penalty.abs() < 1e-6 {
        return;
    }
    let mut seen = std::collections::HashSet::new();
    for &t in context {
        if seen.insert(t) && t < scores.len() {
            scores[t] -= penalty;
        }
    }
}

pub fn apply_top_p(probs: &mut [f32], top_p: f32) {
    if top_p <= 0.0 || top_p >= 1.0 {
        return;
    }
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut cumsum = 0.0f32;
    let mut keep = indexed.len();
    for (i, (_, p)) in indexed.iter().enumerate() {
        cumsum += p;
        if cumsum >= top_p {
            keep = i + 1;
            break;
        }
    }
    let kept: std::collections::HashSet<usize> = indexed.iter().take(keep).map(|(idx, _)| *idx).collect();
    for (i, p) in probs.iter_mut().enumerate() {
        if !kept.contains(&i) {
            *p = 0.0;
        }
    }
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }
}

/// Drop sampling candidates with probability below `min_p` (renormalizes survivors).
pub fn apply_min_p(probs: &mut [f32], min_p: f32) {
    if min_p <= 0.0 {
        return;
    }
    for p in probs.iter_mut() {
        if *p < min_p {
            *p = 0.0;
        }
    }
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }
}

fn sample_next_token(
    scores: &mut [f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    rng: &mut impl Rng,
) -> usize {
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
    apply_top_p(&mut probs, top_p);
    apply_min_p(&mut probs, min_p);
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

fn last_token_scores(
    logits: &mmn_core::Tensor,
    vocab: usize,
    tokens: &[usize],
    config: &GenerateConfig,
) -> Result<Vec<f32>> {
    if logits.shape.len() != 2 {
        return Err(MmnError::Shape {
            message: "generate expected logits [seq, vocab]".into(),
        });
    }
    let last = logits.shape[0] - 1;
    let vocab_size = logits.shape[1].min(vocab);
    let mut scores: Vec<f32> = (0..vocab_size).map(|i| logits.data[[last, i]]).collect();
    apply_repetition_penalty(&mut scores, tokens, config.repetition_penalty);
    apply_frequency_penalty(&mut scores, tokens, config.frequency_penalty);
    apply_presence_penalty(&mut scores, tokens, config.presence_penalty);
    Ok(scores)
}

fn context_window<'a>(tokens: &'a [usize], max_ctx: usize) -> &'a [usize] {
    let start = tokens.len().saturating_sub(max_ctx);
    &tokens[start..]
}

fn forward_logits_after_append(
    model: &Chatbot,
    tokens: &[usize],
    cache: &mut ChatbotKvCache,
    max_ctx: usize,
) -> Result<mmn_core::Tensor> {
    if tokens.len() > max_ctx {
        let overflow = tokens.len() - max_ctx;
        if model.uses_rope() && overflow == 1 && cache.seq_len >= max_ctx {
            model.slide_kv_cache_one(cache)?;
            let last = *tokens
                .last()
                .ok_or_else(|| MmnError::Shape {
                    message: "forward_logits_after_append requires at least one token".into(),
                })?;
            return model.forward_logits_with_kv_cache(&[last], cache);
        }
        model.reset_kv_cache_prefill(context_window(tokens, max_ctx), cache)
    } else {
        let last = *tokens
            .last()
            .ok_or_else(|| MmnError::Shape {
                message: "forward_logits_after_append requires at least one token".into(),
            })?;
        model.forward_logits_with_kv_cache(&[last], cache)
    }
}

fn pick_next_token(
    scores: &mut [f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    rng: &mut impl Rng,
) -> usize {
    if temperature <= 0.0 {
        scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    } else {
        sample_next_token(scores, temperature, top_k, top_p, min_p, rng)
    }
}

fn generate_token_ids_with_kv_cache(
    model: &Chatbot,
    tokens: &mut Vec<usize>,
    prompt_len: usize,
    max_ctx: usize,
    encoder: Option<TextEncoderRef<'_>>,
    config: &GenerateConfig,
) -> Result<Vec<usize>> {
    let vocab = model.shape.vocab_size;
    let mut cache = model.init_kv_cache();
    let mut rng = rand::thread_rng();
    let mut logits =
        model.reset_kv_cache_prefill(context_window(tokens, max_ctx), &mut cache)?;
    let mut scores = last_token_scores(&logits, vocab, tokens, config)?;

    for _ in 0..config.max_new_tokens {
        let next = pick_next_token(
            &mut scores,
            config.temperature,
            config.top_k,
            config.top_p,
            config.min_p,
            &mut rng,
        );
        if config.stop_token_ids.contains(&next) {
            break;
        }
        tokens.push(next);

        if !config.stop_strings.is_empty() {
            let new_text = decode_tokens(&tokens[prompt_len..], encoder);
            let trimmed = truncate_at_stop_strings(&new_text, &config.stop_strings);
            if trimmed.len() < new_text.len() {
                let mut out_ids = Vec::new();
                for i in 1..=(tokens.len() - prompt_len) {
                    let partial = decode_tokens(&tokens[prompt_len..prompt_len + i], encoder);
                    if partial == trimmed {
                        out_ids.extend_from_slice(&tokens[prompt_len..prompt_len + i]);
                        return Ok(out_ids);
                    }
                    if partial.len() > trimmed.len() {
                        break;
                    }
                }
                return Ok(out_ids);
            }
        }

        logits = forward_logits_after_append(model, tokens, &mut cache, max_ctx)?;
        scores = last_token_scores(&logits, vocab, tokens, config)?;
    }

    Ok(tokens[prompt_len..].to_vec())
}

/// Sample new token ids after `prompt` (does not include prompt tokens).
pub fn generate_token_ids(
    model: &Chatbot,
    prompt: &str,
    encoder: Option<TextEncoderRef<'_>>,
    config: &GenerateConfig,
) -> Result<Vec<usize>> {
    let vocab = model.shape.vocab_size;
    let max_ctx = max_context_len(model);
    let prompt_tokens = tokenize_for_generate(prompt, vocab, encoder, max_ctx);
    let mut tokens = prompt_tokens;
    let prompt_len = tokens.len();

    if config.use_kv_cache && !model.vision {
        return generate_token_ids_with_kv_cache(
            model,
            &mut tokens,
            prompt_len,
            max_ctx,
            encoder,
            config,
        );
    }

    let mut rng = rand::thread_rng();

    for _ in 0..config.max_new_tokens {
        let ctx = context_window(&tokens, max_ctx);
        let logits = model.forward_logits(ctx)?;
        let mut scores = last_token_scores(&logits, vocab, &tokens, config)?;
        let next = pick_next_token(
            &mut scores,
            config.temperature,
            config.top_k,
            config.top_p,
            config.min_p,
            &mut rng,
        );

        if config.stop_token_ids.contains(&next) {
            break;
        }
        tokens.push(next);

        if !config.stop_strings.is_empty() {
            let new_text = decode_tokens(&tokens[prompt_len..], encoder);
            let trimmed = truncate_at_stop_strings(&new_text, &config.stop_strings);
            if trimmed.len() < new_text.len() {
                let mut out_ids = Vec::new();
                for i in 1..=(tokens.len() - prompt_len) {
                    let partial = decode_tokens(&tokens[prompt_len..prompt_len + i], encoder);
                    if partial == trimmed {
                        out_ids.extend_from_slice(&tokens[prompt_len..prompt_len + i]);
                        return Ok(out_ids);
                    }
                    if partial.len() > trimmed.len() {
                        break;
                    }
                }
                return Ok(out_ids);
            }
        }
    }

    Ok(tokens[prompt_len..].to_vec())
}

/// Greedy or temperature-sampled continuation from `prompt`.
pub fn generate_text(
    model: &Chatbot,
    prompt: &str,
    encoder: Option<TextEncoderRef<'_>>,
    config: &GenerateConfig,
) -> Result<String> {
    let ids = generate_token_ids(model, prompt, encoder, config)?;
    Ok(decode_tokens(&ids, encoder))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mmn_models::Chatbot;

    #[test]
    fn tokenize_for_generate_allows_long_prompt() {
        let long = "a".repeat(48);
        let ids = tokenize_for_generate(&long, 256, None, 512);
        assert_eq!(ids.len(), 48);
    }

    #[test]
    fn truncate_at_stop_strings_cuts_earliest() {
        let out = truncate_at_stop_strings("hello\nworld", &["\n".into()]);
        assert_eq!(out, "hello");
    }

    #[test]
    fn greedy_generate_is_deterministic() {
        let model = Chatbot::new_with_seed(false, None, 128, Some(1), Some(16), Some(42));
        let cfg = GenerateConfig {
            max_new_tokens: 8,
            temperature: 0.0,
            top_k: 0,
            ..Default::default()
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
        let ids = generate_token_ids(&model, "x", None, &cfg).unwrap();
        assert_eq!(ids.len(), 4);
    }

    #[test]
    fn repetition_penalty_reduces_repeat_mass() {
        let mut scores = vec![2.0f32, 0.1, 0.1, 0.1];
        apply_repetition_penalty(&mut scores, &[0], 2.0);
        assert!(scores[0] < 2.0);
        assert!((scores[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn top_p_trims_tail_mass() {
        let mut probs = vec![0.5f32, 0.3, 0.15, 0.05];
        apply_top_p(&mut probs, 0.8);
        assert!((probs[3] - 0.0).abs() < 1e-6);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn kv_cache_generation_matches_full_forward() {
        let model = Chatbot::new_with_pe_options(
            false, None, 128, Some(1), Some(16), Some(55), true, 64,
        );
        let cfg = GenerateConfig {
            max_new_tokens: 6,
            temperature: 0.0,
            ..Default::default()
        };
        let kv_ids = generate_token_ids(&model, "hello", None, &cfg).unwrap();
        let full_ids = generate_token_ids(
            &model,
            "hello",
            None,
            &GenerateConfig {
                use_kv_cache: false,
                ..cfg.clone()
            },
        )
        .unwrap();
        assert_eq!(kv_ids, full_ids);
    }

    #[test]
    fn min_p_zeros_low_prob_tail() {
        let mut probs = vec![0.7f32, 0.2, 0.09, 0.01];
        apply_min_p(&mut probs, 0.05);
        assert!((probs[3] - 0.0).abs() < 1e-6);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn sliding_window_generates_past_max_ctx() {
        let model = Chatbot::new_with_pe_options(
            false, None, 64, Some(1), Some(16), Some(33), true, 8,
        );
        let cfg = GenerateConfig {
            max_new_tokens: 12,
            temperature: 0.0,
            ..Default::default()
        };
        let ids = generate_token_ids(&model, "ab", None, &cfg).unwrap();
        assert_eq!(ids.len(), 12);
        let full_ids = generate_token_ids(
            &model,
            "ab",
            None,
            &GenerateConfig {
                use_kv_cache: false,
                ..cfg
            },
        )
        .unwrap();
        assert_eq!(ids, full_ids);
    }

    #[test]
    fn frequency_penalty_reduces_repeated_logits() {
        let mut scores = vec![1.0f32, 2.0, 3.0, 4.0];
        apply_frequency_penalty(&mut scores, &[1, 1, 2], 0.5);
        assert!((scores[1] - 1.0).abs() < 1e-5);
        assert!((scores[2] - 2.5).abs() < 1e-5);
        assert!((scores[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn presence_penalty_applies_once_per_type() {
        let mut scores = vec![1.0f32, 2.0, 3.0];
        apply_presence_penalty(&mut scores, &[0, 0, 1], 1.0);
        assert!((scores[0] - 0.0).abs() < 1e-5);
        assert!((scores[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn rope_sliding_kv_generation_matches_full_forward() {
        let model = Chatbot::new_with_position_options(
            false,
            None,
            128,
            Some(1),
            Some(16),
            Some(77),
            false,
            8,
            true,
            10_000.0,
            None,
            None,
        );
        let cfg = GenerateConfig {
            max_new_tokens: 10,
            temperature: 0.0,
            ..Default::default()
        };
        let kv_ids = generate_token_ids(&model, "hi", None, &cfg).unwrap();
        let full_ids = generate_token_ids(
            &model,
            "hi",
            None,
            &GenerateConfig {
                use_kv_cache: false,
                ..cfg
            },
        )
        .unwrap();
        assert_eq!(kv_ids, full_ids);
    }

    #[test]
    fn stop_token_id_limits_output() {
        let model = Chatbot::new_with_seed(false, None, 128, Some(1), Some(16), Some(99));
        let one = generate_token_ids(
            &model,
            "z",
            None,
            &GenerateConfig {
                max_new_tokens: 1,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(one.len(), 1);
        let stopped = generate_token_ids(
            &model,
            "z",
            None,
            &GenerateConfig {
                max_new_tokens: 8,
                stop_token_ids: vec![one[0]],
                ..Default::default()
            },
        )
        .unwrap();
        assert!(stopped.is_empty());
    }
}
