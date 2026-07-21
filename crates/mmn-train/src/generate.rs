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
    /// Locally typical sampling (`0` = off); keep tokens with `-ln(p)` near entropy.
    pub typical_p: f32,
    /// `0` = off; `1` or `2` = Mirostat v2 (v1 treated as v2).
    pub mirostat: u32,
    /// Mirostat target surprise (default `5.0`).
    pub mirostat_tau: f32,
    /// Mirostat learning rate for `mu` (default `0.1`).
    pub mirostat_eta: f32,
    /// Mirostat running surprise estimate (default `2 * tau`).
    pub mirostat_mu: f32,
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
    /// Optional vision prefix patches for `Chatbot(vision=True)` (prefill only).
    pub vision_patches: Option<Vec<Vec<f32>>>,
    /// Constrain decoding to a minimal JSON object/array subset.
    pub json_mode: bool,
    /// Optional grammar name: `"digit"` | `"json"` (`"json"` aliases `json_mode`).
    pub grammar: Option<String>,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 32,
            temperature: 0.0,
            top_k: 0,
            top_p: 0.0,
            min_p: 0.0,
            typical_p: 0.0,
            mirostat: 0,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
            mirostat_mu: 10.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            stop_token_ids: Vec::new(),
            stop_strings: Vec::new(),
            use_kv_cache: true,
            vision_patches: None,
            json_mode: false,
            grammar: None,
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
    if model.use_learned_pos_embed || model.uses_rope() {
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

fn renorm_probs(probs: &mut [f32]) {
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }
}

/// Locally typical sampling (llama.cpp / HF typical): keep tokens whose
/// surprise `-ln(p)` is closest to the distribution entropy until cumulative
/// probability reaches `typical_p`, then zero the rest and renormalize.
pub fn apply_typical_p(probs: &mut [f32], typical_p: f32) {
    if typical_p <= 0.0 || typical_p >= 1.0 {
        return;
    }
    let mut entropy = 0.0f32;
    for &p in probs.iter() {
        if p > 0.0 {
            entropy += -p * p.ln();
        }
    }
    let mut indexed: Vec<(usize, f32, f32)> = probs
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, p)| *p > 0.0)
        .map(|(i, p)| {
            let surprise = -p.ln();
            let shifted = (surprise - entropy).abs();
            (i, p, shifted)
        })
        .collect();
    if indexed.is_empty() {
        return;
    }
    indexed.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    let mut cumsum = 0.0f32;
    let mut keep = indexed.len();
    for (i, (_, p, _)) in indexed.iter().enumerate() {
        cumsum += p;
        if cumsum >= typical_p {
            keep = i + 1;
            break;
        }
    }
    let kept: std::collections::HashSet<usize> =
        indexed.iter().take(keep).map(|(idx, _, _)| *idx).collect();
    for (i, p) in probs.iter_mut().enumerate() {
        if !kept.contains(&i) {
            *p = 0.0;
        }
    }
    renorm_probs(probs);
}

/// Mirostat v2 candidate filter: zero tokens with surprise `-ln(p) > mu`, then renorm.
/// If every token is truncated, keep the highest-probability token.
pub fn apply_mirostat_v2_truncate(probs: &mut [f32], mu: f32) {
    let best = argmax_f32(probs);
    let mut any = false;
    for p in probs.iter_mut() {
        if *p <= 0.0 {
            continue;
        }
        if -p.ln() > mu {
            *p = 0.0;
        } else {
            any = true;
        }
    }
    if !any {
        probs.iter_mut().for_each(|p| *p = 0.0);
        if best < probs.len() {
            probs[best] = 1.0;
        }
    } else {
        renorm_probs(probs);
    }
}

fn sample_from_probs(probs: &[f32], rng: &mut impl Rng) -> usize {
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

fn argmax_f32(scores: &[f32]) -> usize {
    scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn softmax_temperature(scores: &[f32], temperature: f32) -> Vec<f32> {
    let inv_t = 1.0 / temperature.max(1e-6);
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = scores.iter().map(|&s| ((s - max) * inv_t).exp()).collect();
    renorm_probs(&mut probs);
    probs
}

fn sample_next_token(
    scores: &mut [f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    typical_p: f32,
    mirostat: u32,
    mirostat_tau: f32,
    mirostat_eta: f32,
    mirostat_mu: &mut f32,
    rng: &mut impl Rng,
) -> usize {
    let use_mirostat = mirostat != 0;

    if !use_mirostat && top_k > 0 && top_k < scores.len() {
        let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = indexed[top_k.saturating_sub(1)].1;
        for s in scores.iter_mut() {
            if *s < threshold {
                *s = f32::NEG_INFINITY;
            }
        }
    }

    let mut probs = softmax_temperature(scores, temperature);
    if probs.iter().sum::<f32>() <= 0.0 {
        return argmax_f32(scores);
    }

    if use_mirostat {
        apply_mirostat_v2_truncate(&mut probs, *mirostat_mu);
        let next = sample_from_probs(&probs, rng);
        let p_sel = probs[next].max(1e-12);
        let observed_surprise = -p_sel.ln();
        *mirostat_mu -= mirostat_eta * (observed_surprise - mirostat_tau);
        return next;
    }

    apply_top_p(&mut probs, top_p);
    apply_min_p(&mut probs, min_p);
    apply_typical_p(&mut probs, typical_p);
    if probs.iter().sum::<f32>() <= 0.0 {
        return argmax_f32(scores);
    }
    sample_from_probs(&probs, rng)
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

fn wants_json(config: &GenerateConfig) -> bool {
    config.json_mode || config.grammar.as_deref() == Some("json")
}

fn apply_grammar_mask(scores: &mut [f32], generated: &str, config: &GenerateConfig) {
    if config.grammar.as_deref() == Some("digit") {
        crate::json_constrain::apply_digit_mask(scores);
        return;
    }
    if wants_json(config) {
        crate::json_constrain::apply_json_mask(scores, generated);
    }
}

fn scores_with_constraints(
    logits: &mmn_core::Tensor,
    vocab: usize,
    tokens: &[usize],
    prompt_len: usize,
    encoder: Option<TextEncoderRef<'_>>,
    config: &GenerateConfig,
) -> Result<Vec<f32>> {
    let mut scores = last_token_scores(logits, vocab, tokens, config)?;
    if config.grammar.is_some() || config.json_mode {
        let generated = decode_tokens(&tokens[prompt_len..], encoder);
        apply_grammar_mask(&mut scores, &generated, config);
    }
    Ok(scores)
}

fn context_window(tokens: &[usize], max_ctx: usize) -> &[usize] {
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
        if model.uses_rope()
            && overflow == 1
            && cache.seq_len.saturating_sub(cache.n_vision_prefix) >= max_ctx
        {
            model.slide_kv_cache_one(cache)?;
            let last = *tokens
                .last()
                .ok_or_else(|| MmnError::Shape {
                    message: "forward_logits_after_append requires at least one token".into(),
                })?;
            return model.forward_logits_with_kv_cache(&[last], cache);
        }
        let patch_clone = cache.vision_patches.clone();
        model.reset_kv_cache_prefill_with_patches(
            context_window(tokens, max_ctx),
            patch_clone.as_deref(),
            cache,
        )
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
    config: &GenerateConfig,
    mirostat_mu: &mut f32,
    rng: &mut impl Rng,
) -> usize {
    if config.temperature <= 0.0 && config.mirostat == 0 {
        argmax_f32(scores)
    } else {
        sample_next_token(
            scores,
            config.temperature.max(1e-6),
            config.top_k,
            config.top_p,
            config.min_p,
            config.typical_p,
            config.mirostat,
            config.mirostat_tau,
            config.mirostat_eta,
            mirostat_mu,
            rng,
        )
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
    let mut mirostat_mu = config.mirostat_mu;
    let patches = config.vision_patches.as_deref();
    let mut logits = model.reset_kv_cache_prefill_with_patches(
        context_window(tokens, max_ctx),
        patches,
        &mut cache,
    )?;
    let mut scores = scores_with_constraints(&logits, vocab, tokens, prompt_len, encoder, config)?;

    for _ in 0..config.max_new_tokens {
        let next = pick_next_token(&mut scores, config, &mut mirostat_mu, &mut rng);
        if config.stop_token_ids.contains(&next) {
            break;
        }
        tokens.push(next);

        if wants_json(config) {
            let new_text = decode_tokens(&tokens[prompt_len..], encoder);
            if crate::json_constrain::json_is_complete(&new_text) {
                break;
            }
        }

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
        scores = scores_with_constraints(&logits, vocab, tokens, prompt_len, encoder, config)?;
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

    if config.use_kv_cache && model.n_loops <= 1 {
        return generate_token_ids_with_kv_cache(
            model,
            &mut tokens,
            prompt_len,
            max_ctx,
            encoder,
            config,
        );
    }

    let patches = config.vision_patches.as_deref();
    let mut rng = rand::thread_rng();
    let mut mirostat_mu = config.mirostat_mu;

    for _ in 0..config.max_new_tokens {
        let ctx = context_window(&tokens, max_ctx);
        let logits = if patches.is_some() && model.has_vision_patch_encoder() {
            model.forward_logits_with_patches(ctx, patches)?
        } else {
            model.forward_logits(ctx)?
        };
        let mut scores =
            scores_with_constraints(&logits, vocab, &tokens, prompt_len, encoder, config)?;
        let next = pick_next_token(&mut scores, config, &mut mirostat_mu, &mut rng);

        if config.stop_token_ids.contains(&next) {
            break;
        }
        tokens.push(next);

        if wants_json(config) {
            let new_text = decode_tokens(&tokens[prompt_len..], encoder);
            if crate::json_constrain::json_is_complete(&new_text) {
                break;
            }
        }

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
    let text = decode_tokens(&ids, encoder);
    if wants_json(config) {
        Ok(crate::json_constrain::finalize_json(&text))
    } else {
        Ok(text)
    }
}

/// Like `generate_text`, but returns one decoded piece per newly sampled token.
pub fn generate_text_stream(
    model: &Chatbot,
    prompt: &str,
    encoder: Option<TextEncoderRef<'_>>,
    config: &GenerateConfig,
) -> Result<Vec<String>> {
    let ids = generate_token_ids(model, prompt, encoder, config)?;
    let mut pieces = Vec::with_capacity(ids.len());
    let mut prev = String::new();
    for i in 1..=ids.len() {
        let cur = decode_tokens(&ids[..i], encoder);
        let piece = if cur.starts_with(&prev) {
            cur[prev.len()..].to_string()
        } else {
            decode_tokens(&ids[i - 1..i], encoder)
        };
        prev = cur;
        pieces.push(piece);
    }
    Ok(pieces)
}

/// Mean-pool `forward_hidden` over the sequence dimension → `d_model` floats.
pub fn embed_mean_pool(model: &Chatbot, token_ids: &[usize]) -> Result<Vec<f32>> {
    let d_model = model.shape.d_model;
    if token_ids.is_empty() {
        return Ok(vec![0.0; d_model]);
    }
    let h = model.forward_hidden(token_ids)?;
    if h.shape.len() != 2 || h.shape[1] != d_model {
        return Err(MmnError::Shape {
            message: format!(
                "embed_mean_pool expected hidden [seq, {d_model}], got {:?}",
                h.shape
            ),
        });
    }
    let seq = h.shape[0].max(1) as f32;
    let mut out = vec![0.0f32; d_model];
    for s in 0..h.shape[0] {
        for d in 0..d_model {
            out[d] += h.data[[s, d]];
        }
    }
    for v in &mut out {
        *v /= seq;
    }
    Ok(out)
}

/// Format OpenAI/Ollama-style chat messages as ChatML.
pub fn format_chat_messages(messages: &[(String, String)], add_generation_prompt: bool) -> String {
    let mut out = String::new();
    let mut last_role = String::new();
    for (role, content) in messages {
        out.push_str("<|im_start|>");
        out.push_str(role);
        out.push('\n');
        out.push_str(content);
        out.push_str("<|im_end|>\n");
        last_role = role.clone();
    }
    if add_generation_prompt && last_role != "assistant" {
        out.push_str("<|im_start|>assistant\n");
    }
    out
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
    fn vision_kv_generation_matches_full_forward() {
        use mmn_models::vision_patch_from_text;

        let model = Chatbot::new(true, None, 128, Some(1), Some(16));
        let patch = vision_patch_from_text("scene");
        let cfg = GenerateConfig {
            max_new_tokens: 4,
            temperature: 0.0,
            vision_patches: Some(vec![patch]),
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
    fn vision_sliding_window_past_max_ctx_matches_full_forward() {
        use mmn_models::vision_patch_from_text;

        let model = Chatbot::new_with_position_options(
            true,
            None,
            64,
            Some(1),
            Some(16),
            Some(44),
            false,
            4,
            true,
            10_000.0,
            None,
            None,
        );
        let patch = vision_patch_from_text("scene");
        let cfg = GenerateConfig {
            max_new_tokens: 10,
            temperature: 0.0,
            vision_patches: Some(vec![patch]),
            ..Default::default()
        };
        let kv_ids = generate_token_ids(&model, "abcd", None, &cfg).unwrap();
        assert_eq!(kv_ids.len(), 10);
        let full_ids = generate_token_ids(
            &model,
            "abcd",
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

    #[test]
    fn typical_p_zeros_some_mass() {
        let mut probs = vec![0.4f32, 0.3, 0.2, 0.1];
        apply_typical_p(&mut probs, 0.5);
        let zeroed = probs.iter().filter(|&&p| p == 0.0).count();
        assert!(zeroed >= 1, "typical_p should zero at least one token");
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn mirostat_v2_truncates_high_surprise() {
        // High-surprise (low-prob) tail should be zeroed when mu is tight.
        let mut probs = vec![0.7f32, 0.2, 0.08, 0.02];
        apply_mirostat_v2_truncate(&mut probs, 1.0); // -ln(0.08)≈2.5, -ln(0.02)≈3.9 > 1
        assert!(probs[2] == 0.0 || probs[3] == 0.0);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn stream_pieces_join_to_generate_text() {
        let model = Chatbot::new_with_seed(false, None, 128, Some(1), Some(16), Some(7));
        let cfg = GenerateConfig {
            max_new_tokens: 5,
            temperature: 0.0,
            ..Default::default()
        };
        let full = generate_text(&model, "ab", None, &cfg).unwrap();
        let pieces = generate_text_stream(&model, "ab", None, &cfg).unwrap();
        assert_eq!(pieces.len(), 5);
        assert_eq!(pieces.join(""), full);
    }

    #[test]
    fn embed_mean_pool_has_d_model() {
        let model = Chatbot::new_with_seed(false, None, 64, Some(1), Some(32), Some(3));
        let ids = tokenize_for_generate("hello", 64, None, 512);
        let v = embed_mean_pool(&model, &ids).unwrap();
        assert_eq!(v.len(), 32);
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn format_chat_messages_chatml() {
        let msgs = vec![
            ("system".into(), "You are helpful.".into()),
            ("user".into(), "Hi".into()),
        ];
        let text = format_chat_messages(&msgs, true);
        assert!(text.contains("<|im_start|>system"));
        assert!(text.contains("Hi"));
        assert!(text.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn json_mode_generate_is_parseable_shape() {
        let model = Chatbot::new_with_seed(false, None, 256, Some(1), Some(32), Some(11));
        let cfg = GenerateConfig {
            max_new_tokens: 24,
            temperature: 0.8,
            json_mode: true,
            ..Default::default()
        };
        let out = generate_text(&model, r#"Return JSON: {"ok": true}"#, None, &cfg).unwrap();
        let text = out.trim();
        assert!(text.starts_with('{') || text.starts_with('['), "{text}");
        assert!(text.ends_with('}') || text.ends_with(']'), "{text}");
    }

    #[test]
    fn digit_grammar_only_emits_digits() {
        let model = Chatbot::new_with_seed(false, None, 256, Some(1), Some(16), Some(2));
        let cfg = GenerateConfig {
            max_new_tokens: 6,
            temperature: 0.9,
            grammar: Some("digit".into()),
            ..Default::default()
        };
        let out = generate_text(&model, "n=", None, &cfg).unwrap();
        assert!(!out.is_empty());
        assert!(out.chars().all(|c| c.is_ascii_digit()), "{out}");
    }
}
