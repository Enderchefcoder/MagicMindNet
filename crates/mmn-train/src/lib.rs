use mmn_core::{enable_grad, Device, Result};
use mmn_data::{BytePairEncoder, DatasetClassification, DatasetCorpus, DatasetQA};
use mmn_models::{validate_dataset_for_classifier, Chatbot, Classifier};
use mmn_optim::{AdamW, AdamWConfig, HybridOptimizer, MuonConfig};
use rand::Rng;

#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub cuda: bool,
    pub optimizer: String,
    pub learning_rate: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 1,
            batch_size: 8,
            cuda: false,
            optimizer: "hybrid".into(),
            learning_rate: 3e-4,
        }
    }
}

pub fn simple_tokenize(text: &str, vocab_size: usize) -> Vec<usize> {
    text.bytes()
        .map(|b| (b as usize) % vocab_size)
        .take(32)
        .collect()
}

/// Tokenize for LM training: byte fallback or BPE when `bpe` is set (max 32 tokens).
pub fn tokenize_lm(text: &str, vocab_size: usize, bpe: Option<&BytePairEncoder>) -> Vec<usize> {
    match bpe {
        Some(enc) => {
            let mut ids = enc.encode(text);
            ids.truncate(32);
            ids
        }
        None => simple_tokenize(text, vocab_size),
    }
}

/// Truncate input/target token streams to the same length for CE (min length, max 32 each).
pub fn align_qa_token_pairs(tokens: &mut Vec<usize>, targets: &mut Vec<usize>) {
    let n = tokens.len().min(targets.len());
    tokens.truncate(n);
    targets.truncate(n);
    if tokens.is_empty() {
        tokens.push(0);
        targets.push(0);
    }
}

/// Mean CE over all QA samples (aligned token pairs).
/// Mean CE over classification rows with known labels (skips tags not in the model).
pub fn mean_classification_loss(model: &Classifier, dataset: &DatasetClassification) -> Result<f32> {
    let mut total = 0.0f32;
    let mut count = 0usize;
    for (text, tag) in &dataset.samples {
        if let Some(idx) = model.label_index(tag) {
            total += model.loss_on_label(text, idx)?;
            count += 1;
        }
    }
    Ok(if count > 0 {
        total / count as f32
    } else {
        0.0
    })
}

pub fn mean_qa_loss(model: &Chatbot, dataset: &DatasetQA) -> Result<f32> {
    mean_qa_loss_with_bpe(model, dataset, None)
}

pub fn mean_qa_loss_with_bpe(
    model: &Chatbot,
    dataset: &DatasetQA,
    bpe: Option<&BytePairEncoder>,
) -> Result<f32> {
    let vocab = model.shape.vocab_size;
    let mut total = 0.0f32;
    let mut count = 0usize;
    for sample in &dataset.samples {
        let mut tokens = tokenize_lm(&sample.input, vocab, bpe);
        let mut targets = tokenize_lm(&sample.output, vocab, bpe);
        align_qa_token_pairs(&mut tokens, &mut targets);
        total += model.loss_on_batch(&tokens, &targets)?;
        count += 1;
    }
    Ok(if count > 0 {
        total / count as f32
    } else {
        0.0
    })
}

fn corpus_row_lm_pairs(
    text: &str,
    vocab_size: usize,
    bpe: Option<&BytePairEncoder>,
) -> Option<(Vec<usize>, Vec<usize>)> {
    let tokens = tokenize_lm(text, vocab_size, bpe);
    if tokens.len() < 2 {
        return None;
    }
    let input = tokens[..tokens.len() - 1].to_vec();
    let targets = tokens[1..].to_vec();
    Some((input, targets))
}

/// Mean CE over corpus rows (next-token LM: input = tokens[:-1], target = tokens[1:]).
pub fn mean_corpus_loss(model: &Chatbot, dataset: &DatasetCorpus) -> Result<f32> {
    mean_corpus_loss_with_bpe(model, dataset, None)
}

pub fn mean_corpus_loss_with_bpe(
    model: &Chatbot,
    dataset: &DatasetCorpus,
    bpe: Option<&BytePairEncoder>,
) -> Result<f32> {
    let vocab = model.shape.vocab_size;
    let mut total = 0.0f32;
    let mut count = 0usize;
    for row in &dataset.rows {
        if let Some((tokens, targets)) = corpus_row_lm_pairs(&row.text, vocab, bpe) {
            total += model.loss_on_batch(&tokens, &targets)?;
            count += 1;
        }
    }
    Ok(if count > 0 {
        total / count as f32
    } else {
        0.0
    })
}

pub fn train_corpus(model: &mut Chatbot, dataset: &DatasetCorpus, config: &TrainConfig) -> Result<()> {
    train_corpus_with_bpe(model, dataset, config, None)
}

pub fn train_corpus_with_bpe(
    model: &mut Chatbot,
    dataset: &DatasetCorpus,
    config: &TrainConfig,
    bpe: Option<&BytePairEncoder>,
) -> Result<()> {
    let cuda_ok = mmn_cuda::is_available();
    Device::require_cuda_available_checked(config.cuda, cuda_ok)?;
    if config.cuda {
        model.device = Device::Cuda;
    }
    mmn_models::validate_dataset_for_chatbot(&dataset.meta.dataset_type)?;
    enable_grad(true);
    let mut hybrid = HybridOptimizer::new(
        MuonConfig::default(),
        AdamWConfig {
            lr: config.learning_rate,
            ..Default::default()
        },
    );
    let mut adamw = AdamW::new(AdamWConfig {
        lr: config.learning_rate,
        ..Default::default()
    });
    let use_hybrid = config.optimizer == "hybrid";
    let vocab = model.shape.vocab_size;
    let mut param_id = 0usize;
    let batch_size = config.batch_size.max(1);

    for _epoch in 0..config.epochs {
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..dataset.rows.len()).collect();
        for i in 0..indices.len() {
            let j = rng.gen_range(0..indices.len());
            indices.swap(i, j);
        }
        let mut accum = mmn_optim::GradAccumulator::new();
        let mut micro = 0usize;
        let mut valid_steps = 0usize;
        for (i, &idx) in indices.iter().enumerate() {
            let row = &dataset.rows[idx];
            let Some((tokens, targets)) = corpus_row_lm_pairs(&row.text, vocab, bpe) else {
                continue;
            };
            valid_steps += 1;
            if batch_size == 1 {
                model.train_step_lm(
                    &tokens,
                    &targets,
                    &mut hybrid,
                    &mut adamw,
                    use_hybrid,
                    &mut param_id,
                    None,
                )?;
            } else {
                micro += 1;
                model.train_step_lm(
                    &tokens,
                    &targets,
                    &mut hybrid,
                    &mut adamw,
                    use_hybrid,
                    &mut param_id,
                    Some(&mut accum),
                )?;
                let flush = micro >= batch_size || i + 1 == indices.len();
                if flush {
                    model.apply_accumulated_lm_grads(
                        &accum,
                        &mut hybrid,
                        &mut adamw,
                        use_hybrid,
                        &mut param_id,
                    )?;
                    accum.clear();
                    micro = 0;
                }
            }
        }
        if valid_steps == 0 {
            return Err(mmn_core::MmnError::DataMismatch {
                message: "corpus has no rows with at least 2 tokenizable bytes".into(),
                fix: "Add longer text chunks to the corpus rowfile or txtfile.".into(),
                explanation: "Corpus LM training needs input/target token pairs.".into(),
            });
        }
    }
    enable_grad(false);
    Ok(())
}

pub fn train(model: &mut Chatbot, dataset: &DatasetQA, config: &TrainConfig) -> Result<()> {
    train_with_bpe(model, dataset, config, None)
}

pub fn train_with_bpe(
    model: &mut Chatbot,
    dataset: &DatasetQA,
    config: &TrainConfig,
    bpe: Option<&BytePairEncoder>,
) -> Result<()> {
    let cuda_ok = mmn_cuda::is_available();
    Device::require_cuda_available_checked(config.cuda, cuda_ok)?;
    if config.cuda {
        model.device = Device::Cuda;
    }
    mmn_models::validate_dataset_for_chatbot(&dataset.meta.dataset_type)?;
    enable_grad(true);
    let mut hybrid = HybridOptimizer::new(
        MuonConfig::default(),
        AdamWConfig {
            lr: config.learning_rate,
            ..Default::default()
        },
    );
    let mut adamw = AdamW::new(AdamWConfig {
        lr: config.learning_rate,
        ..Default::default()
    });
    let use_hybrid = config.optimizer == "hybrid";
    let vocab = model.shape.vocab_size;
    let mut param_id = 0usize;
    let batch_size = config.batch_size.max(1);

    for _epoch in 0..config.epochs {
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..dataset.samples.len()).collect();
        for i in 0..indices.len() {
            let j = rng.gen_range(0..indices.len());
            indices.swap(i, j);
        }
        let mut accum = mmn_optim::GradAccumulator::new();
        let mut micro = 0usize;
        for (i, &idx) in indices.iter().enumerate() {
            let sample = &dataset.samples[idx];
            let mut tokens = tokenize_lm(&sample.input, vocab, bpe);
            let mut targets = tokenize_lm(&sample.output, vocab, bpe);
            align_qa_token_pairs(&mut tokens, &mut targets);
            if batch_size == 1 {
                model.train_step_lm(
                    &tokens,
                    &targets,
                    &mut hybrid,
                    &mut adamw,
                    use_hybrid,
                    &mut param_id,
                    None,
                )?;
            } else {
                micro += 1;
                model.train_step_lm(
                    &tokens,
                    &targets,
                    &mut hybrid,
                    &mut adamw,
                    use_hybrid,
                    &mut param_id,
                    Some(&mut accum),
                )?;
                let flush = micro >= batch_size || i + 1 == indices.len();
                if flush {
                    model.apply_accumulated_lm_grads(
                        &accum,
                        &mut hybrid,
                        &mut adamw,
                        use_hybrid,
                        &mut param_id,
                    )?;
                    accum.clear();
                    micro = 0;
                }
            }
        }
    }
    enable_grad(false);
    Ok(())
}

pub fn train_classifier(
    model: &mut Classifier,
    dataset: &DatasetClassification,
    config: &TrainConfig,
) -> Result<()> {
    Device::require_cuda_available_checked(config.cuda, mmn_cuda::is_available())?;
    validate_dataset_for_classifier(&dataset.meta.dataset_type)?;
    enable_grad(true);
    let mut adamw = AdamW::new(AdamWConfig {
        lr: config.learning_rate,
        ..Default::default()
    });
    let mut param_id = 0usize;
    let batch_size = config.batch_size.max(1);
    for _epoch in 0..config.epochs {
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..dataset.samples.len()).collect();
        for i in 0..indices.len() {
            let j = rng.gen_range(0..indices.len());
            indices.swap(i, j);
        }
        let total_valid = indices
            .iter()
            .filter(|&&idx| {
                let (_, tag) = &dataset.samples[idx];
                model.label_index(tag).is_some()
            })
            .count();
        let mut accum = mmn_optim::GradAccumulator::new();
        let mut micro = 0usize;
        let mut valid_step = 0usize;
        for &idx in &indices {
            let (text, tag) = &dataset.samples[idx];
            let Some(label_idx) = model.label_index(tag) else {
                continue;
            };
            valid_step += 1;
            if batch_size == 1 {
                model.train_step(text, label_idx, &mut adamw, &mut param_id, None)?;
            } else {
                micro += 1;
                model.train_step(
                    text,
                    label_idx,
                    &mut adamw,
                    &mut param_id,
                    Some(&mut accum),
                )?;
                let flush = micro >= batch_size || valid_step == total_valid;
                if flush {
                    model.apply_accumulated_classifier_grads(
                        &accum,
                        &mut adamw,
                        &mut param_id,
                    )?;
                    accum.clear();
                    micro = 0;
                }
            }
        }
    }
    enable_grad(false);
    Ok(())
}

pub fn rl(
    model: &mut Chatbot,
    dataset: &DatasetQA,
    config: &TrainConfig,
    reward_amount: f32,
    punishment_amount: f32,
    rl_type: &str,
) -> Result<()> {
    rl_with_bpe(
        model,
        dataset,
        config,
        reward_amount,
        punishment_amount,
        rl_type,
        None,
    )
}

pub fn rl_with_bpe(
    model: &mut Chatbot,
    dataset: &DatasetQA,
    config: &TrainConfig,
    reward_amount: f32,
    punishment_amount: f32,
    rl_type: &str,
    bpe: Option<&BytePairEncoder>,
) -> Result<()> {
    Device::require_cuda_available_checked(config.cuda, mmn_cuda::is_available())?;
    let policy = rl_type.to_lowercase();
    enable_grad(true);
    let vocab = model.shape.vocab_size;
    for sample in &dataset.samples {
        let mut tokens = tokenize_lm(&sample.input, vocab, bpe);
        let mut targets = tokenize_lm(&sample.output, vocab, bpe);
        align_qa_token_pairs(&mut tokens, &mut targets);
        let logits = model.forward_logits(&tokens)?;
        let score = if sample.output.contains(' ') {
            reward_amount
        } else {
            -punishment_amount
        };
        let grad_logits = mmn_core::cross_entropy_grad(&logits, &targets)?;
        let scale = match policy.as_str() {
            "policy" | "default" => score * config.learning_rate * 0.1,
            "reward_only" | "selfplay" => {
                if score > 0.0 {
                    reward_amount * config.learning_rate * 0.1
                } else {
                    0.0
                }
            }
            "punish_only" => {
                if score < 0.0 {
                    -punishment_amount * config.learning_rate * 0.1
                } else {
                    0.0
                }
            }
            _ => score * config.learning_rate * 0.1,
        };
        let mut grad = grad_logits;
        grad.mapv_inplace(|g| g * scale);
        let h = model.forward_hidden(&tokens)?;
        let (grad_w, _) = mmn_core::linear_backward(
            h.data.as_ref(),
            model.lm_head.weight.data.as_ref(),
            &grad,
        )?;
        let mut w = model.lm_head.weight.data.as_ref().clone();
        w.zip_mut_with(&grad_w, |wi, gi| *wi += gi);
        model.lm_head.weight = mmn_core::Tensor::from_array(w, true);
    }
    enable_grad(false);
    Ok(())
}

pub fn spin(
    model: &mut Chatbot,
    selfplay_epochs: usize,
    dataset: &DatasetQA,
) -> Result<()> {
    spin_with_bpe(model, selfplay_epochs, dataset, None)
}

pub fn spin_with_bpe(
    model: &mut Chatbot,
    selfplay_epochs: usize,
    dataset: &DatasetQA,
    bpe: Option<&BytePairEncoder>,
) -> Result<()> {
    for _ in 0..selfplay_epochs {
        let cfg = TrainConfig {
            epochs: 1,
            batch_size: 4,
            ..Default::default()
        };
        train_with_bpe(model, dataset, &cfg, bpe)?;
        rl_with_bpe(model, dataset, &cfg, 1.0, 0.5, "selfplay", bpe)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mmn_data::{DatasetQA, DatasetQAConfig};

    fn toy_dataset() -> DatasetQA {
        DatasetQA::load(DatasetQAConfig {
            file: format!(
                "{}/../../tests/fixtures/qa_valid.json",
                env!("CARGO_MANIFEST_DIR")
            ),
            user_row: "input".into(),
            ai_row: "output".into(),
            system_row: Some("systemprompt".into()),
            multiple_turn: true,
            thinktag: "|".into(),
            cot: true,
        })
        .unwrap()
    }

    #[test]
    fn train_classifier_reduces_loss() {
        use mmn_data::{DatasetClassification, DatasetMeta, DatasetType};
        let ds = DatasetClassification {
            meta: DatasetMeta {
                rows: 2,
                format: "test".into(),
                dataset_type: DatasetType::Classification,
            },
            samples: vec![
                ("good day".into(), "Happy".into()),
                ("great day".into(), "Happy".into()),
                ("nice day".into(), "Happy".into()),
            ],
        };
        let mut model = Classifier::from_classification_dataset(&ds, 32);
        let before = model.loss_on_label("good day", 0).unwrap();
        let cfg = TrainConfig {
            epochs: 10,
            learning_rate: 0.08,
            ..Default::default()
        };
        train_classifier(&mut model, &ds, &cfg).unwrap();
        let after = model.loss_on_label("good day", 0).unwrap();
        assert!(after <= before, "classifier loss before={before} after={after}");
    }

    #[test]
    fn chatbot_same_seed_same_loss() {
        let a = Chatbot::new_with_seed(false, None, 64, Some(1), Some(16), Some(99));
        let b = Chatbot::new_with_seed(false, None, 64, Some(1), Some(16), Some(99));
        let mut t = simple_tokenize("hello", 64);
        let mut u = simple_tokenize("world", 64);
        align_qa_token_pairs(&mut t, &mut u);
        let la = a.loss_on_batch(&t, &u).unwrap();
        let lb = b.loss_on_batch(&t, &u).unwrap();
        assert!((la - lb).abs() < 1e-6);
    }

    #[test]
    fn align_qa_token_pairs_truncates_to_shorter() {
        let mut t = vec![1, 2, 3, 4, 5];
        let mut u = vec![9, 8];
        align_qa_token_pairs(&mut t, &mut u);
        assert_eq!(t, vec![1, 2]);
        assert_eq!(u, vec![9, 8]);
    }

    #[test]
    fn train_handles_mismatched_input_output_lengths() {
        use mmn_data::{DatasetMeta, DatasetType, QaSample};
        let ds = DatasetQA {
            meta: DatasetMeta {
                rows: 1,
                format: "test".into(),
                dataset_type: DatasetType::Qa,
            },
            samples: vec![QaSample {
                input: "abcdefghijklmnopqrstuvwxyz".into(),
                output: "short".into(),
                system: None,
            }],
            chatxml: mmn_data::ChatXmlConfig::default(),
        };
        let mut model = Chatbot::new(false, None, 64, Some(1), Some(16));
        let cfg = TrainConfig {
            epochs: 1,
            learning_rate: 0.01,
            ..Default::default()
        };
        train(&mut model, &ds, &cfg).unwrap();
    }

    #[test]
    fn mean_classification_loss_finite() {
        use mmn_data::{DatasetClassification, DatasetMeta, DatasetType};
        let ds = DatasetClassification {
            meta: DatasetMeta {
                rows: 2,
                format: "test".into(),
                dataset_type: DatasetType::Classification,
            },
            samples: vec![
                ("hi".into(), "A".into()),
                ("bye".into(), "B".into()),
            ],
        };
        let model = Classifier::with_labels_seed(vec!["A".into(), "B".into()], 16, Some(1));
        let loss = mean_classification_loss(&model, &ds).unwrap();
        assert!(loss.is_finite() && loss > 0.0);
    }

    #[test]
    fn mean_classification_loss_skips_unknown_tags() {
        use mmn_data::{DatasetClassification, DatasetMeta, DatasetType};
        let ds = DatasetClassification {
            meta: DatasetMeta {
                rows: 3,
                format: "test".into(),
                dataset_type: DatasetType::Classification,
            },
            samples: vec![
                ("good".into(), "A".into()),
                ("bad".into(), "B".into()),
                ("weird".into(), "orphan".into()),
            ],
        };
        let model = Classifier::with_labels_seed(vec!["A".into(), "B".into()], 16, Some(1));
        let mixed = mean_classification_loss(&model, &ds).unwrap();
        let ds_known = DatasetClassification {
            meta: DatasetMeta {
                rows: 2,
                format: "test".into(),
                dataset_type: DatasetType::Classification,
            },
            samples: ds.samples[..2].to_vec(),
        };
        let known_only = mean_classification_loss(&model, &ds_known).unwrap();
        assert!((mixed - known_only).abs() < 1e-5);
    }

    #[test]
    fn train_classifier_reduces_mean_loss() {
        use mmn_data::{DatasetClassification, DatasetMeta, DatasetType};
        let ds = DatasetClassification {
            meta: DatasetMeta {
                rows: 3,
                format: "test".into(),
                dataset_type: DatasetType::Classification,
            },
            samples: vec![
                ("sun".into(), "Happy".into()),
                ("rain".into(), "Sad".into()),
                ("bright".into(), "Happy".into()),
            ],
        };
        let mut model =
            Classifier::from_classification_dataset_seed(&ds, 32, Some(42));
        let before = mean_classification_loss(&model, &ds).unwrap();
        let cfg = TrainConfig {
            epochs: 20,
            learning_rate: 0.01,
            optimizer: "adamw".into(),
            ..Default::default()
        };
        train_classifier(&mut model, &ds, &cfg).unwrap();
        let after = mean_classification_loss(&model, &ds).unwrap();
        assert!(
            after < before,
            "mean classification loss should decrease: before={before} after={after}"
        );
    }

    #[test]
    fn mean_qa_loss_finite_on_fixture() {
        let ds = toy_dataset();
        let model = Chatbot::new(false, None, 256, Some(2), Some(32));
        let loss = mean_qa_loss(&model, &ds).unwrap();
        assert!(loss.is_finite() && loss > 0.0);
    }

    #[test]
    fn train_batch_size_two_accumulates_and_reduces_loss() {
        let ds = toy_dataset();
        let mut model = Chatbot::new(false, None, 256, Some(2), Some(32));
        let mean_before = mean_qa_loss(&model, &ds).unwrap();
        let cfg = TrainConfig {
            epochs: 3,
            batch_size: 2,
            learning_rate: 0.05,
            optimizer: "adamw".into(),
            ..Default::default()
        };
        train(&mut model, &ds, &cfg).unwrap();
        let mean_after = mean_qa_loss(&model, &ds).unwrap();
        assert!(
            mean_after < mean_before,
            "batch_size=2 train should reduce mean loss: before={mean_before} after={mean_after}"
        );
    }

    #[test]
    fn train_classifier_batch_size_two_accumulates_and_reduces_loss() {
        use mmn_data::{DatasetClassification, DatasetMeta, DatasetType};
        let ds = DatasetClassification {
            meta: DatasetMeta {
                rows: 3,
                format: "test".into(),
                dataset_type: DatasetType::Classification,
            },
            samples: vec![
                ("sun".into(), "Happy".into()),
                ("rain".into(), "Sad".into()),
                ("bright".into(), "Happy".into()),
            ],
        };
        let mut model = Classifier::from_classification_dataset_seed(&ds, 32, Some(7));
        let before = mean_classification_loss(&model, &ds).unwrap();
        let cfg = TrainConfig {
            epochs: 15,
            batch_size: 2,
            learning_rate: 0.05,
            optimizer: "adamw".into(),
            ..Default::default()
        };
        train_classifier(&mut model, &ds, &cfg).unwrap();
        let after = mean_classification_loss(&model, &ds).unwrap();
        assert!(
            after < before,
            "batch_size=2 TrainClassifier should reduce mean loss: before={before} after={after}"
        );
    }

    #[test]
    fn train_reduces_loss() {
        let ds = toy_dataset();
        let mut model = Chatbot::new(false, None, 256, Some(2), Some(32));
        let mean_before = mean_qa_loss(&model, &ds).unwrap();
        let mut tokens = simple_tokenize(&ds.samples[0].input, 256);
        let mut targets = simple_tokenize(&ds.samples[0].output, 256);
        align_qa_token_pairs(&mut tokens, &mut targets);
        let before = model.loss_on_batch(&tokens, &targets).unwrap();
        let cfg = TrainConfig {
            epochs: 3,
            batch_size: 2,
            learning_rate: 0.05,
            ..Default::default()
        };
        train(&mut model, &ds, &cfg).unwrap();
        let after = model.loss_on_batch(&tokens, &targets).unwrap();
        let mean_after = mean_qa_loss(&model, &ds).unwrap();
        assert!(
            after <= before,
            "loss should not increase: before={before} after={after}"
        );
        assert!(
            mean_after <= mean_before,
            "mean QA loss should not increase: before={mean_before} after={mean_after}"
        );
    }

    #[test]
    fn rl_changes_lm_head_weight() {
        let ds = toy_dataset();
        let mut model = Chatbot::new(false, None, 256, Some(2), Some(32));
        let w_before = model.lm_head.weight.data[[0, 0]];
        let cfg = TrainConfig {
            learning_rate: 0.05,
            ..Default::default()
        };
        rl(&mut model, &ds, &cfg, 1.0, 0.5, "policy").unwrap();
        let w_after = model.lm_head.weight.data[[0, 0]];
        assert_ne!(w_before, w_after, "RL should update lm_head weights");
    }

    #[test]
    fn rl_reward_only_updates_lm_head() {
        let ds = toy_dataset();
        let mut model = Chatbot::new(false, None, 256, Some(2), Some(32));
        let w_before = model.lm_head.weight.data[[0, 0]];
        let cfg = TrainConfig {
            learning_rate: 0.05,
            ..Default::default()
        };
        rl(&mut model, &ds, &cfg, 1.0, 0.5, "reward_only").unwrap();
        let w_after = model.lm_head.weight.data[[0, 0]];
        assert_ne!(w_before, w_after);
    }

    #[test]
    fn rl_punish_only_updates_lm_head() {
        let ds = toy_dataset();
        let mut model = Chatbot::new(false, None, 256, Some(2), Some(32));
        let w_before = model.lm_head.weight.data[[0, 0]];
        let cfg = TrainConfig {
            learning_rate: 0.05,
            ..Default::default()
        };
        rl(&mut model, &ds, &cfg, 1.0, 0.5, "punish_only").unwrap();
        let w_after = model.lm_head.weight.data[[0, 0]];
        assert_ne!(w_before, w_after);
    }

    #[test]
    fn vision_chatbot_trains_and_keeps_vision_flag() {
        let ds = toy_dataset();
        let mut model = Chatbot::new(true, None, 256, Some(2), Some(32));
        assert!(model.has_vision());
        let loss0 = mean_qa_loss(&model, &ds).unwrap();
        let cfg = TrainConfig {
            epochs: 2,
            learning_rate: 0.05,
            ..Default::default()
        };
        train(&mut model, &ds, &cfg).unwrap();
        assert!(model.has_vision());
        let loss1 = mean_qa_loss(&model, &ds).unwrap();
        assert!(loss1 < loss0, "vision chatbot should reduce mean QA loss");
    }

    #[test]
    fn spin_completes_on_fixture() {
        let ds = toy_dataset();
        let mut model = Chatbot::new(false, None, 256, Some(2), Some(32));
        spin(&mut model, 2, &ds).expect("SPIN should complete");
        let loss = mean_qa_loss(&model, &ds).unwrap();
        assert!(loss.is_finite() && loss > 0.0);
    }

    #[test]
    fn train_corpus_reduces_mean_loss() {
        use mmn_data::{CorpusBatchSize, CorpusRow, DatasetCorpus, DatasetMeta, DatasetType};
        let ds = DatasetCorpus {
            meta: DatasetMeta {
                rows: 2,
                format: "test".into(),
                dataset_type: DatasetType::Corpus,
            },
            rows: vec![
                CorpusRow {
                    text: "hello world train".into(),
                    complexity: 1.0,
                },
                CorpusRow {
                    text: "more text for lm".into(),
                    complexity: 2.0,
                },
            ],
            batch_size: CorpusBatchSize::PerRow,
        };
        let mut model = Chatbot::new(false, None, 256, Some(2), Some(32));
        let before = mean_corpus_loss(&model, &ds).unwrap();
        train_corpus(
            &mut model,
            &ds,
            &TrainConfig {
                epochs: 3,
                batch_size: 2,
                learning_rate: 0.05,
                ..Default::default()
            },
        )
        .unwrap();
        let after = mean_corpus_loss(&model, &ds).unwrap();
        assert!(after < before, "corpus LM loss should drop: {before} -> {after}");
    }

    #[test]
    fn train_corpus_updates_learned_pos_embed() {
        use mmn_data::{CorpusBatchSize, CorpusRow, DatasetCorpus, DatasetMeta, DatasetType};
        let ds = DatasetCorpus {
            meta: DatasetMeta {
                rows: 2,
                format: "test".into(),
                dataset_type: DatasetType::Corpus,
            },
            rows: vec![
                CorpusRow {
                    text: "hello world train".into(),
                    complexity: 1.0,
                },
                CorpusRow {
                    text: "more text for lm".into(),
                    complexity: 2.0,
                },
            ],
            batch_size: CorpusBatchSize::PerRow,
        };
        let mut model = Chatbot::new_with_pe_options(
            false, None, 256, Some(1), Some(32), Some(3), true, 64,
        );
        let pe_before: Vec<f32> = model
            .pos_embed
            .as_ref()
            .unwrap()
            .weight
            .data
            .iter()
            .copied()
            .collect();
        train_corpus(
            &mut model,
            &ds,
            &TrainConfig {
                epochs: 3,
                batch_size: 2,
                learning_rate: 0.05,
                ..Default::default()
            },
        )
        .unwrap();
        let pe_after: Vec<f32> = model
            .pos_embed
            .as_ref()
            .unwrap()
            .weight
            .data
            .iter()
            .copied()
            .collect();
        assert_ne!(pe_before, pe_after);
    }

    #[test]
    fn train_learned_pos_embed_export_import_preserves_mean_loss() {
        use mmn_io::{export_safetensors, import_safetensors};
        use std::fs;
        use std::path::PathBuf;

        let ds = toy_dataset();
        let mut model = Chatbot::new_with_pe_options(
            false, None, 256, Some(1), Some(32), Some(11), true, 64,
        );
        train(
            &mut model,
            &ds,
            &TrainConfig {
                epochs: 3,
                batch_size: 1,
                learning_rate: 0.05,
                optimizer: "adamw".into(),
                ..Default::default()
            },
        )
        .unwrap();
        let loss_before = mean_qa_loss(&model, &ds).unwrap();
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("test_learned_pe_train_export.mmn");
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        export_safetensors(&model, path.to_str().unwrap()).unwrap();
        let loaded = import_safetensors(path.to_str().unwrap(), 256).unwrap();
        assert!(loaded.use_learned_pos_embed);
        assert_eq!(loaded.max_seq_len, 64);
        let loss_after = mean_qa_loss(&loaded, &ds).unwrap();
        assert!(
            (loss_before - loss_after).abs() < 1e-4,
            "trained learned PE mean loss drift after export/import: {loss_before} vs {loss_after}"
        );
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn train_corpus_learned_pos_embed_export_import_preserves_mean_loss() {
        use mmn_data::{CorpusBatchSize, CorpusRow, DatasetCorpus, DatasetMeta, DatasetType};
        use mmn_io::{export_safetensors, import_safetensors};
        use std::fs;
        use std::path::PathBuf;

        let ds = DatasetCorpus {
            meta: DatasetMeta {
                rows: 2,
                format: "test".into(),
                dataset_type: DatasetType::Corpus,
            },
            rows: vec![
                CorpusRow {
                    text: "hello world train".into(),
                    complexity: 1.0,
                },
                CorpusRow {
                    text: "more text for lm".into(),
                    complexity: 2.0,
                },
            ],
            batch_size: CorpusBatchSize::PerRow,
        };
        let mut model = Chatbot::new_with_pe_options(
            false, None, 256, Some(1), Some(32), Some(12), true, 64,
        );
        train_corpus(
            &mut model,
            &ds,
            &TrainConfig {
                epochs: 3,
                batch_size: 2,
                learning_rate: 0.05,
                optimizer: "adamw".into(),
                ..Default::default()
            },
        )
        .unwrap();
        let loss_before = mean_corpus_loss(&model, &ds).unwrap();
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("test_learned_pe_corpus_train_export.mmn");
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        export_safetensors(&model, path.to_str().unwrap()).unwrap();
        let loaded = import_safetensors(path.to_str().unwrap(), 256).unwrap();
        assert!(loaded.use_learned_pos_embed);
        let loss_after = mean_corpus_loss(&loaded, &ds).unwrap();
        assert!(
            (loss_before - loss_after).abs() < 1e-4,
            "corpus trained learned PE mean loss drift: {loss_before} vs {loss_after}"
        );
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn train_learned_pos_embed_quantize_int8_preserves_mean_loss() {
        use mmn_io::quantize_model;

        let ds = toy_dataset();
        let mut model = Chatbot::new_with_pe_options(
            false, None, 256, Some(1), Some(32), Some(14), true, 64,
        );
        train(
            &mut model,
            &ds,
            &TrainConfig {
                epochs: 3,
                batch_size: 1,
                learning_rate: 0.05,
                optimizer: "adamw".into(),
                ..Default::default()
            },
        )
        .unwrap();
        let loss_before = mean_qa_loss(&model, &ds).unwrap();
        quantize_model(&mut model, "int8").unwrap();
        let loss_after = mean_qa_loss(&model, &ds).unwrap();
        assert!(loss_after.is_finite() && loss_after > 0.0);
        let rel = (loss_after - loss_before).abs() / loss_before.max(1e-6);
        assert!(
            rel < 0.5,
            "post-train int8 quantize mean loss drift: {loss_before} -> {loss_after} (rel={rel})"
        );
    }

    #[test]
    fn train_with_bpe_reduces_loss() {
        let ds = toy_dataset();
        let mut texts: Vec<String> = ds
            .samples
            .iter()
            .flat_map(|s| vec![s.input.clone(), s.output.clone()])
            .collect();
        texts.extend(std::iter::repeat("hello hello hello world".to_string()).take(24));
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let bpe = BytePairEncoder::train(&refs, 512, 16);
        assert!(bpe.merge_count() > 0);

        let mut model = Chatbot::new_with_pe_options(
            false, None, 512, Some(1), Some(32), Some(14), false, 64,
        );
        let cfg = TrainConfig {
            epochs: 4,
            batch_size: 1,
            learning_rate: 0.05,
            optimizer: "adamw".into(),
            ..Default::default()
        };

        let loss_before = mean_qa_loss_with_bpe(&model, &ds, Some(&bpe)).unwrap();
        train_with_bpe(&mut model, &ds, &cfg, Some(&bpe)).unwrap();
        let loss_after = mean_qa_loss_with_bpe(&model, &ds, Some(&bpe)).unwrap();
        assert!(loss_after < loss_before, "{loss_before} -> {loss_after}");
    }
}
