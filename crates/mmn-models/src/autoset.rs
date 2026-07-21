//! Autoset parameter budgets → Chatbot shapes.

#[derive(Clone, Debug)]
pub struct ModelShape {
    pub n_layer: usize,
    pub d_model: usize,
    pub n_heads: usize,
    /// Key/value head count for grouped-query attention (`n_kv_heads == n_heads` for standard MHA).
    pub n_kv_heads: usize,
    /// Per-head width. `None` means `d_model / n_heads` (classic layout).
    /// Set when Q/K projections use a wider head (e.g. Qwen3: d_model=1024, n_heads=16, head_dim=128).
    pub head_dim: Option<usize>,
    pub ffn_dim: usize,
    pub vocab_size: usize,
    pub estimated_params: usize,
}

impl ModelShape {
    /// Effective per-head channel count.
    pub fn effective_head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or_else(|| self.d_model.checked_div(self.n_heads).unwrap_or(self.d_model))
    }

    /// Query projection output width (`n_heads * head_dim`).
    pub fn q_dim(&self) -> usize {
        self.n_heads * self.effective_head_dim()
    }

    /// Key/value projection output width (`n_kv_heads * head_dim`).
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.effective_head_dim()
    }
}

/// Autoset presets accepted by `Chatbot(autoset=...)`.
pub const VALID_AUTOSET_BUDGETS: [&str; 6] = [
    "sub-1M", "sub-10M", "sub-50M", "sub-100M", "sub-1B", "sub-10B",
];

/// True when `budget` is a recognized autoset preset (either spelling).
pub fn is_valid_autoset_budget(budget: &str) -> bool {
    matches!(
        budget,
        "sub-1M"
            | "sub_1m"
            | "sub-10M"
            | "sub_10m"
            | "sub-50M"
            | "sub_50m"
            | "sub-100M"
            | "sub_100m"
            | "sub-1B"
            | "sub_1b"
            | "sub-10B"
            | "sub_10b"
    )
}

pub fn autoset(budget: &str, vocab_size: usize) -> ModelShape {
    let param_budget = match budget {
        "sub-1M" | "sub_1m" => 1_000_000,
        "sub-10M" | "sub_10m" => 10_000_000,
        "sub-50M" | "sub_50m" => 50_000_000,
        "sub-100M" | "sub_100m" => 100_000_000,
        "sub-1B" | "sub_1b" => 1_000_000_000,
        "sub-10B" | "sub_10b" => 10_000_000_000,
        _ => 100_000_000,
    };
    solve_shape(param_budget, vocab_size)
}

fn solve_shape(param_budget: usize, vocab_size: usize) -> ModelShape {
    let mut best = ModelShape {
        n_layer: 2,
        d_model: 64,
        n_heads: 1,
        n_kv_heads: 1,
        head_dim: None,
        ffn_dim: 256,
        vocab_size,
        estimated_params: 0,
    };
    // Tiny budgets need smaller dims than the classic grid.
    let d_grid: &[usize] = if param_budget <= 1_000_000 {
        &[32, 48, 64, 96, 128, 192, 256]
    } else if param_budget <= 10_000_000 {
        &[64, 96, 128, 192, 256, 384, 512]
    } else if param_budget <= 50_000_000 {
        &[64, 128, 256, 384, 512, 768]
    } else {
        &[64, 128, 256, 384, 512, 768, 1024, 1536, 2048]
    };
    let layer_grid: &[usize] = if param_budget <= 1_000_000 {
        &[1, 2, 3, 4, 6]
    } else if param_budget <= 10_000_000 {
        &[2, 4, 6, 8, 12]
    } else {
        &[2, 4, 6, 8, 12, 16, 24, 32]
    };
    for &d_model in d_grid {
        let n_heads = (d_model / 64).max(1);
        let ffn_dim = d_model * 4;
        for &n_layer in layer_grid {
            let params =
                estimate_params(n_layer, d_model, ffn_dim, vocab_size, n_heads, n_heads, None);
            if params <= param_budget && params >= best.estimated_params {
                best = ModelShape {
                    n_layer,
                    d_model,
                    n_heads,
                    n_kv_heads: n_heads,
                    head_dim: None,
                    ffn_dim,
                    vocab_size,
                    estimated_params: params,
                };
            }
        }
    }
    if best.estimated_params == 0 {
        best = ModelShape {
            n_layer: 1,
            d_model: 32,
            n_heads: 1,
            n_kv_heads: 1,
            head_dim: None,
            ffn_dim: 128,
            vocab_size,
            estimated_params: estimate_params(1, 32, 128, vocab_size, 1, 1, None),
        };
    }
    best
}

pub fn estimate_params(
    n_layer: usize,
    d_model: usize,
    ffn_dim: usize,
    vocab_size: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: Option<usize>,
) -> usize {
    let head_dim = head_dim.unwrap_or_else(|| d_model.checked_div(n_heads).unwrap_or(d_model));
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let embed = vocab_size * d_model;
    // q + k + v + out
    let attn = d_model * q_dim + 2 * d_model * kv_dim + q_dim * d_model;
    let per_layer = attn + 2 * d_model * ffn_dim + 4 * d_model;
    embed * 2 + per_layer * n_layer
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sub_100m_under_budget() {
        let s = autoset("sub-100M", 32000);
        assert!(s.estimated_params <= 105_000_000);
    }

    #[test]
    fn tiny_presets_under_budget() {
        for (name, budget) in [
            ("sub-1M", 1_000_000usize),
            ("sub-10M", 10_000_000),
            ("sub-50M", 50_000_000),
        ] {
            let s = autoset(name, 4096);
            assert!(
                s.estimated_params <= budget + budget / 20,
                "{name}: {} > {budget}",
                s.estimated_params
            );
            assert!(s.n_layer >= 1);
            assert!(s.d_model >= 16);
        }
    }

    #[test]
    fn estimate_params_custom_head_dim_differs() {
        let classic = estimate_params(1, 16, 64, 32, 4, 2, None);
        let wide = estimate_params(1, 16, 64, 32, 4, 2, Some(8));
        assert!(wide > classic);
    }
}
