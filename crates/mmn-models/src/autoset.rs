#[derive(Clone, Debug)]
pub struct ModelShape {
    pub n_layer: usize,
    pub d_model: usize,
    pub n_heads: usize,
    /// Key/value head count for grouped-query attention (`n_kv_heads == n_heads` for standard MHA).
    pub n_kv_heads: usize,
    pub ffn_dim: usize,
    pub vocab_size: usize,
    pub estimated_params: usize,
}

pub fn autoset(budget: &str, vocab_size: usize) -> ModelShape {
    let param_budget = match budget {
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
        ffn_dim: 256,
        vocab_size,
        estimated_params: 0,
    };
    for d_model in [64usize, 128, 256, 384, 512, 768, 1024, 1536, 2048] {
        let n_heads = (d_model / 64).max(1);
        let ffn_dim = d_model * 4;
        for n_layer in [2usize, 4, 6, 8, 12, 16, 24, 32] {
            let params = estimate_params(n_layer, d_model, ffn_dim, vocab_size, n_heads, n_heads);
            if params <= param_budget && params >= best.estimated_params {
                best = ModelShape {
                    n_layer,
                    d_model,
                    n_heads,
                    n_kv_heads: n_heads,
                    ffn_dim,
                    vocab_size,
                    estimated_params: params,
                };
            }
        }
    }
    if best.estimated_params == 0 {
        best = ModelShape {
            n_layer: 2,
            d_model: 64,
            n_heads: 1,
            n_kv_heads: 1,
            ffn_dim: 256,
            vocab_size,
            estimated_params: estimate_params(2, 64, 256, vocab_size, 1, 1),
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
) -> usize {
    let head_dim = if n_heads > 0 { d_model / n_heads } else { d_model };
    let kv_dim = n_kv_heads * head_dim;
    let embed = vocab_size * d_model;
    let attn = 2 * d_model * d_model + 2 * kv_dim * d_model;
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
}
