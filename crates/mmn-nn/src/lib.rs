use mmn_core::{MmnError, Result, Tensor};
use ndarray::{ArrayD, IxDyn};
use rand::{Rng, SeedableRng};

fn transpose_tensor(t: &Tensor) -> Result<Tensor> {
    let v = t
        .data
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| MmnError::Shape {
            message: e.to_string(),
        })?;
    Ok(Tensor::from_array(v.t().to_owned().into_dyn(), t.requires_grad))
}

pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub in_features: usize,
    pub out_features: usize,
}

pub fn rng_from_seed(seed: Option<u64>) -> rand::rngs::StdRng {
    match seed {
        Some(s) => rand::rngs::StdRng::seed_from_u64(s),
        None => rand::rngs::StdRng::from_entropy(),
    }
}

/// Fixed sinusoidal position encoding `[seq_len, d_model]` (Vaswani et al.).
pub fn sinusoidal_position_encoding(seq_len: usize, d_model: usize) -> Tensor {
    let mut pe = vec![0.0f32; seq_len * d_model];
    for pos in 0..seq_len {
        for i in 0..d_model {
            let div = 10000f32.powf(2.0 * (i / 2) as f32 / d_model as f32);
            let angle = pos as f32 / div;
            pe[pos * d_model + i] = if i % 2 == 0 {
                angle.sin()
            } else {
                angle.cos()
            };
        }
    }
    Tensor::from_array(
        ArrayD::from_shape_vec(IxDyn(&[seq_len, d_model]), pe).unwrap(),
        false,
    )
}

/// Concatenate two `[rows, d_model]` tensors along the sequence (row) axis.
pub fn concat_sequence_rows(top: &Tensor, bottom: &Tensor) -> Result<Tensor> {
    if top.shape.len() != 2 || bottom.shape.len() != 2 {
        return Err(MmnError::Shape {
            message: "concat_sequence_rows expects [rows, d_model] tensors".into(),
        });
    }
    let d_model = top.shape[1];
    if bottom.shape[1] != d_model {
        return Err(MmnError::Shape {
            message: format!(
                "concat_sequence_rows d_model mismatch: {} vs {}",
                d_model, bottom.shape[1]
            ),
        });
    }
    let rows = top.shape[0] + bottom.shape[0];
    let mut data = vec![0.0f32; rows * d_model];
    let top_v = top
        .data
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| MmnError::Shape {
            message: e.to_string(),
        })?;
    let bottom_v = bottom
        .data
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| MmnError::Shape {
            message: e.to_string(),
        })?;
    for r in 0..top.shape[0] {
        for c in 0..d_model {
            data[r * d_model + c] = top_v[[r, c]];
        }
    }
    let off = top.shape[0] * d_model;
    for r in 0..bottom.shape[0] {
        for c in 0..d_model {
            data[off + r * d_model + c] = bottom_v[[r, c]];
        }
    }
    Ok(Tensor::from_array(
        ArrayD::from_shape_vec(IxDyn(&[rows, d_model]), data).unwrap(),
        top.requires_grad || bottom.requires_grad,
    ))
}

/// Add sinusoidal PE to `[seq_len, d_model]` activations (in-place on a new tensor).
pub fn add_sinusoidal_position_encoding(x: &Tensor) -> Result<Tensor> {
    if x.shape.len() != 2 {
        return Err(MmnError::Shape {
            message: "add_sinusoidal_position_encoding expects [seq_len, d_model]".into(),
        });
    }
    let pe = sinusoidal_position_encoding(x.shape[0], x.shape[1]);
    x.add(&pe)
}

/// Default θ for rotary position embedding (LLaMA-style).
pub const DEFAULT_ROPE_THETA: f32 = 10_000.0;

fn rope_cos_sin(seq_len: usize, head_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = vec![0.0f32; seq_len * half];
    let mut sin = vec![0.0f32; seq_len * half];
    for pos in 0..seq_len {
        for i in 0..half {
            let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            cos[pos * half + i] = angle.cos();
            sin[pos * half + i] = angle.sin();
        }
    }
    (cos, sin)
}

/// Apply RoPE to Q and K tensors `[seq_len, d_model]` (rotates within each head).
pub fn apply_rope(q: &Tensor, k: &Tensor, n_heads: usize, theta: f32) -> Result<(Tensor, Tensor)> {
    if q.shape != k.shape || q.shape.len() != 2 {
        return Err(MmnError::Shape {
            message: "apply_rope expects q,k shape [seq_len, d_model]".into(),
        });
    }
    let seq = q.shape[0];
    let d_model = q.shape[1];
    if d_model % n_heads != 0 {
        return Err(MmnError::Shape {
            message: format!("d_model {d_model} not divisible by n_heads {n_heads}"),
        });
    }
    let head_dim = d_model / n_heads;
    if head_dim % 2 != 0 {
        return Err(MmnError::Shape {
            message: format!("head_dim {head_dim} must be even for RoPE"),
        });
    }
    let half = head_dim / 2;
    let (cos, sin) = rope_cos_sin(seq, head_dim, theta);
    let mut q_out = q.data.as_ref().clone();
    let mut k_out = k.data.as_ref().clone();
    for h in 0..n_heads {
        let base = h * head_dim;
        for s in 0..seq {
            for i in 0..half {
                let idx0 = base + 2 * i;
                let idx1 = base + 2 * i + 1;
                let c = cos[s * half + i];
                let sn = sin[s * half + i];
                for (src, dst) in [(&q.data, &mut q_out), (&k.data, &mut k_out)] {
                    let x0 = src[[s, idx0]];
                    let x1 = src[[s, idx1]];
                    dst[[s, idx0]] = x0 * c - x1 * sn;
                    dst[[s, idx1]] = x0 * sn + x1 * c;
                }
            }
        }
    }
    Ok((
        Tensor::from_array(q_out, q.requires_grad),
        Tensor::from_array(k_out, k.requires_grad),
    ))
}

/// Backprop through RoPE: `grad_*` w.r.t. rotated tensors → gradients w.r.t. pre-rotation inputs.
pub fn apply_rope_backward(
    grad_q: &ArrayD<f32>,
    grad_k: &ArrayD<f32>,
    n_heads: usize,
    theta: f32,
    seq_len: usize,
) -> Result<(ArrayD<f32>, ArrayD<f32>)> {
    let d_model = grad_q.shape()[1];
    let head_dim = d_model / n_heads;
    let half = head_dim / 2;
    let (cos, sin) = rope_cos_sin(seq_len, head_dim, theta);
    let mut gq = grad_q.clone();
    let mut gk = grad_k.clone();
    for h in 0..n_heads {
        let base = h * head_dim;
        for s in 0..seq_len {
            for i in 0..half {
                let idx0 = base + 2 * i;
                let idx1 = base + 2 * i + 1;
                let c = cos[s * half + i];
                let sn = sin[s * half + i];
                for grad in [&mut gq, &mut gk] {
                    let gy0 = grad[[s, idx0]];
                    let gy1 = grad[[s, idx1]];
                    grad[[s, idx0]] = gy0 * c + gy1 * sn;
                    grad[[s, idx1]] = -gy0 * sn + gy1 * c;
                }
            }
        }
    }
    Ok((gq, gk))
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self::new_rng(in_features, out_features, &mut rand::thread_rng())
    }

    pub fn new_rng(in_features: usize, out_features: usize, rng: &mut impl Rng) -> Self {
        let scale = (2.0 / (in_features + out_features) as f32).sqrt();
        let w: Vec<f32> = (0..in_features * out_features)
            .map(|_| rng.gen::<f32>() * scale - scale / 2.0)
            .collect();
        let weight = Tensor::from_array(
            ArrayD::from_shape_vec(IxDyn(&[out_features, in_features]), w).unwrap(),
            true,
        );
        let bias = Some(Tensor::from_array(
            ArrayD::zeros(IxDyn(&[out_features])),
            true,
        ));
        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = x.matmul(&transpose_tensor(&self.weight)?)?;
        if let Some(ref b) = self.bias {
            out.add(b)
        } else {
            Ok(out)
        }
    }
}

pub fn gelu(t: &Tensor) -> Tensor {
    let out = t.data.mapv(gelu_scalar);
    Tensor::from_array(out, t.requires_grad)
}

fn gelu_scalar(x: f32) -> f32 {
    let x = x as f64;
    (x * (0.5 * (1.0 + (x * 0.7978845608 * (1.0 + 0.044715 * x * x)).tanh()))) as f32
}

fn gelu_derivative_scalar(x: f32) -> f32 {
    let x = x as f64;
    let inner = 0.7978845608 * (1.0 + 0.044715 * x * x);
    let tanh_in = inner * x;
    let t = tanh_in.tanh();
    let sech2 = 1.0 - t * t;
    let inner_deriv = 0.7978845608 * (1.0 + 3.0 * 0.044715 * x * x);
    let phi = 0.5 * (1.0 + t);
    let phi_prime = 0.5 * sech2 * inner_deriv;
    (phi + x * phi_prime) as f32
}

/// Chain rule through GELU for `out = gelu(x_lin)`.
pub fn gelu_backward(x_lin: &Tensor, grad_out: &ArrayD<f32>) -> ArrayD<f32> {
    let mut grad_in = grad_out.clone();
    grad_in
        .iter_mut()
        .zip(x_lin.data.iter())
        .for_each(|(g, &x)| *g *= gelu_derivative_scalar(x));
    grad_in
}

pub struct LayerNorm {
    pub gamma: Tensor,
    pub beta: Tensor,
    pub normalized_shape: usize,
}

impl LayerNorm {
    pub fn new(size: usize) -> Self {
        Self {
            gamma: Tensor::ones(&[size], true),
            beta: Tensor::zeros(&[size], true),
            normalized_shape: size,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = x.data.as_ref().clone();
        if out.ndim() != 2 {
            return Ok(Tensor::from_array(out, x.requires_grad));
        }
        let dim = out.shape()[1];
        for i in 0..out.shape()[0] {
            let mut row = out.slice_mut(ndarray::s![i, ..]);
            let mean: f32 = row.sum() / dim as f32;
            row.mapv_inplace(|v| v - mean);
            let var: f32 = row.iter().map(|&v| v * v).sum::<f32>() / dim as f32;
            let std = (var + 1e-5f32).sqrt();
            row.mapv_inplace(|v| v / std);
            for j in 0..dim {
                row[j] = row[j] * self.gamma.data[j] + self.beta.data[j];
            }
        }
        Ok(Tensor::from_array(out, x.requires_grad))
    }
}

const LN_EPS: f32 = 1e-5;

fn layernorm_row_backward(
    x_row: &[f32],
    gamma: &[f32],
    grad_out_row: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let dim = x_row.len();
    let mean: f32 = x_row.iter().sum::<f32>() / dim as f32;
    let x_hat: Vec<f32> = x_row.iter().map(|&v| v - mean).collect();
    let var: f32 = x_hat.iter().map(|&v| v * v).sum::<f32>() / dim as f32;
    let std = (var + LN_EPS).sqrt();
    let x_norm: Vec<f32> = x_hat.iter().map(|&v| v / std).collect();

    let mut grad_gamma = vec![0.0f32; dim];
    let mut grad_beta = vec![0.0f32; dim];
    let mut grad_x_norm = vec![0.0f32; dim];
    for j in 0..dim {
        grad_gamma[j] = grad_out_row[j] * x_norm[j];
        grad_beta[j] = grad_out_row[j];
        grad_x_norm[j] = grad_out_row[j] * gamma[j];
    }

    let mut grad_x_hat = vec![0.0f32; dim];
    for j in 0..dim {
        grad_x_hat[j] = grad_x_norm[j] / std;
    }
    let grad_std: f32 = grad_x_norm
        .iter()
        .zip(x_hat.iter())
        .map(|(&gn, &xh)| -gn * xh / (std * std))
        .sum();
    let grad_var = grad_std * 0.5 / std;
    for j in 0..dim {
        grad_x_hat[j] += grad_var * 2.0 * x_hat[j] / dim as f32;
    }
    let grad_mean = -grad_x_hat.iter().sum::<f32>() / dim as f32;
    let grad_x: Vec<f32> = grad_x_hat
        .iter()
        .map(|&g| g + grad_mean)
        .collect();
    (grad_x, grad_gamma, grad_beta)
}

/// Backward through `LayerNorm::forward` for `[batch, dim]` input.
pub fn layernorm_backward(
    ln: &LayerNorm,
    x: &Tensor,
    grad_out: &ArrayD<f32>,
) -> Result<(ArrayD<f32>, ArrayD<f32>, ArrayD<f32>)> {
    if x.shape.len() != 2 {
        return Err(MmnError::Shape {
            message: "layernorm_backward expects [batch, dim]".into(),
        });
    }
    let rows = x.shape[0];
    let dim = x.shape[1];
    let gamma = ln.gamma.data.as_slice().ok_or_else(|| MmnError::Shape {
        message: "gamma must be contiguous".into(),
    })?;
    let mut grad_x = ArrayD::zeros(x.data.raw_dim());
    let mut grad_gamma = ArrayD::zeros(ln.gamma.data.raw_dim());
    let mut grad_beta = ArrayD::zeros(ln.beta.data.raw_dim());

    for i in 0..rows {
        let x_row: Vec<f32> = (0..dim).map(|j| x.data[[i, j]]).collect();
        let grad_row: Vec<f32> = (0..dim).map(|j| grad_out[[i, j]]).collect();
        let (gx, gg, gb) = layernorm_row_backward(&x_row, gamma, &grad_row);
        for j in 0..dim {
            grad_x[[i, j]] = gx[j];
            grad_gamma[[j]] += gg[j];
            grad_beta[[j]] += gb[j];
        }
    }
    Ok((grad_x, grad_gamma, grad_beta))
}

#[cfg(test)]
mod layernorm_tests {
    use super::*;
    use mmn_core::Tensor;
    use ndarray::arr2;

    #[test]
    fn gelu_backward_matches_finite_diff() {
        let x = Tensor::from_array(arr2(&[[0.5, -1.0, 2.0, 0.0]]).into_dyn(), false);
        let eps = 1e-3f32;
        let grad_out = arr2(&[[1.0, 1.0, 1.0, 1.0]]).into_dyn();
        let analytic = gelu_backward(&x, &grad_out);
        for j in 0..4 {
            let x0 = x.data[[0, j]];
            let y_plus = gelu_scalar(x0 + eps);
            let y_minus = gelu_scalar(x0 - eps);
            let numeric = ((y_plus - y_minus) / (2.0 * eps)) as f32;
            assert!(
                (analytic[[0, j]] - numeric).abs() < 0.05,
                "j={j} analytic={} numeric={}",
                analytic[[0, j]],
                numeric
            );
        }
    }

    #[test]
    fn layernorm_row_mean_near_zero() {
        let ln = LayerNorm::new(4);
        let x = Tensor::from_array(arr2(&[[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]).into_dyn(), false);
        let y = ln.forward(&x).unwrap();
        for i in 0..2 {
            let mean: f32 = y.data.slice(ndarray::s![i, ..]).sum() / 4.0;
            assert!(mean.abs() < 1e-4, "row {i} mean {mean}");
        }
    }

    #[test]
    fn layernorm_backward_matches_finite_diff() {
        let ln = LayerNorm::new(4);
        let x = Tensor::from_array(
            arr2(&[[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, -0.5]]).into_dyn(),
            false,
        );
        let eps = 1e-3f32;
        for i in 0..2 {
            for j in 0..4 {
                let mut grad_out = ArrayD::zeros(x.data.raw_dim());
                grad_out[[i, j]] = 1.0;
                let (grad_x, _, _) = layernorm_backward(&ln, &x, &grad_out).unwrap();
                for m in 0..4 {
                    let mut x_plus = x.data.as_ref().clone();
                    x_plus[[i, m]] += eps;
                    let y_plus = ln
                        .forward(&Tensor::from_array(x_plus.clone(), false))
                        .unwrap();
                    let mut x_minus = x.data.as_ref().clone();
                    x_minus[[i, m]] -= eps;
                    let y_minus = ln
                        .forward(&Tensor::from_array(x_minus, false))
                        .unwrap();
                    let numeric = (y_plus.data[[i, j]] - y_minus.data[[i, j]]) / (2.0 * eps);
                    assert!(
                        (grad_x[[i, m]] - numeric).abs() < 0.08,
                        "i={i} out_j={j} in_m={m} analytic={} numeric={}",
                        grad_x[[i, m]],
                        numeric
                    );
                }
            }
        }
    }

    #[test]
    fn layernorm_backward_gamma_beta_finite_diff() {
        let gamma_vals: Vec<f32> = (0..4).map(|j| 1.0 + 0.1 * j as f32).collect();
        let beta_vals: Vec<f32> = (0..4).map(|j| -0.05 * j as f32).collect();
        let ln = {
            let mut ln = LayerNorm::new(4);
            ln.gamma = Tensor::from_array(
                ArrayD::from_shape_vec(IxDyn(&[4]), gamma_vals.clone()).unwrap(),
                true,
            );
            ln.beta = Tensor::from_array(
                ArrayD::from_shape_vec(IxDyn(&[4]), beta_vals.clone()).unwrap(),
                true,
            );
            ln
        };
        let x = Tensor::from_array(
            arr2(&[[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, -0.5]]).into_dyn(),
            false,
        );
        let grad_out = arr2(&[[1.0, 0.5, -0.25, 0.75], [0.2, 0.3, -0.1, 0.4]]).into_dyn();
        let (_, grad_gamma, grad_beta) = layernorm_backward(&ln, &x, &grad_out).unwrap();
        let eps = 1e-3f32;
        let dot_loss = |y: &Tensor| -> f32 {
            y.data
                .iter()
                .zip(grad_out.iter())
                .map(|(yv, gv)| yv * gv)
                .sum()
        };
        for j in 0..4 {
            let mut g_plus = gamma_vals.clone();
            g_plus[j] += eps;
            let mut ln_plus = LayerNorm::new(4);
            ln_plus.gamma =
                Tensor::from_array(ArrayD::from_shape_vec(IxDyn(&[4]), g_plus).unwrap(), true);
            ln_plus.beta = ln.beta.clone();
            let y_plus = ln_plus.forward(&x).unwrap();

            let mut g_minus = gamma_vals.clone();
            g_minus[j] -= eps;
            let mut ln_minus = LayerNorm::new(4);
            ln_minus.gamma =
                Tensor::from_array(ArrayD::from_shape_vec(IxDyn(&[4]), g_minus).unwrap(), true);
            ln_minus.beta = ln.beta.clone();
            let y_minus = ln_minus.forward(&x).unwrap();

            let numeric_gamma = (dot_loss(&y_plus) - dot_loss(&y_minus)) / (2.0 * eps);
            assert!(
                (grad_gamma[[j]] - numeric_gamma).abs() < 0.1,
                "gamma j={j} analytic={} numeric={}",
                grad_gamma[[j]],
                numeric_gamma
            );

            let mut b_plus = beta_vals.clone();
            b_plus[j] += eps;
            let mut ln_plus = LayerNorm::new(4);
            ln_plus.gamma = ln.gamma.clone();
            ln_plus.beta =
                Tensor::from_array(ArrayD::from_shape_vec(IxDyn(&[4]), b_plus).unwrap(), true);
            let y_plus = ln_plus.forward(&x).unwrap();

            let mut b_minus = beta_vals.clone();
            b_minus[j] -= eps;
            let mut ln_minus = LayerNorm::new(4);
            ln_minus.gamma = ln.gamma.clone();
            ln_minus.beta =
                Tensor::from_array(ArrayD::from_shape_vec(IxDyn(&[4]), b_minus).unwrap(), true);
            let y_minus = ln_minus.forward(&x).unwrap();

            let numeric_beta = (dot_loss(&y_plus) - dot_loss(&y_minus)) / (2.0 * eps);
            assert!(
                (grad_beta[[j]] - numeric_beta).abs() < 0.1,
                "beta j={j} analytic={} numeric={}",
                grad_beta[[j]],
                numeric_beta
            );
        }
    }
}

#[cfg(test)]
mod position_encoding_tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn sinusoidal_pe_differs_by_position() {
        let x = Tensor::from_array(arr2(&[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]).into_dyn(), false);
        let y = add_sinusoidal_position_encoding(&x).unwrap();
        assert_ne!(y.data[[0, 0]], y.data[[1, 0]]);
        assert!((y.data[[0, 1]] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn sinusoidal_pe_position_zero_sin_zero() {
        let pe = sinusoidal_position_encoding(3, 4);
        assert!(pe.data[[0, 0]].abs() < 1e-6);
        assert!((pe.data[[0, 1]] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn rope_changes_qk_values() {
        let q = Tensor::from_array(
            ndarray::arr2(&[[1.0, 0.0, 0.5, 0.5], [0.0, 1.0, 0.5, 0.5]]).into_dyn(),
            true,
        );
        let k = q.clone();
        let (qr, _kr) = apply_rope(&q, &k, 2, DEFAULT_ROPE_THETA).unwrap();
        assert_ne!(qr.data[[1, 0]], q.data[[1, 0]]);
        assert_eq!(qr.data[[0, 0]], q.data[[0, 0]]);
    }

    #[test]
    fn rope_backward_matches_finite_diff() {
        let seq = 2usize;
        let d_model = 4usize;
        let n_heads = 2usize;
        let theta = DEFAULT_ROPE_THETA;
        let q = Tensor::from_array(
            ndarray::arr2(&[[0.3, -0.2, 0.7, 0.1], [0.5, 0.4, -0.3, 0.2]]).into_dyn(),
            true,
        );
        let k = q.clone();
        let eps = 1e-3f32;
        let (_qr, _) = apply_rope(&q, &k, n_heads, theta).unwrap();
        let grad_out_q = ndarray::arr2(&[[0.2, -0.1, 0.4, 0.3], [0.1, 0.2, -0.2, 0.5]]).into_dyn();
        let grad_out_k = grad_out_q.clone();
        let (gq, _) =
            apply_rope_backward(&grad_out_q, &grad_out_k, n_heads, theta, seq).unwrap();
        for j in 0..d_model {
            let mut q_plus = q.data.as_ref().clone();
            q_plus[[0, j]] += eps;
            let qp = Tensor::from_array(q_plus.clone(), true);
            let (qr_plus, _) = apply_rope(&qp, &k, n_heads, theta).unwrap();
            let mut q_minus = q.data.as_ref().clone();
            q_minus[[0, j]] -= eps;
            let qm = Tensor::from_array(q_minus, true);
            let (qr_minus, _) = apply_rope(&qm, &k, n_heads, theta).unwrap();
            let numeric = (grad_out_q[[0, 0]] * (qr_plus.data[[0, 0]] - qr_minus.data[[0, 0]])
                + grad_out_q[[0, 1]] * (qr_plus.data[[0, 1]] - qr_minus.data[[0, 1]])
                + grad_out_q[[0, 2]] * (qr_plus.data[[0, 2]] - qr_minus.data[[0, 2]])
                + grad_out_q[[0, 3]] * (qr_plus.data[[0, 3]] - qr_minus.data[[0, 3]]))
                / (2.0 * eps);
            assert!(
                (gq[[0, j]] - numeric).abs() < 0.05,
                "j={j} analytic={} numeric={}",
                gq[[0, j]],
                numeric
            );
        }
    }
}

pub struct MultiHeadAttention {
    pub d_model: usize,
    pub n_heads: usize,
    pub causal: bool,
    /// When set, apply rotary position embedding to Q/K after projection.
    pub rope_theta: Option<f32>,
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, n_heads: usize) -> Self {
        Self::new_rng(d_model, n_heads, &mut rand::thread_rng())
    }

    pub fn new_rng(d_model: usize, n_heads: usize, rng: &mut impl Rng) -> Self {
        Self::new_rng_causal(d_model, n_heads, true, rng)
    }

    pub fn new_rng_causal(
        d_model: usize,
        n_heads: usize,
        causal: bool,
        rng: &mut impl Rng,
    ) -> Self {
        Self::new_rng_causal_rope(d_model, n_heads, causal, None, rng)
    }

    pub fn new_rng_causal_rope(
        d_model: usize,
        n_heads: usize,
        causal: bool,
        rope_theta: Option<f32>,
        rng: &mut impl Rng,
    ) -> Self {
        assert_eq!(d_model % n_heads, 0);
        Self {
            d_model,
            n_heads,
            causal,
            rope_theta,
            q_proj: Linear::new_rng(d_model, d_model, rng),
            k_proj: Linear::new_rng(d_model, d_model, rng),
            v_proj: Linear::new_rng(d_model, d_model, rng),
            out_proj: Linear::new_rng(d_model, d_model, rng),
        }
    }

    fn project_qkv(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor, Option<Tensor>, Option<Tensor>)> {
        let q_lin = self.q_proj.forward(x)?;
        let k_lin = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;
        if let Some(theta) = self.rope_theta {
            let (q, k) = apply_rope(&q_lin, &k_lin, self.n_heads, theta)?;
            Ok((q, k, v, Some(q_lin), Some(k_lin)))
        } else {
            Ok((q_lin.clone(), k_lin.clone(), v, None, None))
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (q, k, v, _, _) = self.project_qkv(x)?;
        let (merged, _) =
            scaled_dot_product_attention_with_cache(&q, &k, &v, self.n_heads, self.causal)?;
        self.out_proj.forward(&merged)
    }

    pub fn forward_with_cache(
        &self,
        x: &Tensor,
    ) -> Result<(
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Option<Tensor>,
        Option<Tensor>,
        SdpAttentionCache,
    )> {
        let (q, k, v, q_lin, k_lin) = self.project_qkv(x)?;
        let (merged, sdp) =
            scaled_dot_product_attention_with_cache(&q, &k, &v, self.n_heads, self.causal)?;
        let out = self.out_proj.forward(&merged)?;
        Ok((out, q, k, v, merged, q_lin, k_lin, sdp))
    }
}

/// Softmax attention weights from the forward pass (`weights[head][query][key]`).
pub struct SdpAttentionCache {
    pub weights: Vec<Vec<Vec<f32>>>,
}

/// Self-attention over sequence rows of `q`, `k`, `v` each `[seq_len, d_model]`.
pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_heads: usize,
    causal: bool,
) -> Result<Tensor> {
    let (out, _) = scaled_dot_product_attention_with_cache(q, k, v, n_heads, causal)?;
    Ok(out)
}

pub fn scaled_dot_product_attention_with_cache(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_heads: usize,
    causal: bool,
) -> Result<(Tensor, SdpAttentionCache)> {
    if q.shape.len() != 2 || k.shape != q.shape || v.shape != q.shape {
        return Err(MmnError::Shape {
            message: "attention expects q,k,v with shape [seq_len, d_model]".into(),
        });
    }
    let seq = q.shape[0];
    let d_model = q.shape[1];
    if d_model % n_heads != 0 {
        return Err(MmnError::Shape {
            message: format!("d_model {d_model} not divisible by n_heads {n_heads}"),
        });
    }
    let head_dim = d_model / n_heads;
    let scale = (head_dim as f32).sqrt().recip();
    let mut out = vec![0.0f32; seq * d_model];
    let mut weights = vec![vec![vec![0.0f32; seq]; seq]; n_heads];

    for h in 0..n_heads {
        let base = h * head_dim;
        for s in 0..seq {
            let mut scores = vec![0.0f32; seq];
            for t in 0..seq {
                if causal && t > s {
                    scores[t] = f32::NEG_INFINITY;
                    continue;
                }
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q.data[[s, base + d]] * k.data[[t, base + d]];
                }
                scores[t] = dot * scale;
            }
            let max = scores
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut row_weights: Vec<f32> = scores.iter().map(|&x| (x - max).exp()).collect();
            let sum: f32 = row_weights.iter().sum();
            if sum > 0.0 {
                for w in &mut row_weights {
                    *w /= sum;
                }
            }
            weights[h][s] = row_weights.clone();
            for d in 0..head_dim {
                let mut ctx = 0.0f32;
                for t in 0..seq {
                    ctx += row_weights[t] * v.data[[t, base + d]];
                }
                out[s * d_model + base + d] = ctx;
            }
        }
    }

    Ok((
        Tensor::from_array(
            ArrayD::from_shape_vec(IxDyn(&[seq, d_model]), out).unwrap(),
            q.requires_grad || k.requires_grad || v.requires_grad,
        ),
        SdpAttentionCache { weights },
    ))
}

/// Backward through scaled dot-product attention (forward must have saved `cache`).
pub fn scaled_dot_product_attention_backward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cache: &SdpAttentionCache,
    grad_out: &ArrayD<f32>,
    n_heads: usize,
) -> Result<(ArrayD<f32>, ArrayD<f32>, ArrayD<f32>)> {
    let seq = q.shape[0];
    let d_model = q.shape[1];
    let head_dim = d_model / n_heads;
    let scale = (head_dim as f32).sqrt().recip();
    let mut grad_q = ArrayD::zeros(q.data.raw_dim());
    let mut grad_k = ArrayD::zeros(k.data.raw_dim());
    let mut grad_v = ArrayD::zeros(v.data.raw_dim());

    for h in 0..n_heads {
        let base = h * head_dim;
        for s in 0..seq {
            let w_row = &cache.weights[h][s];
            let mut grad_scores = vec![0.0f32; seq];
            for t in 0..seq {
                let mut grad_w = 0.0f32;
                for d in 0..head_dim {
                    grad_w += grad_out[[s, base + d]] * v.data[[t, base + d]];
                }
                grad_scores[t] = grad_w;
            }
            let dot: f32 = w_row
                .iter()
                .zip(grad_scores.iter())
                .map(|(w, g)| w * g)
                .sum();
            for t in 0..seq {
                let grad_s = w_row[t] * (grad_scores[t] - dot);
                for d in 0..head_dim {
                    grad_q[[s, base + d]] += grad_s * scale * k.data[[t, base + d]];
                    grad_k[[t, base + d]] += grad_s * scale * q.data[[s, base + d]];
                }
            }
            for t in 0..seq {
                for d in 0..head_dim {
                    grad_v[[t, base + d]] += w_row[t] * grad_out[[s, base + d]];
                }
            }
        }
    }
    Ok((grad_q, grad_k, grad_v))
}

pub struct TransformerBlock {
    pub attn: MultiHeadAttention,
    pub ln1: LayerNorm,
    pub ln2: LayerNorm,
    pub ffn: Linear,
    pub ffn2: Linear,
}

/// Activations cached during `TransformerBlock::forward_with_cache` for backward.
pub struct BlockForwardCache {
    pub x_in: Tensor,
    pub x2: Tensor,
    pub h_ln1: Tensor,
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
    pub merged: Tensor,
    pub h2: Tensor,
    pub f_lin: Tensor,
    pub f_post: Tensor,
    pub sdp: SdpAttentionCache,
    pub q_lin: Option<Tensor>,
    pub k_lin: Option<Tensor>,
}

impl TransformerBlock {
    pub fn new(d_model: usize, n_heads: usize, ffn_dim: usize) -> Self {
        Self::new_rng(d_model, n_heads, ffn_dim, &mut rand::thread_rng())
    }

    pub fn new_rng(d_model: usize, n_heads: usize, ffn_dim: usize, rng: &mut impl Rng) -> Self {
        Self::new_rng_rope(d_model, n_heads, ffn_dim, None, rng)
    }

    pub fn new_rng_rope(
        d_model: usize,
        n_heads: usize,
        ffn_dim: usize,
        rope_theta: Option<f32>,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            attn: MultiHeadAttention::new_rng_causal_rope(d_model, n_heads, true, rope_theta, rng),
            ln1: LayerNorm::new(d_model),
            ln2: LayerNorm::new(d_model),
            ffn: Linear::new_rng(d_model, ffn_dim, rng),
            ffn2: Linear::new_rng(ffn_dim, d_model, rng),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.forward_with_cache(x)?.0)
    }

    /// Returns `(block_output, activations for backward)`.
    pub fn forward_with_cache(&self, x: &Tensor) -> Result<(Tensor, BlockForwardCache)> {
        let x_in = x.clone();
        let h_ln1 = self.ln1.forward(x)?;
        let (a, q, k, v, merged, q_lin, k_lin, sdp) = self.attn.forward_with_cache(&h_ln1)?;
        let x2 = x.add(&a)?;
        let h2 = self.ln2.forward(&x2)?;
        let f_lin = self.ffn.forward(&h2)?;
        let f_post = gelu(&f_lin);
        let ffn_out = self.ffn2.forward(&f_post)?;
        let out = x2.add(&ffn_out)?;
        Ok((
            out,
            BlockForwardCache {
                x_in,
                x2,
                h_ln1,
                q,
                k,
                v,
                merged,
                h2,
                f_lin,
                f_post,
                sdp,
                q_lin,
                k_lin,
            },
        ))
    }

    /// Backward through FFN + attention + LayerNorm.
    /// Returns `(grad_input, [ffn2_w, ffn_w, out_w, q_w, k_w, v_w, ln2_γ, ln2_β, ln1_γ, ln1_β])`.
    pub fn backward_attn_ffn(
        &self,
        cache: &BlockForwardCache,
        grad_out: &ArrayD<f32>,
    ) -> Result<(ArrayD<f32>, [ArrayD<f32>; 10])> {
        use mmn_core::linear_backward;

        let (grad_ffn2_w, grad_f) = linear_backward(
            cache.f_post.data.as_ref(),
            self.ffn2.weight.data.as_ref(),
            grad_out,
        )?;
        let mut grad_x2 = grad_out.to_owned();
        let grad_f_lin = gelu_backward(&cache.f_lin, &grad_f);
        let (grad_ffn_w, grad_h2) = linear_backward(
            cache.h2.data.as_ref(),
            self.ffn.weight.data.as_ref(),
            &grad_f_lin,
        )?;
        let (grad_x2_ln, grad_ln2_gamma, grad_ln2_beta) =
            layernorm_backward(&self.ln2, &cache.x2, &grad_h2)?;
        grad_x2 = &grad_x2 + &grad_x2_ln;
        let grad_attn_out = grad_x2.clone();
        let (grad_out_w, grad_merged) = linear_backward(
            cache.merged.data.as_ref(),
            self.attn.out_proj.weight.data.as_ref(),
            &grad_attn_out,
        )?;
        let (grad_q, grad_k, grad_v) = scaled_dot_product_attention_backward(
            &cache.q,
            &cache.k,
            &cache.v,
            &cache.sdp,
            &grad_merged,
            self.attn.n_heads,
        )?;
        let (grad_q, grad_k) = if let Some(theta) = self.attn.rope_theta {
            apply_rope_backward(
                &grad_q,
                &grad_k,
                self.attn.n_heads,
                theta,
                cache.h_ln1.shape[0],
            )?
        } else {
            (grad_q, grad_k)
        };
        let (grad_q_w, grad_q_in) = linear_backward(
            cache.h_ln1.data.as_ref(),
            self.attn.q_proj.weight.data.as_ref(),
            &grad_q,
        )?;
        let (grad_k_w, grad_k_in) = linear_backward(
            cache.h_ln1.data.as_ref(),
            self.attn.k_proj.weight.data.as_ref(),
            &grad_k,
        )?;
        let (grad_v_w, grad_v_in) = linear_backward(
            cache.h_ln1.data.as_ref(),
            self.attn.v_proj.weight.data.as_ref(),
            &grad_v,
        )?;
        let mut grad_h_ln1 = grad_q_in;
        grad_h_ln1 = &grad_h_ln1 + &grad_k_in;
        grad_h_ln1 = &grad_h_ln1 + &grad_v_in;
        let (grad_x_ln1, grad_ln1_gamma, grad_ln1_beta) =
            layernorm_backward(&self.ln1, &cache.x_in, &grad_h_ln1)?;
        let mut grad_x = grad_x2;
        grad_x = &grad_x + &grad_x_ln1;

        Ok((
            grad_x,
            [
                grad_ffn2_w,
                grad_ffn_w,
                grad_out_w,
                grad_q_w,
                grad_k_w,
                grad_v_w,
                grad_ln2_gamma,
                grad_ln2_beta,
                grad_ln1_gamma,
                grad_ln1_beta,
            ],
        ))
    }

    /// Returns `(block_output, ln2_output, ffn_hidden_pre_gelu, ffn_hidden_post_gelu)`.
    pub fn forward_with_ffn_cache(
        &self,
        x: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let (out, cache) = self.forward_with_cache(x)?;
        Ok((out, cache.h2, cache.f_lin, cache.f_post))
    }
}

#[cfg(test)]
mod attention_tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn attn_forward_preserves_batch_and_d_model() {
        let mut rng = rng_from_seed(Some(11));
        let attn = MultiHeadAttention::new_rng(16, 4, &mut rng);
        let x = Tensor::from_array(
            arr2(&[
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
                [1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            ])
            .into_dyn(),
            false,
        );
        let y = attn.forward(&x).unwrap();
        assert_eq!(y.shape, x.shape);
        assert!(y.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn attn_forward_uses_k_and_v_projections() {
        let mut rng_a = rng_from_seed(Some(22));
        let a = MultiHeadAttention::new_rng(8, 2, &mut rng_a);
        let mut rng_b = rng_from_seed(Some(22));
        let mut b = MultiHeadAttention::new_rng(8, 2, &mut rng_b);
        b.k_proj.weight = Tensor::ones(&[8, 8], false);
        b.v_proj.weight = Tensor::zeros(&[8, 8], false);
        let x = Tensor::from_array(
            arr2(&[
                [1.0, 0.5, -0.2, 0.3, 0.0, 0.1, -0.4, 0.7],
                [0.2, -0.3, 0.4, 0.1, 0.6, -0.5, 0.3, 0.0],
            ])
            .into_dyn(),
            false,
        );
        let y_a = a.forward(&x).unwrap();
        let y_b = b.forward(&x).unwrap();
        assert_ne!(
            y_a.data.as_ref().iter().copied().collect::<Vec<_>>(),
            y_b.data.as_ref().iter().copied().collect::<Vec<_>>()
        );
    }

    #[test]
    fn scaled_dot_product_attention_weights_sum_to_one() {
        let mut rng = rng_from_seed(Some(55));
        let attn = MultiHeadAttention::new_rng(8, 2, &mut rng);
        let x = Tensor::from_array(
            arr2(&[[1.0, 0.0, 0.5, -0.5, 0.2, 0.3, -0.1, 0.4]]).into_dyn(),
            false,
        );
        let q = attn.q_proj.forward(&x).unwrap();
        let k = attn.k_proj.forward(&x).unwrap();
        let v = attn.v_proj.forward(&x).unwrap();
        let y = scaled_dot_product_attention(&q, &k, &v, 2, true).unwrap();
        assert_eq!(y.shape, vec![1, 8]);
        // seq=1 → softmax weight is 1 → merged head slice matches V before out_proj
        for d in 0..8 {
            assert!((y.data[[0, d]] - v.data[[0, d]]).abs() < 1e-5);
        }
    }

    #[test]
    fn scaled_dot_product_uniform_queries_blend_values() {
        let q = Tensor::from_array(arr2(&[[1.0, 0.0], [1.0, 0.0]]).into_dyn(), false);
        let k = q.clone();
        let v = Tensor::from_array(arr2(&[[4.0, 0.0], [0.0, 8.0]]).into_dyn(), false);
        let y = scaled_dot_product_attention(&q, &k, &v, 1, false).unwrap();
        assert!((y.data[[0, 0]] - 2.0).abs() < 1e-4);
        assert!((y.data[[1, 0]] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn scaled_dot_product_causal_masks_future_keys() {
        let q = Tensor::from_array(arr2(&[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]).into_dyn(), false);
        let k = q.clone();
        let v = Tensor::from_array(arr2(&[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]).into_dyn(), false);
        let y_causal = scaled_dot_product_attention(&q, &k, &v, 1, true).unwrap();
        let y_full = scaled_dot_product_attention(&q, &k, &v, 1, false).unwrap();
        assert!((y_causal.data[[0, 0]] - 1.0).abs() < 1e-4);
        assert!((y_full.data[[0, 0]] - 2.0).abs() < 1e-4);
        assert!((y_causal.data[[1, 0]] - 1.5).abs() < 1e-4);
        assert!((y_full.data[[1, 0]] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn scaled_dot_product_attention_backward_bidirectional_finite_diff() {
        let q = Tensor::from_array(arr2(&[[0.3, -0.2], [0.1, 0.4]]).into_dyn(), false);
        let k = Tensor::from_array(arr2(&[[0.2, 0.5], [-0.1, 0.3]]).into_dyn(), false);
        let v = Tensor::from_array(arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn(), false);
        let (y, cache) =
            scaled_dot_product_attention_with_cache(&q, &k, &v, 1, false).unwrap();
        let grad_out = arr2(&[[1.0, 0.5], [0.25, -0.5]]).into_dyn();
        let (grad_q, _, _) =
            scaled_dot_product_attention_backward(&q, &k, &v, &cache, &grad_out, 1).unwrap();
        let eps = 1e-3f32;
        for s in 0..2 {
            for d in 0..2 {
                let mut q_plus = q.data.as_ref().clone();
                q_plus[[s, d]] += eps;
                let y_plus = scaled_dot_product_attention(
                    &Tensor::from_array(q_plus.clone(), false),
                    &k,
                    &v,
                    1,
                    false,
                )
                .unwrap();
                let mut q_minus = q.data.as_ref().clone();
                q_minus[[s, d]] -= eps;
                let y_minus = scaled_dot_product_attention(
                    &Tensor::from_array(q_minus, false),
                    &k,
                    &v,
                    1,
                    false,
                )
                .unwrap();
                let numeric = (y_plus.data[[s, d]] - y_minus.data[[s, d]]) / (2.0 * eps);
                assert!(
                    (grad_q[[s, d]] - numeric).abs() < 0.15,
                    "s={s} d={d} analytic={} numeric={}",
                    grad_q[[s, d]],
                    numeric
                );
            }
        }
        let _ = y;
    }

    #[test]
    fn scaled_dot_product_causal_backward_matches_finite_diff() {
        let q = Tensor::from_array(arr2(&[[0.3, -0.2], [0.1, 0.4]]).into_dyn(), false);
        let k = Tensor::from_array(arr2(&[[0.2, 0.5], [-0.1, 0.3]]).into_dyn(), false);
        let v = Tensor::from_array(arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn(), false);
        let (y, cache) =
            scaled_dot_product_attention_with_cache(&q, &k, &v, 1, true).unwrap();
        let grad_out = arr2(&[[1.0, 0.5], [0.25, -0.5]]).into_dyn();
        let (grad_q, _, _) =
            scaled_dot_product_attention_backward(&q, &k, &v, &cache, &grad_out, 1).unwrap();
        let eps = 1e-3f32;
        for s in 0..2 {
            for d in 0..2 {
                let mut q_plus = q.data.as_ref().clone();
                q_plus[[s, d]] += eps;
                let y_plus = scaled_dot_product_attention(
                    &Tensor::from_array(q_plus.clone(), false),
                    &k,
                    &v,
                    1,
                    true,
                )
                .unwrap();
                let mut q_minus = q.data.as_ref().clone();
                q_minus[[s, d]] -= eps;
                let y_minus = scaled_dot_product_attention(
                    &Tensor::from_array(q_minus, false),
                    &k,
                    &v,
                    1,
                    true,
                )
                .unwrap();
                let numeric = (y_plus.data[[s, d]] - y_minus.data[[s, d]]) / (2.0 * eps);
                assert!(
                    (grad_q[[s, d]] - numeric).abs() < 0.15,
                    "s={s} d={d} analytic={} numeric={}",
                    grad_q[[s, d]],
                    numeric
                );
            }
        }
        let _ = y;
    }

    #[test]
    fn transformer_block_forward_changes_hidden() {
        let mut rng = rng_from_seed(Some(33));
        let block = TransformerBlock::new_rng(16, 4, 64, &mut rng);
        let x = Tensor::randn_rng(&mut rng, &[2, 16], false);
        let before: Vec<f32> = x.data.iter().copied().collect();
        let y = block.forward(&x).unwrap();
        let after: Vec<f32> = y.data.iter().copied().collect();
        assert_ne!(before, after);
        assert!(after.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn transformer_block_forward_with_cache_shapes() {
        let mut rng = rng_from_seed(Some(44));
        let block = TransformerBlock::new_rng(8, 2, 32, &mut rng);
        let x = Tensor::randn_rng(&mut rng, &[3, 8], false);
        let (out, h2, f_lin, f_post) = block.forward_with_ffn_cache(&x).unwrap();
        assert_eq!(out.shape, x.shape);
        assert_eq!(h2.shape, x.shape);
        assert_eq!(f_lin.shape[0], 3);
        assert_eq!(f_post.shape[0], 3);
    }

    #[test]
    fn transformer_block_uses_ffn_residual() {
        let mut rng = rng_from_seed(Some(66));
        let block = TransformerBlock::new_rng(8, 2, 16, &mut rng);
        let x = Tensor::randn_rng(&mut rng, &[2, 8], false);
        let (out, cache) = block.forward_with_cache(&x).unwrap();
        let ffn_only = block.ffn2.forward(&gelu(&block.ffn.forward(&cache.h2).unwrap())).unwrap();
        let expected = cache.x2.add(&ffn_only).unwrap();
        for idx in 0..out.data.len() {
            assert!(
                (out.data.as_slice().unwrap()[idx] - expected.data.as_slice().unwrap()[idx]).abs()
                    < 1e-5,
                "idx={idx}"
            );
        }
    }

    #[test]
    fn transformer_block_backward_input_matches_finite_diff() {
        let mut rng = rng_from_seed(Some(77));
        let block = TransformerBlock::new_rng(8, 2, 16, &mut rng);
        let x = Tensor::from_array(
            arr2(&[
                [0.2, 0.1, -0.3, 0.4, 0.0, 0.5, -0.2, 0.3],
                [0.1, -0.2, 0.3, 0.0, 0.4, -0.1, 0.2, 0.5],
            ])
            .into_dyn(),
            false,
        );
        let (out, cache) = block.forward_with_cache(&x).unwrap();
        let grad_out = arr2(&[
            [1.0, 0.5, -0.25, 0.75, 0.2, 0.3, -0.1, 0.4],
            [0.3, -0.2, 0.1, 0.6, -0.4, 0.2, 0.5, -0.3],
        ])
        .into_dyn();
        let (grad_x, _) = block.backward_attn_ffn(&cache, &grad_out).unwrap();
        let eps = 1e-3f32;
        for i in 0..2 {
            for j in 0..8 {
                let mut x_plus = x.data.as_ref().clone();
                x_plus[[i, j]] += eps;
                let out_plus = block
                    .forward(&Tensor::from_array(x_plus, false))
                    .unwrap();
                let mut x_minus = x.data.as_ref().clone();
                x_minus[[i, j]] -= eps;
                let out_minus = block
                    .forward(&Tensor::from_array(x_minus, false))
                    .unwrap();
                let loss_plus: f32 = out_plus
                    .data
                    .iter()
                    .zip(grad_out.iter())
                    .map(|(y, g)| y * g)
                    .sum();
                let loss_minus: f32 = out_minus
                    .data
                    .iter()
                    .zip(grad_out.iter())
                    .map(|(y, g)| y * g)
                    .sum();
                let numeric = (loss_plus - loss_minus) / (2.0 * eps);
                assert!(
                    (grad_x[[i, j]] - numeric).abs() < 0.25,
                    "i={i} j={j} analytic={} numeric={}",
                    grad_x[[i, j]],
                    numeric
                );
            }
        }
        let _ = out;
    }
}

pub struct Embedding {
    pub weight: Tensor,
    pub vocab_size: usize,
    pub d_model: usize,
}

impl Embedding {
    pub fn new(vocab_size: usize, d_model: usize) -> Self {
        Self::new_rng(vocab_size, d_model, &mut rand::thread_rng())
    }

    pub fn new_rng(vocab_size: usize, d_model: usize, rng: &mut impl Rng) -> Self {
        let w = Tensor::randn_rng(rng, &[vocab_size, d_model], true);
        Self {
            weight: w,
            vocab_size,
            d_model,
        }
    }

    pub fn forward(&self, token_ids: &[usize]) -> Result<Tensor> {
        let batch = token_ids.len();
        let mut rows = Vec::with_capacity(batch * self.d_model);
        for &id in token_ids {
            let row = id.min(self.vocab_size.saturating_sub(1));
            for j in 0..self.d_model {
                rows.push(self.weight.data[[row, j]]);
            }
        }
        Ok(Tensor::from_array(
            ArrayD::from_shape_vec(IxDyn(&[batch, self.d_model]), rows).unwrap(),
            true,
        ))
    }
}

pub struct Conv2d {
    pub weight: Tensor,
    pub in_ch: usize,
    pub out_ch: usize,
    pub kernel: usize,
}

impl Conv2d {
    pub fn new(in_ch: usize, out_ch: usize, kernel: usize) -> Self {
        let _n = in_ch * out_ch * kernel * kernel;
        let w = Tensor::randn(&[out_ch, in_ch, kernel, kernel], true);
        Self {
            weight: w,
            in_ch,
            out_ch,
            kernel,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let shape = &x.shape;
        if shape.len() != 4 {
            return Err(MmnError::Shape {
                message: format!("Conv2d expects NCHW input, got shape {shape:?}"),
            });
        }
        let (batch, in_ch, height, width) = (shape[0], shape[1], shape[2], shape[3]);
        if in_ch != self.in_ch {
            return Err(MmnError::Shape {
                message: format!(
                    "Conv2d input channels {in_ch} do not match layer in_ch {}",
                    self.in_ch
                ),
            });
        }
        let pad = self.kernel / 2;
        let out_h = height + 2 * pad + 1 - self.kernel;
        let out_w = width + 2 * pad + 1 - self.kernel;
        let mut out = ndarray::Array4::<f32>::zeros((batch, self.out_ch, out_h, out_w));
        let w = self.weight.data.view();
        let x_view = x.data.view();
        for n in 0..batch {
            for oc in 0..self.out_ch {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0f32;
                        for ic in 0..self.in_ch {
                            for kh in 0..self.kernel {
                                for kw in 0..self.kernel {
                                    let ih = oh + kh;
                                    let iw = ow + kw;
                                    if ih >= pad
                                        && iw >= pad
                                        && ih < pad + height
                                        && iw < pad + width
                                    {
                                        let xi = ih - pad;
                                        let xj = iw - pad;
                                        sum += x_view[[n, ic, xi, xj]]
                                            * w[[oc, ic, kh, kw]];
                                    }
                                }
                            }
                        }
                        out[[n, oc, oh, ow]] = sum;
                    }
                }
            }
        }
        Ok(Tensor::from_array(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[batch, self.out_ch, out_h, out_w]),
                out.iter().copied().collect(),
            )
            .unwrap(),
            true,
        ))
    }
}

pub struct VaeEncoder {
    pub conv1: Conv2d,
    pub conv2: Conv2d,
}

impl VaeEncoder {
    pub fn new() -> Self {
        Self {
            conv1: Conv2d::new(3, 64, 3),
            conv2: Conv2d::new(64, 4, 3),
        }
    }

    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        self.conv2.forward(&self.conv1.forward(x)?)
    }
}

pub struct UNet2D {
    pub down: Conv2d,
    pub mid: Conv2d,
    pub up: Conv2d,
}

impl UNet2D {
    pub fn new() -> Self {
        Self {
            down: Conv2d::new(4, 64, 3),
            mid: Conv2d::new(64, 64, 3),
            up: Conv2d::new(64, 4, 3),
        }
    }

    pub fn forward(&self, x: &Tensor, _t_emb: &Tensor) -> Result<Tensor> {
        let h = self.down.forward(x)?;
        let h = self.mid.forward(&h)?;
        self.up.forward(&h)
    }
}

#[cfg(test)]
mod conv2d_tests {
    use super::*;
    use ndarray::Array4;

    #[test]
    fn conv2d_same_padding_preserves_spatial_dims() {
        let conv = Conv2d::new(1, 2, 3);
        let mut w = conv
            .weight
            .data
            .as_ref()
            .clone()
            .into_dimensionality::<ndarray::Ix4>()
            .unwrap();
        w.fill(0.0);
        w[[0, 0, 1, 1]] = 1.0;
        w[[1, 0, 1, 1]] = 2.0;
        let mut conv = conv;
        conv.weight = Tensor::from_array(w.into_dyn(), true);
        let x = Tensor::from_array(
            Array4::from_elem((1, 1, 4, 4), 3.0f32).into_dyn(),
            false,
        );
        let y = conv.forward(&x).unwrap();
        assert_eq!(y.shape, vec![1, 2, 4, 4]);
        assert!((y.data[[0, 0, 2, 2]] - 3.0).abs() < 1e-5);
        assert!((y.data[[0, 1, 2, 2]] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn vae_encoder_preserves_8x8_latent_shape() {
        let vae = VaeEncoder::new();
        let x = Tensor::randn(&[1, 3, 8, 8], false);
        let latent = vae.encode(&x).unwrap();
        assert_eq!(latent.shape, vec![1, 4, 8, 8]);
        assert!(latent.data.iter().all(|v| v.is_finite()));
    }
}
