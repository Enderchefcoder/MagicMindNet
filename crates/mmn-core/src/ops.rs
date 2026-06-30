use crate::tensor::Tensor;
use crate::Result;
use ndarray::ArrayD;

/// Gradient of mean cross-entropy w.r.t. logits (softmax + CE).
pub fn cross_entropy_grad(logits: &Tensor, targets: &[usize]) -> Result<ArrayD<f32>> {
    let batch = logits.shape[0];
    let classes = logits.shape.get(1).copied().unwrap_or(1);
    let sm = logits.softmax(1)?;
    let mut grad = sm.data.as_ref().clone();
    let scale = 1.0 / batch.max(1) as f32;
    for (i, &t) in targets.iter().enumerate().take(batch) {
        if t < classes {
            grad[[i, t]] -= 1.0;
        }
    }
    grad.mapv_inplace(|x| x * scale);
    Ok(grad)
}

/// Backward for `out = h @ W^T` with W shape [out_features, in_features].
pub fn linear_backward(
    h: &ArrayD<f32>,
    w: &ArrayD<f32>,
    grad_out: &ArrayD<f32>,
) -> Result<(ArrayD<f32>, ArrayD<f32>)> {
    let h2 = h
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| crate::MmnError::Shape {
            message: e.to_string(),
        })?;
    let w2 = w
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| crate::MmnError::Shape {
            message: e.to_string(),
        })?;
    let g2 = grad_out
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| crate::MmnError::Shape {
            message: e.to_string(),
        })?;
    let grad_w = g2.t().dot(&h2);
    let grad_h = g2.dot(&w2);
    Ok((grad_w.into_dyn(), grad_h.into_dyn()))
}

/// Gradient w.r.t. embedding table from grad on gathered rows `[batch, d_model]`.
pub fn embedding_backward(
    token_ids: &[usize],
    grad_h: &ArrayD<f32>,
    vocab_size: usize,
    d_model: usize,
) -> ArrayD<f32> {
    let mut grad_w = ArrayD::<f32>::zeros(ndarray::IxDyn(&[vocab_size, d_model]));
    for (i, &tid) in token_ids.iter().enumerate() {
        let row = tid.min(vocab_size.saturating_sub(1));
        for j in 0..d_model {
            grad_w[[row, j]] += grad_h[[i, j]];
        }
    }
    grad_w
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use ndarray::arr2;

    #[test]
    fn embedding_backward_accumulates_rows() {
        let grad_h = arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn();
        let g = embedding_backward(&[0, 2], &grad_h, 4, 2);
        assert!((g[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((g[[2, 1]] - 4.0).abs() < 1e-6);
        assert!(g[[1, 0]].abs() < 1e-6);
    }

    #[test]
    fn ce_grad_pushes_down_target_class() {
        let logits = Tensor::from_array(arr2(&[[1.0, 2.0, 0.5]]).into_dyn(), false);
        let g = cross_entropy_grad(&logits, &[1]).unwrap();
        assert!(g[[0, 1]] < 0.0, "gradient at target class should be negative");
    }

    #[test]
    fn linear_backward_grad_shapes_match() {
        let h = arr2(&[[1.0, 2.0]]).into_dyn();
        let w = arr2(&[[0.5, 0.5], [0.1, 0.2]]).into_dyn();
        let grad_out = arr2(&[[1.0, 0.0]]).into_dyn();
        let (grad_w, grad_h) = linear_backward(&h, &w, &grad_out).unwrap();
        assert_eq!(grad_w.shape(), w.shape());
        assert_eq!(grad_h.shape(), h.shape());
    }

    #[test]
    fn ce_grad_averages_over_batch() {
        let logits = Tensor::from_array(arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn(), false);
        let g = cross_entropy_grad(&logits, &[0, 1]).unwrap();
        let sum: f32 = g.iter().sum();
        assert!(sum.abs() < 1e-5, "batch CE grad rows should sum to ~0, got {sum}");
    }
}

