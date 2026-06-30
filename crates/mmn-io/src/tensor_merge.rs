use mmn_core::Tensor;

/// Element-wise mean of two tensors (used by chatbot and classifier merge).
pub(crate) fn average_tensors(a: &Tensor, b: &Tensor) -> Tensor {
    Tensor::from_array(((&*a.data) + (&*b.data)) / 2.0, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn average_tensors_is_elementwise_mean() {
        let a = Tensor::from_array(arr2(&[[1.0, 3.0], [5.0, 7.0]]).into_dyn(), false);
        let b = Tensor::from_array(arr2(&[[3.0, 5.0], [7.0, 9.0]]).into_dyn(), false);
        let m = average_tensors(&a, &b);
        assert!((m.data[[0, 0]] - 2.0).abs() < 1e-6);
        assert!((m.data[[1, 1]] - 8.0).abs() < 1e-6);
    }
}
