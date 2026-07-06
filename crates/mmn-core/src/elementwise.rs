//! NumPy/PyTorch-style tensor operations: elementwise math, reductions,
//! shape manipulation, and comparisons.
//!
//! Unary math ops register autograd nodes in the same style as
//! [`Tensor::relu`]; shape/reduction helpers return detached tensors.

use crate::autograd::{grad_enabled, register_node};
use crate::error::{MmnError, Result};
use crate::tensor::Tensor;
use ndarray::{ArrayD, Axis, IxDyn};
use std::sync::Arc;

impl Tensor {
    fn unary_with_grad<F, G>(&self, op: F, grad_fn: G) -> Tensor
    where
        F: Fn(f32) -> f32,
        G: Fn(f32) -> f32 + Send + Sync + 'static,
    {
        let out = self.data.mapv(&op);
        let req = self.requires_grad;
        let parent = self.node_id.unwrap_or(0);
        let inp = self.data.clone();
        let node_id = if req && grad_enabled() {
            Some(register_node(
                vec![parent],
                Box::new(move |g| {
                    let local = inp.mapv(&grad_fn);
                    vec![g * &local]
                }),
            ))
        } else {
            None
        };
        Tensor {
            data: Arc::new(out),
            shape: self.shape.clone(),
            device: self.device,
            dtype: self.dtype,
            requires_grad: req,
            node_id,
            grad: None,
        }
    }

    fn with_data(&self, data: ArrayD<f32>) -> Tensor {
        let shape = data.shape().to_vec();
        Tensor {
            data: Arc::new(data),
            shape,
            device: self.device,
            dtype: self.dtype,
            requires_grad: self.requires_grad,
            node_id: None,
            grad: None,
        }
    }

    /// Elementwise subtraction with broadcasting.
    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        self.add(&other.neg())
    }

    /// Elementwise division with broadcasting; divides by zero produce ±inf.
    pub fn div(&self, other: &Tensor) -> Result<Tensor> {
        let recip = other.with_data(other.data.mapv(|x| 1.0 / x));
        self.mul(&recip)
    }

    /// Elementwise negation.
    pub fn neg(&self) -> Tensor {
        self.unary_with_grad(|x| -x, |_| -1.0)
    }

    /// Elementwise `e^x`.
    pub fn exp(&self) -> Tensor {
        self.unary_with_grad(|x| x.exp(), |x| x.exp())
    }

    /// Elementwise natural log; non-positive inputs produce -inf/NaN like NumPy.
    pub fn log(&self) -> Tensor {
        self.unary_with_grad(|x| x.ln(), |x| 1.0 / x)
    }

    /// Elementwise square root.
    pub fn sqrt(&self) -> Tensor {
        self.unary_with_grad(|x| x.sqrt(), |x| 0.5 / x.sqrt())
    }

    /// Elementwise absolute value.
    pub fn abs(&self) -> Tensor {
        self.unary_with_grad(|x| x.abs(), |x| x.signum())
    }

    /// Elementwise power with a scalar exponent.
    pub fn pow_scalar(&self, exponent: f32) -> Tensor {
        self.unary_with_grad(
            move |x| x.powf(exponent),
            move |x| exponent * x.powf(exponent - 1.0),
        )
    }

    /// Elementwise sigmoid `1 / (1 + e^-x)`.
    pub fn sigmoid(&self) -> Tensor {
        self.unary_with_grad(
            |x| 1.0 / (1.0 + (-x).exp()),
            |x| {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            },
        )
    }

    /// Elementwise hyperbolic tangent.
    pub fn tanh(&self) -> Tensor {
        self.unary_with_grad(|x| x.tanh(), |x| 1.0 - x.tanh() * x.tanh())
    }

    /// Clamp every element into `[min, max]`.
    pub fn clamp(&self, min: f32, max: f32) -> Tensor {
        self.unary_with_grad(
            move |x| x.clamp(min, max),
            move |x| if x >= min && x <= max { 1.0 } else { 0.0 },
        )
    }

    /// Add a scalar to every element.
    pub fn add_scalar(&self, value: f32) -> Tensor {
        self.unary_with_grad(move |x| x + value, |_| 1.0)
    }

    /// Multiply every element by a scalar.
    pub fn mul_scalar(&self, value: f32) -> Tensor {
        self.unary_with_grad(move |x| x * value, move |_| value)
    }

    /// Reshape into `shape` (element count must match).
    pub fn reshape(&self, shape: &[usize]) -> Result<Tensor> {
        let numel: usize = shape.iter().product();
        if numel != self.numel() {
            return Err(MmnError::Shape {
                message: format!(
                    "cannot reshape {:?} ({} elements) into {:?} ({} elements)",
                    self.shape,
                    self.numel(),
                    shape,
                    numel
                ),
            });
        }
        let standard = self.data.as_standard_layout().into_owned();
        let data = standard
            .into_shape_with_order(IxDyn(shape))
            .map_err(|e| MmnError::Shape {
                message: e.to_string(),
            })?;
        Ok(self.with_data(data))
    }

    /// Reverse all axes (matrix transpose for rank 2, like `numpy.T`).
    pub fn transpose(&self) -> Tensor {
        let axes: Vec<usize> = (0..self.shape.len()).rev().collect();
        let data = self
            .data
            .as_ref()
            .clone()
            .permuted_axes(IxDyn(&axes))
            .as_standard_layout()
            .into_owned();
        self.with_data(data)
    }

    /// Sum along one axis (removing it).
    pub fn sum_axis(&self, axis: usize) -> Result<Tensor> {
        if axis >= self.shape.len() {
            return Err(MmnError::Shape {
                message: format!("sum_axis {axis} out of range for shape {:?}", self.shape),
            });
        }
        Ok(self.with_data(self.data.sum_axis(Axis(axis))))
    }

    /// Mean along one axis (removing it).
    pub fn mean_axis(&self, axis: usize) -> Result<Tensor> {
        if axis >= self.shape.len() {
            return Err(MmnError::Shape {
                message: format!("mean_axis {axis} out of range for shape {:?}", self.shape),
            });
        }
        let mean = self
            .data
            .mean_axis(Axis(axis))
            .ok_or_else(|| MmnError::Shape {
                message: format!("mean_axis {axis} on zero-length axis"),
            })?;
        Ok(self.with_data(mean))
    }

    /// Largest element (NaN-free tensors).
    pub fn max_value(&self) -> f32 {
        self.data.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }

    /// Smallest element (NaN-free tensors).
    pub fn min_value(&self) -> f32 {
        self.data.iter().copied().fold(f32::INFINITY, f32::min)
    }

    /// Flat index of the largest element.
    pub fn argmax(&self) -> usize {
        let mut best = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in self.data.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best = i;
            }
        }
        best
    }

    /// Flat index of the smallest element.
    pub fn argmin(&self) -> usize {
        let mut best = 0usize;
        let mut best_val = f32::INFINITY;
        for (i, &v) in self.data.iter().enumerate() {
            if v < best_val {
                best_val = v;
                best = i;
            }
        }
        best
    }

    /// Per-row argmax for a rank-2 tensor (like `numpy.argmax(a, axis=1)`).
    pub fn argmax_rows(&self) -> Result<Vec<usize>> {
        if self.shape.len() != 2 {
            return Err(MmnError::Shape {
                message: format!("argmax_rows needs rank 2, got shape {:?}", self.shape),
            });
        }
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut out = Vec::with_capacity(rows);
        for i in 0..rows {
            let mut best = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for j in 0..cols {
                let v = self.data[[i, j]];
                if v > best_val {
                    best_val = v;
                    best = j;
                }
            }
            out.push(best);
        }
        Ok(out)
    }

    /// Flatten to rank 1 in C order.
    pub fn flatten(&self) -> Tensor {
        let standard = self.data.as_standard_layout().into_owned();
        let numel = self.numel();
        let data = standard
            .into_shape_with_order(IxDyn(&[numel]))
            .expect("flatten always fits");
        self.with_data(data)
    }

    /// Build a tensor from a flat `f32` vector and shape.
    pub fn from_vec(data: Vec<f32>, shape: &[usize], requires_grad: bool) -> Result<Tensor> {
        let arr = ArrayD::from_shape_vec(IxDyn(shape), data).map_err(|e| MmnError::Shape {
            message: e.to_string(),
        })?;
        Ok(Tensor::from_array(arr, requires_grad))
    }

    /// Copy out the elements as a flat `Vec<f32>` in C order.
    pub fn to_vec(&self) -> Vec<f32> {
        self.data
            .as_standard_layout()
            .into_owned()
            .into_iter()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    fn t2(data: [[f32; 2]; 2]) -> Tensor {
        Tensor::from_array(arr2(&data).into_dyn(), false)
    }

    #[test]
    fn sub_and_div_broadcast() {
        let a = t2([[4.0, 9.0], [1.0, 0.0]]);
        let b = t2([[2.0, 3.0], [1.0, 1.0]]);
        let d = a.sub(&b).unwrap();
        assert_eq!(d.data[[0, 0]], 2.0);
        assert_eq!(d.data[[1, 1]], -1.0);
        let q = a.div(&b).unwrap();
        assert_eq!(q.data[[0, 1]], 3.0);
    }

    #[test]
    fn unary_math_matches_std() {
        let a = t2([[1.0, 4.0], [-2.0, 0.25]]);
        assert!((a.exp().data[[0, 0]] - 1.0f32.exp()).abs() < 1e-6);
        assert!((a.sqrt().data[[0, 1]] - 2.0).abs() < 1e-6);
        assert_eq!(a.abs().data[[1, 0]], 2.0);
        assert_eq!(a.neg().data[[1, 0]], 2.0);
        assert!((a.log().data[[1, 1]] - 0.25f32.ln()).abs() < 1e-6);
        assert!((a.pow_scalar(2.0).data[[0, 1]] - 16.0).abs() < 1e-5);
        assert!((a.sigmoid().data[[0, 0]] - 1.0 / (1.0 + (-1.0f32).exp())).abs() < 1e-6);
        assert!((a.tanh().data[[0, 0]] - 1.0f32.tanh()).abs() < 1e-6);
    }

    #[test]
    fn clamp_and_scalar_ops() {
        let a = t2([[-5.0, 0.5], [2.0, 10.0]]);
        let c = a.clamp(0.0, 1.0);
        assert_eq!(c.data[[0, 0]], 0.0);
        assert_eq!(c.data[[0, 1]], 0.5);
        assert_eq!(c.data[[1, 1]], 1.0);
        assert_eq!(a.add_scalar(1.0).data[[1, 0]], 3.0);
        assert_eq!(a.mul_scalar(-2.0).data[[0, 1]], -1.0);
    }

    #[test]
    fn reshape_transpose_flatten() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false).unwrap();
        let r = a.reshape(&[3, 2]).unwrap();
        assert_eq!(r.shape, vec![3, 2]);
        assert_eq!(r.data[[1, 0]], 3.0);
        assert!(a.reshape(&[4, 2]).is_err());
        let t = a.transpose();
        assert_eq!(t.shape, vec![3, 2]);
        assert_eq!(t.data[[0, 1]], 4.0);
        assert_eq!(t.data[[2, 0]], 3.0);
        let f = a.flatten();
        assert_eq!(f.shape, vec![6]);
        assert_eq!(f.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn reductions_and_argmax() {
        let a = Tensor::from_vec(vec![1.0, 5.0, 2.0, 4.0, 3.0, 9.0], &[2, 3], false).unwrap();
        let s = a.sum_axis(0).unwrap();
        assert_eq!(s.shape, vec![3]);
        assert_eq!(s.data[[1]], 8.0);
        let m = a.mean_axis(1).unwrap();
        assert!((m.data[[0]] - 8.0 / 3.0).abs() < 1e-6);
        assert!(a.sum_axis(2).is_err());
        assert_eq!(a.max_value(), 9.0);
        assert_eq!(a.min_value(), 1.0);
        assert_eq!(a.argmax(), 5);
        assert_eq!(a.argmin(), 0);
        assert_eq!(a.argmax_rows().unwrap(), vec![1, 2]);
        assert!(Tensor::from_vec(vec![1.0], &[1], false)
            .unwrap()
            .argmax_rows()
            .is_err());
    }

    #[test]
    fn exp_backward_flows_local_grad() {
        crate::autograd::enable_grad(true);
        let a = Tensor::from_array(arr2(&[[1.0f32, 2.0]]).into_dyn(), true);
        let aid = a.node_id.unwrap();
        let e = a.exp();
        assert!(e.requires_grad);
        assert!(e.node_id.is_some());
        let grads = crate::autograd::backward(&e, None);
        let ga = grads.iter().find(|(id, _)| *id == aid).map(|(_, g)| g[[0, 0]]);
        assert!((ga.unwrap() - 1.0f32.exp()).abs() < 1e-5);
        crate::autograd::enable_grad(false);
        crate::autograd::clear_tape();
    }

    #[test]
    fn from_vec_shape_mismatch_errors() {
        assert!(Tensor::from_vec(vec![1.0, 2.0], &[3], false).is_err());
    }
}
