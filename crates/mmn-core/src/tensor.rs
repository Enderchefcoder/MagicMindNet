use crate::autograd::{grad_enabled, register_node};
use crate::device::Device;
use crate::dtype::DType;
use crate::error::{MmnError, Result};
use ndarray::{ArrayD, Axis, IxDyn};
use std::sync::Arc;

#[derive(Clone)]
pub struct Tensor {
    pub data: Arc<ArrayD<f32>>,
    pub shape: Vec<usize>,
    pub device: Device,
    pub dtype: DType,
    pub requires_grad: bool,
    pub node_id: Option<u64>,
    pub grad: Option<Arc<ArrayD<f32>>>,
}

impl Tensor {
    pub fn from_array(data: ArrayD<f32>, requires_grad: bool) -> Self {
        let shape = data.shape().to_vec();
        let node_id = if requires_grad && grad_enabled() {
            Some(register_node(vec![], Box::new(|g| vec![g.clone()])))
        } else {
            None
        };
        Self {
            data: Arc::new(data),
            shape,
            device: Device::Cpu,
            dtype: DType::F32,
            requires_grad,
            node_id,
            grad: None,
        }
    }

    pub fn zeros(shape: &[usize], requires_grad: bool) -> Self {
        Self::from_array(ArrayD::zeros(IxDyn(shape)), requires_grad)
    }

    pub fn ones(shape: &[usize], requires_grad: bool) -> Self {
        Self::from_array(ArrayD::ones(IxDyn(shape)), requires_grad)
    }

    pub fn randn(shape: &[usize], requires_grad: bool) -> Self {
        Self::randn_rng(&mut rand::thread_rng(), shape, requires_grad)
    }

    pub fn randn_rng(rng: &mut impl rand::Rng, shape: &[usize], requires_grad: bool) -> Self {
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n)
            .map(|_| rng.gen::<f32>() * 2.0 - 1.0)
            .collect();
        let arr = ArrayD::from_shape_vec(IxDyn(shape), v).unwrap();
        Self::from_array(arr, requires_grad)
    }

    pub fn to_device(&self, device: Device) -> Result<Self> {
        if device == Device::Cuda {
            Device::require_cuda_available(true)?;
        }
        Ok(Self {
            data: self.data.clone(),
            shape: self.shape.clone(),
            device,
            dtype: self.dtype,
            requires_grad: self.requires_grad,
            node_id: self.node_id,
            grad: self.grad.clone(),
        })
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        self.broadcast_bin(other, |a, b| a + b)
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        self.broadcast_bin(other, |a, b| a * b)
    }

    fn broadcast_bin<F>(
        &self,
        other: &Tensor,
        op: F,
    ) -> Result<Tensor>
    where
        F: Fn(&ArrayD<f32>, &ArrayD<f32>) -> ArrayD<f32>,
    {
        let out_shape = broadcast_shape(&self.shape, &other.shape)?;
        let lhs = self
            .data
            .broadcast(IxDyn(&out_shape))
            .ok_or_else(|| MmnError::Shape {
                message: "broadcast failed".into(),
            })?;
        let rhs = other
            .data
            .broadcast(IxDyn(&out_shape))
            .ok_or_else(|| MmnError::Shape {
                message: "broadcast failed".into(),
            })?;
        let out = op(&lhs.to_owned(), &rhs.to_owned());

        let parents = vec![self.node_id.unwrap_or(0), other.node_id.unwrap_or(0)];
        let req = self.requires_grad || other.requires_grad;
        let node_id = if req && grad_enabled() {
            Some(register_node(
                parents.clone(),
                Box::new(move |g| vec![g.clone(), g.clone()]),
            ))
        } else {
            None
        };

        Ok(Tensor {
            data: Arc::new(out),
            shape: out_shape,
            device: self.device,
            dtype: self.dtype,
            requires_grad: req,
            node_id,
            grad: None,
        })
    }

    pub fn broadcast_to(&self, shape: &[usize]) -> Result<ArrayD<f32>> {
        self.data
            .broadcast(IxDyn(shape))
            .map(|v| v.to_owned())
            .ok_or_else(|| MmnError::Shape {
                message: format!("cannot broadcast {:?} to {:?}", self.shape, shape),
            })
    }

    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        let a = self.data.view();
        let b = other.data.view();
        if self.shape.len() < 2 || other.shape.len() < 2 {
            return Err(MmnError::Shape {
                message: "matmul requires rank >= 2".into(),
            });
        }
        let (m, k1) = (self.shape[self.shape.len() - 2], self.shape[self.shape.len() - 1]);
        let (k2, n) = (
            other.shape[other.shape.len() - 2],
            other.shape[other.shape.len() - 1],
        );
        if k1 != k2 {
            return Err(MmnError::Shape {
                message: format!("matmul shape mismatch: {} vs {}", k1, k2),
            });
        }
        let a2 = a
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| MmnError::Shape {
                message: e.to_string(),
            })?;
        let b2 = b
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| MmnError::Shape {
                message: e.to_string(),
            })?;
        let out = a2.dot(&b2);
        let mut out_shape = self.shape[..self.shape.len() - 2].to_vec();
        out_shape.push(m);
        out_shape.push(n);
        let out = out.into_dyn();

        let req = self.requires_grad || other.requires_grad;
        let parents = vec![self.node_id.unwrap_or(0), other.node_id.unwrap_or(0)];
        let node_id = if req && grad_enabled() {
            let sa = self.data.clone();
            let sb = other.data.clone();
            Some(register_node(
                parents,
                Box::new(move |g| {
                    let g2 = g
                        .view()
                        .into_dimensionality::<ndarray::Ix2>()
                        .unwrap()
                        .to_owned();
                    let sa2 = sa
                        .view()
                        .into_dimensionality::<ndarray::Ix2>()
                        .unwrap()
                        .to_owned();
                    let sb2 = sb
                        .view()
                        .into_dimensionality::<ndarray::Ix2>()
                        .unwrap()
                        .to_owned();
                    let da = g2.dot(&sb2.t());
                    let db = sa2.t().dot(&g2);
                    vec![da.into_dyn(), db.into_dyn()]
                }),
            ))
        } else {
            None
        };

        Ok(Tensor {
            data: Arc::new(out),
            shape: out_shape,
            device: self.device,
            dtype: self.dtype,
            requires_grad: req,
            node_id,
            grad: None,
        })
    }

    pub fn relu(&self) -> Tensor {
        let out = self.data.mapv(|x| x.max(0.0));
        let req = self.requires_grad;
        let parent = self.node_id.unwrap_or(0);
        let inp = self.data.clone();
        let node_id = if req && grad_enabled() {
            Some(register_node(
                vec![parent],
                Box::new(move |g| {
                    let mask = inp.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
                    vec![g * &mask]
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

    pub fn softmax(&self, axis: usize) -> Result<Tensor> {
        let mut out = self.data.as_ref().clone();
        // For [batch, classes], normalize each batch row (class logits), not columns.
        if out.ndim() == 2 && axis == 1 {
            let batch = out.shape()[0];
            for i in 0..batch {
                let mut row = out.slice_mut(ndarray::s![i, ..]);
                let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                row.mapv_inplace(|x| (x - max).exp());
                let sum: f32 = row.sum();
                row.mapv_inplace(|x| x / sum);
            }
        } else {
            let ax = Axis(axis);
            for mut row in out.axis_iter_mut(ax) {
                let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                row.mapv_inplace(|x| (x - max).exp());
                let sum: f32 = row.sum();
                row.mapv_inplace(|x| x / sum);
            }
        }
        Ok(Tensor {
            data: Arc::new(out),
            shape: self.shape.clone(),
            device: self.device,
            dtype: self.dtype,
            requires_grad: self.requires_grad,
            node_id: self.node_id,
            grad: None,
        })
    }

    pub fn cross_entropy_loss(&self, targets: &[usize]) -> Result<Tensor> {
        // self: logits [batch, classes]
        let batch = self.shape[0];
        let classes = self.shape.get(1).copied().unwrap_or(1);
        let sm = self.softmax(1)?;
        let mut loss = 0.0f32;
        for (i, &t) in targets.iter().enumerate().take(batch) {
            if t < classes {
                let p = sm.data[[i, t]].max(1e-8);
                loss -= p.ln();
            }
        }
        loss /= batch as f32;
        Ok(Tensor::from_array(
            ArrayD::from_elem(IxDyn(&[]), loss),
            self.requires_grad,
        ))
    }

    pub fn sum(&self) -> Tensor {
        let s = self.data.sum();
        Tensor::from_array(ArrayD::from_elem(IxDyn(&[]), s), self.requires_grad)
    }

    pub fn mean(&self) -> Tensor {
        let n = self.numel() as f32;
        let s = self.data.sum() / n;
        Tensor::from_array(ArrayD::from_elem(IxDyn(&[]), s), self.requires_grad)
    }

    pub fn detach(&self) -> Tensor {
        Tensor {
            data: self.data.clone(),
            shape: self.shape.clone(),
            device: self.device,
            dtype: self.dtype,
            requires_grad: false,
            node_id: None,
            grad: None,
        }
    }
}

fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let len = a.len().max(b.len());
    let mut out = vec![1; len];
    for i in 0..len {
        let ai = if i < len - a.len() {
            1
        } else {
            a[i - (len - a.len())]
        };
        let bi = if i < len - b.len() {
            1
        } else {
            b[i - (len - b.len())]
        };
        if ai != bi && ai != 1 && bi != 1 {
            return Err(MmnError::Shape {
                message: format!("incompatible shapes {:?} vs {:?}", a, b),
            });
        }
        out[i] = ai.max(bi);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn softmax_rows_sum_to_one() {
        let t = Tensor::from_array(arr2(&[[1.0, 2.0, 0.5]]).into_dyn(), false);
        let sm = t.softmax(1).unwrap();
        let s: f32 = sm.data.slice(ndarray::s![0, ..]).sum();
        assert!((s - 1.0).abs() < 1e-5, "row softmax should sum to 1, got {s}");
    }

    #[test]
    fn softmax_batch_rows_each_sum_to_one() {
        let t = Tensor::from_array(
            arr2(&[[1.0, 0.0, 0.0], [0.0, 2.0, 3.0]]).into_dyn(),
            false,
        );
        let sm = t.softmax(1).unwrap();
        for i in 0..2 {
            let s: f32 = sm.data.slice(ndarray::s![i, ..]).sum();
            assert!((s - 1.0).abs() < 1e-5, "row {i} should sum to 1, got {s}");
        }
    }

    #[test]
    fn matmul_shapes() {
        let a = Tensor::from_array(arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn(), false);
        let b = Tensor::from_array(arr2(&[[5.0, 6.0], [7.0, 8.0]]).into_dyn(), false);
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape, vec![2, 2]);
    }

    #[test]
    fn relu_zeros_negatives() {
        let t = Tensor::from_array(arr2(&[[-1.0, 2.0]]).into_dyn(), false);
        let r = t.relu();
        assert_eq!(r.data[[0, 0]], 0.0);
        assert_eq!(r.data[[0, 1]], 2.0);
    }
}
