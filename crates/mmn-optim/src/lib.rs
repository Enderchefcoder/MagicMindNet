use half::bf16;
use ndarray::{Array2, ArrayD, IxDyn};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct AdamWConfig {
    pub lr: f32,
    pub betas: (f32, f32),
    pub eps: f32,
    pub weight_decay: f32,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            lr: 3e-4,
            betas: (0.9, 0.95),
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

pub struct AdamW {
    pub config: AdamWConfig,
    step: usize,
    m: HashMap<usize, ArrayD<f32>>,
    v: HashMap<usize, ArrayD<f32>>,
}

impl AdamW {
    pub fn new(config: AdamWConfig) -> Self {
        Self {
            config,
            step: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }

    pub fn step_param(&mut self, id: usize, param: &mut ArrayD<f32>, grad: &ArrayD<f32>) {
        self.step += 1;
        let t = self.step as f32;
        let (b1, b2) = self.config.betas;
        let m = self.m.entry(id).or_insert_with(|| ArrayD::zeros(param.raw_dim()));
        let v = self.v.entry(id).or_insert_with(|| ArrayD::zeros(param.raw_dim()));
        m.zip_mut_with(grad, |m_, g| {
            *m_ = b1 * *m_ + (1.0 - b1) * g;
        });
        v.zip_mut_with(grad, |v_, g| {
            *v_ = b2 * *v_ + (1.0 - b2) * g * g;
        });
        let m_hat = m.mapv(|x| x / (1.0 - b1.powf(t)));
        let v_hat = v.mapv(|x| x / (1.0 - b2.powf(t)));
        param.zip_mut_with(&(&m_hat / &(&v_hat.mapv(|x| x.sqrt()) + self.config.eps)), |p, upd| {
            *p -= self.config.lr * upd;
        });
        let wd = 1.0 - self.config.lr * self.config.weight_decay;
        param.mapv_inplace(|p| p * wd);
    }
}

#[derive(Clone, Debug)]
pub struct MuonConfig {
    pub lr: f32,
    pub momentum: f32,
    pub nesterov: bool,
    pub ns_steps: usize,
}

impl Default for MuonConfig {
    fn default() -> Self {
        Self {
            lr: 0.02,
            momentum: 0.95,
            nesterov: true,
            ns_steps: 5,
        }
    }
}

/// Newton-Schulz orthogonalization (Keller Jordan Muon quintic iteration).
pub fn newton_schulz5(g: &ArrayD<f32>, steps: usize) -> ArrayD<f32> {
    const A_COEF: f32 = 3.4445;
    const B_COEF: f32 = -4.7750;
    const C_COEF: f32 = 2.0315;
    const EPS: f32 = 1e-7;

    let g2: Array2<f32> = g
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .map(|v| v.to_owned())
        .unwrap_or_else(|_| {
            let n = g.len();
            let side = (n as f32).sqrt().ceil() as usize;
            let mut padded = Array2::<f32>::zeros((side, side));
            padded.iter_mut().zip(g.iter()).for_each(|(p, v)| *p = *v);
            padded
        });

    let (n, m) = (g2.nrows(), g2.ncols());
    let transposed = n > m;
    let mut x = if transposed {
        g2.t().mapv(|v| bf16::from_f32(v).to_f32())
    } else {
        g2.mapv(|v| bf16::from_f32(v).to_f32())
    };

    let norm = x.iter().map(|v| v * v).sum::<f32>().sqrt().max(EPS);
    x.mapv_inplace(|v| v / norm);

    for _ in 0..steps {
        let a_mat = x.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        let x_xt = a_mat.dot(&a_mat.t());
        let a_bf = x_xt.mapv(|v| bf16::from_f32(v).to_f32());
        let a_sq = a_bf.dot(&a_bf).mapv(|v| bf16::from_f32(v).to_f32());
        let b_mat = &a_bf * B_COEF + &a_sq * C_COEF;
        x = &x * A_COEF + b_mat.dot(&a_mat);
        x.mapv_inplace(|v| bf16::from_f32(v).to_f32());
    }

    if transposed {
        x = x.t().to_owned();
    }
    x.into_dyn()
}

pub struct Muon {
    pub config: MuonConfig,
    velocity: HashMap<usize, ArrayD<f32>>,
}

impl Muon {
    pub fn new(config: MuonConfig) -> Self {
        Self {
            config,
            velocity: HashMap::new(),
        }
    }

    pub fn step_matrix(&mut self, id: usize, param: &mut ArrayD<f32>, grad: &ArrayD<f32>) {
        let g2: Array2<f32> = grad
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .map(|v| v.to_owned())
            .unwrap_or_else(|_| {
                let side = (grad.len() as f32).sqrt().ceil() as usize;
                Array2::from_shape_vec((side, side), grad.iter().cloned().collect()).unwrap()
            });
        let v = self
            .velocity
            .entry(id)
            .or_insert_with(|| ArrayD::zeros(IxDyn(g2.shape())));
        v.zip_mut_with(&g2, |vel, g| {
            *vel = self.config.momentum * *vel + (1.0 - self.config.momentum) * g;
        });
        let update = if self.config.nesterov {
            let v2 = v
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap()
                .to_owned();
            let mom = self.config.momentum;
            let nesterov = g2.mapv(|x| x * (1.0 - mom)) + v2.mapv(|x| x * mom);
            newton_schulz5(&nesterov.into_dyn(), self.config.ns_steps)
        } else {
            newton_schulz5(v, self.config.ns_steps)
        };
        let fan_out = param.shape()[0] as f32;
        let fan_in = param.shape().get(1).copied().unwrap_or(1) as f32;
        let scale = (fan_out / fan_in).sqrt();
        param.zip_mut_with(&update, |p, u| {
            *p -= self.config.lr * scale * u;
        });
    }
}

pub enum ParamKind {
    Matrix,
    Other,
}

pub struct HybridOptimizer {
    pub muon: Muon,
    pub adamw: AdamW,
}

impl HybridOptimizer {
    pub fn new(muon: MuonConfig, adamw: AdamWConfig) -> Self {
        Self {
            muon: Muon::new(muon),
            adamw: AdamW::new(adamw),
        }
    }

    pub fn classify_param(shape: &[usize]) -> ParamKind {
        if shape.len() == 2 && shape[0] > 1 && shape[1] > 1 {
            ParamKind::Matrix
        } else {
            ParamKind::Other
        }
    }

    pub fn step(&mut self, id: usize, param: &mut ArrayD<f32>, grad: &ArrayD<f32>) {
        match Self::classify_param(param.shape()) {
            ParamKind::Matrix => self.muon.step_matrix(id, param, grad),
            ParamKind::Other => self.adamw.step_param(id, param, grad),
        }
    }
}

/// Sum per-parameter gradients across micro-batches; apply once with averaged grads.
#[derive(Default)]
pub struct GradAccumulator {
    grads: Vec<ArrayD<f32>>,
    micro_batches: usize,
    param_index: usize,
}

impl GradAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.grads.clear();
        self.micro_batches = 0;
        self.param_index = 0;
    }

    pub fn micro_batches(&self) -> usize {
        self.micro_batches
    }

    pub fn begin_micro_batch(&mut self) {
        self.param_index = 0;
    }

    pub fn finish_micro_batch(&mut self) {
        self.micro_batches += 1;
        self.param_index = 0;
    }

    pub fn add_param_grad(&mut self, grad: &ArrayD<f32>) {
        if self.param_index < self.grads.len() {
            self.grads[self.param_index]
                .zip_mut_with(grad, |acc, g| *acc += *g);
        } else {
            self.grads.push(grad.clone());
        }
        self.param_index += 1;
    }

    pub fn averaged_grad(&self, index: usize) -> ArrayD<f32> {
        let n = self.micro_batches.max(1) as f32;
        self.grads[index].mapv(|x| x / n)
    }

    pub fn len(&self) -> usize {
        self.grads.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn adamw_changes_weights() {
        let mut p = arr2(&[[1.0]]).into_dyn();
        let g = arr2(&[[0.5]]).into_dyn();
        let mut opt = AdamW::new(AdamWConfig {
            lr: 0.1,
            ..Default::default()
        });
        let before = p[[0, 0]];
        opt.step_param(0, &mut p, &g);
        assert_ne!(p[[0, 0]], before);
    }

    #[test]
    fn grad_accumulator_averages_two_micro_batches() {
        let mut acc = GradAccumulator::new();
        acc.begin_micro_batch();
        acc.add_param_grad(&arr2(&[[1.0]]).into_dyn());
        acc.finish_micro_batch();
        acc.begin_micro_batch();
        acc.add_param_grad(&arr2(&[[3.0]]).into_dyn());
        acc.finish_micro_batch();
        let avg = acc.averaged_grad(0);
        assert!((avg[[0, 0]] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn newton_schulz_output_finite() {
        let g = arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn();
        let o = newton_schulz5(&g, 5);
        assert!(o.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn classify_param_matrix_vs_other() {
        assert!(matches!(
            HybridOptimizer::classify_param(&[64, 32]),
            ParamKind::Matrix
        ));
        assert!(matches!(
            HybridOptimizer::classify_param(&[16]),
            ParamKind::Other
        ));
        assert!(matches!(
            HybridOptimizer::classify_param(&[16, 1]),
            ParamKind::Other
        ));
        assert!(matches!(
            HybridOptimizer::classify_param(&[1, 3, 3, 3]),
            ParamKind::Other
        ));
    }

    #[test]
    fn newton_schulz_output_nonzero() {
        let g = ndarray::Array2::from_shape_fn((4, 4), |(i, j)| (i + j) as f32 * 0.1).into_dyn();
        let o = newton_schulz5(&g, 5);
        let s: f32 = o.iter().map(|x| x.abs()).sum();
        assert!(s > 1e-6, "newton_schulz output sum abs = {s}");
    }

    #[test]
    fn muon_step_matrix_changes_weights() {
        let mut opt = Muon::new(MuonConfig {
            lr: 0.1,
            ..Default::default()
        });
        let mut p = ndarray::Array2::from_shape_fn((4, 4), |(i, j)| 1.0 + (i + j) as f32 * 0.01).into_dyn();
        let g = ndarray::Array2::from_shape_fn((4, 4), |(i, j)| 0.1 + (i as f32) * 0.02 + (j as f32) * 0.03)
            .into_dyn();
        let before = p.clone();
        opt.step_matrix(0, &mut p, &g);
        assert_ne!(p, before);
    }

    #[test]
    fn hybrid_routes_matrix_to_muon() {
        let mut opt = HybridOptimizer::new(
            MuonConfig {
                lr: 0.1,
                ..Default::default()
            },
            AdamWConfig::default(),
        );
        let mut p = ndarray::Array2::from_shape_fn((4, 4), |(i, j)| 1.0 + (i + j) as f32 * 0.01).into_dyn();
        let g = ndarray::Array2::from_shape_fn((4, 4), |(i, j)| 0.1 + (i as f32) * 0.02 + (j as f32) * 0.03)
            .into_dyn();
        let before = p.clone();
        opt.step(0, &mut p, &g);
        assert_ne!(p, before);
    }

    #[test]
    fn hybrid_vector_step_uses_adamw() {
        let mut opt = HybridOptimizer::new(MuonConfig::default(), AdamWConfig::default());
        let mut p = arr2(&[[1.0], [2.0], [3.0]]).into_dyn();
        let g = arr2(&[[0.5], [0.5], [0.5]]).into_dyn();
        let before = p[[0, 0]];
        opt.step(1, &mut p, &g);
        assert_ne!(p[[0, 0]], before);
    }

    #[test]
    fn grad_accumulator_clear_resets_state() {
        let mut acc = GradAccumulator::new();
        acc.begin_micro_batch();
        acc.add_param_grad(&arr2(&[[1.0]]).into_dyn());
        acc.finish_micro_batch();
        assert_eq!(acc.micro_batches(), 1);
        acc.clear();
        assert_eq!(acc.micro_batches(), 0);
        assert_eq!(acc.len(), 0);
    }
}
