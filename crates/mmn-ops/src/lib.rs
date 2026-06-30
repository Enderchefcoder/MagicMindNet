use mmn_core::{Result, Tensor};

pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    a.add(b)
}

pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    a.matmul(b)
}

pub fn relu(t: &Tensor) -> Tensor {
    t.relu()
}

pub fn softmax(t: &Tensor, axis: usize) -> Result<Tensor> {
    t.softmax(axis)
}
