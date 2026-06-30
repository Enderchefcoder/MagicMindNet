use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};

static NODE_ID: AtomicU64 = AtomicU64::new(1);

fn next_id() -> u64 {
    NODE_ID.fetch_add(1, Ordering::Relaxed)
}

pub type BackwardFn = Box<dyn Fn(&ArrayD<f32>) -> Vec<ArrayD<f32>>>;

pub struct Node {
    pub id: u64,
    pub parents: Vec<u64>,
    pub backward: Option<BackwardFn>,
}

thread_local! {
    static TAPE: RefCell<Vec<Node>> = RefCell::new(Vec::new());
    static ENABLED: RefCell<bool> = RefCell::new(false);
}

pub fn enable_grad(enabled: bool) {
    ENABLED.with(|e| *e.borrow_mut() = enabled);
    if enabled {
        clear_tape();
    }
}

pub fn grad_enabled() -> bool {
    ENABLED.with(|e| *e.borrow())
}

pub fn clear_tape() {
    TAPE.with(|t| t.borrow_mut().clear());
}

pub fn register_node(parents: Vec<u64>, backward: BackwardFn) -> u64 {
    let id = next_id();
    if grad_enabled() {
        TAPE.with(|t| {
            t.borrow_mut().push(Node {
                id,
                parents,
                backward: Some(backward),
            });
        });
    }
    id
}

pub fn backward(root: &Tensor, grad: Option<ArrayD<f32>>) -> Vec<(u64, ArrayD<f32>)> {
    let mut grads: std::collections::HashMap<u64, ArrayD<f32>> = std::collections::HashMap::new();
    let root_id = root.node_id.unwrap_or(0);
    let init = grad.unwrap_or_else(|| ArrayD::ones(IxDyn(&root.shape)).into());
    grads.insert(root_id, init);

    let mut tape = TAPE.with(|t| std::mem::take(&mut *t.borrow_mut()));
    tape.reverse();
    for node in tape {
        if let Some(g) = grads.get(&node.id).cloned() {
            if let Some(bw) = node.backward {
                let parent_grads = bw(&g);
                for (pid, pg) in node.parents.iter().zip(parent_grads.into_iter()) {
                    grads
                        .entry(*pid)
                        .and_modify(|acc| {
                            if acc.shape() == pg.shape() {
                                *acc = &*acc + &pg;
                            }
                        })
                        .or_insert(pg);
                }
            }
        }
    }
    grads.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use ndarray::arr2;

    #[test]
    fn tape_accumulates_grad() {
        enable_grad(true);
        let a = Tensor::from_array(arr2(&[[1.0, 2.0]]).into_dyn(), true);
        let b = Tensor::from_array(arr2(&[[3.0, 4.0]]).into_dyn(), true);
        let c = a.add(&b).unwrap();
        let grads = backward(&c, None);
        assert!(!grads.is_empty());
        enable_grad(false);
        clear_tape();
    }

    #[test]
    fn backward_add_splits_grad_to_parents() {
        enable_grad(true);
        let a = Tensor::from_array(arr2(&[[1.0]]).into_dyn(), true);
        let b = Tensor::from_array(arr2(&[[2.0]]).into_dyn(), true);
        let aid = a.node_id.unwrap();
        let bid = b.node_id.unwrap();
        let c = a.add(&b).unwrap();
        let grads = backward(&c, None);
        let ga = grads
            .iter()
            .find(|(id, _)| *id == aid)
            .map(|(_, g)| g[[0, 0]]);
        let gb = grads
            .iter()
            .find(|(id, _)| *id == bid)
            .map(|(_, g)| g[[0, 0]]);
        assert_eq!(ga, Some(1.0));
        assert_eq!(gb, Some(1.0));
        enable_grad(false);
        clear_tape();
    }
}
