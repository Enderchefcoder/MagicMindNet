pub mod autograd;
pub mod device;
pub mod dtype;
pub mod error;
pub mod ops;
pub mod tensor;

pub use autograd::{backward, clear_tape, enable_grad, grad_enabled};
pub use device::Device;
pub use dtype::DType;
pub use error::{MmnError, Result};
pub use ops::{cross_entropy_grad, embedding_backward, linear_backward};
pub use tensor::Tensor;
