//! KLENT zero-search self-play collection and closed-form policy/value targets.
//!
//! Pure Rust crate: drives ONNX inference through `great_kingdom_onnx` and
//! contains no PyO3 dependency.

mod batch;
mod error;
mod rng;
mod targets;

pub use batch::KlentZeroSearchBatch;
pub use error::KlentError;
pub use targets::{analytical_policy_row, masked_state_value_row};
