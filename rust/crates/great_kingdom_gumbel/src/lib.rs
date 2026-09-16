//! Gumbel AlphaZero MCTS search, batched self-play, and arena evaluation.
//!
//! Pure Rust crate: no ONNX Runtime binding knowledge beyond the abstract
//! `GumbelEvaluator` trait, and no PyO3 dependency.

mod arena_batch;
mod batch;
mod config;
mod debug;
mod error;
mod evaluator;
mod node;
mod policy;
mod profile;
mod result;
pub(crate) mod rng;
mod root;
mod sampling;
mod search;
mod selection;
mod sequential_halving;

pub use arena_batch::{ArenaLeafEvaluator, GumbelArenaBatch};
pub use batch::GumbelSelfPlayBatch;
pub use config::GumbelConfig;
pub use error::{GumbelError, GumbelOutcome};
pub use evaluator::GumbelEvaluator;
pub use result::GumbelResult;
pub use search::{GumbelEvalBatch, GumbelSearch};
