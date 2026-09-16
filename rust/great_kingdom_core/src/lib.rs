//! `great_kingdom_core` PyO3 facade.
//!
//! This crate is the only Python C-extension boundary. It re-exports the
//! classes, functions, and constants of the pure Rust domain crates under the
//! unchanged `great_kingdom_core` module while keeping all game, feature, ONNX,
//! and search logic in dependency-light crates.

mod errors;
mod eval_request;
mod game;
mod gumbel;
mod klent;
mod onnx;
mod python_eval;

use pyo3::prelude::*;

use great_kingdom_engine::game::{ACTION_SPACE, BOARD_CELLS, BOARD_SIZE, PASS_ACTION};
use great_kingdom_features::FEATURE_CHANNELS;

use eval_request::EvalRequest;
use game::GameState;
use gumbel::{GumbelArenaBatch, GumbelConfig, GumbelResult, GumbelSearch, GumbelSelfPlayBatch};
use klent::KlentZeroSearchBatch;
use onnx::OnnxEvaluator;

#[pyfunction]
#[must_use]
pub fn action_space() -> usize {
    ACTION_SPACE
}

#[pyfunction]
#[must_use]
pub fn rayon_thread_count() -> usize {
    rayon::current_num_threads()
}

#[pymodule]
fn great_kingdom_core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<GameState>()?;
    module.add_class::<EvalRequest>()?;
    module.add_class::<GumbelConfig>()?;
    module.add_class::<GumbelResult>()?;
    module.add_class::<GumbelSearch>()?;
    module.add_class::<GumbelSelfPlayBatch>()?;
    module.add_class::<GumbelArenaBatch>()?;
    module.add_class::<OnnxEvaluator>()?;
    module.add_class::<KlentZeroSearchBatch>()?;
    module.add_function(wrap_pyfunction!(action_space, module)?)?;
    module.add_function(wrap_pyfunction!(rayon_thread_count, module)?)?;
    module.add_function(wrap_pyfunction!(klent::klent_analytical_policy, module)?)?;
    module.add_function(wrap_pyfunction!(klent::klent_masked_state_value, module)?)?;
    module.add("BOARD_SIZE", BOARD_SIZE)?;
    module.add("BOARD_CELLS", BOARD_CELLS)?;
    module.add("PASS_ACTION", PASS_ACTION)?;
    module.add("FEATURE_CHANNELS", FEATURE_CHANNELS)?;
    Ok(())
}
