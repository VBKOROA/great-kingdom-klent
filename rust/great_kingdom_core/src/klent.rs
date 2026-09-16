//! PyO3 bindings for the KLENT zero-search actor and analytical targets.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use great_kingdom_klent::{
    KlentZeroSearchBatch as CoreKlentZeroSearchBatch, analytical_policy_row,
    masked_state_value_row,
};

use crate::errors::klent_error_to_pyerr;

#[pyclass(unsendable)]
pub struct KlentZeroSearchBatch {
    inner: CoreKlentZeroSearchBatch,
}

#[pymethods]
impl KlentZeroSearchBatch {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        path,
        games,
        seed,
        device = "cpu",
        max_batch_size = 256,
        alpha = 0.03,
        beta = 0.1,
        max_turns = 200,
    ))]
    fn py_new(
        path: &str,
        games: usize,
        seed: u64,
        device: &str,
        max_batch_size: usize,
        alpha: f32,
        beta: f32,
        max_turns: usize,
    ) -> PyResult<Self> {
        CoreKlentZeroSearchBatch::new(
            path,
            games,
            seed,
            device,
            max_batch_size,
            alpha,
            beta,
            max_turns,
        )
        .map(|inner| Self { inner })
        .map_err(klent_error_to_pyerr)
    }

    #[must_use]
    fn game_count(&self) -> usize {
        self.inner.game_count()
    }

    #[must_use]
    fn seeds(&self) -> Vec<u64> {
        self.inner.seeds()
    }

    fn run(&mut self) -> PyResult<()> {
        self.inner.run().map_err(klent_error_to_pyerr)
    }

    #[must_use]
    fn winners(&self) -> Vec<Option<u8>> {
        self.inner.winners()
    }

    #[must_use]
    fn end_reasons(&self) -> Vec<Option<u8>> {
        self.inner.end_reasons()
    }

    #[must_use]
    fn territory_scores(&self) -> Vec<(u8, u8)> {
        self.inner.territory_scores()
    }

    #[must_use]
    fn move_counts(&self) -> Vec<usize> {
        self.inner.move_counts()
    }

    fn turn_players(&self, game_index: usize) -> PyResult<Vec<u8>> {
        self.inner
            .turn_players(game_index)
            .map_err(klent_error_to_pyerr)
    }

    fn turn_actions(&self, game_index: usize) -> PyResult<Vec<usize>> {
        self.inner
            .turn_actions(game_index)
            .map_err(klent_error_to_pyerr)
    }

    fn policy_targets(&self, game_index: usize) -> PyResult<Vec<Vec<f32>>> {
        self.inner
            .policy_targets(game_index)
            .map_err(klent_error_to_pyerr)
    }

    fn state_values(&self, game_index: usize) -> PyResult<Vec<f32>> {
        self.inner
            .state_values(game_index)
            .map_err(klent_error_to_pyerr)
    }
}

#[pyfunction]
#[pyo3(signature = (policy_logits, q_values, legal_mask, alpha, beta))]
pub fn klent_analytical_policy(
    policy_logits: Vec<f32>,
    q_values: Vec<f32>,
    legal_mask: Vec<bool>,
    alpha: f32,
    beta: f32,
) -> PyResult<Vec<f32>> {
    analytical_policy_row(&policy_logits, &q_values, &legal_mask, alpha, beta)
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

#[pyfunction]
#[pyo3(signature = (policy_logits, q_values, legal_mask))]
pub fn klent_masked_state_value(
    policy_logits: Vec<f32>,
    q_values: Vec<f32>,
    legal_mask: Vec<bool>,
) -> PyResult<f32> {
    masked_state_value_row(&policy_logits, &q_values, &legal_mask)
        .map_err(|err| PyValueError::new_err(err.to_string()))
}
