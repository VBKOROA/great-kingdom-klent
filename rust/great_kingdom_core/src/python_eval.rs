//! Adapters that let pure Gumbel search call Python evaluator callbacks.

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use great_kingdom_engine::game::{ACTION_SPACE, GameState as CoreGameState};
use great_kingdom_features::EvalRequest as CoreEvalRequest;
use great_kingdom_gumbel::{ArenaLeafEvaluator, GumbelError, GumbelEvalBatch, GumbelEvaluator};

use crate::eval_request::EvalRequest;

pub(crate) struct PythonGumbelEvaluator<'a, 'py> {
    callback: &'a Bound<'py, PyAny>,
}

impl<'a, 'py> PythonGumbelEvaluator<'a, 'py> {
    pub(crate) const fn new(callback: &'a Bound<'py, PyAny>) -> Self {
        Self { callback }
    }
}

impl GumbelEvaluator for PythonGumbelEvaluator<'_, '_> {
    fn check_signals(&self) -> Result<(), GumbelError> {
        self.callback
            .py()
            .check_signals()
            .map_err(GumbelError::external)
    }

    fn evaluate(&mut self, request: CoreEvalRequest) -> Result<GumbelEvalBatch, GumbelError> {
        let py = self.callback.py();
        let request = Py::new(py, EvalRequest::from_inner(request)).map_err(GumbelError::external)?;
        let response = self
            .callback
            .call1((request,))
            .map_err(GumbelError::external)?;
        parse_gumbel_eval_response(&response).map_err(GumbelError::external)
    }
}

pub(crate) struct PythonArenaLeafEvaluator<'a, 'py> {
    callback: &'a Bound<'py, PyAny>,
}

impl<'a, 'py> PythonArenaLeafEvaluator<'a, 'py> {
    pub(crate) const fn new(callback: &'a Bound<'py, PyAny>) -> Self {
        Self { callback }
    }
}

impl ArenaLeafEvaluator for PythonArenaLeafEvaluator<'_, '_> {
    fn check_signals(&mut self) -> Result<(), GumbelError> {
        self.callback
            .py()
            .check_signals()
            .map_err(GumbelError::external)
    }

    fn evaluate(
        &mut self,
        states: Vec<CoreGameState>,
        game_indexes: Vec<usize>,
        _active_games: usize,
        _wave: u64,
    ) -> Result<GumbelEvalBatch, GumbelError> {
        let request = CoreEvalRequest::new_with_game_indexes(states, game_indexes);
        let py = self.callback.py();
        let request = Py::new(py, EvalRequest::from_inner(request)).map_err(GumbelError::external)?;
        let response = self
            .callback
            .call1((request,))
            .map_err(GumbelError::external)?;
        parse_gumbel_eval_response(&response).map_err(GumbelError::external)
    }
}

pub(crate) fn parse_gumbel_eval_response(
    response: &Bound<'_, PyAny>,
) -> PyResult<GumbelEvalBatch> {
    if let Ok((policy_obj, value_obj)) = response.extract::<(Bound<'_, PyAny>, Bound<'_, PyAny>)>()
    {
        if let Ok(eval) = parse_gumbel_eval_response_buffers(&policy_obj, &value_obj) {
            return Ok(eval);
        }
    }

    let (policy_rows, values): (Vec<Vec<f32>>, Vec<f32>) = response.extract()?;
    let mut policies = Vec::with_capacity(policy_rows.len());
    for (row_index, row) in policy_rows.into_iter().enumerate() {
        policies.push(parse_gumbel_policy_row(row, row_index)?);
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("values must be finite"));
    }
    Ok(GumbelEvalBatch::new(policies, values))
}

fn parse_gumbel_eval_response_buffers(
    policy_obj: &Bound<'_, PyAny>,
    value_obj: &Bound<'_, PyAny>,
) -> PyResult<GumbelEvalBatch> {
    let py = policy_obj.py();
    let policy_buffer = PyBuffer::<f32>::get(policy_obj)?;
    let value_buffer = PyBuffer::<f32>::get(value_obj)?;
    if !policy_buffer.is_c_contiguous() || !value_buffer.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "policy/value buffers must be C-contiguous float32 arrays",
        ));
    }
    let policy_count = policy_buffer.item_count();
    if policy_count % ACTION_SPACE != 0 {
        return Err(PyValueError::new_err(format!(
            "policy buffer length must be divisible by {ACTION_SPACE}, got {policy_count}",
        )));
    }
    let batch_size = policy_count / ACTION_SPACE;
    if value_buffer.item_count() != batch_size {
        return Err(PyValueError::new_err(format!(
            "expected {batch_size} values, got {}",
            value_buffer.item_count()
        )));
    }

    let policy_values = policy_buffer.to_vec(py)?;
    let values = value_buffer.to_vec(py)?;
    let mut policies = Vec::with_capacity(batch_size);
    for (row_index, row) in policy_values.chunks_exact(ACTION_SPACE).enumerate() {
        if row.iter().any(|logit| !logit.is_finite()) {
            return Err(PyValueError::new_err(format!(
                "policy row {row_index} contains non-finite logits"
            )));
        }
        let mut policy = [0.0; ACTION_SPACE];
        policy.copy_from_slice(row);
        policies.push(policy);
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("values must be finite"));
    }
    Ok(GumbelEvalBatch::new(policies, values))
}

fn parse_gumbel_policy_row(row: Vec<f32>, row_index: usize) -> PyResult<[f32; ACTION_SPACE]> {
    if row.len() != ACTION_SPACE {
        return Err(PyValueError::new_err(format!(
            "policy row {row_index} must have length {ACTION_SPACE}, got {}",
            row.len()
        )));
    }
    if row.iter().any(|logit| !logit.is_finite()) {
        return Err(PyValueError::new_err("policy logits must be finite"));
    }
    let mut policy = [0.0; ACTION_SPACE];
    policy.copy_from_slice(&row);
    Ok(policy)
}
