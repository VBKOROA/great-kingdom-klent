//! PyO3 bindings for the Gumbel search, self-play batch, and arena batch.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use great_kingdom_gumbel::{
    GumbelArenaBatch as CoreArenaBatch, GumbelConfig as CoreConfig,
    GumbelEvaluator as _, GumbelResult as CoreResult, GumbelSearch as CoreSearch,
    GumbelSelfPlayBatch as CoreSelfPlayBatch,
};

use crate::errors::gumbel_error_to_pyerr;
use crate::eval_request::EvalRequest;
use crate::game::GameState;
use crate::onnx::OnnxEvaluator;
use crate::python_eval::{PythonArenaLeafEvaluator, PythonGumbelEvaluator};

#[allow(clippy::too_many_arguments)]
fn build_config(
    simulations: u32,
    max_considered_actions: usize,
    c_visit: f32,
    c_scale: f32,
    seed: u64,
    gumbel_scale: f32,
    policy_target_temperature: f32,
    policy_target_c_visit: Option<f32>,
    policy_target_c_scale: Option<f32>,
) -> PyResult<CoreConfig> {
    let policy_target_c_visit = policy_target_c_visit
        .ok_or_else(|| PyValueError::new_err("policy_target_c_visit must be set"))?;
    let policy_target_c_scale = policy_target_c_scale
        .ok_or_else(|| PyValueError::new_err("policy_target_c_scale must be set"))?;
    let config = CoreConfig::new_with_full_config(
        simulations,
        max_considered_actions,
        c_visit,
        c_scale,
        seed,
        gumbel_scale,
        policy_target_temperature,
        policy_target_c_visit,
        policy_target_c_scale,
    );
    config.validate().map_err(gumbel_error_to_pyerr)?;
    Ok(config)
}

fn map_results(results: Vec<Option<CoreResult>>) -> Vec<Option<GumbelResult>> {
    results
        .into_iter()
        .map(|result| result.map(GumbelResult::from_inner))
        .collect()
}

fn map_results_with_root_logits(
    pair: (Vec<Option<CoreResult>>, Vec<Vec<f32>>),
) -> (Vec<Option<GumbelResult>>, Vec<Vec<f32>>) {
    (map_results(pair.0), pair.1)
}

#[pyclass]
#[derive(Clone, Copy, Debug)]
pub struct GumbelConfig {
    inner: CoreConfig,
}

#[pymethods]
impl GumbelConfig {
    #[new]
    #[pyo3(signature = (
        simulations = 128,
        max_considered_actions = 16,
        c_visit = 50.0,
        c_scale = 1.0,
        seed = 0,
        gumbel_scale = 1.0,
        policy_target_temperature = 1.0,
        policy_target_c_visit = None,
        policy_target_c_scale = None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn py_new(
        simulations: u32,
        max_considered_actions: usize,
        c_visit: f32,
        c_scale: f32,
        seed: u64,
        gumbel_scale: f32,
        policy_target_temperature: f32,
        policy_target_c_visit: Option<f32>,
        policy_target_c_scale: Option<f32>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: build_config(
                simulations,
                max_considered_actions,
                c_visit,
                c_scale,
                seed,
                gumbel_scale,
                policy_target_temperature,
                policy_target_c_visit,
                policy_target_c_scale,
            )?,
        })
    }

    #[must_use]
    pub fn simulations(&self) -> u32 {
        self.inner.simulations
    }

    #[must_use]
    pub fn max_considered_actions(&self) -> usize {
        self.inner.max_considered_actions
    }

    #[must_use]
    pub fn c_visit(&self) -> f32 {
        self.inner.c_visit
    }

    #[must_use]
    pub fn c_scale(&self) -> f32 {
        self.inner.c_scale
    }

    #[must_use]
    pub fn gumbel_scale(&self) -> f32 {
        self.inner.gumbel_scale
    }

    #[must_use]
    pub fn policy_target_c_visit(&self) -> f32 {
        self.inner.policy_target_c_visit
    }

    #[must_use]
    pub fn policy_target_c_scale(&self) -> f32 {
        self.inner.policy_target_c_scale
    }

    #[must_use]
    pub fn policy_target_temperature(&self) -> f32 {
        self.inner.policy_target_temperature
    }

    #[must_use]
    pub fn seed(&self) -> u64 {
        self.inner.seed
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct GumbelResult {
    pub(crate) inner: CoreResult,
}

impl GumbelResult {
    pub(crate) const fn from_inner(inner: CoreResult) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl GumbelResult {
    #[must_use]
    pub fn selected_action(&self) -> Option<usize> {
        self.inner.selected_action
    }

    #[must_use]
    pub fn policy_target(&self) -> Vec<f32> {
        self.inner.policy_target.to_vec()
    }

    #[must_use]
    pub fn visit_counts(&self) -> Vec<u32> {
        self.inner.visit_counts.to_vec()
    }

    #[must_use]
    pub fn root_value(&self) -> f32 {
        self.inner.root_value
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct GumbelSearch {
    inner: CoreSearch,
}

#[pymethods]
impl GumbelSearch {
    #[new]
    #[pyo3(signature = (
        simulations = 128,
        max_considered_actions = 16,
        c_visit = 50.0,
        c_scale = 1.0,
        seed = 0,
        gumbel_scale = 1.0,
        policy_target_temperature = 1.0,
        policy_target_c_visit = None,
        policy_target_c_scale = None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn py_new(
        simulations: u32,
        max_considered_actions: usize,
        c_visit: f32,
        c_scale: f32,
        seed: u64,
        gumbel_scale: f32,
        policy_target_temperature: f32,
        policy_target_c_visit: Option<f32>,
        policy_target_c_scale: Option<f32>,
    ) -> PyResult<Self> {
        let config = build_config(
            simulations,
            max_considered_actions,
            c_visit,
            c_scale,
            seed,
            gumbel_scale,
            policy_target_temperature,
            policy_target_c_visit,
            policy_target_c_scale,
        )?;
        Ok(Self {
            inner: CoreSearch::new(config),
        })
    }

    #[must_use]
    pub fn simulations(&self) -> u32 {
        self.inner.config.simulations
    }

    #[must_use]
    pub fn max_considered_actions(&self) -> usize {
        self.inner.config.max_considered_actions
    }

    #[must_use]
    pub fn c_visit(&self) -> f32 {
        self.inner.config.c_visit
    }

    #[must_use]
    pub fn c_scale(&self) -> f32 {
        self.inner.config.c_scale
    }

    #[must_use]
    pub fn gumbel_scale(&self) -> f32 {
        self.inner.config.gumbel_scale
    }

    #[must_use]
    pub fn policy_target_c_visit(&self) -> f32 {
        self.inner.config.policy_target_c_visit
    }

    #[must_use]
    pub fn policy_target_c_scale(&self) -> f32 {
        self.inner.config.policy_target_c_scale
    }

    #[must_use]
    pub fn policy_target_temperature(&self) -> f32 {
        self.inner.config.policy_target_temperature
    }

    #[must_use]
    pub fn seed(&self) -> u64 {
        self.inner.seed()
    }

    pub fn set_simulations(&mut self, simulations: u32) -> PyResult<()> {
        self.inner
            .set_simulations(simulations)
            .map_err(gumbel_error_to_pyerr)
    }

    pub fn set_max_considered_actions(&mut self, max_considered_actions: usize) -> PyResult<()> {
        self.inner
            .set_max_considered_actions(max_considered_actions)
            .map_err(gumbel_error_to_pyerr)
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.inner.set_seed(seed);
    }

    pub fn set_gumbel_scale(&mut self, gumbel_scale: f32) -> PyResult<()> {
        self.inner
            .set_gumbel_scale(gumbel_scale)
            .map_err(gumbel_error_to_pyerr)
    }

    pub fn search_with_logits(
        &mut self,
        state: &GameState,
        policy_logits: Vec<f32>,
    ) -> PyResult<GumbelResult> {
        self.inner
            .search_with_logits(&state.inner, &policy_logits)
            .map(GumbelResult::from_inner)
            .map_err(gumbel_error_to_pyerr)
    }

    #[pyo3(signature = (state, policy_logits, evaluator, root_value, leaf_batch_size = 16))]
    pub fn search_with_logits_and_evaluator(
        &mut self,
        state: &GameState,
        policy_logits: Vec<f32>,
        evaluator: &Bound<'_, PyAny>,
        root_value: f32,
        leaf_batch_size: usize,
    ) -> PyResult<GumbelResult> {
        if leaf_batch_size == 0 {
            return Err(PyValueError::new_err("leaf_batch_size must be positive"));
        }
        let mut adapter = PythonGumbelEvaluator::new(evaluator);
        self.inner
            .result_from_logits_with_evaluator(
                &state.inner,
                &policy_logits,
                leaf_batch_size,
                root_value,
                |request| adapter.evaluate(request),
            )
            .map(GumbelResult::from_inner)
            .map_err(gumbel_error_to_pyerr)
    }

    pub fn search_with_priors(
        &mut self,
        state: &GameState,
        priors: Vec<f32>,
    ) -> PyResult<GumbelResult> {
        self.inner
            .search_with_priors(&state.inner, &priors)
            .map(GumbelResult::from_inner)
            .map_err(gumbel_error_to_pyerr)
    }

    #[pyo3(signature = (state, priors, evaluator, root_value, leaf_batch_size = 16))]
    pub fn search_with_priors_and_evaluator(
        &mut self,
        state: &GameState,
        priors: Vec<f32>,
        evaluator: &Bound<'_, PyAny>,
        root_value: f32,
        leaf_batch_size: usize,
    ) -> PyResult<GumbelResult> {
        if leaf_batch_size == 0 {
            return Err(PyValueError::new_err("leaf_batch_size must be positive"));
        }
        let mut adapter = PythonGumbelEvaluator::new(evaluator);
        self.inner
            .result_from_priors_with_evaluator(
                &state.inner,
                &priors,
                leaf_batch_size,
                root_value,
                |request| adapter.evaluate(request),
            )
            .map(GumbelResult::from_inner)
            .map_err(gumbel_error_to_pyerr)
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct GumbelSelfPlayBatch {
    inner: CoreSelfPlayBatch,
}

#[pymethods]
impl GumbelSelfPlayBatch {
    #[new]
    #[pyo3(signature = (
        game_count,
        simulations = 128,
        max_considered_actions = 16,
        c_visit = 50.0,
        c_scale = 1.0,
        seed = 2026,
        gumbel_scale = 1.0,
        policy_target_temperature = 1.0,
        policy_target_c_visit = None,
        policy_target_c_scale = None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn py_new(
        game_count: usize,
        simulations: u32,
        max_considered_actions: usize,
        c_visit: f32,
        c_scale: f32,
        seed: u64,
        gumbel_scale: f32,
        policy_target_temperature: f32,
        policy_target_c_visit: Option<f32>,
        policy_target_c_scale: Option<f32>,
    ) -> PyResult<Self> {
        CoreSelfPlayBatch::py_new(
            game_count,
            simulations,
            max_considered_actions,
            c_visit,
            c_scale,
            seed,
            gumbel_scale,
            policy_target_temperature,
            policy_target_c_visit,
            policy_target_c_scale,
        )
        .map(|inner| Self { inner })
        .map_err(gumbel_error_to_pyerr)
    }

    #[staticmethod]
    #[pyo3(signature = (
        action_histories,
        simulations = 128,
        max_considered_actions = 16,
        c_visit = 50.0,
        c_scale = 1.0,
        seed = 2026,
        gumbel_scale = 1.0,
        policy_target_temperature = 1.0,
        policy_target_c_visit = None,
        policy_target_c_scale = None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn from_action_histories(
        action_histories: Vec<Vec<usize>>,
        simulations: u32,
        max_considered_actions: usize,
        c_visit: f32,
        c_scale: f32,
        seed: u64,
        gumbel_scale: f32,
        policy_target_temperature: f32,
        policy_target_c_visit: Option<f32>,
        policy_target_c_scale: Option<f32>,
    ) -> PyResult<Self> {
        CoreSelfPlayBatch::from_action_histories(
            action_histories,
            simulations,
            max_considered_actions,
            c_visit,
            c_scale,
            seed,
            gumbel_scale,
            policy_target_temperature,
            policy_target_c_visit,
            policy_target_c_scale,
        )
        .map(|inner| Self { inner })
        .map_err(gumbel_error_to_pyerr)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[must_use]
    pub fn active_count(&self) -> usize {
        self.inner.active_count()
    }

    #[must_use]
    pub fn active_game_indexes(&self) -> Vec<usize> {
        self.inner.active_game_indexes()
    }

    #[must_use]
    pub fn active_eval_request(&self) -> EvalRequest {
        EvalRequest::from_inner(self.inner.active_eval_request())
    }

    pub fn feature_rows_for_game_indexes(
        &self,
        game_indexes: Vec<usize>,
    ) -> PyResult<Vec<Vec<f32>>> {
        self.inner
            .feature_rows_for_game_indexes(game_indexes)
            .map_err(gumbel_error_to_pyerr)
    }

    #[must_use]
    pub fn current_players(&self) -> Vec<u8> {
        self.inner.current_players()
    }

    #[must_use]
    pub fn is_terminal(&self) -> Vec<bool> {
        self.inner.is_terminal()
    }

    #[must_use]
    pub fn winners(&self) -> Vec<Option<u8>> {
        self.inner.winners()
    }

    #[must_use]
    pub fn end_reasons(&self) -> Vec<Option<u8>> {
        self.inner.end_reasons()
    }

    #[must_use]
    pub fn territory_scores(&self) -> Vec<(u8, u8)> {
        self.inner.territory_scores()
    }

    #[must_use]
    pub fn boards(&self) -> Vec<Vec<u8>> {
        self.inner.boards()
    }

    pub fn search_active_with_logits(
        &mut self,
        policy_logits: Vec<Vec<f32>>,
    ) -> PyResult<Vec<Option<GumbelResult>>> {
        self.inner
            .search_active_with_logits(policy_logits)
            .map(map_results)
            .map_err(gumbel_error_to_pyerr)
    }

    #[pyo3(signature = (policy_logits, evaluator, root_values, leaf_batch_size = 16))]
    pub fn search_active_with_logits_and_evaluator(
        &mut self,
        policy_logits: Vec<Vec<f32>>,
        evaluator: &Bound<'_, PyAny>,
        root_values: Vec<f32>,
        leaf_batch_size: usize,
    ) -> PyResult<Vec<Option<GumbelResult>>> {
        if leaf_batch_size == 0 {
            return Err(PyValueError::new_err("leaf_batch_size must be positive"));
        }
        let mut adapter = PythonGumbelEvaluator::new(evaluator);
        self.inner
            .search_active_with_evaluator(
                policy_logits,
                true,
                &mut adapter,
                leaf_batch_size,
                &root_values,
            )
            .map(map_results)
            .map_err(gumbel_error_to_pyerr)
    }

    #[pyo3(signature = (evaluator, leaf_batch_size = 16))]
    pub fn search_active_with_onnx_evaluator(
        &mut self,
        mut evaluator: PyRefMut<'_, OnnxEvaluator>,
        leaf_batch_size: usize,
    ) -> PyResult<Vec<Option<GumbelResult>>> {
        self.inner
            .search_active_with_onnx_evaluator(&mut evaluator.inner, leaf_batch_size)
            .map(map_results)
            .map_err(gumbel_error_to_pyerr)
    }

    #[pyo3(signature = (evaluator, leaf_batch_size = 16))]
    pub fn search_active_with_onnx_evaluator_and_root_logits(
        &mut self,
        mut evaluator: PyRefMut<'_, OnnxEvaluator>,
        leaf_batch_size: usize,
    ) -> PyResult<(Vec<Option<GumbelResult>>, Vec<Vec<f32>>)> {
        self.inner
            .search_active_with_onnx_evaluator_and_root_logits(&mut evaluator.inner, leaf_batch_size)
            .map(map_results_with_root_logits)
            .map_err(gumbel_error_to_pyerr)
    }

    #[pyo3(signature = (evaluator, root_logit_game_indexes, leaf_batch_size = 16))]
    pub fn search_active_with_onnx_evaluator_and_selected_root_logits(
        &mut self,
        mut evaluator: PyRefMut<'_, OnnxEvaluator>,
        root_logit_game_indexes: Vec<usize>,
        leaf_batch_size: usize,
    ) -> PyResult<(Vec<Option<GumbelResult>>, Vec<Vec<f32>>)> {
        self.inner
            .search_active_with_onnx_evaluator_and_selected_root_logits(
                &mut evaluator.inner,
                root_logit_game_indexes,
                leaf_batch_size,
            )
            .map(map_results_with_root_logits)
            .map_err(gumbel_error_to_pyerr)
    }

    #[pyo3(signature = (raw_evaluator, ema_evaluator, raw_players, leaf_batch_size = 16))]
    pub fn search_active_with_onnx_raw_ema_evaluators_and_root_logits(
        &mut self,
        mut raw_evaluator: PyRefMut<'_, OnnxEvaluator>,
        mut ema_evaluator: PyRefMut<'_, OnnxEvaluator>,
        raw_players: Vec<u8>,
        leaf_batch_size: usize,
    ) -> PyResult<(Vec<Option<GumbelResult>>, Vec<Vec<f32>>)> {
        self.inner
            .search_active_with_onnx_raw_ema_evaluators_and_root_logits(
                &mut raw_evaluator.inner,
                &mut ema_evaluator.inner,
                raw_players,
                leaf_batch_size,
            )
            .map(map_results_with_root_logits)
            .map_err(gumbel_error_to_pyerr)
    }

    #[pyo3(signature = (
        raw_evaluator,
        ema_evaluator,
        raw_players,
        root_logit_game_indexes,
        leaf_batch_size = 16
    ))]
    pub fn search_active_with_onnx_raw_ema_evaluators_and_selected_root_logits(
        &mut self,
        mut raw_evaluator: PyRefMut<'_, OnnxEvaluator>,
        mut ema_evaluator: PyRefMut<'_, OnnxEvaluator>,
        raw_players: Vec<u8>,
        root_logit_game_indexes: Vec<usize>,
        leaf_batch_size: usize,
    ) -> PyResult<(Vec<Option<GumbelResult>>, Vec<Vec<f32>>)> {
        self.inner
            .search_active_with_onnx_raw_ema_evaluators_and_selected_root_logits(
                &mut raw_evaluator.inner,
                &mut ema_evaluator.inner,
                raw_players,
                root_logit_game_indexes,
                leaf_batch_size,
            )
            .map(map_results_with_root_logits)
            .map_err(gumbel_error_to_pyerr)
    }

    pub fn search_active_with_priors(
        &mut self,
        priors: Vec<Vec<f32>>,
    ) -> PyResult<Vec<Option<GumbelResult>>> {
        self.inner
            .search_active_with_priors(priors)
            .map(map_results)
            .map_err(gumbel_error_to_pyerr)
    }

    #[pyo3(signature = (priors, evaluator, root_values, leaf_batch_size = 16))]
    pub fn search_active_with_priors_and_evaluator(
        &mut self,
        priors: Vec<Vec<f32>>,
        evaluator: &Bound<'_, PyAny>,
        root_values: Vec<f32>,
        leaf_batch_size: usize,
    ) -> PyResult<Vec<Option<GumbelResult>>> {
        if leaf_batch_size == 0 {
            return Err(PyValueError::new_err("leaf_batch_size must be positive"));
        }
        let mut adapter = PythonGumbelEvaluator::new(evaluator);
        self.inner
            .search_active_with_evaluator(
                priors,
                false,
                &mut adapter,
                leaf_batch_size,
                &root_values,
            )
            .map(map_results)
            .map_err(gumbel_error_to_pyerr)
    }

    pub fn apply_actions(&mut self, actions: Vec<Option<usize>>) -> PyResult<Vec<Option<u8>>> {
        self.inner
            .apply_actions(actions)
            .map_err(gumbel_error_to_pyerr)
    }

    pub fn set_simulations(&mut self, simulations: Vec<Option<u32>>) -> PyResult<()> {
        self.inner
            .set_simulations(simulations)
            .map_err(gumbel_error_to_pyerr)
    }

    pub fn set_max_considered_actions(
        &mut self,
        max_considered_actions: Vec<Option<usize>>,
    ) -> PyResult<()> {
        self.inner
            .set_max_considered_actions(max_considered_actions)
            .map_err(gumbel_error_to_pyerr)
    }

    pub fn set_seeds(&mut self, seeds: Vec<Option<u64>>) -> PyResult<()> {
        self.inner.set_seeds(seeds).map_err(gumbel_error_to_pyerr)
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct GumbelArenaBatch {
    inner: CoreArenaBatch,
}

#[pymethods]
impl GumbelArenaBatch {
    #[new]
    #[pyo3(signature = (
        game_count,
        seed_start = 0,
        game_index_start = 0,
        simulations = 128,
        max_considered_actions = 16,
        c_visit = 50.0,
        c_scale = 1.0,
        seed = 2026,
        gumbel_scale = 1.0,
        policy_target_temperature = 1.0,
        policy_target_c_visit = None,
        policy_target_c_scale = None,
        paired_seeds = false,
        candidate_c_scale = None,
        candidate_policy_target_c_scale = None,
        best_c_scale = None,
        best_policy_target_c_scale = None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn py_new(
        game_count: usize,
        seed_start: u64,
        game_index_start: usize,
        simulations: u32,
        max_considered_actions: usize,
        c_visit: f32,
        c_scale: f32,
        seed: u64,
        gumbel_scale: f32,
        policy_target_temperature: f32,
        policy_target_c_visit: Option<f32>,
        policy_target_c_scale: Option<f32>,
        paired_seeds: bool,
        candidate_c_scale: Option<f32>,
        candidate_policy_target_c_scale: Option<f32>,
        best_c_scale: Option<f32>,
        best_policy_target_c_scale: Option<f32>,
    ) -> PyResult<Self> {
        CoreArenaBatch::py_new(
            game_count,
            seed_start,
            game_index_start,
            simulations,
            max_considered_actions,
            c_visit,
            c_scale,
            seed,
            gumbel_scale,
            policy_target_temperature,
            policy_target_c_visit,
            policy_target_c_scale,
            paired_seeds,
            candidate_c_scale,
            candidate_policy_target_c_scale,
            best_c_scale,
            best_policy_target_c_scale,
        )
        .map(|inner| Self { inner })
        .map_err(gumbel_error_to_pyerr)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[must_use]
    pub fn active_count(&self) -> usize {
        self.inner.active_count()
    }

    #[must_use]
    pub fn active_game_indexes(&self) -> Vec<usize> {
        self.inner.active_game_indexes()
    }

    #[must_use]
    pub fn active_eval_request(&self) -> EvalRequest {
        EvalRequest::from_inner(self.inner.active_eval_request())
    }

    #[must_use]
    pub fn active_legal_masks(&self) -> Vec<Vec<bool>> {
        self.inner.active_legal_masks()
    }

    #[must_use]
    pub fn current_players(&self) -> Vec<u8> {
        self.inner.current_players()
    }

    #[must_use]
    pub fn candidate_players(&self) -> Vec<u8> {
        self.inner.candidate_players()
    }

    #[must_use]
    pub fn seeds(&self) -> Vec<u64> {
        self.inner.seeds()
    }

    #[must_use]
    pub fn is_terminal(&self) -> Vec<bool> {
        self.inner.is_terminal()
    }

    #[must_use]
    pub fn winners(&self) -> Vec<Option<u8>> {
        self.inner.winners()
    }

    #[must_use]
    pub fn end_reasons(&self) -> Vec<Option<u8>> {
        self.inner.end_reasons()
    }

    #[must_use]
    pub fn territory_scores(&self) -> Vec<(u8, u8)> {
        self.inner.territory_scores()
    }

    #[pyo3(signature = (policy_logits, evaluator, root_values, leaf_batch_size = 16))]
    pub fn search_active_with_logits_and_evaluator(
        &mut self,
        policy_logits: Vec<Vec<f32>>,
        evaluator: &Bound<'_, PyAny>,
        root_values: Vec<f32>,
        leaf_batch_size: usize,
    ) -> PyResult<Vec<Option<GumbelResult>>> {
        if leaf_batch_size == 0 {
            return Err(PyValueError::new_err("leaf_batch_size must be positive"));
        }
        let mut adapter = PythonArenaLeafEvaluator::new(evaluator);
        self.inner
            .search_active_with_logits_evaluator(
                policy_logits,
                &root_values,
                leaf_batch_size,
                &mut adapter,
            )
            .map(map_results)
            .map_err(gumbel_error_to_pyerr)
    }

    #[pyo3(signature = (candidate_evaluator, best_evaluator, leaf_batch_size = 16))]
    pub fn search_active_with_onnx_evaluators(
        &mut self,
        mut candidate_evaluator: PyRefMut<'_, OnnxEvaluator>,
        mut best_evaluator: PyRefMut<'_, OnnxEvaluator>,
        leaf_batch_size: usize,
    ) -> PyResult<(Vec<Option<GumbelResult>>, Vec<Vec<f32>>)> {
        self.inner
            .search_active_with_onnx_evaluators(
                &mut candidate_evaluator.inner,
                &mut best_evaluator.inner,
                leaf_batch_size,
            )
            .map(map_results_with_root_logits)
            .map_err(gumbel_error_to_pyerr)
    }

    pub fn apply_actions(&mut self, actions: Vec<Option<usize>>) -> PyResult<Vec<Option<u8>>> {
        self.inner
            .apply_actions(actions)
            .map_err(gumbel_error_to_pyerr)
    }

    pub fn set_gumbel_scale(&mut self, gumbel_scale: f32) -> PyResult<()> {
        self.inner
            .set_gumbel_scale(gumbel_scale)
            .map_err(gumbel_error_to_pyerr)
    }
}
