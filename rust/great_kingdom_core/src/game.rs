//! PyO3 binding for the pure game-state engine type.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use great_kingdom_engine::game::{
    Action, BOARD_SIZE, GameState as CoreGameState,
};
use great_kingdom_features::{FEATURE_CHANNELS, GameStateFeatures};

#[pyclass]
#[derive(Clone, Debug)]
pub struct GameState {
    pub(crate) inner: CoreGameState,
}

#[pymethods]
impl GameState {
    #[new]
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: CoreGameState::new(),
        }
    }

    #[must_use]
    pub fn current_player(&self) -> u8 {
        self.inner.current_player()
    }

    #[must_use]
    pub fn blue_used(&self) -> u8 {
        self.inner.blue_used()
    }

    #[must_use]
    pub fn orange_used(&self) -> u8 {
        self.inner.orange_used()
    }

    #[must_use]
    pub fn previous_pass(&self) -> bool {
        self.inner.previous_pass()
    }

    #[must_use]
    pub fn board(&self) -> Vec<u8> {
        self.inner.board()
    }

    #[must_use]
    pub fn cell_at(&self, index: usize) -> Option<u8> {
        self.inner.cell_at(index)
    }

    #[must_use]
    pub fn legal_actions(&self) -> Vec<usize> {
        self.inner.legal_actions()
    }

    #[must_use]
    pub fn legal_mask(&self) -> Vec<bool> {
        self.inner.legal_mask()
    }

    #[must_use]
    pub fn feature_planes(&self) -> Vec<f32> {
        self.inner.feature_planes()
    }

    #[must_use]
    pub fn feature_shape(&self) -> (usize, usize, usize) {
        (FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    }

    #[must_use]
    pub fn territory_scores(&self) -> (u8, u8) {
        self.inner.territory_scores()
    }

    pub fn apply_action(&mut self, action_index: usize) -> PyResult<Option<u8>> {
        self.inner
            .apply(Action::from_index(action_index).ok_or_else(|| {
                PyValueError::new_err(format!("invalid action index: {action_index}"))
            })?)
            .map(|outcome| outcome.map(|outcome| outcome.winner as u8))
            .map_err(|err| PyValueError::new_err(format!("invalid action: {err:?}")))
    }

    #[must_use]
    pub fn is_terminal(&self) -> bool {
        self.inner.is_terminal()
    }

    #[must_use]
    pub fn winner(&self) -> Option<u8> {
        self.inner.winner()
    }

    #[must_use]
    pub fn end_reason(&self) -> Option<u8> {
        self.inner.end_reason()
    }
}
