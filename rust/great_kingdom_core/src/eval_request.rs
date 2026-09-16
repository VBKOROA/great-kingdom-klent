//! PyO3 binding for the pure evaluation request container.

use std::borrow::Cow;

use pyo3::prelude::*;
use pyo3::types::PyBytes;

use great_kingdom_features::EvalRequest as CoreEvalRequest;

use crate::errors::feature_error_to_pyerr;

#[pyclass]
#[derive(Clone, Debug)]
pub struct EvalRequest {
    pub(crate) inner: CoreEvalRequest,
}

impl EvalRequest {
    pub(crate) const fn from_inner(inner: CoreEvalRequest) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl EvalRequest {
    #[staticmethod]
    pub fn from_feature_rows(feature_rows: Vec<Vec<f32>>) -> PyResult<Self> {
        CoreEvalRequest::from_feature_rows(feature_rows)
            .map(Self::from_inner)
            .map_err(feature_error_to_pyerr)
    }

    #[staticmethod]
    pub fn from_feature_plane_bytes(
        row_count: usize,
        feature_bytes: &Bound<'_, PyBytes>,
    ) -> PyResult<Self> {
        CoreEvalRequest::from_feature_plane_bytes(row_count, feature_bytes.as_bytes())
            .map(Self::from_inner)
            .map_err(feature_error_to_pyerr)
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
    pub fn feature_planes(&self) -> Vec<Vec<f32>> {
        self.inner.feature_planes()
    }

    #[must_use]
    pub fn feature_plane_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        match self.inner.feature_plane_bytes() {
            Cow::Borrowed(bytes) => PyBytes::new(py, bytes),
            Cow::Owned(bytes) => PyBytes::new(py, &bytes),
        }
    }

    #[must_use]
    pub fn legal_masks(&self) -> Vec<Vec<bool>> {
        self.inner.legal_masks()
    }

    #[must_use]
    pub fn legal_mask_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        match self.inner.legal_mask_bytes() {
            Cow::Borrowed(bytes) => PyBytes::new(py, bytes),
            Cow::Owned(bytes) => PyBytes::new(py, &bytes),
        }
    }

    #[must_use]
    pub fn current_players(&self) -> Vec<u8> {
        self.inner.current_players()
    }

    #[must_use]
    pub fn game_indexes(&self) -> Vec<usize> {
        self.inner.game_indexes()
    }
}
