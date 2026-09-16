//! PyO3 binding for the pure ONNX evaluator.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use great_kingdom_onnx::{OnnxEvaluator as CoreOnnxEvaluator, OnnxEvaluatorConfig, parse_device};

use crate::errors::onnx_error_to_runtime;
use crate::eval_request::EvalRequest;

#[pyclass(unsendable)]
pub struct OnnxEvaluator {
    pub(crate) inner: CoreOnnxEvaluator,
}

#[pymethods]
impl OnnxEvaluator {
    #[new]
    #[pyo3(signature = (path, device = "cpu", max_batch_size = 256))]
    fn py_new(path: &str, device: &str, max_batch_size: usize) -> PyResult<Self> {
        let device = parse_device(device).map_err(|err| PyValueError::new_err(err.to_string()))?;
        CoreOnnxEvaluator::load(
            path,
            OnnxEvaluatorConfig {
                device,
                max_batch_size,
            },
        )
        .map(|inner| Self { inner })
        .map_err(onnx_error_to_runtime)
    }

    #[must_use]
    fn device(&self) -> &'static str {
        self.inner.device()
    }

    #[must_use]
    fn max_batch_size(&self) -> usize {
        self.inner.max_batch_size()
    }

    fn evaluate(&mut self, request: &EvalRequest) -> PyResult<(Vec<Vec<f32>>, Vec<f32>)> {
        self.inner
            .evaluate(&request.inner)
            .map_err(onnx_error_to_runtime)
    }

    fn evaluate_with_q(
        &mut self,
        request: &EvalRequest,
    ) -> PyResult<(Vec<Vec<f32>>, Vec<f32>, Vec<Vec<f32>>)> {
        self.inner
            .evaluate_with_q(&request.inner)
            .map_err(onnx_error_to_runtime)
    }

    #[must_use]
    fn has_q_values_output(&self) -> bool {
        self.inner.has_q_values_output()
    }

    #[must_use]
    fn output_names(&self) -> Vec<String> {
        self.inner.output_names()
    }
}
