//! Conversion of pure domain errors into Python exceptions.
//!
//! The mapping preserves the exception types raised by the previous monolithic
//! extension: configuration/input problems become `ValueError`, runtime
//! failures become `RuntimeError`, and embedded Python exceptions (raised by a
//! user-supplied evaluator) are re-raised unchanged.

use pyo3::PyErr;
use pyo3::exceptions::{PyRuntimeError, PyValueError};

use great_kingdom_features::FeatureError;
use great_kingdom_gumbel::GumbelError;
use great_kingdom_klent::KlentError;
use great_kingdom_onnx::OnnxError;

pub(crate) fn feature_error_to_pyerr(error: FeatureError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

pub(crate) fn onnx_error_to_runtime(error: OnnxError) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

pub(crate) fn gumbel_error_to_pyerr(error: GumbelError) -> PyErr {
    match error {
        GumbelError::Message(message) => PyValueError::new_err(message),
        GumbelError::Runtime(message) => PyRuntimeError::new_err(message),
        GumbelError::External(boxed) => match boxed.downcast::<PyErr>() {
            Ok(pyerr) => *pyerr,
            Err(other) => PyRuntimeError::new_err(other.to_string()),
        },
    }
}

pub(crate) fn klent_error_to_pyerr(error: KlentError) -> PyErr {
    match error {
        KlentError::InvalidInput(message) => PyValueError::new_err(message),
        KlentError::Runtime(message) => PyRuntimeError::new_err(message),
        KlentError::Onnx(error) => PyRuntimeError::new_err(error.to_string()),
    }
}
