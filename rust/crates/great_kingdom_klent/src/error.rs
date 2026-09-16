//! Error type for the KLENT zero-search domain.

use std::fmt;

use great_kingdom_onnx::OnnxError;

#[derive(Debug)]
pub enum KlentError {
    /// Invalid constructor arguments or out-of-range accessors (maps to
    /// `ValueError` at the Python boundary).
    InvalidInput(String),
    /// Runtime failure during self-play collection (maps to `RuntimeError`).
    Runtime(String),
    /// Underlying ONNX Runtime failure.
    Onnx(OnnxError),
}

impl fmt::Display for KlentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) | Self::Runtime(message) => formatter.write_str(message),
            Self::Onnx(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for KlentError {}

impl From<OnnxError> for KlentError {
    fn from(error: OnnxError) -> Self {
        Self::Onnx(error)
    }
}
