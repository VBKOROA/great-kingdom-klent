//! Error type for the pure Gumbel search domain.

use std::fmt;

/// Error produced by Gumbel search and configuration operations.
///
/// `External` carries opaque errors coming from evaluator callbacks (for
/// example a Python exception raised by a user-supplied evaluator). The facade
/// can downcast it back to the original error to preserve tracebacks.
#[derive(Debug)]
pub enum GumbelError {
    /// Invalid input or configuration (maps to `ValueError`).
    Message(String),
    /// Internal runtime failure (maps to `RuntimeError`).
    Runtime(String),
    External(Box<dyn std::error::Error + Send + Sync>),
}

impl GumbelError {
    #[must_use]
    pub fn message(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }

    #[must_use]
    pub fn runtime(message: impl Into<String>) -> Self {
        Self::Runtime(message.into())
    }

    #[must_use]
    pub fn external(error: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::External(Box::new(error))
    }
}

impl fmt::Display for GumbelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Message(message) | Self::Runtime(message) => formatter.write_str(message),
            Self::External(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for GumbelError {}

impl From<String> for GumbelError {
    fn from(message: String) -> Self {
        Self::Message(message)
    }
}

impl From<&str> for GumbelError {
    fn from(message: &str) -> Self {
        Self::Message(message.to_string())
    }
}

impl From<great_kingdom_onnx::OnnxError> for GumbelError {
    fn from(error: great_kingdom_onnx::OnnxError) -> Self {
        Self::External(Box::new(error))
    }
}

pub type GumbelOutcome<T> = Result<T, GumbelError>;
