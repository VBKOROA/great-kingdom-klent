//! ONNX Runtime inference infrastructure: session pooling, device selection,
//! dynamic batching, and output parsing. No PyO3 dependency.

mod evaluator;
mod profile;

pub use evaluator::{
    NetworkOutput, OnnxDevice, OnnxError, OnnxEvaluator, OnnxEvaluatorConfig, parse_device,
};
