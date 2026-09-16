use great_kingdom_features::EvalRequest;
use great_kingdom_onnx::OnnxEvaluator;

use super::search::GumbelEvalBatch;
use crate::error::GumbelError;

/// Abstraction over a leaf evaluator (ONNX Runtime or a host-provided
/// callback) used by the batched Gumbel search.
pub trait GumbelEvaluator {
    fn check_signals(&self) -> Result<(), GumbelError> {
        Ok(())
    }

    fn needs_legal_masks(&self) -> bool {
        true
    }

    fn needs_game_indexes(&self) -> bool {
        false
    }

    fn set_batch_profile_context(&mut self, _wave: u64, _active_games: usize, _leaves: usize) {}

    fn evaluate(&mut self, request: EvalRequest) -> Result<GumbelEvalBatch, GumbelError>;
}

pub(crate) struct OnnxGumbelEvaluator<'a> {
    evaluator: &'a mut OnnxEvaluator,
}

impl<'a> OnnxGumbelEvaluator<'a> {
    pub(crate) const fn new(evaluator: &'a mut OnnxEvaluator) -> Self {
        Self { evaluator }
    }
}

impl GumbelEvaluator for OnnxGumbelEvaluator<'_> {
    fn needs_legal_masks(&self) -> bool {
        false
    }

    fn set_batch_profile_context(&mut self, wave: u64, active_games: usize, leaves: usize) {
        self.evaluator
            .set_gumbel_leaf_profile_context(wave, active_games, leaves);
    }

    fn evaluate(&mut self, request: EvalRequest) -> Result<GumbelEvalBatch, GumbelError> {
        let output = self.evaluator.evaluate_request(&request)?;
        Ok(GumbelEvalBatch::new(output.policy_logits, output.values))
    }
}
