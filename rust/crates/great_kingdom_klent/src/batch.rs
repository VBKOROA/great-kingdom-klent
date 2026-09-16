//! KLENT zero-search batched self-play actor.
//!
//! Each turn evaluates every active game state in one batched forward pass,
//! computes the closed-form target policy `pi'` and `E_pi'[Q]` in FP32, and
//! samples an action from `pi'`. Lambda returns are left to the caller, which
//! owns the shared target implementation.

use great_kingdom_engine::game::{Action, GameState};
use great_kingdom_features::EvalRequest;
use great_kingdom_onnx::{OnnxEvaluator, OnnxEvaluatorConfig, parse_device};

use crate::error::KlentError;
use crate::rng::SplitMix64;
use crate::targets::{analytical_policy_row, masked_value_row};

#[derive(Clone, Debug)]
pub(crate) struct KlentTurn {
    player: u8,
    action: usize,
    policy_target: Vec<f32>,
    state_value: f32,
}

pub struct KlentZeroSearchBatch {
    evaluator: OnnxEvaluator,
    games: Vec<GameState>,
    rngs: Vec<SplitMix64>,
    records: Vec<Vec<KlentTurn>>,
    seeds: Vec<u64>,
    alpha: f32,
    beta: f32,
    max_turns: usize,
    ran: bool,
}

impl KlentZeroSearchBatch {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        path: &str,
        games: usize,
        seed: u64,
        device: &str,
        max_batch_size: usize,
        alpha: f32,
        beta: f32,
        max_turns: usize,
    ) -> Result<Self, KlentError> {
        validate_klent_coefficients(alpha, beta)?;
        if games == 0 {
            return Err(KlentError::InvalidInput("games must be positive".to_string()));
        }
        if max_batch_size == 0 {
            return Err(KlentError::InvalidInput(
                "max_batch_size must be positive".to_string(),
            ));
        }
        if max_turns == 0 {
            return Err(KlentError::InvalidInput(
                "max_turns must be positive".to_string(),
            ));
        }

        let evaluator = OnnxEvaluator::load(
            path,
            OnnxEvaluatorConfig {
                device: parse_device(device)?,
                max_batch_size,
            },
        )
        .map_err(|err| KlentError::Runtime(err.to_string()))?;
        if !evaluator.supports_q_values_output() {
            return Err(KlentError::InvalidInput(
                "zero-search actor requires a 3-output ONNX model with a 'q_values' output"
                    .to_string(),
            ));
        }

        let seeds = (0..games)
            .map(|game_index| seed.wrapping_add(game_index as u64))
            .collect::<Vec<_>>();
        Ok(Self {
            evaluator,
            games: vec![GameState::new(); games],
            rngs: seeds.iter().copied().map(SplitMix64::new).collect(),
            records: vec![Vec::new(); games],
            seeds,
            alpha,
            beta,
            max_turns,
            ran: false,
        })
    }

    #[must_use]
    pub fn game_count(&self) -> usize {
        self.games.len()
    }

    #[must_use]
    pub fn seeds(&self) -> Vec<u64> {
        self.seeds.clone()
    }

    pub fn run(&mut self) -> Result<(), KlentError> {
        if self.ran {
            return Err(KlentError::InvalidInput(
                "KlentZeroSearchBatch has already run".to_string(),
            ));
        }
        for _turn in 0..self.max_turns {
            let active = (0..self.games.len())
                .filter(|game_index| !self.games[*game_index].is_terminal())
                .collect::<Vec<_>>();
            if active.is_empty() {
                break;
            }

            let request = EvalRequest::new_with_precomputed_features(
                active
                    .iter()
                    .map(|game_index| self.games[*game_index].clone())
                    .collect(),
            );
            let output = self.evaluator.evaluate_request_with_q(&request)?;
            let q_values = output
                .q_values
                .expect("q values are validated at construction");

            for (row, game_index) in active.iter().enumerate() {
                let legal_mask = self.games[*game_index].legal_mask();
                let policy_target = analytical_policy_row(
                    &output.policy_logits[row],
                    &q_values[row],
                    &legal_mask,
                    self.alpha,
                    self.beta,
                )?;
                let state_value = masked_value_row(&policy_target, &q_values[row]);
                let action = sample_from_target(&policy_target, &mut self.rngs[*game_index]);
                let player = self.games[*game_index].current_player();
                self.games[*game_index]
                    .apply(Action::from_index(action).expect("sampled action is in range"))
                    .map_err(|err| {
                        KlentError::Runtime(format!("invalid sampled action: {err:?}"))
                    })?;
                self.records[*game_index].push(KlentTurn {
                    player,
                    action,
                    policy_target,
                    state_value,
                });
            }
        }

        if let Some(game_index) = self.games.iter().position(|game| !game.is_terminal()) {
            return Err(KlentError::Runtime(format!(
                "game {game_index} exceeded max_turns={} without a terminal outcome",
                self.max_turns
            )));
        }
        self.ran = true;
        Ok(())
    }

    #[must_use]
    pub fn winners(&self) -> Vec<Option<u8>> {
        self.games.iter().map(GameState::winner).collect()
    }

    #[must_use]
    pub fn end_reasons(&self) -> Vec<Option<u8>> {
        self.games.iter().map(GameState::end_reason).collect()
    }

    #[must_use]
    pub fn territory_scores(&self) -> Vec<(u8, u8)> {
        self.games.iter().map(GameState::territory_scores).collect()
    }

    #[must_use]
    pub fn move_counts(&self) -> Vec<usize> {
        self.records.iter().map(Vec::len).collect()
    }

    pub fn turn_players(&self, game_index: usize) -> Result<Vec<u8>, KlentError> {
        Ok(self
            .game_records(game_index)?
            .iter()
            .map(|t| t.player)
            .collect())
    }

    pub fn turn_actions(&self, game_index: usize) -> Result<Vec<usize>, KlentError> {
        Ok(self
            .game_records(game_index)?
            .iter()
            .map(|t| t.action)
            .collect())
    }

    pub fn policy_targets(&self, game_index: usize) -> Result<Vec<Vec<f32>>, KlentError> {
        Ok(self
            .game_records(game_index)?
            .iter()
            .map(|t| t.policy_target.clone())
            .collect())
    }

    pub fn state_values(&self, game_index: usize) -> Result<Vec<f32>, KlentError> {
        Ok(self
            .game_records(game_index)?
            .iter()
            .map(|t| t.state_value)
            .collect())
    }

    fn game_records(&self, game_index: usize) -> Result<&[KlentTurn], KlentError> {
        self.records
            .get(game_index)
            .map(Vec::as_slice)
            .ok_or_else(|| {
                KlentError::InvalidInput(format!("game index {game_index} is out of range"))
            })
    }
}

pub(crate) fn validate_klent_coefficients(alpha: f32, beta: f32) -> Result<(), KlentError> {
    if !alpha.is_finite() || alpha < 0.0 {
        return Err(KlentError::InvalidInput(format!(
            "alpha must be finite and non-negative, got {alpha}"
        )));
    }
    if !beta.is_finite() || beta < 0.0 {
        return Err(KlentError::InvalidInput(format!(
            "beta must be finite and non-negative, got {beta}"
        )));
    }
    if alpha + beta <= 0.0 {
        return Err(KlentError::InvalidInput(
            "alpha + beta must be positive".to_string(),
        ));
    }
    Ok(())
}

fn sample_from_target(target: &[f32], rng: &mut SplitMix64) -> usize {
    let total: f32 = target.iter().sum();
    let mut threshold = rng.next_open01() * total;
    let mut fallback = 0;
    for (index, probability) in target.iter().enumerate() {
        if *probability <= 0.0 {
            continue;
        }
        fallback = index;
        threshold -= *probability;
        if threshold <= 0.0 {
            return index;
        }
    }
    fallback
}

#[cfg(test)]
mod tests {
    use super::*;
    use great_kingdom_engine::game::ACTION_SPACE;

    #[test]
    fn sampler_stays_on_legal_actions() {
        let mut target = vec![0.0; ACTION_SPACE];
        target[10] = 0.25;
        target[81] = 0.75;
        let mut rng = SplitMix64::new(7);

        for _ in 0..64 {
            let action = sample_from_target(&target, &mut rng);
            assert!(action == 10 || action == 81);
        }
    }

    #[test]
    fn coefficient_validation_rejects_invalid_values() {
        assert!(validate_klent_coefficients(-0.1, 0.1).is_err());
        assert!(validate_klent_coefficients(0.0, 0.0).is_err());
        assert!(validate_klent_coefficients(f32::NAN, 0.1).is_err());
        assert!(validate_klent_coefficients(0.03, 0.1).is_ok());
    }
}
