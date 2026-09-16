//! Closed-form KLENT policy and value targets.

use great_kingdom_engine::game::ACTION_SPACE;
use great_kingdom_onnx::OnnxError;

pub fn analytical_policy_row(
    policy_logits: &[f32],
    q_values: &[f32],
    legal_mask: &[bool],
    alpha: f32,
    beta: f32,
) -> Result<Vec<f32>, OnnxError> {
    validate_target_inputs(policy_logits, q_values, legal_mask, alpha, beta)?;

    let log_normalizer = legal_log_normalizer(policy_logits, legal_mask)?;
    let denominator = alpha + beta;
    let mut max_score = f32::NEG_INFINITY;
    for index in 0..ACTION_SPACE {
        if !legal_mask[index] {
            continue;
        }
        let log_pi = policy_logits[index] - log_normalizer;
        let score = (q_values[index] + beta * log_pi) / denominator;
        max_score = max_score.max(score);
    }

    let mut target = vec![0.0; ACTION_SPACE];
    let mut exp_sum = 0.0;
    for index in 0..ACTION_SPACE {
        if !legal_mask[index] {
            continue;
        }
        let log_pi = policy_logits[index] - log_normalizer;
        let score = (q_values[index] + beta * log_pi) / denominator;
        let weight = (score - max_score).exp();
        target[index] = weight;
        exp_sum += weight;
    }
    for index in 0..ACTION_SPACE {
        if legal_mask[index] {
            target[index] /= exp_sum;
        }
    }
    Ok(target)
}

pub fn masked_state_value_row(
    policy_logits: &[f32],
    q_values: &[f32],
    legal_mask: &[bool],
) -> Result<f32, OnnxError> {
    validate_target_inputs(policy_logits, q_values, legal_mask, 1.0, 0.0)?;
    let probabilities = legal_softmax(policy_logits, legal_mask)?;
    Ok(masked_value_row(&probabilities, q_values))
}

pub(crate) fn masked_value_row(probabilities: &[f32], q_values: &[f32]) -> f32 {
    probabilities
        .iter()
        .zip(q_values.iter())
        .map(|(probability, q_value)| probability * q_value)
        .sum()
}

fn validate_target_inputs(
    policy_logits: &[f32],
    q_values: &[f32],
    legal_mask: &[bool],
    alpha: f32,
    beta: f32,
) -> Result<(), OnnxError> {
    if policy_logits.len() != ACTION_SPACE || q_values.len() != ACTION_SPACE {
        return Err(OnnxError::InvalidRequest(format!(
            "expected {ACTION_SPACE} policy logits and q values, got {} and {}",
            policy_logits.len(),
            q_values.len()
        )));
    }
    if legal_mask.len() != ACTION_SPACE {
        return Err(OnnxError::InvalidRequest(format!(
            "expected {ACTION_SPACE} legal mask entries, got {}",
            legal_mask.len()
        )));
    }
    crate::batch::validate_klent_coefficients(alpha, beta)
        .map_err(|err| OnnxError::InvalidRequest(err.to_string()))?;
    Ok(())
}

fn legal_log_normalizer(policy_logits: &[f32], legal_mask: &[bool]) -> Result<f32, OnnxError> {
    let max_legal = legal_mask
        .iter()
        .enumerate()
        .filter(|(_, is_legal)| **is_legal)
        .map(|(index, _)| policy_logits[index])
        .fold(f32::NEG_INFINITY, f32::max);
    if !max_legal.is_finite() {
        return Err(OnnxError::InvalidRequest(
            "legal mask must contain at least one legal action".to_string(),
        ));
    }
    let exp_sum: f32 = legal_mask
        .iter()
        .enumerate()
        .filter(|(_, is_legal)| **is_legal)
        .map(|(index, _)| (policy_logits[index] - max_legal).exp())
        .sum();
    Ok(max_legal + exp_sum.ln())
}

fn legal_softmax(policy_logits: &[f32], legal_mask: &[bool]) -> Result<Vec<f32>, OnnxError> {
    let log_normalizer = legal_log_normalizer(policy_logits, legal_mask)?;
    let mut probabilities = vec![0.0; ACTION_SPACE];
    for index in 0..ACTION_SPACE {
        if legal_mask[index] {
            probabilities[index] = (policy_logits[index] - log_normalizer).exp();
        }
    }
    Ok(probabilities)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    fn mask_of(indexes: &[usize]) -> Vec<bool> {
        let mut mask = vec![false; ACTION_SPACE];
        for index in indexes {
            mask[*index] = true;
        }
        mask
    }

    #[test]
    fn analytical_policy_normalizes_over_legal_actions_only() {
        let logits = vec![0.5; ACTION_SPACE];
        let q_values = vec![0.0; ACTION_SPACE];
        let mask = mask_of(&[0, 5, 81]);

        let target = analytical_policy_row(&logits, &q_values, &mask, 0.03, 0.1)
            .expect("target should compute");

        let total: f32 = target.iter().sum();
        assert!((total - 1.0).abs() < 1e-6);
        assert_eq!(target[0], target[5]);
        assert_eq!(target[5], target[81]);
        assert_eq!(target[1], 0.0);
    }

    #[test]
    fn analytical_policy_prefers_larger_q_values() {
        let logits = vec![0.0; ACTION_SPACE];
        let mut q_values = vec![0.0; ACTION_SPACE];
        q_values[3] = 1.0;
        let mask = mask_of(&[3, 7]);

        let target = analytical_policy_row(&logits, &q_values, &mask, 0.03, 0.1)
            .expect("target should compute");

        assert!(target[3] > target[7]);
        assert_eq!(target[0], 0.0);
    }

    #[test]
    fn analytical_policy_rejects_empty_legal_mask() {
        let error = analytical_policy_row(
            &vec![0.0; ACTION_SPACE],
            &vec![0.0; ACTION_SPACE],
            &[false; ACTION_SPACE],
            0.03,
            0.1,
        )
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "legal mask must contain at least one legal action"
        );
    }

    #[test]
    fn masked_state_value_uses_legal_softmax() {
        let mut logits = vec![0.0; ACTION_SPACE];
        logits[0] = 2.0;
        let mut q_values = vec![0.0; ACTION_SPACE];
        q_values[0] = 0.5;
        q_values[1] = -1.0;
        let mask = mask_of(&[0, 1]);

        let value =
            masked_state_value_row(&logits, &q_values, &mask).expect("value should compute");

        let probability_0 = (2.0_f32).exp() / (2.0_f32.exp() + 1.0);
        let expected = probability_0 * 0.5 + (1.0 - probability_0) * -1.0;
        assert!((value - expected).abs() < 1e-6);
    }
}
