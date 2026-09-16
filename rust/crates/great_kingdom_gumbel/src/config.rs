use crate::error::GumbelError;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GumbelConfig {
    pub simulations: u32,
    pub max_considered_actions: usize,
    pub c_visit: f32,
    pub c_scale: f32,
    pub gumbel_scale: f32,
    pub policy_target_c_visit: f32,
    pub policy_target_c_scale: f32,
    pub policy_target_temperature: f32,
    pub seed: u64,
}

impl Default for GumbelConfig {
    fn default() -> Self {
        Self {
            simulations: 128,
            max_considered_actions: 16,
            c_visit: 50.0,
            c_scale: 1.0,
            gumbel_scale: 1.0,
            policy_target_c_visit: 50.0,
            policy_target_c_scale: 1.0,
            policy_target_temperature: 1.0,
            seed: 0,
        }
    }
}

impl GumbelConfig {
    #[must_use]
    pub const fn new(
        simulations: u32,
        max_considered_actions: usize,
        c_visit: f32,
        c_scale: f32,
        seed: u64,
    ) -> Self {
        Self {
            simulations,
            max_considered_actions,
            c_visit,
            c_scale,
            gumbel_scale: 1.0,
            policy_target_c_visit: c_visit,
            policy_target_c_scale: c_scale,
            policy_target_temperature: 1.0,
            seed,
        }
    }

    #[must_use]
    pub const fn new_with_policy_target_temperature(
        simulations: u32,
        max_considered_actions: usize,
        c_visit: f32,
        c_scale: f32,
        seed: u64,
        policy_target_temperature: f32,
    ) -> Self {
        Self {
            simulations,
            max_considered_actions,
            c_visit,
            c_scale,
            gumbel_scale: 1.0,
            policy_target_c_visit: c_visit,
            policy_target_c_scale: c_scale,
            policy_target_temperature,
            seed,
        }
    }

    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub const fn new_with_policy_target_config(
        simulations: u32,
        max_considered_actions: usize,
        c_visit: f32,
        c_scale: f32,
        seed: u64,
        policy_target_temperature: f32,
        policy_target_c_visit: f32,
        policy_target_c_scale: f32,
    ) -> Self {
        Self {
            simulations,
            max_considered_actions,
            c_visit,
            c_scale,
            gumbel_scale: 1.0,
            policy_target_c_visit,
            policy_target_c_scale,
            policy_target_temperature,
            seed,
        }
    }

    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub const fn new_with_full_config(
        simulations: u32,
        max_considered_actions: usize,
        c_visit: f32,
        c_scale: f32,
        seed: u64,
        gumbel_scale: f32,
        policy_target_temperature: f32,
        policy_target_c_visit: f32,
        policy_target_c_scale: f32,
    ) -> Self {
        Self {
            simulations,
            max_considered_actions,
            c_visit,
            c_scale,
            gumbel_scale,
            policy_target_c_visit,
            policy_target_c_scale,
            policy_target_temperature,
            seed,
        }
    }

    pub fn validate(&self) -> Result<(), GumbelError> {
        if self.simulations == 0 {
            return Err(GumbelError::message("simulations must be positive"));
        }
        if self.max_considered_actions == 0 {
            return Err(GumbelError::message(
                "max_considered_actions must be positive",
            ));
        }
        if !self.c_visit.is_finite() || self.c_visit <= 0.0 {
            return Err(GumbelError::message(
                "c_visit must be a finite positive value",
            ));
        }
        if !self.c_scale.is_finite() || self.c_scale <= 0.0 {
            return Err(GumbelError::message(
                "c_scale must be a finite positive value",
            ));
        }
        if !self.gumbel_scale.is_finite() || self.gumbel_scale < 0.0 {
            return Err(GumbelError::message(
                "gumbel_scale must be a finite non-negative value",
            ));
        }
        if !self.policy_target_c_visit.is_finite() || self.policy_target_c_visit <= 0.0 {
            return Err(GumbelError::message(
                "policy_target_c_visit must be a finite positive value",
            ));
        }
        if !self.policy_target_c_scale.is_finite() || self.policy_target_c_scale <= 0.0 {
            return Err(GumbelError::message(
                "policy_target_c_scale must be a finite positive value",
            ));
        }
        if !self.policy_target_temperature.is_finite() || self.policy_target_temperature <= 0.0 {
            return Err(GumbelError::message(
                "policy_target_temperature must be a finite positive value",
            ));
        }
        Ok(())
    }
}
