//! Neural-network observation and feature engineering for Great Kingdom.
//!
//! Owns the feature-plane projection and the batched evaluation request
//! container. Depends only on the pure game engine and `rayon`.

mod eval_request;
mod features;

pub use eval_request::{EvalRequest, FeatureError};
pub use features::{FEATURE_CHANNELS, FeatureChannel, GameStateFeatures};

pub use great_kingdom_engine as engine;
