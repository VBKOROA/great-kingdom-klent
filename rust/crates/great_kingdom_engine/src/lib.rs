//! Pure Great Kingdom game engine: board representation, move legality,
//! capture resolution, and territory adjudication. This crate has no
//! dependencies on neural-network, tensor, ONNX Runtime, or PyO3 code.

pub mod game;
pub mod rules;
pub mod territory;

pub use game::{
    ACTION_SPACE, Action, BOARD_CELLS, BOARD_SIZE, CASTLES_PER_PLAYER, CENTER_INDEX, Cell,
    GameEndReason, GameOutcome, GameState, InvalidAction, PASS_ACTION, Player,
};
