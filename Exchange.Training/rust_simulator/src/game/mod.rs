//! Game simulation module.
//!
//! This module contains the game state, simulator, and tensor computation
//! logic for the EXCHANGE game.

pub mod simulator;
pub mod state;
pub mod tensor;

pub use simulator::GameSimulator;
pub use state::{AttackMaps, Bitboard, GameState};
pub use tensor::{compute_tensor, TensorComputer, TENSOR_BUFFER};
// Re-export for convenience
pub use crate::core::INPUT_CHANNELS;
pub use half::f16;
