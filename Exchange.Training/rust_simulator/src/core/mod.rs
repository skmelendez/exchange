//! Core game primitives for EXCHANGE.
//!
//! This module contains the fundamental types and constants that
//! define the game: pieces, moves, teams, and configuration values.

pub mod channels;
pub mod constants;
pub mod maps;
pub mod moves;
pub mod piece;
pub mod rewards;
pub mod types;
pub mod zobrist;

// Re-export commonly used types at the module level
pub use channels::INPUT_CHANNELS;
pub use constants::*;
pub use maps::CENTER_WEIGHTS;
pub use moves::{encode_move, Move, UndoInfo};
pub use piece::Piece;
pub use types::{AbilityId, MoveType, PieceType, Team};
pub use rewards::{calculate_damage_reward, calculate_shuffle_penalty, calculate_win_value};
pub use zobrist::{hp_to_bucket, piece_key, royal_decree_key, side_to_move_key, ZOBRIST};
