//! Move representation and undo information.
//!
//! This module contains the Move struct for representing game actions
//! and UndoInfo for the make/unmake pattern used in search.

use super::piece::Piece;
use super::types::{AbilityId, MoveType, Team};

// ============================================================================
// Move Structure
// ============================================================================

/// Represents a single game action (move, attack, or ability use).
#[derive(Debug, Clone, Copy)]
pub struct Move {
    /// Index of the piece making the move
    pub piece_idx: u8,
    /// Type of move being made
    pub move_type: MoveType,
    /// Starting X position
    pub from_x: i8,
    /// Starting Y position
    pub from_y: i8,
    /// Destination X position (or attack target X for Attack type)
    pub to_x: i8,
    /// Destination Y position (or attack target Y for Attack type)
    pub to_y: i8,
    /// Attack target X for MoveAndAttack (-1 if not applicable)
    pub attack_x: i8,
    /// Attack target Y for MoveAndAttack (-1 if not applicable)
    pub attack_y: i8,
    /// Ability target X (for abilities with targeting)
    pub ability_target_x: i8,
    /// Ability target Y (for abilities with targeting)
    pub ability_target_y: i8,
    /// Which ability is being used (if any)
    pub ability_id: Option<AbilityId>,
}

impl Move {
    /// Create a pure movement move (no attack).
    #[inline]
    pub fn new_move(piece_idx: u8, from: (i8, i8), to: (i8, i8)) -> Self {
        Move {
            piece_idx,
            move_type: MoveType::Move,
            from_x: from.0,
            from_y: from.1,
            to_x: to.0,
            to_y: to.1,
            attack_x: -1,
            attack_y: -1,
            ability_target_x: -1,
            ability_target_y: -1,
            ability_id: None,
        }
    }

    /// Create an attack move (stationary attack).
    #[inline]
    pub fn new_attack(piece_idx: u8, from: (i8, i8), target: (i8, i8)) -> Self {
        Move {
            piece_idx,
            move_type: MoveType::Attack,
            from_x: from.0,
            from_y: from.1,
            to_x: target.0,
            to_y: target.1,
            attack_x: -1,
            attack_y: -1,
            ability_target_x: -1,
            ability_target_y: -1,
            ability_id: None,
        }
    }

    /// Create a move-and-attack (e.g., Knight Skirmish).
    #[inline]
    pub fn new_move_and_attack(
        piece_idx: u8,
        from: (i8, i8),
        to: (i8, i8),
        attack: (i8, i8),
    ) -> Self {
        Move {
            piece_idx,
            move_type: MoveType::MoveAndAttack,
            from_x: from.0,
            from_y: from.1,
            to_x: to.0,
            to_y: to.1,
            attack_x: attack.0,
            attack_y: attack.1,
            ability_target_x: -1,
            ability_target_y: -1,
            ability_id: None,
        }
    }

    /// Create an ability use move.
    #[inline]
    pub fn new_ability(piece_idx: u8, from: (i8, i8), to: (i8, i8), ability: AbilityId) -> Self {
        Move {
            piece_idx,
            move_type: MoveType::Ability,
            from_x: from.0,
            from_y: from.1,
            to_x: to.0,
            to_y: to.1,
            attack_x: -1,
            attack_y: -1,
            ability_target_x: -1,
            ability_target_y: -1,
            ability_id: Some(ability),
        }
    }

    /// Create an ability use with additional targeting.
    #[inline]
    pub fn new_ability_with_target(
        piece_idx: u8,
        from: (i8, i8),
        to: (i8, i8),
        attack: Option<(i8, i8)>,
        target: Option<(i8, i8)>,
        ability: AbilityId,
    ) -> Self {
        Move {
            piece_idx,
            move_type: MoveType::Ability,
            from_x: from.0,
            from_y: from.1,
            to_x: to.0,
            to_y: to.1,
            attack_x: attack.map_or(-1, |p| p.0),
            attack_y: attack.map_or(-1, |p| p.1),
            ability_target_x: target.map_or(-1, |p| p.0),
            ability_target_y: target.map_or(-1, |p| p.1),
            ability_id: Some(ability),
        }
    }

    /// Check if this move has an attack component.
    #[inline]
    pub fn has_attack(&self) -> bool {
        match self.move_type {
            MoveType::Attack | MoveType::MoveAndAttack => true,
            MoveType::Ability => self.attack_x >= 0,
            _ => false,
        }
    }

    /// Get the attack target position if any.
    #[inline]
    pub fn attack_target(&self) -> Option<(i8, i8)> {
        match self.move_type {
            MoveType::Attack => Some((self.to_x, self.to_y)),
            MoveType::MoveAndAttack => Some((self.attack_x, self.attack_y)),
            MoveType::Ability if self.attack_x >= 0 => Some((self.attack_x, self.attack_y)),
            _ => None,
        }
    }
}

// ============================================================================
// Move Encoding (for neural network policy)
// ============================================================================

/// Encode a move to a policy index for the neural network.
/// Policy encoding: from_square * 64 + to_square (for basic moves)
#[inline]
pub fn encode_move(mv: &Move) -> usize {
    let from_idx = (mv.from_y as usize) * 8 + (mv.from_x as usize);
    let to_idx = (mv.to_y as usize) * 8 + (mv.to_x as usize);
    let base = from_idx * 64 + to_idx;

    // MoveAndAttack uses extended encoding
    if mv.move_type == MoveType::MoveAndAttack && mv.attack_x >= 0 {
        let attack_idx = (mv.attack_y as usize) * 8 + (mv.attack_x as usize);
        return 4096 + from_idx * 8 + (attack_idx % 8);
    }

    base
}

// ============================================================================
// Undo Info (for make/unmake pattern)
// ============================================================================

/// Information needed to undo a move (for search algorithms).
///
/// The make/unmake pattern is faster than cloning the entire game state
/// for each move in the search tree.
#[derive(Clone)]
pub struct UndoInfo {
    // Piece state before move
    pub piece_idx: u8,
    pub piece_snapshot: Piece,
    pub old_board_from: Option<u8>,
    pub old_board_to: Option<u8>,

    // Target piece state (if attack)
    pub target_idx: Option<u8>,
    pub target_snapshot: Option<Piece>,
    pub old_board_attack: Option<u8>,

    // Secondary target (for abilities like Consecration, Interpose)
    pub secondary_idx: Option<u8>,
    pub secondary_snapshot: Option<Piece>,

    // Game state
    pub old_hash: u64,
    pub old_white_royal_decree_turns: i8,
    pub old_black_royal_decree_turns: i8,
    pub old_moves_without_damage: i32,
    pub old_side_to_move: Team,
    pub old_turn_number: i32,

    // For Rook's Interpose damage split
    pub interpose_rook_idx: Option<u8>,
    pub interpose_rook_snapshot: Option<Piece>,
}

impl UndoInfo {
    /// Create a new UndoInfo with minimal initialization.
    /// Fields should be set by the caller during make_move.
    pub fn new(piece_idx: u8, piece: Piece) -> Self {
        UndoInfo {
            piece_idx,
            piece_snapshot: piece,
            old_board_from: None,
            old_board_to: None,
            target_idx: None,
            target_snapshot: None,
            old_board_attack: None,
            secondary_idx: None,
            secondary_snapshot: None,
            old_hash: 0,
            old_white_royal_decree_turns: 0,
            old_black_royal_decree_turns: 0,
            old_moves_without_damage: 0,
            old_side_to_move: Team::White,
            old_turn_number: 0,
            interpose_rook_idx: None,
            interpose_rook_snapshot: None,
        }
    }
}
