//! Piece representation and operations.
//!
//! The Piece struct is a compact 12-byte representation optimized for
//! cache efficiency and fast copying during MCTS simulations.

use super::constants::{FLAG_ABILITY_USED, FLAG_DEAD, FLAG_INTERPOSE, FLAG_WAS_PAWN};
use super::types::{PieceType, Team};

// ============================================================================
// Piece Structure
// ============================================================================

/// A compact piece representation (12 bytes).
///
/// Memory layout:
/// - Bytes 0-1: position (x, y)
/// - Byte 2: type_team_flags (type: 3 bits, team: 1 bit, flags: 4 bits)
/// - Byte 3: ability_cooldown
/// - Bytes 4-5: current_hp
/// - Bytes 6-7: max_hp
/// - Bytes 8-9: base_damage
/// - Byte 10: advance_cooldown_turns
/// - Byte 11: ability_uses_remaining
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Piece {
    /// Position X coordinate (0-7)
    pub x: i8,
    /// Position Y coordinate (0-7)
    pub y: i8,
    /// Packed: piece_type (3 bits) | team (1 bit) | flags (4 bits)
    type_team_flags: u8,
    /// Turns until ability can be used again
    pub ability_cooldown: i8,
    /// Current health points
    pub current_hp: i16,
    /// Maximum health points
    pub max_hp: i16,
    /// Base damage dealt per attack
    pub base_damage: i16,
    /// Turns until Advance can be used again (Pawn only)
    pub advance_cooldown_turns: i8,
    /// Remaining ability uses (-1 = unlimited, 0 = none left)
    pub ability_uses_remaining: i8,
}

impl Piece {
    /// Create a new piece with default stats for its type.
    pub fn new(piece_type: PieceType, team: Team, x: i8, y: i8) -> Self {
        let (max_hp, base_damage, _, ability_max_uses) = piece_type.stats();

        // ability_max_uses: 0 = unlimited (use -1 internally), N = limited uses
        let uses_remaining = if ability_max_uses == 0 { -1 } else { ability_max_uses };

        Piece {
            x,
            y,
            type_team_flags: (piece_type as u8) | ((team as u8) << 3),
            ability_cooldown: 0,
            current_hp: max_hp,
            max_hp,
            base_damage,
            advance_cooldown_turns: 0,
            ability_uses_remaining: uses_remaining,
        }
    }

    // ========================================================================
    // Type and Team Accessors
    // ========================================================================

    /// Get the piece type.
    #[inline]
    pub fn piece_type(&self) -> PieceType {
        match self.type_team_flags & 0x07 {
            0 => PieceType::King,
            1 => PieceType::Queen,
            2 => PieceType::Rook,
            3 => PieceType::Bishop,
            4 => PieceType::Knight,
            _ => PieceType::Pawn,
        }
    }

    /// Get the piece's team.
    #[inline]
    pub fn team(&self) -> Team {
        if (self.type_team_flags >> 3) & 1 == 0 {
            Team::White
        } else {
            Team::Black
        }
    }

    /// Set the piece type (used for promotion).
    #[inline]
    pub fn set_piece_type(&mut self, pt: PieceType) {
        self.type_team_flags = (self.type_team_flags & 0xF8) | (pt as u8);
    }

    // ========================================================================
    // State Checks
    // ========================================================================

    /// Check if the piece is still alive.
    #[inline]
    pub fn is_alive(&self) -> bool {
        self.current_hp > 0 && (self.type_team_flags & FLAG_DEAD) == 0
    }

    /// Check if ability has been used this match (for tracking).
    #[inline]
    pub fn ability_used_this_match(&self) -> bool {
        (self.type_team_flags & FLAG_ABILITY_USED) != 0
    }

    /// Mark ability as used this match.
    #[inline]
    #[allow(dead_code)]
    pub fn set_ability_used(&mut self, used: bool) {
        if used {
            self.type_team_flags |= FLAG_ABILITY_USED;
        } else {
            self.type_team_flags &= !FLAG_ABILITY_USED;
        }
    }

    /// Check if Rook's Interpose ability is active.
    #[inline]
    pub fn interpose_active(&self) -> bool {
        (self.type_team_flags & FLAG_INTERPOSE) != 0
    }

    /// Set Interpose active state.
    #[inline]
    pub fn set_interpose(&mut self, active: bool) {
        if active {
            self.type_team_flags |= FLAG_INTERPOSE;
        } else {
            self.type_team_flags &= !FLAG_INTERPOSE;
        }
    }

    /// Check if this piece was originally a pawn (for promotion tracking).
    #[inline]
    pub fn was_pawn(&self) -> bool {
        (self.type_team_flags & FLAG_WAS_PAWN) != 0
    }

    /// Mark piece as having been a pawn.
    #[inline]
    pub fn set_was_pawn(&mut self, was: bool) {
        if was {
            self.type_team_flags |= FLAG_WAS_PAWN;
        } else {
            self.type_team_flags &= !FLAG_WAS_PAWN;
        }
    }

    /// Mark piece as dead.
    #[inline]
    pub fn set_dead(&mut self, dead: bool) {
        if dead {
            self.type_team_flags |= FLAG_DEAD;
        } else {
            self.type_team_flags &= !FLAG_DEAD;
        }
    }

    // ========================================================================
    // Ability Management
    // ========================================================================

    /// Check if the piece can use its ability (has charges and off cooldown).
    pub fn can_use_ability(&self) -> bool {
        // Check uses remaining first (0 = no charges left)
        if self.ability_uses_remaining == 0 {
            return false;
        }
        // Check cooldown
        self.ability_cooldown <= 0
    }

    /// Consume one ability use.
    #[inline]
    pub fn use_ability_charge(&mut self) {
        if self.ability_uses_remaining > 0 {
            self.ability_uses_remaining -= 1;
        }
    }

    // ========================================================================
    // Hashing Support
    // ========================================================================

    /// Get HP bucket (0-7) for Zobrist hashing.
    /// This reduces hash key space while still differentiating health states.
    #[inline]
    pub fn hp_bucket(&self) -> usize {
        if self.max_hp == 0 {
            return 0;
        }
        let ratio = (self.current_hp as f32 / self.max_hp as f32 * 7.0) as usize;
        ratio.min(7)
    }

    /// Get square index (0-63) for board lookups.
    #[inline]
    pub fn square(&self) -> usize {
        (self.y as usize) * 8 + (self.x as usize)
    }
}
