//! Core type definitions for the EXCHANGE game.
//!
//! These enums represent the fundamental game concepts:
//! pieces, teams, move types, and abilities.

use super::constants::PIECE_STATS;

// ============================================================================
// Piece Types
// ============================================================================

/// The six chess-like piece types in EXCHANGE.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PieceType {
    King = 0,
    Queen = 1,
    Rook = 2,
    Bishop = 3,
    Knight = 4,
    Pawn = 5,
}

impl PieceType {
    /// Get piece stats: (max_hp, base_damage, cooldown_max, ability_max_uses)
    #[inline]
    pub fn stats(self) -> (i16, i16, i8, i8) {
        PIECE_STATS[self as usize]
    }

    /// Get the maximum HP for this piece type
    #[inline]
    pub fn max_hp(self) -> i16 {
        PIECE_STATS[self as usize].0
    }

    /// Get the base damage for this piece type
    #[inline]
    pub fn base_damage(self) -> i16 {
        PIECE_STATS[self as usize].1
    }

    /// Get the ability cooldown for this piece type
    #[inline]
    pub fn cooldown_max(self) -> i8 {
        PIECE_STATS[self as usize].2
    }

    /// Get the maximum ability uses for this piece type (0 = unlimited)
    #[inline]
    pub fn ability_max_uses(self) -> i8 {
        PIECE_STATS[self as usize].3
    }
}

// ============================================================================
// Teams
// ============================================================================

/// The two opposing teams.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Team {
    White = 0,
    Black = 1,
}

impl Team {
    /// Get the opposing team.
    #[inline]
    pub fn opposite(self) -> Team {
        match self {
            Team::White => Team::Black,
            Team::Black => Team::White,
        }
    }

    /// Convert from integer (0 = White, 1 = Black)
    #[inline]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Team::White),
            1 => Some(Team::Black),
            _ => None,
        }
    }
}

// ============================================================================
// Move Types
// ============================================================================

/// The different types of moves a piece can make.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MoveType {
    /// Pure movement (no attack)
    Move = 0,
    /// Attack in place (no movement)
    Attack = 1,
    /// Move then attack (e.g., Knight Skirmish)
    MoveAndAttack = 2,
    /// Use a special ability
    Ability = 3,
}

// ============================================================================
// Abilities
// ============================================================================

/// The unique ability for each piece type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AbilityId {
    /// King: Royal Decree - +2 damage for all allies for 2 turns
    RoyalDecree = 0,
    /// Queen: Overextend - Attack twice but take damage
    Overextend = 1,
    /// Rook: Interpose - Block attacks targeting adjacent allies
    Interpose = 2,
    /// Bishop: Consecration - Heal all adjacent allies
    Consecration = 3,
    /// Knight: Skirmish - Move after attacking
    Skirmish = 4,
    /// Pawn: Advance - Move 2 squares forward
    Advance = 5,
}

impl AbilityId {
    /// Get the ability for a given piece type.
    #[inline]
    pub fn for_piece_type(pt: PieceType) -> Self {
        match pt {
            PieceType::King => AbilityId::RoyalDecree,
            PieceType::Queen => AbilityId::Overextend,
            PieceType::Rook => AbilityId::Interpose,
            PieceType::Bishop => AbilityId::Consecration,
            PieceType::Knight => AbilityId::Skirmish,
            PieceType::Pawn => AbilityId::Advance,
        }
    }

    /// Get the name of this ability.
    #[inline]
    pub fn name(self) -> &'static str {
        match self {
            AbilityId::RoyalDecree => "Royal Decree",
            AbilityId::Overextend => "Overextend",
            AbilityId::Interpose => "Interpose",
            AbilityId::Consecration => "Consecration",
            AbilityId::Skirmish => "Skirmish",
            AbilityId::Advance => "Advance",
        }
    }
}
