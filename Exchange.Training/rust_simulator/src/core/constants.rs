//! Game constants and configuration values.
//!
//! This module contains all the magic numbers and tunable parameters
//! for the EXCHANGE game. Centralizing them here makes experimentation easier.

// ============================================================================
// Board Configuration
// ============================================================================

/// Board dimensions (8x8 chess-like grid)
pub const BOARD_SIZE: usize = 8;

/// Maximum pieces per side (16 each = 32 total)
pub const MAX_PIECES: usize = 32;

// Note: INPUT_CHANNELS is defined in core/channels.rs with full channel documentation

// ============================================================================
// Game Rules
// ============================================================================

// NOTE: 30-move no-damage draw rule REMOVED - was being exploited by AI for passive play
// Only insufficient material and threefold repetition trigger draws now

/// Position repetitions required for draw
pub const DRAW_REPETITION_COUNT: usize = 3;

// ============================================================================
// Piece Configuration
// ============================================================================

/// Material values for position evaluation: [King, Queen, Rook, Bishop, Knight, Pawn]
pub const PIECE_VALUES: [i32; 6] = [0, 9, 5, 3, 3, 1];

/// Piece stats: (max_hp, base_damage, cooldown_max, ability_max_uses)
/// - cooldown_max: turns between ability uses (0 = can use again immediately if charges remain)
/// - ability_max_uses: 0 = unlimited, N = limited uses per match
pub const PIECE_STATS: [(i16, i16, i8, i8); 6] = [
    (25, 1, 0, 3),   // King - 3 charges, no cooldown (Royal Decree: +2 damage for 2 turns)
    (10, 3, 3, 0),   // Queen - unlimited uses (Overextend: self-limits via damage taken)
    (13, 2, 3, 5),   // Rook - 5 uses per match (Interpose: block attacks on allies)
    (10, 2, 3, 3),   // Bishop - 3 uses per match (Consecration: heal adjacent allies)
    (11, 2, 3, 5),   // Knight - 5 uses per match (Skirmish: move after attack)
    (7, 1, 1, 0),    // Pawn - unlimited uses (Advance: move 2 squares forward)
];

// ============================================================================
// Movement Directions
// ============================================================================

/// Cardinal directions: up, down, right, left
pub const CARDINAL: [(i8, i8); 4] = [(0, 1), (0, -1), (1, 0), (-1, 0)];

/// Diagonal directions: NE, SE, NW, SW
pub const DIAGONAL: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

/// All 8 directions (cardinal + diagonal)
pub const ALL_DIRS: [(i8, i8); 8] = [
    (0, 1), (0, -1), (1, 0), (-1, 0),
    (1, 1), (1, -1), (-1, 1), (-1, -1),
];

/// Knight move offsets (L-shaped)
pub const KNIGHT_MOVES: [(i8, i8); 8] = [
    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
    (1, -2), (1, 2), (2, -1), (2, 1),
];

// ============================================================================
// Piece Flags (bit flags for compact storage)
// ============================================================================

/// Piece has used its ability this match (for limited-use abilities)
pub const FLAG_ABILITY_USED: u8 = 0x10;

/// Rook's Interpose ability is active (blocking attacks)
pub const FLAG_INTERPOSE: u8 = 0x20;

/// Piece was originally a pawn (for promotion tracking)
pub const FLAG_WAS_PAWN: u8 = 0x40;

/// Piece is dead/captured
pub const FLAG_DEAD: u8 = 0x80;

// ============================================================================
// Utility Functions
// ============================================================================

/// Check if coordinates are within board bounds
#[inline]
pub fn is_on_board(x: i8, y: i8) -> bool {
    x >= 0 && x < 8 && y >= 0 && y < 8
}

/// Convert (x, y) to square index (0-63)
#[inline]
pub fn sq_idx(x: i8, y: i8) -> usize {
    (y as usize) * 8 + (x as usize)
}

/// Convert (x, y) to bitboard bit position
#[inline]
pub fn sq_bit(x: i8, y: i8) -> u64 {
    1u64 << (y * 8 + x)
}
