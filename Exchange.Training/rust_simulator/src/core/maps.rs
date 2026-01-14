//! Static board maps and weight tables.
//!
//! This module contains pre-computed board patterns and weight tables
//! used for evaluation and tensor computation.

use super::constants::BOARD_SIZE;

// ============================================================================
// Center Control Weights
// ============================================================================

/// Center control weight map.
/// Higher values = more important for center control.
/// Used for computing center control channel in tensor.
pub const CENTER_WEIGHTS: [[f32; BOARD_SIZE]; BOARD_SIZE] = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
];

// ============================================================================
// Piece-Square Tables (for future evaluation use)
// ============================================================================

/// Pawn position bonuses (from White's perspective).
/// Encourages advancement and central control.
#[allow(dead_code)]
pub const PAWN_PST: [[i16; BOARD_SIZE]; BOARD_SIZE] = [
    [  0,   0,   0,   0,   0,   0,   0,   0],  // Rank 1 (never here)
    [  5,  10,  10, -20, -20,  10,  10,   5],  // Rank 2
    [  5,  -5, -10,   0,   0, -10,  -5,   5],  // Rank 3
    [  0,   0,   0,  20,  20,   0,   0,   0],  // Rank 4
    [  5,   5,  10,  25,  25,  10,   5,   5],  // Rank 5
    [ 10,  10,  20,  30,  30,  20,  10,  10],  // Rank 6
    [ 50,  50,  50,  50,  50,  50,  50,  50],  // Rank 7 (about to promote)
    [  0,   0,   0,   0,   0,   0,   0,   0],  // Rank 8 (promoted)
];

/// Knight position bonuses.
/// Knights prefer central squares and avoid edges.
#[allow(dead_code)]
pub const KNIGHT_PST: [[i16; BOARD_SIZE]; BOARD_SIZE] = [
    [-50, -40, -30, -30, -30, -30, -40, -50],
    [-40, -20,   0,   5,   5,   0, -20, -40],
    [-30,   5,  10,  15,  15,  10,   5, -30],
    [-30,   0,  15,  20,  20,  15,   0, -30],
    [-30,   5,  15,  20,  20,  15,   5, -30],
    [-30,   0,  10,  15,  15,  10,   0, -30],
    [-40, -20,   0,   0,   0,   0, -20, -40],
    [-50, -40, -30, -30, -30, -30, -40, -50],
];

/// Bishop position bonuses.
/// Bishops prefer long diagonals and avoid corners.
#[allow(dead_code)]
pub const BISHOP_PST: [[i16; BOARD_SIZE]; BOARD_SIZE] = [
    [-20, -10, -10, -10, -10, -10, -10, -20],
    [-10,   5,   0,   0,   0,   0,   5, -10],
    [-10,  10,  10,  10,  10,  10,  10, -10],
    [-10,   0,  10,  10,  10,  10,   0, -10],
    [-10,   5,   5,  10,  10,   5,   5, -10],
    [-10,   0,   5,  10,  10,   5,   0, -10],
    [-10,   0,   0,   0,   0,   0,   0, -10],
    [-20, -10, -10, -10, -10, -10, -10, -20],
];

/// Rook position bonuses.
/// Rooks prefer open files and 7th rank.
#[allow(dead_code)]
pub const ROOK_PST: [[i16; BOARD_SIZE]; BOARD_SIZE] = [
    [  0,   0,   0,   5,   5,   0,   0,   0],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [  5,  10,  10,  10,  10,  10,  10,   5],  // 7th rank bonus
    [  0,   0,   0,   0,   0,   0,   0,   0],
];

/// Queen position bonuses.
/// Queens are flexible but prefer avoiding early development to edges.
#[allow(dead_code)]
pub const QUEEN_PST: [[i16; BOARD_SIZE]; BOARD_SIZE] = [
    [-20, -10, -10,  -5,  -5, -10, -10, -20],
    [-10,   0,   5,   0,   0,   0,   0, -10],
    [-10,   5,   5,   5,   5,   5,   0, -10],
    [  0,   0,   5,   5,   5,   5,   0,  -5],
    [ -5,   0,   5,   5,   5,   5,   0,  -5],
    [-10,   0,   5,   5,   5,   5,   0, -10],
    [-10,   0,   0,   0,   0,   0,   0, -10],
    [-20, -10, -10,  -5,  -5, -10, -10, -20],
];

/// King position bonuses (middlegame).
/// King prefers castled position and avoids center.
#[allow(dead_code)]
pub const KING_PST_MIDDLEGAME: [[i16; BOARD_SIZE]; BOARD_SIZE] = [
    [ 20,  30,  10,   0,   0,  10,  30,  20],  // Castled positions
    [ 20,  20,   0,   0,   0,   0,  20,  20],
    [-10, -20, -20, -20, -20, -20, -20, -10],
    [-20, -30, -30, -40, -40, -30, -30, -20],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
];

/// King position bonuses (endgame).
/// King becomes active and prefers central squares.
#[allow(dead_code)]
pub const KING_PST_ENDGAME: [[i16; BOARD_SIZE]; BOARD_SIZE] = [
    [-50, -30, -30, -30, -30, -30, -30, -50],
    [-30, -30,   0,   0,   0,   0, -30, -30],
    [-30, -10,  20,  30,  30,  20, -10, -30],
    [-30, -10,  30,  40,  40,  30, -10, -30],
    [-30, -10,  30,  40,  40,  30, -10, -30],
    [-30, -10,  20,  30,  30,  20, -10, -30],
    [-30, -20, -10,   0,   0, -10, -20, -30],
    [-50, -40, -30, -20, -20, -30, -40, -50],
];

// ============================================================================
// Distance Tables
// ============================================================================

/// Manhattan distance from center (average of distance to d4, d5, e4, e5)
#[allow(dead_code)]
pub const CENTER_DISTANCE: [[u8; BOARD_SIZE]; BOARD_SIZE] = [
    [6, 5, 4, 3, 3, 4, 5, 6],
    [5, 4, 3, 2, 2, 3, 4, 5],
    [4, 3, 2, 1, 1, 2, 3, 4],
    [3, 2, 1, 0, 0, 1, 2, 3],
    [3, 2, 1, 0, 0, 1, 2, 3],
    [4, 3, 2, 1, 1, 2, 3, 4],
    [5, 4, 3, 2, 2, 3, 4, 5],
    [6, 5, 4, 3, 3, 4, 5, 6],
];

/// Precomputed Manhattan distances between all square pairs.
/// Access: MANHATTAN_DISTANCE[from_sq][to_sq]
#[allow(dead_code)]
pub static MANHATTAN_DISTANCE: [[u8; 64]; 64] = {
    let mut table = [[0u8; 64]; 64];
    let mut from = 0;
    while from < 64 {
        let mut to = 0;
        while to < 64 {
            let from_x = (from % 8) as i8;
            let from_y = (from / 8) as i8;
            let to_x = (to % 8) as i8;
            let to_y = (to / 8) as i8;
            let dx = if from_x > to_x { from_x - to_x } else { to_x - from_x };
            let dy = if from_y > to_y { from_y - to_y } else { to_y - from_y };
            table[from][to] = (dx + dy) as u8;
            to += 1;
        }
        from += 1;
    }
    table
};

// ============================================================================
// Helper Functions
// ============================================================================

/// Get center weight for a square.
#[inline]
pub fn center_weight(x: usize, y: usize) -> f32 {
    CENTER_WEIGHTS[y][x]
}

/// Get PST value for a piece at a position.
/// Automatically flips for black pieces.
#[allow(dead_code)]
#[inline]
pub fn pst_value(piece_type: usize, x: usize, y: usize, is_white: bool) -> i16 {
    let effective_y = if is_white { y } else { 7 - y };
    match piece_type {
        0 => KING_PST_MIDDLEGAME[effective_y][x], // King
        1 => QUEEN_PST[effective_y][x],           // Queen
        2 => ROOK_PST[effective_y][x],            // Rook
        3 => BISHOP_PST[effective_y][x],          // Bishop
        4 => KNIGHT_PST[effective_y][x],          // Knight
        5 => PAWN_PST[effective_y][x],            // Pawn
        _ => 0,
    }
}
