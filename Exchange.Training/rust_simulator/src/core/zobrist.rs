//! Zobrist hashing for fast position comparison.
//!
//! Zobrist hashing provides O(1) incremental hash updates when pieces
//! move or change state. This enables efficient transposition detection
//! and position repetition checking.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::LazyLock;

// ============================================================================
// Zobrist Keys
// ============================================================================

/// Pre-computed random keys for Zobrist hashing.
pub struct ZobristKeys {
    /// Keys indexed by [square][piece_type][team][hp_bucket]
    /// HP is bucketed to 8 levels for reasonable collision avoidance
    pub pieces: [[[[u64; 8]; 2]; 6]; 64],
    /// Key for side to move (XOR when it's Black's turn)
    pub side_to_move: u64,
    /// Keys for Royal Decree active [team]
    pub royal_decree: [u64; 2],
}

impl ZobristKeys {
    /// Generate deterministic pseudo-random keys.
    fn new() -> Self {
        let mut keys = ZobristKeys {
            pieces: [[[[0u64; 8]; 2]; 6]; 64],
            side_to_move: 0,
            royal_decree: [0; 2],
        };

        // Use a deterministic seed for reproducibility
        let mut seed = 0x12345678u64;
        let mut next_key = || {
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            seed = hasher.finish();
            seed
        };

        // Generate keys for all piece configurations
        for sq in 0..64 {
            for pt in 0..6 {
                for team in 0..2 {
                    for hp in 0..8 {
                        keys.pieces[sq][pt][team][hp] = next_key();
                    }
                }
            }
        }

        keys.side_to_move = next_key();
        keys.royal_decree[0] = next_key();
        keys.royal_decree[1] = next_key();

        keys
    }
}

/// Global Zobrist keys - initialized once at startup.
pub static ZOBRIST: LazyLock<ZobristKeys> = LazyLock::new(ZobristKeys::new);

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert HP to bucket index (0-7) for Zobrist hashing.
/// This reduces the key space while still differentiating health states.
#[inline]
pub fn hp_to_bucket(hp: i16, max_hp: i16) -> usize {
    if max_hp == 0 {
        return 0;
    }
    let ratio = (hp as f32) / (max_hp as f32);
    ((ratio * 7.0) as usize).min(7)
}

/// Compute Zobrist key contribution for a piece at a position.
#[inline]
pub fn piece_key(sq: usize, piece_type: usize, team: usize, hp_bucket: usize) -> u64 {
    ZOBRIST.pieces[sq][piece_type][team][hp_bucket]
}

/// Get the side-to-move key (XOR to flip perspective).
#[inline]
pub fn side_to_move_key() -> u64 {
    ZOBRIST.side_to_move
}

/// Get the Royal Decree active key for a team.
#[inline]
pub fn royal_decree_key(team: usize) -> u64 {
    ZOBRIST.royal_decree[team]
}
