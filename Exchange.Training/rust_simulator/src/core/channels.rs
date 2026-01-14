//! Neural network input channel definitions.
//!
//! This module defines all 75 input channels used by the neural network.
//! Each channel is an 8x8 plane encoding specific game information.

// ============================================================================
// Channel Count
// ============================================================================

/// Total number of input channels for the neural network.
pub const INPUT_CHANNELS: usize = 75;

// ============================================================================
// Piece Position Channels (0-11)
// ============================================================================

/// Channel range for White piece positions (one-hot by piece type)
pub mod white_pieces {
    pub const KING: usize = 0;
    pub const QUEEN: usize = 1;
    pub const ROOK: usize = 2;
    pub const BISHOP: usize = 3;
    pub const KNIGHT: usize = 4;
    pub const PAWN: usize = 5;
}

/// Channel range for Black piece positions (one-hot by piece type)
pub mod black_pieces {
    pub const KING: usize = 6;
    pub const QUEEN: usize = 7;
    pub const ROOK: usize = 8;
    pub const BISHOP: usize = 9;
    pub const KNIGHT: usize = 10;
    pub const PAWN: usize = 11;
}

/// Get piece position channel for a given piece type and team
#[inline]
pub fn piece_channel(piece_type: usize, is_white: bool) -> usize {
    if is_white { piece_type } else { 6 + piece_type }
}

// ============================================================================
// HP Ratio Channels (12-17)
// ============================================================================

/// HP ratio channels by piece type (current_hp / max_hp)
pub mod hp_ratio {
    pub const KING: usize = 12;
    pub const QUEEN: usize = 13;
    pub const ROOK: usize = 14;
    pub const BISHOP: usize = 15;
    pub const KNIGHT: usize = 16;
    pub const PAWN: usize = 17;
}

/// Get HP ratio channel for a piece type
#[inline]
pub fn hp_channel(piece_type: usize) -> usize {
    12 + piece_type
}

// ============================================================================
// Ability Cooldown Channels (18-23)
// ============================================================================

/// Ability cooldown channels by piece type (cooldown / max_cooldown)
pub mod ability_cooldown {
    pub const KING: usize = 18;
    pub const QUEEN: usize = 19;
    pub const ROOK: usize = 20;
    pub const BISHOP: usize = 21;
    pub const KNIGHT: usize = 22;
    pub const PAWN: usize = 23;
}

/// Get ability cooldown channel for a piece type
#[inline]
pub fn cooldown_channel(piece_type: usize) -> usize {
    18 + piece_type
}

// ============================================================================
// Game State Channels (24-27)
// ============================================================================

/// Royal Decree active for current side (filled plane: 1.0 if active)
pub const ROYAL_DECREE_ACTIVE: usize = 24;

/// Interpose active markers (per-square: 1.0 if rook has interpose up)
pub const INTERPOSE_ACTIVE: usize = 25;

/// Side to move (filled plane: 1.0 = White, 0.0 = Black)
pub const SIDE_TO_MOVE: usize = 26;

/// AI playing as (filled plane: 1.0 = White, 0.0 = Black)
pub const PLAYING_AS: usize = 27;

// ============================================================================
// Strategic Feature Channels (28-45)
// ============================================================================

/// Turn progress (filled plane: turn_number / 300, clamped to 1.0)
pub const TURN_PROGRESS: usize = 28;

/// Shuffle tracking (filled plane: moves_without_damage / 30) - used for shuffle penalty, not draw
pub const DRAW_PROGRESS: usize = 29;

/// Repetition count (filled plane: position_repeats / 3, clamped to 1.0)
pub const REPETITION_COUNT: usize = 30;

/// HP balance (filled plane: white_hp / total_hp)
pub const HP_BALANCE: usize = 31;

/// Piece count balance (filled plane: white_pieces / total_pieces)
pub const PIECE_COUNT_BALANCE: usize = 32;

/// Material balance (filled plane: (white_mat - black_mat) normalized)
pub const MATERIAL_BALANCE: usize = 33;

/// White attack map (per-square: 1.0 if attacked by white)
pub const WHITE_ATTACKS: usize = 34;

/// Black attack map (per-square: 1.0 if attacked by black)
pub const BLACK_ATTACKS: usize = 35;

/// Contested squares (per-square: 1.0 if attacked by both)
pub const CONTESTED_SQUARES: usize = 36;

/// White king zone (3x3 around white king)
pub const WHITE_KING_ZONE: usize = 37;

/// Black king zone (3x3 around black king)
pub const BLACK_KING_ZONE: usize = 38;

/// White pawn advancement (per-pawn: y / 7)
pub const WHITE_PAWN_ADVANCEMENT: usize = 39;

/// Black pawn advancement (per-pawn: (7 - y) / 7)
pub const BLACK_PAWN_ADVANCEMENT: usize = 40;

/// White abilities ready ratio (filled plane)
pub const WHITE_ABILITIES_READY: usize = 41;

/// Black abilities ready ratio (filled plane)
pub const BLACK_ABILITIES_READY: usize = 42;

/// Low HP targets (per-square: 1.0 if piece HP < 40%)
pub const LOW_HP_TARGETS: usize = 43;

/// Center control balance (per-square: weighted control difference)
pub const CENTER_CONTROL: usize = 44;

/// King tropism (filled plane: 1.0 - manhattan_distance / 14)
pub const KING_TROPISM: usize = 45;

// ============================================================================
// Tactical Feature Channels (46-57)
// ============================================================================

/// Hanging white pieces (attacked by black, not defended by white)
pub const WHITE_HANGING: usize = 46;

/// Hanging black pieces (attacked by white, not defended by black)
pub const BLACK_HANGING: usize = 47;

/// Defended white pieces (attacked by friendly)
pub const WHITE_DEFENDED: usize = 48;

/// Defended black pieces (attacked by friendly)
pub const BLACK_DEFENDED: usize = 49;

/// Attackers on white king zone (filled plane: count / 6)
pub const WHITE_KING_ATTACKERS: usize = 50;

/// Attackers on black king zone (filled plane: count / 6)
pub const BLACK_KING_ATTACKERS: usize = 51;

/// Safe squares for white king (not attacked by black)
pub const WHITE_KING_SAFE_SQUARES: usize = 52;

/// Safe squares for black king (not attacked by white)
pub const BLACK_KING_SAFE_SQUARES: usize = 53;

/// White passed pawns (no enemy pawns ahead)
pub const WHITE_PASSED_PAWNS: usize = 54;

/// Black passed pawns (no enemy pawns ahead)
pub const BLACK_PASSED_PAWNS: usize = 55;

/// Open files (no pawns on file)
pub const OPEN_FILES: usize = 56;

/// Damage potential balance (per-square: normalized attack damage)
pub const DAMAGE_POTENTIAL: usize = 57;

// ============================================================================
// Ability Tracking Channels (58-61)
// ============================================================================

/// Ability charges remaining (per-piece: uses / max_uses)
pub const ABILITY_CHARGES: usize = 58;

/// Royal Decree turns remaining (filled plane: turns / 2)
pub const ROYAL_DECREE_TURNS: usize = 59;

/// RD combo potential (pieces that can attack with RD active)
pub const RD_COMBO_POTENTIAL: usize = 60;

/// Promoted pieces (Queens that were pawns)
pub const PROMOTED_PIECES: usize = 61;

// ============================================================================
// New Tactical & Strategic Channels (62-74)
// ============================================================================

/// Threat map for white pieces (per-square: 1.0 if white piece can be attacked)
pub const THREAT_MAP_WHITE: usize = 62;

/// Threat map for black pieces (per-square: 1.0 if black piece can be attacked)
pub const THREAT_MAP_BLACK: usize = 63;

/// Interpose coverage for white (per-square: 1.0 if Rook has LOS protection)
pub const INTERPOSE_COVERAGE_WHITE: usize = 64;

/// Interpose coverage for black (per-square: 1.0 if Rook has LOS protection)
pub const INTERPOSE_COVERAGE_BLACK: usize = 65;

/// Consecration targets (per-square: HP deficit ratio for pieces in Bishop heal range)
pub const CONSECRATION_TARGETS: usize = 66;

/// Forcing moves (per-square: 1.0 if piece requires immediate response)
pub const FORCING_MOVES: usize = 67;

/// Enhanced RD combo potential (per-square: attack opportunity during RD)
pub const RD_COMBO_ENHANCED: usize = 68;

/// Pawn promotion distance (per-pawn: (7 - dist_to_back_rank) / 7)
pub const PAWN_PROMOTION_DIST: usize = 69;

/// Enemy Royal Decree turns remaining (filled plane: turns / 2)
pub const ENEMY_RD_TURNS: usize = 70;

/// Enemy abilities ready ratio (filled plane: ready_count / piece_count)
pub const ENEMY_ABILITIES_READY: usize = 71;

/// King proximity map (per-piece: 1.0 - manhattan_dist / 14)
pub const KING_PROXIMITY_MAP: usize = 72;

/// Safe attack squares (per-square: 1.0 if we can attack without counter)
pub const SAFE_ATTACK_SQUARES: usize = 73;

/// Enemy ability charges (per-square: charges / max_charges for enemy pieces)
pub const ENEMY_ABILITY_CHARGES: usize = 74;

// ============================================================================
// Channel Groups (for iteration)
// ============================================================================

/// All piece position channels
pub const PIECE_CHANNELS: std::ops::Range<usize> = 0..12;

/// All HP ratio channels
pub const HP_CHANNELS: std::ops::Range<usize> = 12..18;

/// All ability cooldown channels
pub const COOLDOWN_CHANNELS: std::ops::Range<usize> = 18..24;

/// All game state channels
pub const STATE_CHANNELS: std::ops::Range<usize> = 24..28;

/// All strategic feature channels
pub const STRATEGIC_CHANNELS: std::ops::Range<usize> = 28..46;

/// All tactical feature channels
pub const TACTICAL_CHANNELS: std::ops::Range<usize> = 46..58;

/// All ability tracking channels (original 58-61)
pub const ABILITY_CHANNELS: std::ops::Range<usize> = 58..62;

/// All new tactical/strategic channels (62-74)
pub const NEW_TACTICAL_CHANNELS: std::ops::Range<usize> = 62..75;
