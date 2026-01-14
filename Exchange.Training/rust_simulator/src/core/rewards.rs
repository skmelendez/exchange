//! Reward and penalty configuration for training.
//!
//! All tunable reward values centralized here for easy experimentation.
//! These values shape the AI's behavior during self-play training.

// ============================================================================
// Aggression Rewards
// ============================================================================

/// Base damage reward per HP dealt
pub const DAMAGE_REWARD_BASE: f32 = 0.01;

/// Late-game damage multiplier (at turn 100)
pub const DAMAGE_REWARD_LATE_MULTIPLIER: f32 = 2.0;

/// Turn at which late multiplier is fully applied
pub const DAMAGE_REWARD_SCALE_TURN: i32 = 100;

/// Reward per attack action chosen
pub const ATTACK_ACTION_REWARD: f32 = 0.02;

// ============================================================================
// Shuffle Penalties
// ============================================================================

/// Base penalty for first no-damage move
pub const SHUFFLE_PENALTY_BASE: f32 = 0.01;

/// Multiplier per consecutive no-damage move (exponential: base * mult^moves)
pub const SHUFFLE_PENALTY_MULTIPLIER: f32 = 2.0;

/// Maximum shuffle penalty (caps at loss equivalent)
pub const SHUFFLE_PENALTY_MAX: f32 = 1.0;

// ============================================================================
// King Hunt Rewards
// ============================================================================

/// Reward per square closer to enemy king (normalized by max distance 14)
pub const KING_PROXIMITY_REWARD: f32 = 0.02;

/// Reward per piece that can attack enemy king next turn
pub const KING_ATTACK_OPPORTUNITY_REWARD: f32 = 0.05;

// ============================================================================
// Royal Decree (King Ability)
// ============================================================================

/// Small reward for activating Royal Decree
pub const RD_ACTIVATION_REWARD: f32 = 0.05;

/// Reward per attack during Royal Decree (TRIPLED from original 0.05)
pub const RD_COMBO_BONUS_PER_ATTACK: f32 = 0.15;

/// Maximum combo bonus cap
pub const RD_COMBO_BONUS_MAX: f32 = 0.60;

/// Penalty if Royal Decree expires with zero attacks made
pub const RD_WASTE_PENALTY: f32 = 0.10;

// ============================================================================
// Ability Rewards
// ============================================================================

/// Rook: reward per damage blocked by Interpose
pub const INTERPOSE_DAMAGE_BLOCKED_REWARD: f32 = 0.02;

/// Bishop: max reward for efficient heal (scales with efficiency)
pub const CONSECRATION_EFFICIENCY_REWARD: f32 = 0.10;

/// Knight: reward per Skirmish use
pub const SKIRMISH_REWARD: f32 = 0.03;

/// Queen: reward multiplier for net damage (dealt - 2 self)
pub const OVEREXTEND_NET_DAMAGE_REWARD: f32 = 0.05;

/// Pawn: reward per advance
pub const PAWN_ADVANCE_REWARD: f32 = 0.02;

/// Pawn: big bonus for promotion
pub const PAWN_PROMOTION_REWARD: f32 = 0.20;

/// Pawn: reward per threat square created
pub const PAWN_THREAT_REWARD: f32 = 0.01;

// ============================================================================
// Game Outcomes
// ============================================================================

/// Base win value
pub const WIN_VALUE: f32 = 1.0;

/// Early win bonus (added for wins before EARLY_WIN_THRESHOLD)
pub const EARLY_WIN_BONUS: f32 = 0.5;

/// Turn threshold for early win bonus (wins before this get bonus)
pub const EARLY_WIN_THRESHOLD: i32 = 50;

/// Turn at which early win bonus reaches 0 (linear decay)
pub const EARLY_WIN_DECAY_END: i32 = 100;

/// Base loss penalty
pub const LOSS_PENALTY: f32 = 1.0;

/// Draw penalty (only for insufficient material draws now)
pub const DRAW_PENALTY: f32 = 0.5;

// ============================================================================
// Helper Functions
// ============================================================================

/// Calculate damage reward with late-game scaling.
///
/// Formula: damage * base * (1.0 + turn / scale_turn), clamped to 2x max
#[inline]
pub fn calculate_damage_reward(damage: i32, turn_number: i32) -> f32 {
    let multiplier = (1.0 + turn_number as f32 / DAMAGE_REWARD_SCALE_TURN as f32)
        .min(DAMAGE_REWARD_LATE_MULTIPLIER);
    damage as f32 * DAMAGE_REWARD_BASE * multiplier
}

/// Calculate exponential shuffle penalty.
///
/// Formula: base * (multiplier ^ consecutive_moves), capped at max
#[inline]
pub fn calculate_shuffle_penalty(consecutive_no_damage_moves: i32) -> f32 {
    if consecutive_no_damage_moves <= 0 {
        return 0.0;
    }
    let penalty = SHUFFLE_PENALTY_BASE
        * SHUFFLE_PENALTY_MULTIPLIER.powi(consecutive_no_damage_moves);
    penalty.min(SHUFFLE_PENALTY_MAX)
}

/// Calculate win value with early-win bonus.
///
/// Wins before turn 50: +1.5 (50% bonus)
/// Wins turn 50-100: linear decay to +1.0
/// Wins after turn 100: +1.0
#[inline]
pub fn calculate_win_value(turn_number: i32) -> f32 {
    if turn_number <= EARLY_WIN_THRESHOLD {
        WIN_VALUE + EARLY_WIN_BONUS
    } else if turn_number >= EARLY_WIN_DECAY_END {
        WIN_VALUE
    } else {
        let progress = (turn_number - EARLY_WIN_THRESHOLD) as f32
            / (EARLY_WIN_DECAY_END - EARLY_WIN_THRESHOLD) as f32;
        WIN_VALUE + EARLY_WIN_BONUS * (1.0 - progress)
    }
}
