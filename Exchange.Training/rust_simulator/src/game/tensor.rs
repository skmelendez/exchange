//! Tensor computation for neural network input.
//!
//! This module contains the compute_tensor function that converts
//! a game state into the 75-channel tensor representation used by the model.
//!
//! Uses f16 (half precision) for memory efficiency.

use std::cell::RefCell;
use half::f16;

use crate::core::{
    sq_idx, PieceType, Team, BOARD_SIZE, PIECE_STATS, PIECE_VALUES,
    CENTER_WEIGHTS,
    channels::{
        INPUT_CHANNELS,
        // Piece channels
        piece_channel, hp_channel, cooldown_channel,
        // State channels
        ROYAL_DECREE_ACTIVE, INTERPOSE_ACTIVE, SIDE_TO_MOVE, PLAYING_AS,
        // Strategic channels
        TURN_PROGRESS, DRAW_PROGRESS, REPETITION_COUNT, HP_BALANCE,
        PIECE_COUNT_BALANCE, MATERIAL_BALANCE, WHITE_ATTACKS, BLACK_ATTACKS,
        CONTESTED_SQUARES, WHITE_KING_ZONE, BLACK_KING_ZONE,
        WHITE_PAWN_ADVANCEMENT, BLACK_PAWN_ADVANCEMENT,
        WHITE_ABILITIES_READY, BLACK_ABILITIES_READY, LOW_HP_TARGETS,
        CENTER_CONTROL, KING_TROPISM,
        // Tactical channels
        WHITE_HANGING, BLACK_HANGING, WHITE_DEFENDED, BLACK_DEFENDED,
        WHITE_KING_ATTACKERS, BLACK_KING_ATTACKERS,
        WHITE_KING_SAFE_SQUARES, BLACK_KING_SAFE_SQUARES,
        WHITE_PASSED_PAWNS, BLACK_PASSED_PAWNS, OPEN_FILES, DAMAGE_POTENTIAL,
        // Ability channels
        ABILITY_CHARGES, ROYAL_DECREE_TURNS, RD_COMBO_POTENTIAL, PROMOTED_PIECES,
        // New tactical/strategic channels (62-74)
        THREAT_MAP_WHITE, THREAT_MAP_BLACK,
        INTERPOSE_COVERAGE_WHITE, INTERPOSE_COVERAGE_BLACK,
        CONSECRATION_TARGETS, FORCING_MOVES, RD_COMBO_ENHANCED,
        PAWN_PROMOTION_DIST, ENEMY_RD_TURNS, ENEMY_ABILITIES_READY,
        KING_PROXIMITY_MAP, SAFE_ATTACK_SQUARES, ENEMY_ABILITY_CHARGES,
    },
};
use crate::game::state::{AttackMaps, GameState};
use crate::game::simulator::GameSimulator;

// Thread-local buffers for tensor computation to avoid repeated allocations
// We compute in f32 for speed, then convert to f16 for storage
thread_local! {
    static COMPUTE_BUFFER: RefCell<Vec<f32>> = RefCell::new(vec![0.0; INPUT_CHANNELS * BOARD_SIZE * BOARD_SIZE]);
    pub static TENSOR_BUFFER: RefCell<Vec<f16>> = RefCell::new(vec![f16::ZERO; INPUT_CHANNELS * BOARD_SIZE * BOARD_SIZE]);
}

/// Compute the tensor representation of a game state.
///
/// Returns a flat Vec<f16> of size INPUT_CHANNELS * BOARD_SIZE * BOARD_SIZE.
/// Computation is done in f32 for speed, then converted to f16.
/// See `core/channels.rs` for the complete channel layout documentation.
pub fn compute_tensor(state: &GameState) -> Vec<f16> {
    COMPUTE_BUFFER.with(|buf| {
        let mut tensor = buf.borrow_mut();

        // Zero out the buffer
        tensor.fill(0.0);

        // Helper closures (compute in f32 for speed)
        let set = |t: &mut [f32], c: usize, y: i8, x: i8, v: f32| {
            if x >= 0 && x < 8 && y >= 0 && y < 8 {
                t[c * 64 + (y as usize) * 8 + (x as usize)] = v;
            }
        };

        let fill = |t: &mut [f32], c: usize, v: f32| {
            let start = c * 64;
            for i in 0..64 {
                t[start + i] = v;
            }
        };

        // Collect pieces by team
        let mut white_pieces: Vec<u8> = Vec::with_capacity(16);
        let mut black_pieces: Vec<u8> = Vec::with_capacity(16);
        let mut white_king_idx: Option<u8> = None;
        let mut black_king_idx: Option<u8> = None;

        for idx in 0..state.piece_count {
            let piece = &state.pieces[idx as usize];
            if !piece.is_alive() { continue; }
            match piece.team() {
                Team::White => {
                    white_pieces.push(idx);
                    if piece.piece_type() == PieceType::King {
                        white_king_idx = Some(idx);
                    }
                }
                Team::Black => {
                    black_pieces.push(idx);
                    if piece.piece_type() == PieceType::King {
                        black_king_idx = Some(idx);
                    }
                }
            }
        }

        // ====================================================================
        // PIECE CHANNELS (0-23)
        // ====================================================================
        for &idx in white_pieces.iter().chain(black_pieces.iter()) {
            let piece = &state.pieces[idx as usize];
            let x = piece.x;
            let y = piece.y;
            let pt = piece.piece_type() as usize;
            let is_white = piece.team() == Team::White;

            // Piece position (channels 0-11)
            set(&mut tensor, piece_channel(pt, is_white), y, x, 1.0);

            // HP ratio (channels 12-17)
            let hp_ratio = piece.current_hp as f32 / piece.max_hp as f32;
            set(&mut tensor, hp_channel(pt), y, x, hp_ratio);

            // Ability cooldown (channels 18-23)
            let (_, _, cd_max, _) = PIECE_STATS[pt];
            if cd_max > 0 {
                let cd_ratio = piece.ability_cooldown as f32 / cd_max as f32;
                set(&mut tensor, cooldown_channel(pt), y, x, cd_ratio);
            } else if cd_max == -1 {
                set(&mut tensor, cooldown_channel(pt), y, x, if piece.ability_used_this_match() { 1.0 } else { 0.0 });
            }

            // Interpose active marker (channel 25)
            if piece.interpose_active() {
                set(&mut tensor, INTERPOSE_ACTIVE, y, x, 1.0);
            }
        }

        // ====================================================================
        // STATE CHANNELS (24-27)
        // ====================================================================

        // Royal Decree active (channel 24)
        if state.get_royal_decree_bonus(state.side_to_move) > 0 {
            fill(&mut tensor, ROYAL_DECREE_ACTIVE, 1.0);
        }

        // Side to move (channel 26)
        if state.side_to_move == Team::White {
            fill(&mut tensor, SIDE_TO_MOVE, 1.0);
        }

        // Playing as (channel 27)
        if state.playing_as == Team::White {
            fill(&mut tensor, PLAYING_AS, 1.0);
        }

        // ====================================================================
        // STRATEGIC CHANNELS (28-45)
        // ====================================================================

        // Turn progress (channel 28)
        fill(&mut tensor, TURN_PROGRESS, (state.turn_number as f32 / 300.0).min(1.0));

        // Draw progress (channel 29)
        fill(&mut tensor, DRAW_PROGRESS, state.moves_without_damage as f32 / 30.0);

        // Repetition count (channel 30)
        let current_hash = state.hash;
        let rep_count = state.position_history.iter().filter(|&&h| h == current_hash).count();
        fill(&mut tensor, REPETITION_COUNT, (rep_count as f32 / 3.0).min(1.0));

        // HP balance (channel 31)
        let white_hp: i32 = white_pieces.iter().map(|&i| state.pieces[i as usize].current_hp as i32).sum();
        let black_hp: i32 = black_pieces.iter().map(|&i| state.pieces[i as usize].current_hp as i32).sum();
        let total_hp = white_hp + black_hp;
        fill(&mut tensor, HP_BALANCE, if total_hp > 0 { white_hp as f32 / total_hp as f32 } else { 0.5 });

        // Piece count balance (channel 32)
        let total_pieces = white_pieces.len() + black_pieces.len();
        fill(&mut tensor, PIECE_COUNT_BALANCE, if total_pieces > 0 { white_pieces.len() as f32 / total_pieces as f32 } else { 0.5 });

        // Material balance (channel 33)
        let white_mat: i32 = white_pieces.iter().map(|&i| PIECE_VALUES[state.pieces[i as usize].piece_type() as usize]).sum();
        let black_mat: i32 = black_pieces.iter().map(|&i| PIECE_VALUES[state.pieces[i as usize].piece_type() as usize]).sum();
        let mat_diff = (white_mat - black_mat) as f32 / 39.0;
        fill(&mut tensor, MATERIAL_BALANCE, (mat_diff + 1.0) / 2.0);

        // Compute attack maps using bitboards
        let attacks = AttackMaps::compute(state);

        // Attack maps (channels 34-36)
        for sq in 0..64 {
            let x = (sq % 8) as i8;
            let y = (sq / 8) as i8;
            if (attacks.white >> sq) & 1 != 0 {
                set(&mut tensor, WHITE_ATTACKS, y, x, 1.0);
            }
            if (attacks.black >> sq) & 1 != 0 {
                set(&mut tensor, BLACK_ATTACKS, y, x, 1.0);
            }
            if ((attacks.white & attacks.black) >> sq) & 1 != 0 {
                set(&mut tensor, CONTESTED_SQUARES, y, x, 1.0);
            }
        }

        // King zones (channels 37-38)
        if let Some(wk_idx) = white_king_idx {
            let wk = &state.pieces[wk_idx as usize];
            for dx in -1..=1 {
                for dy in -1..=1 {
                    let nx = wk.x + dx;
                    let ny = wk.y + dy;
                    if nx >= 0 && nx < 8 && ny >= 0 && ny < 8 {
                        set(&mut tensor, WHITE_KING_ZONE, ny, nx, 1.0);
                    }
                }
            }
        }
        if let Some(bk_idx) = black_king_idx {
            let bk = &state.pieces[bk_idx as usize];
            for dx in -1..=1 {
                for dy in -1..=1 {
                    let nx = bk.x + dx;
                    let ny = bk.y + dy;
                    if nx >= 0 && nx < 8 && ny >= 0 && ny < 8 {
                        set(&mut tensor, BLACK_KING_ZONE, ny, nx, 1.0);
                    }
                }
            }
        }

        // Pawn advancement (channels 39-40)
        for &idx in &white_pieces {
            let p = &state.pieces[idx as usize];
            if p.piece_type() == PieceType::Pawn {
                set(&mut tensor, WHITE_PAWN_ADVANCEMENT, p.y, p.x, p.y as f32 / 7.0);
            }
        }
        for &idx in &black_pieces {
            let p = &state.pieces[idx as usize];
            if p.piece_type() == PieceType::Pawn {
                set(&mut tensor, BLACK_PAWN_ADVANCEMENT, p.y, p.x, (7 - p.y) as f32 / 7.0);
            }
        }

        // Abilities ready (channels 41-42)
        let white_ready: usize = white_pieces.iter().filter(|&&i| state.pieces[i as usize].can_use_ability()).count();
        let black_ready: usize = black_pieces.iter().filter(|&&i| state.pieces[i as usize].can_use_ability()).count();
        if !white_pieces.is_empty() {
            fill(&mut tensor, WHITE_ABILITIES_READY, white_ready as f32 / white_pieces.len() as f32);
        }
        if !black_pieces.is_empty() {
            fill(&mut tensor, BLACK_ABILITIES_READY, black_ready as f32 / black_pieces.len() as f32);
        }

        // Low HP targets (channel 43)
        for &idx in white_pieces.iter().chain(black_pieces.iter()) {
            let p = &state.pieces[idx as usize];
            if (p.current_hp as f32) / (p.max_hp as f32) < 0.4 {
                set(&mut tensor, LOW_HP_TARGETS, p.y, p.x, 1.0);
            }
        }

        // Center control (channel 44) - uses CENTER_WEIGHTS from core/maps.rs
        for sq in 0..64 {
            let x = sq % 8;
            let y = sq / 8;
            let w = CENTER_WEIGHTS[y][x];
            if w > 0.0 {
                let white_ctrl = if (attacks.white >> sq) & 1 != 0 { w * 0.5 } else { 0.0 };
                let black_ctrl = if (attacks.black >> sq) & 1 != 0 { w * 0.5 } else { 0.0 };
                tensor[CENTER_CONTROL * 64 + sq] = ((white_ctrl - black_ctrl) + 1.0) / 2.0;
            } else {
                tensor[CENTER_CONTROL * 64 + sq] = 0.5;
            }
        }

        // King tropism (channel 45)
        if let (Some(wk_idx), Some(bk_idx)) = (white_king_idx, black_king_idx) {
            let wk = &state.pieces[wk_idx as usize];
            let bk = &state.pieces[bk_idx as usize];
            let dist = (wk.x - bk.x).abs() + (wk.y - bk.y).abs();
            fill(&mut tensor, KING_TROPISM, 1.0 - dist as f32 / 14.0);
        }

        // ====================================================================
        // TACTICAL CHANNELS (46-57)
        // ====================================================================

        // Hanging pieces (channels 46-47)
        for &idx in &white_pieces {
            let p = &state.pieces[idx as usize];
            let sq = sq_idx(p.x, p.y);
            if (attacks.black >> sq) & 1 != 0 && (attacks.white >> sq) & 1 == 0 {
                set(&mut tensor, WHITE_HANGING, p.y, p.x, 1.0);
            }
        }
        for &idx in &black_pieces {
            let p = &state.pieces[idx as usize];
            let sq = sq_idx(p.x, p.y);
            if (attacks.white >> sq) & 1 != 0 && (attacks.black >> sq) & 1 == 0 {
                set(&mut tensor, BLACK_HANGING, p.y, p.x, 1.0);
            }
        }

        // Defended pieces (channels 48-49)
        for &idx in &white_pieces {
            let p = &state.pieces[idx as usize];
            let sq = sq_idx(p.x, p.y);
            if (attacks.white >> sq) & 1 != 0 {
                set(&mut tensor, WHITE_DEFENDED, p.y, p.x, 1.0);
            }
        }
        for &idx in &black_pieces {
            let p = &state.pieces[idx as usize];
            let sq = sq_idx(p.x, p.y);
            if (attacks.black >> sq) & 1 != 0 {
                set(&mut tensor, BLACK_DEFENDED, p.y, p.x, 1.0);
            }
        }

        // King attackers (channels 50-51)
        if let Some(wk_idx) = white_king_idx {
            let wk = &state.pieces[wk_idx as usize];
            let mut wk_zone: u64 = 0;
            for dx in -1..=1 {
                for dy in -1..=1 {
                    let nx = wk.x + dx;
                    let ny = wk.y + dy;
                    if nx >= 0 && nx < 8 && ny >= 0 && ny < 8 {
                        wk_zone |= crate::core::sq_bit(nx, ny);
                    }
                }
            }
            let mut attackers = 0;
            for i in 0..black_pieces.len() {
                if attacks.black_per_piece[i] & wk_zone != 0 {
                    attackers += 1;
                }
            }
            fill(&mut tensor, WHITE_KING_ATTACKERS, (attackers as f32 / 6.0).min(1.0));
        }
        if let Some(bk_idx) = black_king_idx {
            let bk = &state.pieces[bk_idx as usize];
            let mut bk_zone: u64 = 0;
            for dx in -1..=1 {
                for dy in -1..=1 {
                    let nx = bk.x + dx;
                    let ny = bk.y + dy;
                    if nx >= 0 && nx < 8 && ny >= 0 && ny < 8 {
                        bk_zone |= crate::core::sq_bit(nx, ny);
                    }
                }
            }
            let mut attackers = 0;
            for i in 0..white_pieces.len() {
                if attacks.white_per_piece[i] & bk_zone != 0 {
                    attackers += 1;
                }
            }
            fill(&mut tensor, BLACK_KING_ATTACKERS, (attackers as f32 / 6.0).min(1.0));
        }

        // Safe king squares (channels 52-53)
        if let Some(wk_idx) = white_king_idx {
            let wk = &state.pieces[wk_idx as usize];
            for dx in -1..=1 {
                for dy in -1..=1 {
                    if dx == 0 && dy == 0 { continue; }
                    let nx = wk.x + dx;
                    let ny = wk.y + dy;
                    if nx >= 0 && nx < 8 && ny >= 0 && ny < 8 {
                        let sq = sq_idx(nx, ny);
                        if (attacks.black >> sq) & 1 == 0 {
                            let occ = state.board[ny as usize][nx as usize];
                            if occ.is_none() || state.pieces[occ.unwrap() as usize].team() != Team::White {
                                set(&mut tensor, WHITE_KING_SAFE_SQUARES, ny, nx, 1.0);
                            }
                        }
                    }
                }
            }
        }
        if let Some(bk_idx) = black_king_idx {
            let bk = &state.pieces[bk_idx as usize];
            for dx in -1..=1 {
                for dy in -1..=1 {
                    if dx == 0 && dy == 0 { continue; }
                    let nx = bk.x + dx;
                    let ny = bk.y + dy;
                    if nx >= 0 && nx < 8 && ny >= 0 && ny < 8 {
                        let sq = sq_idx(nx, ny);
                        if (attacks.white >> sq) & 1 == 0 {
                            let occ = state.board[ny as usize][nx as usize];
                            if occ.is_none() || state.pieces[occ.unwrap() as usize].team() != Team::Black {
                                set(&mut tensor, BLACK_KING_SAFE_SQUARES, ny, nx, 1.0);
                            }
                        }
                    }
                }
            }
        }

        // Passed pawns (channels 54-55)
        for &idx in &white_pieces {
            let p = &state.pieces[idx as usize];
            if p.piece_type() == PieceType::Pawn {
                let mut is_passed = true;
                'outer: for check_y in (p.y + 1)..8 {
                    for check_x in [p.x - 1, p.x, p.x + 1] {
                        if check_x >= 0 && check_x < 8 {
                            if let Some(blocker_idx) = state.get_piece_at(check_x, check_y) {
                                let blocker = &state.pieces[blocker_idx as usize];
                                if blocker.team() == Team::Black && blocker.piece_type() == PieceType::Pawn {
                                    is_passed = false;
                                    break 'outer;
                                }
                            }
                        }
                    }
                }
                if is_passed {
                    set(&mut tensor, WHITE_PASSED_PAWNS, p.y, p.x, 1.0);
                }
            }
        }
        for &idx in &black_pieces {
            let p = &state.pieces[idx as usize];
            if p.piece_type() == PieceType::Pawn {
                let mut is_passed = true;
                'outer: for check_y in (0..p.y).rev() {
                    for check_x in [p.x - 1, p.x, p.x + 1] {
                        if check_x >= 0 && check_x < 8 {
                            if let Some(blocker_idx) = state.get_piece_at(check_x, check_y) {
                                let blocker = &state.pieces[blocker_idx as usize];
                                if blocker.team() == Team::White && blocker.piece_type() == PieceType::Pawn {
                                    is_passed = false;
                                    break 'outer;
                                }
                            }
                        }
                    }
                }
                if is_passed {
                    set(&mut tensor, BLACK_PASSED_PAWNS, p.y, p.x, 1.0);
                }
            }
        }

        // Open files (channel 56)
        for file_x in 0i8..8 {
            let mut has_pawn = false;
            for check_y in 0i8..8 {
                if let Some(idx) = state.board[check_y as usize][file_x as usize] {
                    if state.pieces[idx as usize].piece_type() == PieceType::Pawn {
                        has_pawn = true;
                        break;
                    }
                }
            }
            if !has_pawn {
                for y in 0i8..8 {
                    set(&mut tensor, OPEN_FILES, y, file_x, 1.0);
                }
            }
        }

        // Damage potential (channel 57)
        for (i, &idx) in white_pieces.iter().enumerate() {
            let p = &state.pieces[idx as usize];
            let dmg = p.base_damage as f32 / 10.0;
            let mut bb = attacks.white_per_piece[i];
            while bb != 0 {
                let sq = bb.trailing_zeros() as usize;
                tensor[DAMAGE_POTENTIAL * 64 + sq] += dmg;
                bb &= bb - 1;
            }
        }
        for (i, &idx) in black_pieces.iter().enumerate() {
            let p = &state.pieces[idx as usize];
            let dmg = p.base_damage as f32 / 10.0;
            let mut bb = attacks.black_per_piece[i];
            while bb != 0 {
                let sq = bb.trailing_zeros() as usize;
                tensor[DAMAGE_POTENTIAL * 64 + sq] -= dmg;
                bb &= bb - 1;
            }
        }
        // Normalize to 0-1
        for i in 0..64 {
            tensor[DAMAGE_POTENTIAL * 64 + i] = (tensor[DAMAGE_POTENTIAL * 64 + i] + 1.0) / 2.0;
        }

        // ====================================================================
        // ABILITY TRACKING CHANNELS (58-61)
        // ====================================================================

        // Ability charges remaining (channel 58)
        for &idx in white_pieces.iter().chain(black_pieces.iter()) {
            let piece = &state.pieces[idx as usize];
            let pt = piece.piece_type() as usize;
            let (_, _, _, max_uses) = PIECE_STATS[pt];
            if max_uses == 0 {
                // Unlimited ability - always show as "full"
                set(&mut tensor, ABILITY_CHARGES, piece.y, piece.x, 1.0);
            } else if piece.ability_uses_remaining >= 0 {
                // Limited uses - normalize by max
                set(&mut tensor, ABILITY_CHARGES, piece.y, piece.x, piece.ability_uses_remaining as f32 / max_uses as f32);
            }
        }

        // Royal Decree turns remaining (channel 59)
        let rd_turns = if state.side_to_move == Team::White {
            state.white_royal_decree_turns
        } else {
            state.black_royal_decree_turns
        };
        fill(&mut tensor, ROYAL_DECREE_TURNS, rd_turns as f32 / 2.0);

        // RD combo potential (channel 60)
        if state.white_royal_decree_turns > 0 {
            for (i, &idx) in white_pieces.iter().enumerate() {
                let piece = &state.pieces[idx as usize];
                let piece_attacks = attacks.white_per_piece[i];
                let has_enemy_target = black_pieces.iter().any(|&bi| {
                    let bp = &state.pieces[bi as usize];
                    let sq = sq_idx(bp.x, bp.y);
                    (piece_attacks >> sq) & 1 != 0
                });
                if has_enemy_target {
                    set(&mut tensor, RD_COMBO_POTENTIAL, piece.y, piece.x, 1.0);
                }
            }
        }
        if state.black_royal_decree_turns > 0 {
            for (i, &idx) in black_pieces.iter().enumerate() {
                let piece = &state.pieces[idx as usize];
                let piece_attacks = attacks.black_per_piece[i];
                let has_enemy_target = white_pieces.iter().any(|&wi| {
                    let wp = &state.pieces[wi as usize];
                    let sq = sq_idx(wp.x, wp.y);
                    (piece_attacks >> sq) & 1 != 0
                });
                if has_enemy_target {
                    set(&mut tensor, RD_COMBO_POTENTIAL, piece.y, piece.x, 1.0);
                }
            }
        }

        // Promoted pieces (channel 61)
        for &idx in white_pieces.iter().chain(black_pieces.iter()) {
            let piece = &state.pieces[idx as usize];
            if piece.was_pawn() {
                set(&mut tensor, PROMOTED_PIECES, piece.y, piece.x, 1.0);
            }
        }

        // ====================================================================
        // NEW TACTICAL/STRATEGIC CHANNELS (62-74)
        // ====================================================================

        // Threat map - which pieces can be attacked next turn (channels 62-63)
        // Channel 62: White pieces that black can attack
        for &idx in &white_pieces {
            let p = &state.pieces[idx as usize];
            let sq = sq_idx(p.x, p.y);
            if (attacks.black >> sq) & 1 != 0 {
                set(&mut tensor, THREAT_MAP_WHITE, p.y, p.x, 1.0);
            }
        }
        // Channel 63: Black pieces that white can attack
        for &idx in &black_pieces {
            let p = &state.pieces[idx as usize];
            let sq = sq_idx(p.x, p.y);
            if (attacks.white >> sq) & 1 != 0 {
                set(&mut tensor, THREAT_MAP_BLACK, p.y, p.x, 1.0);
            }
        }

        // Interpose coverage - pieces with Rook LOS protection (channels 64-65)
        // A piece has coverage if a friendly Rook with charges is within 3 squares orthogonally
        for &idx in &white_pieces {
            let p = &state.pieces[idx as usize];
            if p.piece_type() == PieceType::Rook { continue; } // Rooks don't protect themselves

            // Check if any white Rook can protect this piece
            for &rook_idx in &white_pieces {
                let rook = &state.pieces[rook_idx as usize];
                if rook.piece_type() != PieceType::Rook { continue; }
                if !rook.can_use_ability() { continue; }

                // Check orthogonal LOS within 3 squares
                let dx = p.x - rook.x;
                let dy = p.y - rook.y;
                if (dx == 0 || dy == 0) && dx != dy { // Orthogonal, not same square
                    let dist = dx.abs() + dy.abs();
                    if dist > 0 && dist <= 3 {
                        // Check for blocking pieces
                        let step_x = if dx == 0 { 0 } else { dx.signum() };
                        let step_y = if dy == 0 { 0 } else { dy.signum() };
                        let mut blocked = false;
                        let mut cx = rook.x + step_x;
                        let mut cy = rook.y + step_y;
                        while (cx, cy) != (p.x, p.y) {
                            if state.get_piece_at(cx, cy).is_some() {
                                blocked = true;
                                break;
                            }
                            cx += step_x;
                            cy += step_y;
                        }
                        if !blocked {
                            set(&mut tensor, INTERPOSE_COVERAGE_WHITE, p.y, p.x, 1.0);
                            break;
                        }
                    }
                }
            }
        }
        // Same for black pieces
        for &idx in &black_pieces {
            let p = &state.pieces[idx as usize];
            if p.piece_type() == PieceType::Rook { continue; }

            for &rook_idx in &black_pieces {
                let rook = &state.pieces[rook_idx as usize];
                if rook.piece_type() != PieceType::Rook { continue; }
                if !rook.can_use_ability() { continue; }

                let dx = p.x - rook.x;
                let dy = p.y - rook.y;
                if (dx == 0 || dy == 0) && dx != dy {
                    let dist = dx.abs() + dy.abs();
                    if dist > 0 && dist <= 3 {
                        let step_x = if dx == 0 { 0 } else { dx.signum() };
                        let step_y = if dy == 0 { 0 } else { dy.signum() };
                        let mut blocked = false;
                        let mut cx = rook.x + step_x;
                        let mut cy = rook.y + step_y;
                        while (cx, cy) != (p.x, p.y) {
                            if state.get_piece_at(cx, cy).is_some() {
                                blocked = true;
                                break;
                            }
                            cx += step_x;
                            cy += step_y;
                        }
                        if !blocked {
                            set(&mut tensor, INTERPOSE_COVERAGE_BLACK, p.y, p.x, 1.0);
                            break;
                        }
                    }
                }
            }
        }

        // Consecration targets - damaged pieces in Bishop heal range (channel 66)
        // Bishops heal diagonally 1-3 squares
        for &idx in white_pieces.iter().chain(black_pieces.iter()) {
            let piece = &state.pieces[idx as usize];
            let team = piece.team();
            let hp_deficit = (piece.max_hp - piece.current_hp) as f32 / piece.max_hp as f32;
            if hp_deficit <= 0.0 { continue; } // Not damaged

            // Check if any friendly Bishop can heal this piece
            let friendly_pieces = if team == Team::White { &white_pieces } else { &black_pieces };
            for &bishop_idx in friendly_pieces {
                let bishop = &state.pieces[bishop_idx as usize];
                if bishop.piece_type() != PieceType::Bishop { continue; }
                if !bishop.can_use_ability() { continue; }

                // Check diagonal distance 1-3
                let dx = (piece.x - bishop.x).abs();
                let dy = (piece.y - bishop.y).abs();
                if dx == dy && dx >= 1 && dx <= 3 {
                    set(&mut tensor, CONSECRATION_TARGETS, piece.y, piece.x, hp_deficit);
                    break;
                }
            }
        }

        // Forcing moves - pieces under attack by higher-value attacker (channel 67)
        for &idx in &white_pieces {
            let p = &state.pieces[idx as usize];
            let sq = sq_idx(p.x, p.y);
            if (attacks.black >> sq) & 1 == 0 { continue; } // Not under attack

            let piece_value = PIECE_VALUES[p.piece_type() as usize];
            // Check if any attacker is lower value (forcing us to respond)
            for (i, &attacker_idx) in black_pieces.iter().enumerate() {
                let attacker = &state.pieces[attacker_idx as usize];
                if (attacks.black_per_piece[i] >> sq) & 1 != 0 {
                    let attacker_value = PIECE_VALUES[attacker.piece_type() as usize];
                    if attacker_value < piece_value || p.piece_type() == PieceType::King {
                        set(&mut tensor, FORCING_MOVES, p.y, p.x, 1.0);
                        break;
                    }
                }
            }
        }
        for &idx in &black_pieces {
            let p = &state.pieces[idx as usize];
            let sq = sq_idx(p.x, p.y);
            if (attacks.white >> sq) & 1 == 0 { continue; }

            let piece_value = PIECE_VALUES[p.piece_type() as usize];
            for (i, &attacker_idx) in white_pieces.iter().enumerate() {
                let attacker = &state.pieces[attacker_idx as usize];
                if (attacks.white_per_piece[i] >> sq) & 1 != 0 {
                    let attacker_value = PIECE_VALUES[attacker.piece_type() as usize];
                    if attacker_value < piece_value || p.piece_type() == PieceType::King {
                        set(&mut tensor, FORCING_MOVES, p.y, p.x, 1.0);
                        break;
                    }
                }
            }
        }

        // Enhanced RD combo potential (channel 68)
        // Shows how many enemies each piece can attack during RD, normalized
        if state.white_royal_decree_turns > 0 {
            for (i, &idx) in white_pieces.iter().enumerate() {
                let piece = &state.pieces[idx as usize];
                let piece_attacks = attacks.white_per_piece[i];
                let mut enemy_count = 0;
                for &bi in &black_pieces {
                    let bp = &state.pieces[bi as usize];
                    let sq = sq_idx(bp.x, bp.y);
                    if (piece_attacks >> sq) & 1 != 0 {
                        enemy_count += 1;
                    }
                }
                if enemy_count > 0 {
                    set(&mut tensor, RD_COMBO_ENHANCED, piece.y, piece.x, (enemy_count as f32 / 6.0).min(1.0));
                }
            }
        }
        if state.black_royal_decree_turns > 0 {
            for (i, &idx) in black_pieces.iter().enumerate() {
                let piece = &state.pieces[idx as usize];
                let piece_attacks = attacks.black_per_piece[i];
                let mut enemy_count = 0;
                for &wi in &white_pieces {
                    let wp = &state.pieces[wi as usize];
                    let sq = sq_idx(wp.x, wp.y);
                    if (piece_attacks >> sq) & 1 != 0 {
                        enemy_count += 1;
                    }
                }
                if enemy_count > 0 {
                    set(&mut tensor, RD_COMBO_ENHANCED, piece.y, piece.x, (enemy_count as f32 / 6.0).min(1.0));
                }
            }
        }

        // Pawn promotion distance (channel 69)
        for &idx in &white_pieces {
            let p = &state.pieces[idx as usize];
            if p.piece_type() == PieceType::Pawn {
                let dist_to_promotion = 7 - p.y; // White promotes at y=7
                set(&mut tensor, PAWN_PROMOTION_DIST, p.y, p.x, (7 - dist_to_promotion) as f32 / 7.0);
            }
        }
        for &idx in &black_pieces {
            let p = &state.pieces[idx as usize];
            if p.piece_type() == PieceType::Pawn {
                let dist_to_promotion = p.y; // Black promotes at y=0
                set(&mut tensor, PAWN_PROMOTION_DIST, p.y, p.x, (7 - dist_to_promotion) as f32 / 7.0);
            }
        }

        // Enemy RD turns remaining (channel 70)
        let enemy_rd_turns = if state.side_to_move == Team::White {
            state.black_royal_decree_turns
        } else {
            state.white_royal_decree_turns
        };
        fill(&mut tensor, ENEMY_RD_TURNS, enemy_rd_turns as f32 / 2.0);

        // Enemy abilities ready (channel 71)
        let enemy_pieces = if state.side_to_move == Team::White { &black_pieces } else { &white_pieces };
        let enemy_ready: usize = enemy_pieces.iter().filter(|&&i| state.pieces[i as usize].can_use_ability()).count();
        if !enemy_pieces.is_empty() {
            fill(&mut tensor, ENEMY_ABILITIES_READY, enemy_ready as f32 / enemy_pieces.len() as f32);
        }

        // King proximity map - distance from each piece to enemy king (channel 72)
        if let Some(bk_idx) = black_king_idx {
            let bk = &state.pieces[bk_idx as usize];
            for &idx in &white_pieces {
                let p = &state.pieces[idx as usize];
                let dist = (p.x - bk.x).abs() + (p.y - bk.y).abs();
                set(&mut tensor, KING_PROXIMITY_MAP, p.y, p.x, 1.0 - dist as f32 / 14.0);
            }
        }
        if let Some(wk_idx) = white_king_idx {
            let wk = &state.pieces[wk_idx as usize];
            for &idx in &black_pieces {
                let p = &state.pieces[idx as usize];
                let dist = (p.x - wk.x).abs() + (p.y - wk.y).abs();
                set(&mut tensor, KING_PROXIMITY_MAP, p.y, p.x, 1.0 - dist as f32 / 14.0);
            }
        }

        // Safe attack squares - where we can attack without counter (channel 73)
        // For each square we attack, check if enemy can counter-attack that square
        for sq in 0..64 {
            let x = (sq % 8) as i8;
            let y = (sq / 8) as i8;

            // If white attacks this square and black doesn't, it's safe for white
            if (attacks.white >> sq) & 1 != 0 && (attacks.black >> sq) & 1 == 0 {
                // Check if there's a black piece there (actual target)
                if let Some(piece_idx) = state.get_piece_at(x, y) {
                    if state.pieces[piece_idx as usize].team() == Team::Black {
                        set(&mut tensor, SAFE_ATTACK_SQUARES, y, x, 1.0);
                    }
                }
            }
            // If black attacks and white doesn't, safe for black
            if (attacks.black >> sq) & 1 != 0 && (attacks.white >> sq) & 1 == 0 {
                if let Some(piece_idx) = state.get_piece_at(x, y) {
                    if state.pieces[piece_idx as usize].team() == Team::White {
                        set(&mut tensor, SAFE_ATTACK_SQUARES, y, x, 1.0);
                    }
                }
            }
        }

        // Enemy ability charges (channel 74)
        for &idx in enemy_pieces {
            let piece = &state.pieces[idx as usize];
            let pt = piece.piece_type() as usize;
            let (_, _, _, max_uses) = PIECE_STATS[pt];
            if max_uses == 0 {
                set(&mut tensor, ENEMY_ABILITY_CHARGES, piece.y, piece.x, 1.0);
            } else if piece.ability_uses_remaining >= 0 {
                set(&mut tensor, ENEMY_ABILITY_CHARGES, piece.y, piece.x, piece.ability_uses_remaining as f32 / max_uses as f32);
            }
        }

        // Convert f32 -> f16 and return
        // Use slice conversion for efficiency
        tensor.iter().map(|&v| f16::from_f32(v)).collect()
    })
}

/// Wrapper struct for tensor computation that owns a GameSimulator.
/// Used internally by MCTS and 1-ply evaluation.
pub struct TensorComputer {
    pub sim: GameSimulator,
}

impl TensorComputer {
    pub fn new(state: GameState) -> Self {
        Self {
            sim: GameSimulator::with_state(state),
        }
    }

    pub fn compute_tensor(&self) -> Vec<f16> {
        compute_tensor(&self.sim.state)
    }
}
