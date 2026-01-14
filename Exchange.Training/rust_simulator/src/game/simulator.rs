//! Game simulator - the core game engine.
//!
//! GameSimulator handles move generation, move execution, and game state updates.
//! It uses the make/unmake pattern for efficient search.

use std::cell::RefCell;

use crate::core::{
    is_on_board, sq_idx, AbilityId, Move, MoveType, PieceType, Team, UndoInfo,
    ALL_DIRS, CARDINAL, DIAGONAL, KNIGHT_MOVES, PIECE_STATS, ZOBRIST,
};
use super::state::GameState;

// Thread-local move buffer to avoid allocations
thread_local! {
    pub static MOVE_BUFFER: RefCell<Vec<Move>> = RefCell::new(Vec::with_capacity(256));
}

// ============================================================================
// GameSimulator
// ============================================================================

/// The core game engine for EXCHANGE.
pub struct GameSimulator {
    /// Current game state
    pub state: GameState,
    /// Random number generator for dice rolls
    rng: fastrand::Rng,
}

impl GameSimulator {
    /// Create a new game with standard starting position.
    pub fn new() -> Self {
        GameSimulator {
            state: GameState::new(),
            rng: fastrand::Rng::new(),
        }
    }

    /// Create a simulator with a specific game state.
    pub fn with_state(state: GameState) -> Self {
        GameSimulator {
            state,
            rng: fastrand::Rng::new(),
        }
    }

    /// Reset to starting position.
    pub fn reset(&mut self) -> &GameState {
        self.state = GameState::new();
        &self.state
    }

    /// Set the random seed for deterministic play.
    pub fn set_seed(&mut self, seed: u64) {
        self.rng = fastrand::Rng::with_seed(seed);
    }

    /// Check if a position is occupied.
    #[inline]
    fn is_occupied(&self, x: i8, y: i8) -> bool {
        self.state.get_piece_at(x, y).is_some()
    }

    // ========================================================================
    // Move Generation
    // ========================================================================

    /// Generate all legal moves for a team.
    pub fn generate_moves(&self, team: Team) -> Vec<Move> {
        MOVE_BUFFER.with(|buf| {
            let mut moves = buf.borrow_mut();
            moves.clear();

            let piece_indices = self.state.get_pieces_for_team(team);

            for &piece_idx in piece_indices {
                let piece = &self.state.pieces[piece_idx as usize];
                if !piece.is_alive() { continue; }

                // Generate attacks first
                if piece.piece_type() != PieceType::Knight {
                    self.generate_attacks(piece_idx, &mut moves);
                }

                // Generate regular moves
                self.generate_piece_moves(piece_idx, &mut moves);

                // Knight combos
                if piece.piece_type() == PieceType::Knight {
                    self.generate_knight_combos(piece_idx, &mut moves);
                }

                // Abilities
                if piece.can_use_ability() {
                    self.generate_ability_moves(piece_idx, &mut moves);
                }
            }

            moves.clone()
        })
    }

    fn generate_piece_moves(&self, piece_idx: u8, moves: &mut Vec<Move>) {
        let piece = &self.state.pieces[piece_idx as usize];
        let from = (piece.x, piece.y);

        match piece.piece_type() {
            PieceType::King => self.gen_king_moves(piece_idx, from, moves),
            PieceType::Queen => self.gen_sliding_moves(piece_idx, from, &ALL_DIRS, 7, moves),
            PieceType::Rook => self.gen_sliding_moves(piece_idx, from, &CARDINAL, 7, moves),
            PieceType::Bishop => self.gen_sliding_moves(piece_idx, from, &DIAGONAL, 7, moves),
            PieceType::Knight => self.gen_knight_moves(piece_idx, from, moves),
            PieceType::Pawn => self.gen_pawn_moves(piece_idx, from, piece.team(), moves),
        }
    }

    fn gen_king_moves(&self, idx: u8, from: (i8, i8), moves: &mut Vec<Move>) {
        for (dx, dy) in ALL_DIRS {
            let nx = from.0 + dx;
            let ny = from.1 + dy;
            if is_on_board(nx, ny) && !self.is_occupied(nx, ny) {
                moves.push(Move::new_move(idx, from, (nx, ny)));
            }
        }
    }

    fn gen_sliding_moves(&self, idx: u8, from: (i8, i8), dirs: &[(i8, i8)], max_dist: i8, moves: &mut Vec<Move>) {
        for &(dx, dy) in dirs {
            for dist in 1..=max_dist {
                let nx = from.0 + dx * dist;
                let ny = from.1 + dy * dist;
                if !is_on_board(nx, ny) { break; }
                if self.is_occupied(nx, ny) { break; }
                moves.push(Move::new_move(idx, from, (nx, ny)));
            }
        }
    }

    fn gen_knight_moves(&self, idx: u8, from: (i8, i8), moves: &mut Vec<Move>) {
        for (dx, dy) in KNIGHT_MOVES {
            let nx = from.0 + dx;
            let ny = from.1 + dy;
            if is_on_board(nx, ny) && !self.is_occupied(nx, ny) {
                moves.push(Move::new_move(idx, from, (nx, ny)));
            }
        }
    }

    fn gen_pawn_moves(&self, idx: u8, from: (i8, i8), team: Team, moves: &mut Vec<Move>) {
        let dir: i8 = if team == Team::White { 1 } else { -1 };
        let start_row: i8 = if team == Team::White { 1 } else { 6 };

        let ny = from.1 + dir;
        if is_on_board(from.0, ny) && !self.is_occupied(from.0, ny) {
            moves.push(Move::new_move(idx, from, (from.0, ny)));

            if from.1 == start_row {
                let ny2 = from.1 + 2 * dir;
                if !self.is_occupied(from.0, ny2) {
                    moves.push(Move::new_move(idx, from, (from.0, ny2)));
                }
            }
        }
    }

    fn generate_attacks(&self, piece_idx: u8, moves: &mut Vec<Move>) {
        let piece = &self.state.pieces[piece_idx as usize];
        let from = (piece.x, piece.y);
        let team = piece.team();

        match piece.piece_type() {
            PieceType::King => {
                for (dx, dy) in ALL_DIRS {
                    let nx = from.0 + dx;
                    let ny = from.1 + dy;
                    if is_on_board(nx, ny) {
                        if let Some(target_idx) = self.state.get_piece_at(nx, ny) {
                            if self.state.pieces[target_idx as usize].team() != team {
                                moves.push(Move::new_attack(piece_idx, from, (nx, ny)));
                            }
                        }
                    }
                }
            }
            PieceType::Queen => {
                const QUEEN_RANGE: i8 = 3;
                for (dx, dy) in ALL_DIRS {
                    for dist in 1..=QUEEN_RANGE {
                        let nx = from.0 + dx * dist;
                        let ny = from.1 + dy * dist;
                        if !is_on_board(nx, ny) { break; }
                        if let Some(target_idx) = self.state.get_piece_at(nx, ny) {
                            if self.state.pieces[target_idx as usize].team() != team {
                                moves.push(Move::new_attack(piece_idx, from, (nx, ny)));
                            }
                            break;
                        }
                    }
                }
            }
            PieceType::Rook => {
                const ROOK_RANGE: i8 = 2;
                // Adjacent (all 8 dirs)
                for (dx, dy) in ALL_DIRS {
                    let nx = from.0 + dx;
                    let ny = from.1 + dy;
                    if is_on_board(nx, ny) {
                        if let Some(target_idx) = self.state.get_piece_at(nx, ny) {
                            if self.state.pieces[target_idx as usize].team() != team {
                                moves.push(Move::new_attack(piece_idx, from, (nx, ny)));
                            }
                        }
                    }
                }
                // Cardinal extended
                for (dx, dy) in CARDINAL {
                    for dist in 2..=ROOK_RANGE {
                        let nx = from.0 + dx * dist;
                        let ny = from.1 + dy * dist;
                        if !is_on_board(nx, ny) { break; }
                        if let Some(target_idx) = self.state.get_piece_at(nx, ny) {
                            if self.state.pieces[target_idx as usize].team() != team {
                                moves.push(Move::new_attack(piece_idx, from, (nx, ny)));
                            }
                            break;
                        }
                        // Check if blocked
                        let check_x = from.0 + dx * (dist - 1);
                        let check_y = from.1 + dy * (dist - 1);
                        if dist > 1 && self.state.board[check_y as usize][check_x as usize].is_some() {
                            break;
                        }
                    }
                }
            }
            PieceType::Bishop => {
                const BISHOP_MIN: i8 = 2;
                const BISHOP_MAX: i8 = 3;
                for (dx, dy) in DIAGONAL {
                    for dist in 1..=BISHOP_MAX {
                        let nx = from.0 + dx * dist;
                        let ny = from.1 + dy * dist;
                        if !is_on_board(nx, ny) { break; }
                        if let Some(target_idx) = self.state.get_piece_at(nx, ny) {
                            if dist >= BISHOP_MIN && self.state.pieces[target_idx as usize].team() != team {
                                moves.push(Move::new_attack(piece_idx, from, (nx, ny)));
                            }
                            break;
                        }
                    }
                }
            }
            PieceType::Knight => {
                // Knight attacks via combos only
            }
            PieceType::Pawn => {
                let dir: i8 = if team == Team::White { 1 } else { -1 };
                for dx in [-1i8, 1] {
                    let nx = from.0 + dx;
                    let ny = from.1 + dir;
                    if is_on_board(nx, ny) {
                        if let Some(target_idx) = self.state.get_piece_at(nx, ny) {
                            if self.state.pieces[target_idx as usize].team() != team {
                                moves.push(Move::new_attack(piece_idx, from, (nx, ny)));
                            }
                        }
                    }
                }
            }
        }
    }

    fn generate_knight_combos(&self, piece_idx: u8, moves: &mut Vec<Move>) {
        let piece = &self.state.pieces[piece_idx as usize];
        let from = (piece.x, piece.y);
        let team = piece.team();

        for (dx, dy) in KNIGHT_MOVES {
            let mx = from.0 + dx;
            let my = from.1 + dy;
            if !is_on_board(mx, my) || self.is_occupied(mx, my) { continue; }

            // From landing position, check cardinal adjacent for attacks
            for (adx, ady) in CARDINAL {
                let ax = mx + adx;
                let ay = my + ady;
                if !is_on_board(ax, ay) { continue; }
                if let Some(target_idx) = self.state.get_piece_at(ax, ay) {
                    if self.state.pieces[target_idx as usize].team() != team {
                        moves.push(Move::new_move_and_attack(piece_idx, from, (mx, my), (ax, ay)));
                    }
                }
            }
        }
    }

    fn generate_ability_moves(&self, piece_idx: u8, moves: &mut Vec<Move>) {
        let piece = &self.state.pieces[piece_idx as usize];
        let ability_id = AbilityId::for_piece_type(piece.piece_type());
        let from = (piece.x, piece.y);
        let team = piece.team();

        match ability_id {
            AbilityId::RoyalDecree => {
                moves.push(Move::new_ability(piece_idx, from, from, AbilityId::RoyalDecree));
            }
            AbilityId::Overextend => {
                // Move then attack
                for (dx, dy) in ALL_DIRS {
                    for dist in 1..8 {
                        let mx = from.0 + dx * dist;
                        let my = from.1 + dy * dist;
                        if !is_on_board(mx, my) { break; }
                        if self.is_occupied(mx, my) { break; }

                        // From move position, find attack targets
                        for (adx, ady) in ALL_DIRS {
                            for adist in 1..=3 {
                                let ax = mx + adx * adist;
                                let ay = my + ady * adist;
                                if !is_on_board(ax, ay) { break; }
                                if let Some(target_idx) = self.state.get_piece_at(ax, ay) {
                                    if self.state.pieces[target_idx as usize].team() != team {
                                        moves.push(Move::new_ability_with_target(
                                            piece_idx, from, (mx, my), Some((ax, ay)), None, AbilityId::Overextend
                                        ));
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            AbilityId::Interpose => {
                // REWORKED: Interpose is now REACTIVE - no proactive ability to activate.
                // It triggers automatically when an ally within orthogonal LOS (3 squares)
                // takes damage and Rook has uses remaining and is not on cooldown.
                // See apply_damage_with_interpose for the reactive trigger logic.
                // No moves to generate.
            }
            AbilityId::Consecration => {
                for (dx, dy) in DIAGONAL {
                    for dist in 1..=3 {
                        let nx = from.0 + dx * dist;
                        let ny = from.1 + dy * dist;
                        if !is_on_board(nx, ny) { break; }
                        if let Some(target_idx) = self.state.get_piece_at(nx, ny) {
                            let target = &self.state.pieces[target_idx as usize];
                            if target.team() == team && target.current_hp < target.max_hp {
                                moves.push(Move::new_ability_with_target(
                                    piece_idx, from, from, None, Some((nx, ny)), AbilityId::Consecration
                                ));
                            }
                            break;
                        }
                    }
                }
            }
            AbilityId::Skirmish => {
                // Attack then reposition
                for (dx, dy) in CARDINAL {
                    let ax = from.0 + dx;
                    let ay = from.1 + dy;
                    if let Some(target_idx) = self.state.get_piece_at(ax, ay) {
                        if self.state.pieces[target_idx as usize].team() != team {
                            for (rdx, rdy) in ALL_DIRS {
                                let rx = from.0 + rdx;
                                let ry = from.1 + rdy;
                                if is_on_board(rx, ry) && !self.is_occupied(rx, ry) {
                                    moves.push(Move::new_ability_with_target(
                                        piece_idx, from, (rx, ry), Some((ax, ay)), None, AbilityId::Skirmish
                                    ));
                                }
                            }
                        }
                    }
                }
            }
            AbilityId::Advance => {
                if piece.advance_cooldown_turns > 0 { return; }

                let dir: i8 = if team == Team::White { 1 } else { -1 };
                let ny1 = from.1 + dir;
                if is_on_board(from.0, ny1) && !self.is_occupied(from.0, ny1) {
                    let ny2 = from.1 + 2 * dir;
                    if is_on_board(from.0, ny2) && !self.is_occupied(from.0, ny2) {
                        moves.push(Move::new_ability(piece_idx, from, (from.0, ny2), AbilityId::Advance));
                    }
                }
            }
        }
    }

    // ========================================================================
    // Move Execution
    // ========================================================================

    /// Execute a move and return (damage_dealt, undo_info).
    pub fn make_move(&mut self, mv: &Move) -> (i32, UndoInfo) {
        let mut undo = UndoInfo::new(mv.piece_idx, self.state.pieces[mv.piece_idx as usize]);
        undo.old_board_from = self.state.board[mv.from_y as usize][mv.from_x as usize];
        undo.old_board_to = self.state.board[mv.to_y as usize][mv.to_x as usize];
        undo.old_hash = self.state.hash;
        undo.old_white_royal_decree_turns = self.state.white_royal_decree_turns;
        undo.old_black_royal_decree_turns = self.state.black_royal_decree_turns;
        undo.old_moves_without_damage = self.state.moves_without_damage;
        undo.old_side_to_move = self.state.side_to_move;
        undo.old_turn_number = self.state.turn_number;

        let mut damage_dealt = 0i32;

        match mv.move_type {
            MoveType::Move => {
                self.move_piece(mv.piece_idx, mv.from_x, mv.from_y, mv.to_x, mv.to_y);
                self.check_promotion(mv.piece_idx);
            }
            MoveType::Attack => {
                if let Some(target_idx) = self.state.get_piece_at(mv.to_x, mv.to_y) {
                    undo.target_idx = Some(target_idx);
                    undo.target_snapshot = Some(self.state.pieces[target_idx as usize]);
                    undo.old_board_attack = self.state.board[mv.to_y as usize][mv.to_x as usize];
                    damage_dealt = self.apply_damage(mv.piece_idx, target_idx, &mut undo);
                }
            }
            MoveType::MoveAndAttack => {
                self.move_piece(mv.piece_idx, mv.from_x, mv.from_y, mv.to_x, mv.to_y);
                if mv.attack_x >= 0 {
                    if let Some(target_idx) = self.state.get_piece_at(mv.attack_x, mv.attack_y) {
                        undo.target_idx = Some(target_idx);
                        undo.target_snapshot = Some(self.state.pieces[target_idx as usize]);
                        undo.old_board_attack = self.state.board[mv.attack_y as usize][mv.attack_x as usize];
                        damage_dealt = self.apply_damage(mv.piece_idx, target_idx, &mut undo);
                    }
                }
            }
            MoveType::Ability => {
                damage_dealt = self.execute_ability(mv, &mut undo);
            }
        }

        // Update game state
        if damage_dealt > 0 {
            self.state.moves_without_damage = 0;
        } else {
            self.state.moves_without_damage += 1;
        }

        // Switch sides
        self.state.hash ^= ZOBRIST.side_to_move;
        self.state.side_to_move = self.state.side_to_move.opposite();
        self.state.turn_number += 1;

        // Decrement Royal Decree turns for the team that just moved
        let moved_team = self.state.pieces[mv.piece_idx as usize].team();
        let zobrist = &*ZOBRIST;
        match moved_team {
            Team::White => {
                if self.state.white_royal_decree_turns > 0 {
                    if self.state.white_royal_decree_turns == 1 {
                        self.state.hash ^= zobrist.royal_decree[0];
                    }
                    self.state.white_royal_decree_turns -= 1;
                }
            }
            Team::Black => {
                if self.state.black_royal_decree_turns > 0 {
                    if self.state.black_royal_decree_turns == 1 {
                        self.state.hash ^= zobrist.royal_decree[1];
                    }
                    self.state.black_royal_decree_turns -= 1;
                }
            }
        }

        // Tick cooldowns
        for idx in 0..self.state.piece_count {
            let piece = &mut self.state.pieces[idx as usize];
            if piece.team() == moved_team && piece.is_alive() {
                if piece.ability_cooldown > 0 {
                    piece.ability_cooldown -= 1;
                }
                // NOTE: Interpose is now reactive (triggers automatically on ally damage)
                // so we no longer need to deactivate it. Keeping the field for compatibility.
            }
        }

        // Decrement Advance cooldown
        for idx in 0..self.state.piece_count {
            let piece = &mut self.state.pieces[idx as usize];
            if piece.team() == self.state.side_to_move
                && piece.piece_type() == PieceType::Pawn
                && piece.advance_cooldown_turns > 0
            {
                piece.advance_cooldown_turns -= 1;
            }
        }

        self.state.record_position();
        self.state.check_terminal();

        (damage_dealt, undo)
    }

    /// Undo a move using saved undo info.
    pub fn unmake_move(&mut self, undo: &UndoInfo) {
        // Restore game state
        self.state.hash = undo.old_hash;
        self.state.white_royal_decree_turns = undo.old_white_royal_decree_turns;
        self.state.black_royal_decree_turns = undo.old_black_royal_decree_turns;
        self.state.moves_without_damage = undo.old_moves_without_damage;
        self.state.side_to_move = undo.old_side_to_move;
        self.state.turn_number = undo.old_turn_number;
        self.state.is_terminal = false;
        self.state.is_draw = false;
        self.state.winner = None;

        // Pop position history
        self.state.position_history.pop();

        // Restore board state at destination
        let current_piece = &self.state.pieces[undo.piece_idx as usize];
        self.state.board[current_piece.y as usize][current_piece.x as usize] = undo.old_board_to;

        // Restore pieces
        self.state.pieces[undo.piece_idx as usize] = undo.piece_snapshot;

        // Restore board state at source
        self.state.board[undo.piece_snapshot.y as usize][undo.piece_snapshot.x as usize] = undo.old_board_from;

        if let Some(target_snapshot) = &undo.target_snapshot {
            let target_idx = undo.target_idx.unwrap();
            self.state.pieces[target_idx as usize] = *target_snapshot;
            self.state.board[target_snapshot.y as usize][target_snapshot.x as usize] = Some(target_idx);
        }

        if let Some(secondary_snapshot) = &undo.secondary_snapshot {
            let secondary_idx = undo.secondary_idx.unwrap();
            self.state.pieces[secondary_idx as usize] = *secondary_snapshot;
        }

        if let Some(rook_snapshot) = &undo.interpose_rook_snapshot {
            let rook_idx = undo.interpose_rook_idx.unwrap();
            self.state.pieces[rook_idx as usize] = *rook_snapshot;
        }
    }

    #[inline]
    fn move_piece(&mut self, idx: u8, from_x: i8, from_y: i8, to_x: i8, to_y: i8) {
        self.state.hash_move_piece(idx, from_x, from_y, to_x, to_y);
        self.state.board[from_y as usize][from_x as usize] = None;
        self.state.board[to_y as usize][to_x as usize] = Some(idx);
        self.state.pieces[idx as usize].x = to_x;
        self.state.pieces[idx as usize].y = to_y;
    }

    fn apply_damage(&mut self, attacker_idx: u8, target_idx: u8, undo: &mut UndoInfo) -> i32 {
        let dice_roll = (self.rng.u32(0..6) + 1) as i16;
        let attacker = &self.state.pieces[attacker_idx as usize];
        let mut damage = attacker.base_damage + dice_roll;

        // Royal Decree bonus (+2)
        damage += self.state.get_royal_decree_bonus(attacker.team());

        self.apply_damage_with_interpose(target_idx, damage, undo)
    }

    fn apply_damage_with_interpose(&mut self, target_idx: u8, damage: i16, undo: &mut UndoInfo) -> i32 {
        // REWORKED INTERPOSE: Now triggers automatically (reactive) when:
        // 1. Target is an ally of a Rook
        // 2. Rook has orthogonal LOS to target (same row or column)
        // 3. Distance is 1-3 squares
        // 4. No blocking pieces in the path
        // 5. Rook has ability uses remaining and is not on cooldown

        const INTERPOSE_RANGE: i8 = 3;

        let target = &self.state.pieces[target_idx as usize];
        let target_team = target.team();
        let target_x = target.x;
        let target_y = target.y;
        let old_hp_bucket = target.hp_bucket();

        // Find a Rook that can reactively Interpose
        let mut interpose_rook: Option<u8> = None;
        for idx in 0..self.state.piece_count {
            let p = &self.state.pieces[idx as usize];
            if p.piece_type() != PieceType::Rook { continue; }
            if p.team() != target_team { continue; }
            if idx == target_idx as u8 { continue; }
            if !p.is_alive() { continue; }
            if !p.can_use_ability() { continue; }  // On cooldown or no charges left

            // Check orthogonal LOS (same row or column)
            let dx = p.x - target_x;
            let dy = p.y - target_y;

            // Must be orthogonal (one axis zero, other non-zero)
            if !((dx == 0) != (dy == 0)) { continue; }  // XOR - exactly one must be 0

            let dist = dx.abs() + dy.abs();
            if dist < 1 || dist > INTERPOSE_RANGE { continue; }

            // Check for blocking pieces in the path
            let step_x: i8 = if dx == 0 { 0 } else if dx > 0 { 1 } else { -1 };
            let step_y: i8 = if dy == 0 { 0 } else if dy > 0 { 1 } else { -1 };

            let mut blocked = false;
            let mut check_x = target_x + step_x;
            let mut check_y = target_y + step_y;
            while (check_x, check_y) != (p.x, p.y) {
                if self.state.get_piece_at(check_x, check_y).is_some() {
                    blocked = true;
                    break;
                }
                check_x += step_x;
                check_y += step_y;
            }

            if blocked { continue; }

            // Found a valid Rook!
            interpose_rook = Some(idx);
            break;
        }

        if let Some(rook_idx) = interpose_rook {
            undo.interpose_rook_idx = Some(rook_idx);
            undo.interpose_rook_snapshot = Some(self.state.pieces[rook_idx as usize]);

            let rook_old_bucket = self.state.pieces[rook_idx as usize].hp_bucket();

            let half = damage / 2;
            let remainder = damage % 2;

            self.state.pieces[target_idx as usize].current_hp -= half;
            self.state.pieces[rook_idx as usize].current_hp -= half + remainder;

            // Consume a use and apply cooldown (reactive trigger)
            if self.state.pieces[rook_idx as usize].ability_uses_remaining > 0 {
                self.state.pieces[rook_idx as usize].ability_uses_remaining -= 1;
            }
            self.state.pieces[rook_idx as usize].ability_cooldown = PIECE_STATS[PieceType::Rook as usize].2;

            // Update hash for HP changes
            let target_new_bucket = self.state.pieces[target_idx as usize].hp_bucket();
            let rook_new_bucket = self.state.pieces[rook_idx as usize].hp_bucket();

            self.state.hash_hp_change(target_idx, old_hp_bucket, target_new_bucket);
            self.state.hash_hp_change(rook_idx, rook_old_bucket, rook_new_bucket);

            // Check deaths
            if self.state.pieces[target_idx as usize].current_hp <= 0 {
                self.state.hash_remove_piece(target_idx);
                self.state.board[target_y as usize][target_x as usize] = None;
            }
            if self.state.pieces[rook_idx as usize].current_hp <= 0 {
                let rook_x = self.state.pieces[rook_idx as usize].x;
                let rook_y = self.state.pieces[rook_idx as usize].y;
                self.state.hash_remove_piece(rook_idx);
                self.state.board[rook_y as usize][rook_x as usize] = None;
            }
        } else {
            self.state.pieces[target_idx as usize].current_hp -= damage;

            let new_hp_bucket = self.state.pieces[target_idx as usize].hp_bucket();
            self.state.hash_hp_change(target_idx, old_hp_bucket, new_hp_bucket);

            if self.state.pieces[target_idx as usize].current_hp <= 0 {
                self.state.hash_remove_piece(target_idx);
                self.state.board[target_y as usize][target_x as usize] = None;
            }
        }

        damage as i32
    }

    fn execute_ability(&mut self, mv: &Move, undo: &mut UndoInfo) -> i32 {
        let ability_id = mv.ability_id.expect("Ability move must have ability_id");
        let piece_team = self.state.pieces[mv.piece_idx as usize].team();
        let mut damage_dealt = 0i32;

        match ability_id {
            AbilityId::RoyalDecree => {
                let zobrist = &*ZOBRIST;
                match piece_team {
                    Team::White => {
                        if self.state.white_royal_decree_turns == 0 {
                            self.state.hash ^= zobrist.royal_decree[0];
                        }
                        self.state.white_royal_decree_turns = 2;
                    }
                    Team::Black => {
                        if self.state.black_royal_decree_turns == 0 {
                            self.state.hash ^= zobrist.royal_decree[1];
                        }
                        self.state.black_royal_decree_turns = 2;
                    }
                }
                if self.state.pieces[mv.piece_idx as usize].ability_uses_remaining > 0 {
                    self.state.pieces[mv.piece_idx as usize].ability_uses_remaining -= 1;
                }
            }
            AbilityId::Overextend => {
                self.move_piece(mv.piece_idx, mv.from_x, mv.from_y, mv.to_x, mv.to_y);

                if mv.attack_x >= 0 {
                    if let Some(target_idx) = self.state.get_piece_at(mv.attack_x, mv.attack_y) {
                        undo.target_idx = Some(target_idx);
                        undo.target_snapshot = Some(self.state.pieces[target_idx as usize]);
                        damage_dealt = self.apply_damage(mv.piece_idx, target_idx, undo);
                    }
                }

                // Self-damage
                let old_bucket = self.state.pieces[mv.piece_idx as usize].hp_bucket();
                self.state.pieces[mv.piece_idx as usize].current_hp -= 2;
                let new_bucket = self.state.pieces[mv.piece_idx as usize].hp_bucket();
                self.state.hash_hp_change(mv.piece_idx, old_bucket, new_bucket);

                self.state.pieces[mv.piece_idx as usize].ability_cooldown = PIECE_STATS[PieceType::Queen as usize].2;
            }
            AbilityId::Interpose => {
                // REWORKED: Interpose is now reactive - this code should never execute
                // since we no longer generate proactive Interpose moves.
                // Keeping for backwards compatibility with saved replays.
            }
            AbilityId::Consecration => {
                if mv.ability_target_x >= 0 {
                    if let Some(target_idx) = self.state.get_piece_at(mv.ability_target_x, mv.ability_target_y) {
                        let target = &self.state.pieces[target_idx as usize];
                        if target.team() == piece_team {
                            undo.secondary_idx = Some(target_idx);
                            undo.secondary_snapshot = Some(*target);

                            let old_bucket = target.hp_bucket();
                            let mut heal = (self.rng.u32(0..6) + 1) as i16;
                            heal += self.state.get_royal_decree_bonus(piece_team);

                            let target = &mut self.state.pieces[target_idx as usize];
                            target.current_hp = (target.current_hp + heal).min(target.max_hp);

                            let new_bucket = target.hp_bucket();
                            self.state.hash_hp_change(target_idx, old_bucket, new_bucket);
                        }
                    }
                }
                self.state.pieces[mv.piece_idx as usize].ability_cooldown = PIECE_STATS[PieceType::Bishop as usize].2;
                if self.state.pieces[mv.piece_idx as usize].ability_uses_remaining > 0 {
                    self.state.pieces[mv.piece_idx as usize].ability_uses_remaining -= 1;
                }
            }
            AbilityId::Skirmish => {
                if mv.attack_x >= 0 {
                    if let Some(target_idx) = self.state.get_piece_at(mv.attack_x, mv.attack_y) {
                        undo.target_idx = Some(target_idx);
                        undo.target_snapshot = Some(self.state.pieces[target_idx as usize]);
                        damage_dealt = self.apply_damage(mv.piece_idx, target_idx, undo);
                    }
                }
                self.move_piece(mv.piece_idx, mv.from_x, mv.from_y, mv.to_x, mv.to_y);
                self.state.pieces[mv.piece_idx as usize].ability_cooldown = PIECE_STATS[PieceType::Knight as usize].2;
                if self.state.pieces[mv.piece_idx as usize].ability_uses_remaining > 0 {
                    self.state.pieces[mv.piece_idx as usize].ability_uses_remaining -= 1;
                }
            }
            AbilityId::Advance => {
                self.move_piece(mv.piece_idx, mv.from_x, mv.from_y, mv.to_x, mv.to_y);
                self.check_promotion(mv.piece_idx);
                self.state.pieces[mv.piece_idx as usize].ability_cooldown = PIECE_STATS[PieceType::Pawn as usize].2;
                self.state.pieces[mv.piece_idx as usize].advance_cooldown_turns = 2;
            }
        }

        damage_dealt
    }

    fn check_promotion(&mut self, piece_idx: u8) {
        let piece = &self.state.pieces[piece_idx as usize];
        if piece.piece_type() != PieceType::Pawn { return; }

        let promotion_row = if piece.team() == Team::White { 7 } else { 0 };
        if piece.y == promotion_row {
            let hp_ratio = piece.current_hp as f32 / piece.max_hp as f32;
            let old_bucket = piece.hp_bucket();
            let (queen_max_hp, queen_damage, _, _) = PieceType::Queen.stats();

            // Update hash for piece type change
            let sq = sq_idx(piece.x, piece.y);
            let team = piece.team() as usize;
            self.state.hash ^= ZOBRIST.pieces[sq][PieceType::Pawn as usize][team][old_bucket];

            let piece = &mut self.state.pieces[piece_idx as usize];
            piece.set_piece_type(PieceType::Queen);
            piece.max_hp = queen_max_hp;
            piece.current_hp = (queen_max_hp as f32 * hp_ratio).max(1.0) as i16;
            piece.base_damage = queen_damage;
            piece.set_was_pawn(true);

            let new_bucket = piece.hp_bucket();
            self.state.hash ^= ZOBRIST.pieces[sq][PieceType::Queen as usize][team][new_bucket];
        }
    }

    /// Simple make_move that returns only damage (for compatibility).
    pub fn make_move_simple(&mut self, mv: &Move) -> i32 {
        let (damage, _) = self.make_move(mv);
        damage
    }
}

impl Default for GameSimulator {
    fn default() -> Self {
        Self::new()
    }
}
