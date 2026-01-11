//! High-performance EXCHANGE game simulator for neural network training.
//!
//! OPTIMIZED VERSION with:
//! - O(1) board lookup via 8x8 array
//! - Zobrist hashing with incremental updates
//! - Make/Unmake pattern to avoid cloning
//! - Bitboard attack maps for fast tactical analysis
//! - Thread-local pre-allocated buffers
//! - Compact piece representation

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::PyArray3;
use ndarray::Array3;
use std::cell::RefCell;
use std::sync::LazyLock;

// ============================================================================
// Constants
// ============================================================================

const BOARD_SIZE: usize = 8;
const DRAW_MOVES_WITHOUT_DAMAGE: i32 = 30;
const DRAW_REPETITION_COUNT: usize = 3;
const INPUT_CHANNELS: usize = 58;
const MAX_PIECES: usize = 32;

// Piece values for material calculation
const PIECE_VALUES: [i32; 6] = [0, 9, 5, 3, 3, 1];  // K, Q, R, B, N, P

// Direction vectors
const CARDINAL: [(i8, i8); 4] = [(0, 1), (0, -1), (1, 0), (-1, 0)];
const DIAGONAL: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
const ALL_DIRS: [(i8, i8); 8] = [
    (0, 1), (0, -1), (1, 0), (-1, 0),
    (1, 1), (1, -1), (-1, 1), (-1, -1),
];
const KNIGHT_MOVES: [(i8, i8); 8] = [
    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
    (1, -2), (1, 2), (2, -1), (2, 1),
];

// Piece stats: (max_hp, base_damage, cooldown_max)
// cooldown_max = -1 means once per match
const PIECE_STATS: [(i16, i16, i8); 6] = [
    (25, 1, -1),  // King
    (10, 3, 3),   // Queen
    (13, 2, 3),   // Rook
    (10, 2, 3),   // Bishop
    (11, 2, 3),   // Knight
    (7, 1, 1),    // Pawn
];

// ============================================================================
// Zobrist Hashing - Pre-computed random keys for O(1) incremental updates
// ============================================================================

struct ZobristKeys {
    // [square][piece_type][team][hp_bucket] - HP bucketed to 8 levels
    pieces: [[[[u64; 8]; 2]; 6]; 64],
    side_to_move: u64,
    royal_decree: [u64; 2],  // [team]
}

impl ZobristKeys {
    fn new() -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut keys = ZobristKeys {
            pieces: [[[[0u64; 8]; 2]; 6]; 64],
            side_to_move: 0,
            royal_decree: [0; 2],
        };

        // Generate deterministic pseudo-random keys
        let mut seed = 0x12345678u64;
        let mut next_key = || {
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            seed = hasher.finish();
            seed
        };

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

static ZOBRIST: LazyLock<ZobristKeys> = LazyLock::new(ZobristKeys::new);

// ============================================================================
// Bitboard - 64-bit representation for fast attack queries
// ============================================================================

type Bitboard = u64;

#[inline]
fn sq_bit(x: i8, y: i8) -> Bitboard {
    1u64 << (y * 8 + x)
}

#[inline]
fn sq_idx(x: i8, y: i8) -> usize {
    (y as usize) * 8 + (x as usize)
}

#[inline]
fn is_on_board(x: i8, y: i8) -> bool {
    x >= 0 && x < 8 && y >= 0 && y < 8
}

// ============================================================================
// Thread-local buffers to avoid allocations in hot paths
// ============================================================================

thread_local! {
    static MOVE_BUFFER: RefCell<Vec<Move>> = RefCell::new(Vec::with_capacity(256));
    static TENSOR_BUFFER: RefCell<Vec<f32>> = RefCell::new(vec![0.0; INPUT_CHANNELS * 64]);
}

// ============================================================================
// Enums
// ============================================================================

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
    #[inline]
    fn stats(self) -> (i16, i16, i8) {
        PIECE_STATS[self as usize]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Team {
    White = 0,
    Black = 1,
}

impl Team {
    #[inline]
    fn opposite(self) -> Team {
        match self {
            Team::White => Team::Black,
            Team::Black => Team::White,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MoveType {
    Move = 0,
    Attack = 1,
    MoveAndAttack = 2,
    Ability = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AbilityId {
    RoyalDecree = 0,
    Overextend = 1,
    Interpose = 2,
    Consecration = 3,
    Skirmish = 4,
    Advance = 5,
}

impl AbilityId {
    #[inline]
    fn for_piece_type(pt: PieceType) -> Self {
        match pt {
            PieceType::King => AbilityId::RoyalDecree,
            PieceType::Queen => AbilityId::Overextend,
            PieceType::Rook => AbilityId::Interpose,
            PieceType::Bishop => AbilityId::Consecration,
            PieceType::Knight => AbilityId::Skirmish,
            PieceType::Pawn => AbilityId::Advance,
        }
    }
}

// ============================================================================
// Compact Piece - 16 bytes instead of 40
// ============================================================================

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Piece {
    // Byte 0: position x
    pub x: i8,
    // Byte 1: position y
    pub y: i8,
    // Byte 2: piece_type (3 bits) | team (1 bit) | flags (4 bits)
    type_team_flags: u8,
    // Byte 3: ability_cooldown
    pub ability_cooldown: i8,
    // Bytes 4-5: current_hp
    pub current_hp: i16,
    // Bytes 6-7: max_hp
    pub max_hp: i16,
    // Bytes 8-9: base_damage
    pub base_damage: i16,
    // Byte 10: advance_cooldown_turns
    pub advance_cooldown_turns: i8,
    // Byte 11: padding
    _pad: u8,
}

// Flag bits in type_team_flags
const FLAG_ABILITY_USED: u8 = 0x10;
const FLAG_INTERPOSE: u8 = 0x20;
const FLAG_WAS_PAWN: u8 = 0x40;
const FLAG_DEAD: u8 = 0x80;

impl Piece {
    pub fn new(piece_type: PieceType, team: Team, x: i8, y: i8) -> Self {
        let (max_hp, base_damage, _) = piece_type.stats();
        Piece {
            x,
            y,
            type_team_flags: (piece_type as u8) | ((team as u8) << 3),
            ability_cooldown: 0,
            current_hp: max_hp,
            max_hp,
            base_damage,
            advance_cooldown_turns: 0,
            _pad: 0,
        }
    }

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

    #[inline]
    pub fn team(&self) -> Team {
        if (self.type_team_flags >> 3) & 1 == 0 {
            Team::White
        } else {
            Team::Black
        }
    }

    #[inline]
    fn set_piece_type(&mut self, pt: PieceType) {
        self.type_team_flags = (self.type_team_flags & 0xF8) | (pt as u8);
    }

    #[inline]
    pub fn is_alive(&self) -> bool {
        self.current_hp > 0 && (self.type_team_flags & FLAG_DEAD) == 0
    }

    #[inline]
    pub fn ability_used_this_match(&self) -> bool {
        (self.type_team_flags & FLAG_ABILITY_USED) != 0
    }

    #[inline]
    fn set_ability_used(&mut self, used: bool) {
        if used {
            self.type_team_flags |= FLAG_ABILITY_USED;
        } else {
            self.type_team_flags &= !FLAG_ABILITY_USED;
        }
    }

    #[inline]
    pub fn interpose_active(&self) -> bool {
        (self.type_team_flags & FLAG_INTERPOSE) != 0
    }

    #[inline]
    fn set_interpose(&mut self, active: bool) {
        if active {
            self.type_team_flags |= FLAG_INTERPOSE;
        } else {
            self.type_team_flags &= !FLAG_INTERPOSE;
        }
    }

    #[inline]
    pub fn was_pawn(&self) -> bool {
        (self.type_team_flags & FLAG_WAS_PAWN) != 0
    }

    #[inline]
    fn set_was_pawn(&mut self, was: bool) {
        if was {
            self.type_team_flags |= FLAG_WAS_PAWN;
        } else {
            self.type_team_flags &= !FLAG_WAS_PAWN;
        }
    }

    pub fn can_use_ability(&self) -> bool {
        let (_, _, cooldown_max) = self.piece_type().stats();
        if cooldown_max == -1 {
            !self.ability_used_this_match()
        } else {
            self.ability_cooldown == 0
        }
    }

    /// Get HP bucket (0-7) for Zobrist hashing
    #[inline]
    fn hp_bucket(&self) -> usize {
        if self.max_hp == 0 { return 0; }
        let ratio = (self.current_hp as f32 / self.max_hp as f32 * 7.0) as usize;
        ratio.min(7)
    }
}

// ============================================================================
// Move
// ============================================================================

#[derive(Debug, Clone, Copy)]
pub struct Move {
    pub piece_idx: u8,
    pub move_type: MoveType,
    pub from_x: i8,
    pub from_y: i8,
    pub to_x: i8,
    pub to_y: i8,
    pub attack_x: i8,
    pub attack_y: i8,
    pub ability_target_x: i8,
    pub ability_target_y: i8,
    pub ability_id: Option<AbilityId>,
}

impl Move {
    #[inline]
    fn new_move(piece_idx: u8, from: (i8, i8), to: (i8, i8)) -> Self {
        Move {
            piece_idx,
            move_type: MoveType::Move,
            from_x: from.0, from_y: from.1,
            to_x: to.0, to_y: to.1,
            attack_x: -1, attack_y: -1,
            ability_target_x: -1, ability_target_y: -1,
            ability_id: None,
        }
    }

    #[inline]
    fn new_attack(piece_idx: u8, from: (i8, i8), target: (i8, i8)) -> Self {
        Move {
            piece_idx,
            move_type: MoveType::Attack,
            from_x: from.0, from_y: from.1,
            to_x: target.0, to_y: target.1,
            attack_x: -1, attack_y: -1,
            ability_target_x: -1, ability_target_y: -1,
            ability_id: None,
        }
    }

    #[inline]
    fn new_move_and_attack(piece_idx: u8, from: (i8, i8), to: (i8, i8), attack: (i8, i8)) -> Self {
        Move {
            piece_idx,
            move_type: MoveType::MoveAndAttack,
            from_x: from.0, from_y: from.1,
            to_x: to.0, to_y: to.1,
            attack_x: attack.0, attack_y: attack.1,
            ability_target_x: -1, ability_target_y: -1,
            ability_id: None,
        }
    }

    #[inline]
    fn new_ability(piece_idx: u8, from: (i8, i8), to: (i8, i8), ability: AbilityId) -> Self {
        Move {
            piece_idx,
            move_type: MoveType::Ability,
            from_x: from.0, from_y: from.1,
            to_x: to.0, to_y: to.1,
            attack_x: -1, attack_y: -1,
            ability_target_x: -1, ability_target_y: -1,
            ability_id: Some(ability),
        }
    }

    #[inline]
    fn new_ability_with_target(
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
            from_x: from.0, from_y: from.1,
            to_x: to.0, to_y: to.1,
            attack_x: attack.map_or(-1, |p| p.0),
            attack_y: attack.map_or(-1, |p| p.1),
            ability_target_x: target.map_or(-1, |p| p.0),
            ability_target_y: target.map_or(-1, |p| p.1),
            ability_id: Some(ability),
        }
    }
}

// ============================================================================
// Undo Info - For make/unmake pattern
// ============================================================================

#[derive(Clone)]
pub struct UndoInfo {
    // Piece state before move
    piece_idx: u8,
    piece_snapshot: Piece,
    old_board_from: Option<u8>,
    old_board_to: Option<u8>,

    // Target piece state (if attack)
    target_idx: Option<u8>,
    target_snapshot: Option<Piece>,
    old_board_attack: Option<u8>,

    // Secondary target (for abilities like Consecration, Interpose)
    secondary_idx: Option<u8>,
    secondary_snapshot: Option<Piece>,

    // Game state
    old_hash: u64,
    old_royal_decree: Option<Team>,
    old_moves_without_damage: i32,
    old_side_to_move: Team,
    old_turn_number: i32,

    // For Rook's interpose damage split
    interpose_rook_idx: Option<u8>,
    interpose_rook_snapshot: Option<Piece>,
}

// ============================================================================
// GameState - Optimized with board array and Zobrist hash
// ============================================================================

#[derive(Clone)]
pub struct GameState {
    pub pieces: [Piece; MAX_PIECES],
    pub piece_count: u8,

    // O(1) lookup: board[y][x] = Some(piece_idx) or None
    pub board: [[Option<u8>; 8]; 8],

    // Zobrist hash for fast repetition detection
    pub hash: u64,

    pub side_to_move: Team,
    pub royal_decree_active: Option<Team>,
    pub turn_number: i32,
    pub moves_without_damage: i32,
    pub position_history: Vec<u64>,
    pub playing_as: Team,
    pub winner: Option<Team>,
    pub is_terminal: bool,
    pub is_draw: bool,

    // Cached piece indices by team (faster iteration)
    white_pieces: [u8; 16],
    white_count: u8,
    black_pieces: [u8; 16],
    black_count: u8,
}

impl GameState {
    pub fn new() -> Self {
        let mut state = GameState {
            pieces: [Piece::new(PieceType::Pawn, Team::White, 0, 0); MAX_PIECES],
            piece_count: 0,
            board: [[None; 8]; 8],
            hash: 0,
            side_to_move: Team::White,
            royal_decree_active: None,
            turn_number: 0,
            moves_without_damage: 0,
            position_history: Vec::with_capacity(100),
            playing_as: Team::White,
            winner: None,
            is_terminal: false,
            is_draw: false,
            white_pieces: [0; 16],
            white_count: 0,
            black_pieces: [0; 16],
            black_count: 0,
        };

        // Standard starting position
        let back_rank = [
            PieceType::Rook, PieceType::Knight, PieceType::Bishop, PieceType::Queen,
            PieceType::King, PieceType::Bishop, PieceType::Knight, PieceType::Rook,
        ];

        // White pieces (rows 0-1)
        for (x, &pt) in back_rank.iter().enumerate() {
            state.add_piece(Piece::new(pt, Team::White, x as i8, 0));
        }
        for x in 0..8 {
            state.add_piece(Piece::new(PieceType::Pawn, Team::White, x, 1));
        }

        // Black pieces (rows 6-7)
        for (x, &pt) in back_rank.iter().enumerate() {
            state.add_piece(Piece::new(pt, Team::Black, x as i8, 7));
        }
        for x in 0..8 {
            state.add_piece(Piece::new(PieceType::Pawn, Team::Black, x, 6));
        }

        // Initialize hash
        state.hash = state.compute_full_hash();

        state
    }

    fn add_piece(&mut self, piece: Piece) {
        let idx = self.piece_count;
        self.pieces[idx as usize] = piece;
        self.board[piece.y as usize][piece.x as usize] = Some(idx);

        match piece.team() {
            Team::White => {
                self.white_pieces[self.white_count as usize] = idx;
                self.white_count += 1;
            }
            Team::Black => {
                self.black_pieces[self.black_count as usize] = idx;
                self.black_count += 1;
            }
        }

        self.piece_count += 1;
    }

    /// Compute full hash (used for initialization)
    fn compute_full_hash(&self) -> u64 {
        let zobrist = &*ZOBRIST;
        let mut hash = 0u64;

        for idx in 0..self.piece_count {
            let p = &self.pieces[idx as usize];
            if p.is_alive() {
                let sq = sq_idx(p.x, p.y);
                let pt = p.piece_type() as usize;
                let team = p.team() as usize;
                let hp = p.hp_bucket();
                hash ^= zobrist.pieces[sq][pt][team][hp];
            }
        }

        if self.side_to_move == Team::Black {
            hash ^= zobrist.side_to_move;
        }

        if let Some(team) = self.royal_decree_active {
            hash ^= zobrist.royal_decree[team as usize];
        }

        hash
    }

    /// Update hash for piece movement
    #[inline]
    fn hash_move_piece(&mut self, idx: u8, old_x: i8, old_y: i8, new_x: i8, new_y: i8) {
        let zobrist = &*ZOBRIST;
        let p = &self.pieces[idx as usize];
        let pt = p.piece_type() as usize;
        let team = p.team() as usize;
        let hp = p.hp_bucket();

        // Remove from old position
        self.hash ^= zobrist.pieces[sq_idx(old_x, old_y)][pt][team][hp];
        // Add to new position
        self.hash ^= zobrist.pieces[sq_idx(new_x, new_y)][pt][team][hp];
    }

    /// Update hash for HP change
    #[inline]
    fn hash_hp_change(&mut self, idx: u8, old_hp_bucket: usize, new_hp_bucket: usize) {
        if old_hp_bucket == new_hp_bucket { return; }

        let zobrist = &*ZOBRIST;
        let p = &self.pieces[idx as usize];
        let sq = sq_idx(p.x, p.y);
        let pt = p.piece_type() as usize;
        let team = p.team() as usize;

        self.hash ^= zobrist.pieces[sq][pt][team][old_hp_bucket];
        self.hash ^= zobrist.pieces[sq][pt][team][new_hp_bucket];
    }

    /// Update hash for piece removal (death)
    #[inline]
    fn hash_remove_piece(&mut self, idx: u8) {
        let zobrist = &*ZOBRIST;
        let p = &self.pieces[idx as usize];
        let sq = sq_idx(p.x, p.y);
        let pt = p.piece_type() as usize;
        let team = p.team() as usize;
        let hp = p.hp_bucket();

        self.hash ^= zobrist.pieces[sq][pt][team][hp];
    }

    #[inline]
    pub fn get_piece_at(&self, x: i8, y: i8) -> Option<u8> {
        if !is_on_board(x, y) { return None; }
        self.board[y as usize][x as usize].filter(|&idx| self.pieces[idx as usize].is_alive())
    }

    pub fn get_pieces_for_team(&self, team: Team) -> &[u8] {
        match team {
            Team::White => &self.white_pieces[..self.white_count as usize],
            Team::Black => &self.black_pieces[..self.black_count as usize],
        }
    }

    pub fn get_king(&self, team: Team) -> Option<u8> {
        let pieces = self.get_pieces_for_team(team);
        for &idx in pieces {
            let p = &self.pieces[idx as usize];
            if p.is_alive() && p.piece_type() == PieceType::King {
                return Some(idx);
            }
        }
        None
    }

    pub fn record_position(&mut self) {
        self.position_history.push(self.hash);
    }

    pub fn is_threefold_repetition(&self) -> bool {
        if self.position_history.len() < DRAW_REPETITION_COUNT {
            return false;
        }
        self.position_history.iter().filter(|&&h| h == self.hash).count() >= DRAW_REPETITION_COUNT
    }

    pub fn check_terminal(&mut self) {
        let white_king = self.get_king(Team::White);
        let black_king = self.get_king(Team::Black);

        if white_king.is_none() {
            self.is_terminal = true;
            self.winner = Some(Team::Black);
            return;
        }
        if black_king.is_none() {
            self.is_terminal = true;
            self.winner = Some(Team::White);
            return;
        }

        // Count non-king pieces
        let mut non_king_count = 0;
        for idx in 0..self.piece_count {
            let p = &self.pieces[idx as usize];
            if p.is_alive() && p.piece_type() != PieceType::King {
                non_king_count += 1;
            }
        }
        if non_king_count == 0 {
            self.is_terminal = true;
            self.is_draw = true;
            self.winner = None;
            return;
        }

        if self.moves_without_damage >= DRAW_MOVES_WITHOUT_DAMAGE {
            self.is_terminal = true;
            self.is_draw = true;
            self.winner = None;
            return;
        }

        if self.is_threefold_repetition() {
            self.is_terminal = true;
            self.is_draw = true;
            self.winner = None;
        }
    }

    /// Clone for MCTS without position history
    pub fn clone_for_mcts(&self) -> Self {
        let mut new_state = self.clone();
        new_state.position_history.clear();
        new_state
    }
}

impl Default for GameState {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Attack Maps - Bitboard-based for O(1) queries
// ============================================================================

pub struct AttackMaps {
    pub white: Bitboard,
    pub black: Bitboard,
    pub white_per_piece: [Bitboard; 16],
    pub black_per_piece: [Bitboard; 16],
}

impl AttackMaps {
    pub fn compute(state: &GameState) -> Self {
        let mut maps = AttackMaps {
            white: 0,
            black: 0,
            white_per_piece: [0; 16],
            black_per_piece: [0; 16],
        };

        for (i, &idx) in state.get_pieces_for_team(Team::White).iter().enumerate() {
            let bb = Self::piece_attacks(state, idx);
            maps.white_per_piece[i] = bb;
            maps.white |= bb;
        }

        for (i, &idx) in state.get_pieces_for_team(Team::Black).iter().enumerate() {
            let bb = Self::piece_attacks(state, idx);
            maps.black_per_piece[i] = bb;
            maps.black |= bb;
        }

        maps
    }

    fn piece_attacks(state: &GameState, idx: u8) -> Bitboard {
        let p = &state.pieces[idx as usize];
        if !p.is_alive() { return 0; }

        let x = p.x;
        let y = p.y;
        let mut bb = 0u64;

        match p.piece_type() {
            PieceType::King => {
                for (dx, dy) in ALL_DIRS {
                    let nx = x + dx;
                    let ny = y + dy;
                    if is_on_board(nx, ny) {
                        bb |= sq_bit(nx, ny);
                    }
                }
            }
            PieceType::Queen => {
                for (dx, dy) in ALL_DIRS {
                    for dist in 1..8 {
                        let nx = x + dx * dist;
                        let ny = y + dy * dist;
                        if !is_on_board(nx, ny) { break; }
                        bb |= sq_bit(nx, ny);
                        if state.board[ny as usize][nx as usize].is_some() { break; }
                    }
                }
            }
            PieceType::Rook => {
                for (dx, dy) in CARDINAL {
                    for dist in 1..8 {
                        let nx = x + dx * dist;
                        let ny = y + dy * dist;
                        if !is_on_board(nx, ny) { break; }
                        bb |= sq_bit(nx, ny);
                        if state.board[ny as usize][nx as usize].is_some() { break; }
                    }
                }
            }
            PieceType::Bishop => {
                for (dx, dy) in DIAGONAL {
                    for dist in 1..8 {
                        let nx = x + dx * dist;
                        let ny = y + dy * dist;
                        if !is_on_board(nx, ny) { break; }
                        bb |= sq_bit(nx, ny);
                        if state.board[ny as usize][nx as usize].is_some() { break; }
                    }
                }
            }
            PieceType::Knight => {
                for (dx, dy) in KNIGHT_MOVES {
                    let nx = x + dx;
                    let ny = y + dy;
                    if is_on_board(nx, ny) {
                        bb |= sq_bit(nx, ny);
                    }
                }
            }
            PieceType::Pawn => {
                let dir: i8 = if p.team() == Team::White { 1 } else { -1 };
                for dx in [-1i8, 1] {
                    let nx = x + dx;
                    let ny = y + dir;
                    if is_on_board(nx, ny) {
                        bb |= sq_bit(nx, ny);
                    }
                }
            }
        }

        bb
    }
}

// ============================================================================
// GameSimulator - Core game engine
// ============================================================================

pub struct GameSimulator {
    pub state: GameState,
    rng: fastrand::Rng,
}

impl GameSimulator {
    pub fn new() -> Self {
        GameSimulator {
            state: GameState::new(),
            rng: fastrand::Rng::new(),
        }
    }

    pub fn with_state(state: GameState) -> Self {
        GameSimulator {
            state,
            rng: fastrand::Rng::new(),
        }
    }

    pub fn reset(&mut self) -> &GameState {
        self.state = GameState::new();
        &self.state
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.rng = fastrand::Rng::with_seed(seed);
    }

    #[inline]
    fn is_occupied(&self, x: i8, y: i8) -> bool {
        self.state.get_piece_at(x, y).is_some()
    }

    // ========================================================================
    // Move Generation
    // ========================================================================

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
                // Knight attacks via combos
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
                let has_adjacent_ally = ALL_DIRS.iter().any(|&(dx, dy)| {
                    let nx = from.0 + dx;
                    let ny = from.1 + dy;
                    if let Some(ally_idx) = self.state.get_piece_at(nx, ny) {
                        ally_idx != piece_idx && self.state.pieces[ally_idx as usize].team() == team
                    } else {
                        false
                    }
                });
                if has_adjacent_ally {
                    moves.push(Move::new_ability(piece_idx, from, from, AbilityId::Interpose));
                }
            }
            AbilityId::Consecration => {
                for (dx, dy) in DIAGONAL {
                    for dist in 1..8 {
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
    // Move Execution with Make/Unmake
    // ========================================================================

    pub fn make_move(&mut self, mv: &Move) -> (i32, UndoInfo) {
        let mut undo = UndoInfo {
            piece_idx: mv.piece_idx,
            piece_snapshot: self.state.pieces[mv.piece_idx as usize],
            old_board_from: self.state.board[mv.from_y as usize][mv.from_x as usize],
            old_board_to: self.state.board[mv.to_y as usize][mv.to_x as usize],
            target_idx: None,
            target_snapshot: None,
            old_board_attack: None,
            secondary_idx: None,
            secondary_snapshot: None,
            old_hash: self.state.hash,
            old_royal_decree: self.state.royal_decree_active,
            old_moves_without_damage: self.state.moves_without_damage,
            old_side_to_move: self.state.side_to_move,
            old_turn_number: self.state.turn_number,
            interpose_rook_idx: None,
            interpose_rook_snapshot: None,
        };

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

        // Tick cooldowns
        let moved_team = self.state.pieces[mv.piece_idx as usize].team();
        for idx in 0..self.state.piece_count {
            let piece = &mut self.state.pieces[idx as usize];
            if piece.team() == moved_team && piece.is_alive() {
                if piece.ability_cooldown > 0 {
                    piece.ability_cooldown -= 1;
                }
                if piece.piece_type() == PieceType::Rook {
                    piece.set_interpose(false);
                }
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

    pub fn unmake_move(&mut self, undo: &UndoInfo) {
        // Restore game state
        self.state.hash = undo.old_hash;
        self.state.royal_decree_active = undo.old_royal_decree;
        self.state.moves_without_damage = undo.old_moves_without_damage;
        self.state.side_to_move = undo.old_side_to_move;
        self.state.turn_number = undo.old_turn_number;
        self.state.is_terminal = false;
        self.state.is_draw = false;
        self.state.winner = None;

        // Pop position history
        self.state.position_history.pop();

        // Restore pieces
        self.state.pieces[undo.piece_idx as usize] = undo.piece_snapshot;

        // Restore board
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

        if self.state.royal_decree_active == Some(attacker.team()) {
            damage += 1;
        }

        self.apply_damage_with_interpose(target_idx, damage, undo)
    }

    fn apply_damage_with_interpose(&mut self, target_idx: u8, damage: i16, undo: &mut UndoInfo) -> i32 {
        let target = &self.state.pieces[target_idx as usize];
        let target_team = target.team();
        let target_x = target.x;
        let target_y = target.y;
        let old_hp_bucket = target.hp_bucket();

        // Find interposing rook
        let mut interpose_rook: Option<u8> = None;
        for idx in 0..self.state.piece_count {
            let p = &self.state.pieces[idx as usize];
            if p.piece_type() == PieceType::Rook
                && p.interpose_active()
                && p.team() == target_team
                && idx != target_idx as u8
                && p.is_alive()
            {
                let dx = (p.x - target_x).abs();
                let dy = (p.y - target_y).abs();
                if dx <= 1 && dy <= 1 && (dx + dy) > 0 {
                    interpose_rook = Some(idx);
                    break;
                }
            }
        }

        if let Some(rook_idx) = interpose_rook {
            undo.interpose_rook_idx = Some(rook_idx);
            undo.interpose_rook_snapshot = Some(self.state.pieces[rook_idx as usize]);

            let rook_old_bucket = self.state.pieces[rook_idx as usize].hp_bucket();

            let half = damage / 2;
            let remainder = damage % 2;

            self.state.pieces[target_idx as usize].current_hp -= half;
            self.state.pieces[rook_idx as usize].current_hp -= half + remainder;

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
                if let Some(old_team) = self.state.royal_decree_active {
                    self.state.hash ^= ZOBRIST.royal_decree[old_team as usize];
                }
                self.state.hash ^= ZOBRIST.royal_decree[piece_team as usize];
                self.state.royal_decree_active = Some(piece_team);
                self.state.pieces[mv.piece_idx as usize].set_ability_used(true);
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
                self.state.pieces[mv.piece_idx as usize].set_interpose(true);
                self.state.pieces[mv.piece_idx as usize].ability_cooldown = PIECE_STATS[PieceType::Rook as usize].2;
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
                            if self.state.royal_decree_active == Some(piece_team) {
                                heal += 1;
                            }

                            let target = &mut self.state.pieces[target_idx as usize];
                            target.current_hp = (target.current_hp + heal).min(target.max_hp);

                            let new_bucket = target.hp_bucket();
                            self.state.hash_hp_change(target_idx, old_bucket, new_bucket);
                        }
                    }
                }
                self.state.pieces[mv.piece_idx as usize].ability_cooldown = PIECE_STATS[PieceType::Bishop as usize].2;
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
            let (queen_max_hp, queen_damage, _) = PieceType::Queen.stats();

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

    /// Simple make_move that returns only damage (for compatibility)
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

// ============================================================================
// Python Bindings (PyO3)
// ============================================================================

#[pyclass(name = "RustSimulator")]
pub struct PySimulator {
    sim: GameSimulator,
}

#[pyclass(name = "RustMove")]
#[derive(Clone)]
pub struct PyMove {
    #[pyo3(get)]
    pub piece_idx: usize,
    #[pyo3(get)]
    pub move_type: u8,
    #[pyo3(get)]
    pub from_x: i8,
    #[pyo3(get)]
    pub from_y: i8,
    #[pyo3(get)]
    pub to_x: i8,
    #[pyo3(get)]
    pub to_y: i8,
    #[pyo3(get)]
    pub attack_x: Option<i8>,
    #[pyo3(get)]
    pub attack_y: Option<i8>,
    #[pyo3(get)]
    pub ability_target_x: Option<i8>,
    #[pyo3(get)]
    pub ability_target_y: Option<i8>,
    #[pyo3(get)]
    pub ability_id: Option<u8>,
    internal_move: Move,
}

impl PyMove {
    fn from_move(m: &Move) -> Self {
        PyMove {
            piece_idx: m.piece_idx as usize,
            move_type: m.move_type as u8,
            from_x: m.from_x,
            from_y: m.from_y,
            to_x: m.to_x,
            to_y: m.to_y,
            attack_x: if m.attack_x >= 0 { Some(m.attack_x) } else { None },
            attack_y: if m.attack_y >= 0 { Some(m.attack_y) } else { None },
            ability_target_x: if m.ability_target_x >= 0 { Some(m.ability_target_x) } else { None },
            ability_target_y: if m.ability_target_y >= 0 { Some(m.ability_target_y) } else { None },
            ability_id: m.ability_id.map(|a| a as u8),
            internal_move: *m,
        }
    }
}

#[pyclass(name = "RustPiece")]
#[derive(Clone)]
pub struct PyPiece {
    #[pyo3(get)]
    pub piece_type: u8,
    #[pyo3(get)]
    pub team: u8,
    #[pyo3(get)]
    pub x: i8,
    #[pyo3(get)]
    pub y: i8,
    #[pyo3(get)]
    pub current_hp: i16,
    #[pyo3(get)]
    pub max_hp: i16,
    #[pyo3(get)]
    pub base_damage: i16,
    #[pyo3(get)]
    pub ability_cooldown: i8,
    #[pyo3(get)]
    pub ability_used_this_match: bool,
    #[pyo3(get)]
    pub interpose_active: bool,
    #[pyo3(get)]
    pub was_pawn: bool,
    #[pyo3(get)]
    pub advance_cooldown_turns: i8,
}

impl PyPiece {
    fn from_piece(p: &Piece) -> Self {
        PyPiece {
            piece_type: p.piece_type() as u8,
            team: p.team() as u8,
            x: p.x,
            y: p.y,
            current_hp: p.current_hp,
            max_hp: p.max_hp,
            base_damage: p.base_damage,
            ability_cooldown: p.ability_cooldown,
            ability_used_this_match: p.ability_used_this_match(),
            interpose_active: p.interpose_active(),
            was_pawn: p.was_pawn(),
            advance_cooldown_turns: p.advance_cooldown_turns,
        }
    }
}

#[pymethods]
impl PySimulator {
    #[new]
    fn new() -> Self {
        PySimulator {
            sim: GameSimulator::new(),
        }
    }

    fn reset(&mut self) {
        self.sim.reset();
    }

    fn set_seed(&mut self, seed: u64) {
        self.sim.set_seed(seed);
    }

    fn clone_sim(&self) -> PySimulator {
        PySimulator {
            sim: GameSimulator::with_state(self.sim.state.clone_for_mcts()),
        }
    }

    #[getter]
    fn side_to_move(&self) -> u8 {
        self.sim.state.side_to_move as u8
    }

    #[getter]
    fn is_terminal(&self) -> bool {
        self.sim.state.is_terminal
    }

    #[getter]
    fn winner(&self) -> Option<u8> {
        self.sim.state.winner.map(|t| t as u8)
    }

    #[getter]
    fn is_draw(&self) -> bool {
        self.sim.state.is_draw
    }

    #[getter]
    fn turn_number(&self) -> i32 {
        self.sim.state.turn_number
    }

    fn get_pieces(&self) -> Vec<PyPiece> {
        (0..self.sim.state.piece_count)
            .map(|i| PyPiece::from_piece(&self.sim.state.pieces[i as usize]))
            .collect()
    }

    fn get_piece_at(&self, x: i8, y: i8) -> Option<PyPiece> {
        self.sim.state.get_piece_at(x, y)
            .map(|idx| PyPiece::from_piece(&self.sim.state.pieces[idx as usize]))
    }

    fn generate_moves(&self) -> Vec<PyMove> {
        self.sim.generate_moves(self.sim.state.side_to_move)
            .iter()
            .map(PyMove::from_move)
            .collect()
    }

    fn make_move(&mut self, mv: &PyMove) -> i32 {
        self.sim.make_move_simple(&mv.internal_move)
    }

    fn make_move_by_index(&mut self, move_index: usize) -> PyResult<i32> {
        let moves = self.sim.generate_moves(self.sim.state.side_to_move);
        if move_index >= moves.len() {
            return Err(PyValueError::new_err(format!(
                "Move index {} out of range (0..{})",
                move_index, moves.len()
            )));
        }
        Ok(self.sim.make_move_simple(&moves[move_index]))
    }

    fn num_moves(&self) -> usize {
        self.sim.generate_moves(self.sim.state.side_to_move).len()
    }

    #[getter]
    fn moves_without_damage(&self) -> i32 {
        self.sim.state.moves_without_damage
    }

    fn to_tensor<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f32>>> {
        let tensor = self.compute_tensor();
        let arr = Array3::from_shape_vec((INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), tensor)
            .map_err(|e| PyValueError::new_err(format!("Failed to create tensor: {}", e)))?;
        Ok(PyArray3::from_owned_array_bound(py, arr))
    }
}

impl PySimulator {
    fn compute_tensor(&self) -> Vec<f32> {
        TENSOR_BUFFER.with(|buf| {
            let mut tensor = buf.borrow_mut();

            // Zero out the buffer
            tensor.fill(0.0);

            let state = &self.sim.state;

            // Helper closures
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

            // === CHANNELS 0-27: Basic piece info ===
            for &idx in white_pieces.iter().chain(black_pieces.iter()) {
                let piece = &state.pieces[idx as usize];
                let x = piece.x;
                let y = piece.y;
                let pt = piece.piece_type() as usize;
                let team_offset = if piece.team() == Team::White { 0 } else { 6 };

                set(&mut tensor, team_offset + pt, y, x, 1.0);

                let hp_ratio = piece.current_hp as f32 / piece.max_hp as f32;
                set(&mut tensor, 12 + pt, y, x, hp_ratio);

                let (_, _, cd_max) = PIECE_STATS[pt];
                if cd_max > 0 {
                    let cd_ratio = piece.ability_cooldown as f32 / cd_max as f32;
                    set(&mut tensor, 18 + pt, y, x, cd_ratio);
                } else if cd_max == -1 {
                    set(&mut tensor, 18 + pt, y, x, if piece.ability_used_this_match() { 1.0 } else { 0.0 });
                }

                if piece.interpose_active() {
                    set(&mut tensor, 25, y, x, 1.0);
                }
            }

            if state.royal_decree_active == Some(state.side_to_move) {
                fill(&mut tensor, 24, 1.0);
            }

            if state.side_to_move == Team::White {
                fill(&mut tensor, 26, 1.0);
            }

            fill(&mut tensor, 27, 1.0);

            // === CHANNELS 28-45: Strategic ===
            fill(&mut tensor, 28, (state.turn_number as f32 / 300.0).min(1.0));
            fill(&mut tensor, 29, state.moves_without_damage as f32 / 30.0);
            fill(&mut tensor, 30, 0.0);

            let white_hp: i32 = white_pieces.iter().map(|&i| state.pieces[i as usize].current_hp as i32).sum();
            let black_hp: i32 = black_pieces.iter().map(|&i| state.pieces[i as usize].current_hp as i32).sum();
            let total_hp = white_hp + black_hp;
            fill(&mut tensor, 31, if total_hp > 0 { white_hp as f32 / total_hp as f32 } else { 0.5 });

            let total_pieces = white_pieces.len() + black_pieces.len();
            fill(&mut tensor, 32, if total_pieces > 0 { white_pieces.len() as f32 / total_pieces as f32 } else { 0.5 });

            let white_mat: i32 = white_pieces.iter().map(|&i| PIECE_VALUES[state.pieces[i as usize].piece_type() as usize]).sum();
            let black_mat: i32 = black_pieces.iter().map(|&i| PIECE_VALUES[state.pieces[i as usize].piece_type() as usize]).sum();
            let mat_diff = (white_mat - black_mat) as f32 / 39.0;
            fill(&mut tensor, 33, (mat_diff + 1.0) / 2.0);

            // Compute attack maps using bitboards
            let attacks = AttackMaps::compute(state);

            // Attack maps (34-35)
            for sq in 0..64 {
                let x = (sq % 8) as i8;
                let y = (sq / 8) as i8;
                if (attacks.white >> sq) & 1 != 0 {
                    set(&mut tensor, 34, y, x, 1.0);
                }
                if (attacks.black >> sq) & 1 != 0 {
                    set(&mut tensor, 35, y, x, 1.0);
                }
                if ((attacks.white & attacks.black) >> sq) & 1 != 0 {
                    set(&mut tensor, 36, y, x, 1.0);
                }
            }

            // King zones (37-38)
            if let Some(wk_idx) = white_king_idx {
                let wk = &state.pieces[wk_idx as usize];
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        let nx = wk.x + dx;
                        let ny = wk.y + dy;
                        if nx >= 0 && nx < 8 && ny >= 0 && ny < 8 {
                            set(&mut tensor, 37, ny, nx, 1.0);
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
                            set(&mut tensor, 38, ny, nx, 1.0);
                        }
                    }
                }
            }

            // Pawn advancement (39-40)
            for &idx in &white_pieces {
                let p = &state.pieces[idx as usize];
                if p.piece_type() == PieceType::Pawn {
                    set(&mut tensor, 39, p.y, p.x, p.y as f32 / 7.0);
                }
            }
            for &idx in &black_pieces {
                let p = &state.pieces[idx as usize];
                if p.piece_type() == PieceType::Pawn {
                    set(&mut tensor, 40, p.y, p.x, (7 - p.y) as f32 / 7.0);
                }
            }

            // Abilities ready (41-42)
            let white_ready: usize = white_pieces.iter().filter(|&&i| state.pieces[i as usize].can_use_ability()).count();
            let black_ready: usize = black_pieces.iter().filter(|&&i| state.pieces[i as usize].can_use_ability()).count();
            if !white_pieces.is_empty() {
                fill(&mut tensor, 41, white_ready as f32 / white_pieces.len() as f32);
            }
            if !black_pieces.is_empty() {
                fill(&mut tensor, 42, black_ready as f32 / black_pieces.len() as f32);
            }

            // Low HP targets (43)
            for &idx in white_pieces.iter().chain(black_pieces.iter()) {
                let p = &state.pieces[idx as usize];
                if (p.current_hp as f32) / (p.max_hp as f32) < 0.4 {
                    set(&mut tensor, 43, p.y, p.x, 1.0);
                }
            }

            // Center control (44)
            let center_weights: [[f32; 8]; 8] = [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ];
            for sq in 0..64 {
                let x = sq % 8;
                let y = sq / 8;
                let w = center_weights[y][x];
                if w > 0.0 {
                    let white_ctrl = if (attacks.white >> sq) & 1 != 0 { w * 0.5 } else { 0.0 };
                    let black_ctrl = if (attacks.black >> sq) & 1 != 0 { w * 0.5 } else { 0.0 };
                    tensor[44 * 64 + sq] = ((white_ctrl - black_ctrl) + 1.0) / 2.0;
                } else {
                    tensor[44 * 64 + sq] = 0.5;
                }
            }

            // King tropism (45)
            if let (Some(wk_idx), Some(bk_idx)) = (white_king_idx, black_king_idx) {
                let wk = &state.pieces[wk_idx as usize];
                let bk = &state.pieces[bk_idx as usize];
                let dist = (wk.x - bk.x).abs() + (wk.y - bk.y).abs();
                fill(&mut tensor, 45, 1.0 - dist as f32 / 14.0);
            }

            // === CHANNELS 46-57: Tactical ===

            // Hanging pieces (46-47)
            for &idx in &white_pieces {
                let p = &state.pieces[idx as usize];
                let sq = sq_idx(p.x, p.y);
                if (attacks.black >> sq) & 1 != 0 && (attacks.white >> sq) & 1 == 0 {
                    set(&mut tensor, 46, p.y, p.x, 1.0);
                }
            }
            for &idx in &black_pieces {
                let p = &state.pieces[idx as usize];
                let sq = sq_idx(p.x, p.y);
                if (attacks.white >> sq) & 1 != 0 && (attacks.black >> sq) & 1 == 0 {
                    set(&mut tensor, 47, p.y, p.x, 1.0);
                }
            }

            // Defended pieces (48-49)
            for &idx in &white_pieces {
                let p = &state.pieces[idx as usize];
                let sq = sq_idx(p.x, p.y);
                if (attacks.white >> sq) & 1 != 0 {
                    set(&mut tensor, 48, p.y, p.x, 1.0);
                }
            }
            for &idx in &black_pieces {
                let p = &state.pieces[idx as usize];
                let sq = sq_idx(p.x, p.y);
                if (attacks.black >> sq) & 1 != 0 {
                    set(&mut tensor, 49, p.y, p.x, 1.0);
                }
            }

            // King attackers (50-51)
            if let Some(wk_idx) = white_king_idx {
                let wk = &state.pieces[wk_idx as usize];
                let mut wk_zone: u64 = 0;
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        let nx = wk.x + dx;
                        let ny = wk.y + dy;
                        if nx >= 0 && nx < 8 && ny >= 0 && ny < 8 {
                            wk_zone |= sq_bit(nx, ny);
                        }
                    }
                }
                let mut attackers = 0;
                for i in 0..black_pieces.len() {
                    if attacks.black_per_piece[i] & wk_zone != 0 {
                        attackers += 1;
                    }
                }
                fill(&mut tensor, 50, (attackers as f32 / 6.0).min(1.0));
            }
            if let Some(bk_idx) = black_king_idx {
                let bk = &state.pieces[bk_idx as usize];
                let mut bk_zone: u64 = 0;
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        let nx = bk.x + dx;
                        let ny = bk.y + dy;
                        if nx >= 0 && nx < 8 && ny >= 0 && ny < 8 {
                            bk_zone |= sq_bit(nx, ny);
                        }
                    }
                }
                let mut attackers = 0;
                for i in 0..white_pieces.len() {
                    if attacks.white_per_piece[i] & bk_zone != 0 {
                        attackers += 1;
                    }
                }
                fill(&mut tensor, 51, (attackers as f32 / 6.0).min(1.0));
            }

            // Safe king squares (52-53)
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
                                    set(&mut tensor, 52, ny, nx, 1.0);
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
                                    set(&mut tensor, 53, ny, nx, 1.0);
                                }
                            }
                        }
                    }
                }
            }

            // Passed pawns (54-55)
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
                        set(&mut tensor, 54, p.y, p.x, 1.0);
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
                        set(&mut tensor, 55, p.y, p.x, 1.0);
                    }
                }
            }

            // Open files (56)
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
                        set(&mut tensor, 56, y, file_x, 1.0);
                    }
                }
            }

            // Damage potential (57)
            for (i, &idx) in white_pieces.iter().enumerate() {
                let p = &state.pieces[idx as usize];
                let dmg = p.base_damage as f32 / 10.0;
                let mut bb = attacks.white_per_piece[i];
                while bb != 0 {
                    let sq = bb.trailing_zeros() as usize;
                    tensor[57 * 64 + sq] += dmg;
                    bb &= bb - 1;
                }
            }
            for (i, &idx) in black_pieces.iter().enumerate() {
                let p = &state.pieces[idx as usize];
                let dmg = p.base_damage as f32 / 10.0;
                let mut bb = attacks.black_per_piece[i];
                while bb != 0 {
                    let sq = bb.trailing_zeros() as usize;
                    tensor[57 * 64 + sq] -= dmg;
                    bb &= bb - 1;
                }
            }
            // Normalize to 0-1
            for i in 0..64 {
                tensor[57 * 64 + i] = (tensor[57 * 64 + i] + 1.0) / 2.0;
            }

            tensor.clone()
        })
    }
}

#[pymodule]
fn exchange_simulator(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySimulator>()?;
    m.add_class::<PyMove>()?;
    m.add_class::<PyPiece>()?;
    Ok(())
}
