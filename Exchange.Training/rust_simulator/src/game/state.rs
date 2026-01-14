//! Game state representation with Zobrist hashing.
//!
//! GameState contains all the information about the current game position:
//! pieces, board, hash, turn counters, and terminal conditions.

use crate::core::{
    is_on_board, sq_bit, sq_idx, Piece, PieceType, Team,
    ALL_DIRS, CARDINAL, DIAGONAL, DRAW_REPETITION_COUNT,
    KNIGHT_MOVES, MAX_PIECES, ZOBRIST,
};

/// Bitboard type for efficient attack map queries.
pub type Bitboard = u64;

// ============================================================================
// GameState
// ============================================================================

/// The complete state of an EXCHANGE game.
#[derive(Clone)]
pub struct GameState {
    /// All pieces in the game (up to MAX_PIECES)
    pub pieces: [Piece; MAX_PIECES],
    /// Number of pieces currently in play
    pub piece_count: u8,

    /// O(1) board lookup: board[y][x] = Some(piece_idx) or None
    pub board: [[Option<u8>; 8]; 8],

    /// Zobrist hash for fast position comparison
    pub hash: u64,

    /// Which team is to move
    pub side_to_move: Team,
    /// Turns remaining for White's Royal Decree (+2 bonus while > 0)
    pub white_royal_decree_turns: i8,
    /// Turns remaining for Black's Royal Decree
    pub black_royal_decree_turns: i8,
    /// Current turn number
    pub turn_number: i32,
    /// Consecutive moves without damage (for draw detection)
    pub moves_without_damage: i32,
    /// History of position hashes (for repetition detection)
    pub position_history: Vec<u64>,
    /// Which team the AI is playing as
    pub playing_as: Team,
    /// Winner of the game (if terminal)
    pub winner: Option<Team>,
    /// Whether the game has ended
    pub is_terminal: bool,
    /// Whether the game ended in a draw
    pub is_draw: bool,

    /// Cached piece indices for White (faster iteration)
    pub white_pieces: [u8; 16],
    pub white_count: u8,
    /// Cached piece indices for Black
    pub black_pieces: [u8; 16],
    pub black_count: u8,
}

impl GameState {
    /// Create a new game with standard starting position.
    pub fn new() -> Self {
        let mut state = GameState {
            pieces: [Piece::new(PieceType::Pawn, Team::White, 0, 0); MAX_PIECES],
            piece_count: 0,
            board: [[None; 8]; 8],
            hash: 0,
            side_to_move: Team::White,
            white_royal_decree_turns: 0,
            black_royal_decree_turns: 0,
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

    /// Add a piece to the game.
    pub fn add_piece(&mut self, piece: Piece) {
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

    /// Compute the full Zobrist hash (used for initialization).
    pub fn compute_full_hash(&self) -> u64 {
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

        // Include Royal Decree state in hash
        if self.white_royal_decree_turns > 0 {
            hash ^= zobrist.royal_decree[0];
        }
        if self.black_royal_decree_turns > 0 {
            hash ^= zobrist.royal_decree[1];
        }

        hash
    }

    /// Update hash for piece movement.
    #[inline]
    pub fn hash_move_piece(&mut self, idx: u8, old_x: i8, old_y: i8, new_x: i8, new_y: i8) {
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

    /// Update hash for HP change.
    #[inline]
    pub fn hash_hp_change(&mut self, idx: u8, old_hp_bucket: usize, new_hp_bucket: usize) {
        if old_hp_bucket == new_hp_bucket { return; }

        let zobrist = &*ZOBRIST;
        let p = &self.pieces[idx as usize];
        let sq = sq_idx(p.x, p.y);
        let pt = p.piece_type() as usize;
        let team = p.team() as usize;

        self.hash ^= zobrist.pieces[sq][pt][team][old_hp_bucket];
        self.hash ^= zobrist.pieces[sq][pt][team][new_hp_bucket];
    }

    /// Update hash for piece removal (death).
    #[inline]
    pub fn hash_remove_piece(&mut self, idx: u8) {
        let zobrist = &*ZOBRIST;
        let p = &self.pieces[idx as usize];
        let sq = sq_idx(p.x, p.y);
        let pt = p.piece_type() as usize;
        let team = p.team() as usize;
        let hp = p.hp_bucket();

        self.hash ^= zobrist.pieces[sq][pt][team][hp];
    }

    /// Get piece at position (if alive).
    #[inline]
    pub fn get_piece_at(&self, x: i8, y: i8) -> Option<u8> {
        if !is_on_board(x, y) { return None; }
        self.board[y as usize][x as usize].filter(|&idx| self.pieces[idx as usize].is_alive())
    }

    /// Get all piece indices for a team.
    pub fn get_pieces_for_team(&self, team: Team) -> &[u8] {
        match team {
            Team::White => &self.white_pieces[..self.white_count as usize],
            Team::Black => &self.black_pieces[..self.black_count as usize],
        }
    }

    /// Find the king for a team.
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

    /// Get Royal Decree bonus (+2 if active, 0 otherwise).
    #[inline]
    pub fn get_royal_decree_bonus(&self, team: Team) -> i16 {
        let turns = match team {
            Team::White => self.white_royal_decree_turns,
            Team::Black => self.black_royal_decree_turns,
        };
        if turns > 0 { 2 } else { 0 }
    }

    /// Record current position in history.
    pub fn record_position(&mut self) {
        self.position_history.push(self.hash);
    }

    /// Check for threefold repetition.
    pub fn is_threefold_repetition(&self) -> bool {
        if self.position_history.len() < DRAW_REPETITION_COUNT {
            return false;
        }
        self.position_history.iter().filter(|&&h| h == self.hash).count() >= DRAW_REPETITION_COUNT
    }

    /// Check and update terminal conditions.
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

        // Draw condition 2: Threefold repetition
        if self.is_threefold_repetition() {
            self.is_terminal = true;
            self.is_draw = true;
            self.winner = None;
        }
    }

    /// Clone for MCTS without position history (saves memory).
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
// Attack Maps
// ============================================================================

/// Bitboard-based attack maps for O(1) attack queries.
pub struct AttackMaps {
    pub white: Bitboard,
    pub black: Bitboard,
    pub white_per_piece: [Bitboard; 16],
    pub black_per_piece: [Bitboard; 16],
}

impl AttackMaps {
    /// Compute attack maps for the current game state.
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

    /// Compute attack bitboard for a single piece.
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
