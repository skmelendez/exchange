//! Python-exposed types.
//!
//! This module contains PyO3-wrapped types for Python interop:
//! PyMove, PyPiece.

use pyo3::prelude::*;

use crate::core::{Move, MoveType, Piece, PieceType, Team};

/// Python-exposed move representation.
#[pyclass]
#[derive(Clone)]
pub struct PyMove {
    #[pyo3(get)]
    pub piece_idx: u8,
    #[pyo3(get)]
    pub move_type: String,
    #[pyo3(get)]
    pub from_x: i8,
    #[pyo3(get)]
    pub from_y: i8,
    #[pyo3(get)]
    pub to_x: i8,
    #[pyo3(get)]
    pub to_y: i8,
    #[pyo3(get)]
    pub attack_x: i8,
    #[pyo3(get)]
    pub attack_y: i8,
    #[pyo3(get)]
    pub ability_target_x: i8,
    #[pyo3(get)]
    pub ability_target_y: i8,
    #[pyo3(get)]
    pub ability_id: Option<String>,
}

#[pymethods]
impl PyMove {
    #[new]
    #[pyo3(signature = (
        piece_idx,
        move_type,
        from_x,
        from_y,
        to_x,
        to_y,
        attack_x = -1,
        attack_y = -1,
        ability_target_x = -1,
        ability_target_y = -1,
        ability_id = None
    ))]
    fn new(
        piece_idx: u8,
        move_type: String,
        from_x: i8,
        from_y: i8,
        to_x: i8,
        to_y: i8,
        attack_x: i8,
        attack_y: i8,
        ability_target_x: i8,
        ability_target_y: i8,
        ability_id: Option<String>,
    ) -> Self {
        PyMove {
            piece_idx,
            move_type,
            from_x,
            from_y,
            to_x,
            to_y,
            attack_x,
            attack_y,
            ability_target_x,
            ability_target_y,
            ability_id,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PyMove(piece={}, type={}, from=({},{}), to=({},{}))",
            self.piece_idx, self.move_type, self.from_x, self.from_y, self.to_x, self.to_y
        )
    }
}

impl PyMove {
    /// Convert from internal Move to PyMove.
    pub fn from_move(m: &Move) -> Self {
        let move_type = match m.move_type {
            MoveType::Move => "Move",
            MoveType::Attack => "Attack",
            MoveType::MoveAndAttack => "MoveAndAttack",
            MoveType::Ability => "Ability",
        }.to_string();

        let ability_id = m.ability_id.map(|a| {
            match a {
                crate::core::AbilityId::RoyalDecree => "RoyalDecree",
                crate::core::AbilityId::Overextend => "Overextend",
                crate::core::AbilityId::Interpose => "Interpose",
                crate::core::AbilityId::Consecration => "Consecration",
                crate::core::AbilityId::Skirmish => "Skirmish",
                crate::core::AbilityId::Advance => "Advance",
            }.to_string()
        });

        PyMove {
            piece_idx: m.piece_idx,
            move_type,
            from_x: m.from_x,
            from_y: m.from_y,
            to_x: m.to_x,
            to_y: m.to_y,
            attack_x: m.attack_x,
            attack_y: m.attack_y,
            ability_target_x: m.ability_target_x,
            ability_target_y: m.ability_target_y,
            ability_id,
        }
    }

    /// Convert from PyMove to internal Move.
    pub fn to_move(&self) -> Move {
        use crate::core::AbilityId;

        let move_type = match self.move_type.as_str() {
            "Move" => MoveType::Move,
            "Attack" => MoveType::Attack,
            "MoveAndAttack" => MoveType::MoveAndAttack,
            "Ability" => MoveType::Ability,
            _ => MoveType::Move,
        };

        let ability_id = self.ability_id.as_ref().map(|s| {
            match s.as_str() {
                "RoyalDecree" => AbilityId::RoyalDecree,
                "Overextend" => AbilityId::Overextend,
                "Interpose" => AbilityId::Interpose,
                "Consecration" => AbilityId::Consecration,
                "Skirmish" => AbilityId::Skirmish,
                "Advance" => AbilityId::Advance,
                _ => AbilityId::RoyalDecree,
            }
        });

        Move {
            piece_idx: self.piece_idx,
            move_type,
            from_x: self.from_x,
            from_y: self.from_y,
            to_x: self.to_x,
            to_y: self.to_y,
            attack_x: self.attack_x,
            attack_y: self.attack_y,
            ability_target_x: self.ability_target_x,
            ability_target_y: self.ability_target_y,
            ability_id,
        }
    }
}

/// Python-exposed piece representation.
#[pyclass]
#[derive(Clone)]
pub struct PyPiece {
    #[pyo3(get)]
    pub piece_type: String,
    #[pyo3(get)]
    pub team: String,
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
    pub is_alive: bool,
    #[pyo3(get)]
    pub ability_used_this_match: bool,
    #[pyo3(get)]
    pub interpose_active: bool,
    #[pyo3(get)]
    pub was_pawn: bool,
}

#[pymethods]
impl PyPiece {
    fn __repr__(&self) -> String {
        format!(
            "PyPiece({} {} at ({},{}), HP: {}/{})",
            self.team, self.piece_type, self.x, self.y, self.current_hp, self.max_hp
        )
    }
}

impl PyPiece {
    /// Convert from internal Piece to PyPiece.
    pub fn from_piece(p: &Piece) -> Self {
        let piece_type = match p.piece_type() {
            PieceType::King => "King",
            PieceType::Queen => "Queen",
            PieceType::Rook => "Rook",
            PieceType::Bishop => "Bishop",
            PieceType::Knight => "Knight",
            PieceType::Pawn => "Pawn",
        }.to_string();

        let team = match p.team() {
            Team::White => "White",
            Team::Black => "Black",
        }.to_string();

        PyPiece {
            piece_type,
            team,
            x: p.x,
            y: p.y,
            current_hp: p.current_hp,
            max_hp: p.max_hp,
            base_damage: p.base_damage,
            ability_cooldown: p.ability_cooldown,
            is_alive: p.is_alive(),
            ability_used_this_match: p.ability_used_this_match(),
            interpose_active: p.interpose_active(),
            was_pawn: p.was_pawn(),
        }
    }
}
