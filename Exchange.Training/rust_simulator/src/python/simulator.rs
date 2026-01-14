//! Python-exposed simulator.
//!
//! This module contains the PySimulator class that wraps GameSimulator
//! for Python interop.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray3, IntoPyArray};
use ndarray::Array3;

use crate::core::{Team, BOARD_SIZE};
use crate::game::{compute_tensor, f16, GameSimulator, INPUT_CHANNELS};
use super::types::{PyMove, PyPiece};

/// Python-exposed game simulator.
#[pyclass]
pub struct PySimulator {
    pub sim: GameSimulator,
}

#[pymethods]
impl PySimulator {
    #[new]
    fn new() -> Self {
        PySimulator {
            sim: GameSimulator::new(),
        }
    }

    /// Clone the simulator.
    fn clone(&self) -> Self {
        PySimulator {
            sim: GameSimulator::with_state(self.sim.state.clone()),
        }
    }

    /// Reset to initial position.
    fn reset(&mut self) {
        self.sim = GameSimulator::new();
    }

    /// Get legal moves for a team.
    fn get_legal_moves(&self, team: &str) -> Vec<PyMove> {
        let team = match team {
            "White" => Team::White,
            "Black" => Team::Black,
            _ => Team::White,
        };
        self.sim.generate_moves(team)
            .into_iter()
            .map(|m| PyMove::from_move(&m))
            .collect()
    }

    /// Make a move and return (damage_dealt, interpose_blocked).
    ///
    /// Returns:
    /// - damage_dealt: Total damage dealt by this move
    /// - interpose_blocked: Damage absorbed by Interposing Rook (0 if no Interpose)
    fn make_move(&mut self, py_move: &PyMove) -> (i32, i32) {
        let mv = py_move.to_move();
        let (damage, undo) = self.sim.make_move(&mv);

        // Calculate interpose blocked amount from snapshot comparison
        let interpose_blocked = if let Some(rook_snapshot) = &undo.interpose_rook_snapshot {
            let rook_idx = undo.interpose_rook_idx.unwrap() as usize;
            let old_hp = rook_snapshot.current_hp;
            let new_hp = self.sim.state.pieces[rook_idx].current_hp;
            (old_hp - new_hp) as i32
        } else {
            0
        };

        (damage, interpose_blocked)
    }

    /// Make a move by index from legal moves.
    /// Returns (damage_dealt, interpose_blocked).
    fn make_move_by_index(&mut self, move_index: usize) -> PyResult<(i32, i32)> {
        let moves = self.sim.generate_moves(self.sim.state.side_to_move);
        if move_index >= moves.len() {
            return Err(PyValueError::new_err(format!(
                "Move index {} out of range (0..{})",
                move_index, moves.len()
            )));
        }
        let (damage, undo) = self.sim.make_move(&moves[move_index]);

        let interpose_blocked = if let Some(rook_snapshot) = &undo.interpose_rook_snapshot {
            let rook_idx = undo.interpose_rook_idx.unwrap() as usize;
            let old_hp = rook_snapshot.current_hp;
            let new_hp = self.sim.state.pieces[rook_idx].current_hp;
            (old_hp - new_hp) as i32
        } else {
            0
        };

        Ok((damage, interpose_blocked))
    }

    /// Get all pieces.
    fn get_pieces(&self) -> Vec<PyPiece> {
        self.sim.state.pieces[..self.sim.state.piece_count as usize]
            .iter()
            .map(|p| PyPiece::from_piece(p))
            .collect()
    }

    /// Get the current side to move.
    fn side_to_move(&self) -> String {
        match self.sim.state.side_to_move {
            Team::White => "White".to_string(),
            Team::Black => "Black".to_string(),
        }
    }

    /// Check if the game is over.
    fn is_terminal(&self) -> bool {
        self.sim.state.is_terminal
    }

    /// Check if the game is a draw.
    fn is_draw(&self) -> bool {
        self.sim.state.is_draw
    }

    /// Get the winner (if any).
    fn winner(&self) -> Option<String> {
        self.sim.state.winner.map(|w| match w {
            Team::White => "White".to_string(),
            Team::Black => "Black".to_string(),
        })
    }

    /// Get the current turn number.
    fn turn_number(&self) -> i32 {
        self.sim.state.turn_number
    }

    /// Get the moves without damage counter.
    fn moves_without_damage(&self) -> i32 {
        self.sim.state.moves_without_damage
    }

    /// Get the position hash.
    fn position_hash(&self) -> u64 {
        self.sim.state.hash
    }

    /// Set which team the AI is playing as.
    fn set_playing_as(&mut self, team: &str) {
        self.sim.state.playing_as = match team {
            "White" => Team::White,
            "Black" => Team::Black,
            _ => Team::White,
        };
    }

    /// Get the team the AI is playing as.
    fn playing_as(&self) -> String {
        match self.sim.state.playing_as {
            Team::White => "White".to_string(),
            Team::Black => "Black".to_string(),
        }
    }

    /// Get the repetition count for a position that would result from making a move.
    fn get_repetition_count(&self, move_index: usize) -> PyResult<usize> {
        let moves = self.sim.generate_moves(self.sim.state.side_to_move);
        if move_index >= moves.len() {
            return Err(PyValueError::new_err(format!(
                "Move index {} out of range (0..{})",
                move_index, moves.len()
            )));
        }

        // Clone state (preserving position history), make move, check repetition count
        let mut sim_copy = GameSimulator::with_state(self.sim.state.clone());
        sim_copy.make_move_simple(&moves[move_index]);
        let new_hash = sim_copy.state.hash;

        // Count how many times this position appears in history
        let count = sim_copy.state.position_history.iter().filter(|&&h| h == new_hash).count();
        Ok(count)
    }

    /// Convert state to tensor for neural network input.
    /// Internal computation uses f16, converted to f32 for Python/numpy compatibility.
    fn to_tensor<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f32>>> {
        let tensor_f16 = compute_tensor(&self.sim.state);
        // Convert f16 -> f32 for numpy (numpy doesn't support f16 directly in rust bindings)
        let tensor: Vec<f32> = tensor_f16.iter().map(|&v| v.to_f32()).collect();
        let arr = Array3::from_shape_vec((INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), tensor)
            .map_err(|e| PyValueError::new_err(format!("Failed to create tensor: {}", e)))?;
        Ok(arr.into_pyarray_bound(py))
    }

    /// Get White's Royal Decree turns remaining.
    fn white_royal_decree_turns(&self) -> i8 {
        self.sim.state.white_royal_decree_turns
    }

    /// Get Black's Royal Decree turns remaining.
    fn black_royal_decree_turns(&self) -> i8 {
        self.sim.state.black_royal_decree_turns
    }

    /// Get the number of legal moves for a team.
    fn num_legal_moves(&self, team: &str) -> usize {
        let team = match team {
            "White" => Team::White,
            "Black" => Team::Black,
            _ => Team::White,
        };
        self.sim.generate_moves(team).len()
    }

    /// Set the random seed for reproducibility.
    fn set_seed(&self, seed: u64) {
        fastrand::seed(seed);
    }
}

impl PySimulator {
    /// Internal helper to compute tensor (f16).
    pub fn compute_tensor(&self) -> Vec<f16> {
        compute_tensor(&self.sim.state)
    }
}
