//! 1-ply evaluation functions.
//!
//! This module contains functions for 1-ply lookahead evaluation,
//! which evaluates all legal moves and selects the best one based
//! on the neural network's position evaluation.

use half::f16;
use rayon::prelude::*;

use crate::core::Move;
use crate::game::{GameSimulator, GameState, TensorComputer, INPUT_CHANNELS};
use super::OnnxEvaluator;

/// Run 1-ply evaluation with native ONNX - NO PYTHON IN THE LOOP!
pub fn run_1ply_native_impl(
    states: &[GameState],
    evaluator: &OnnxEvaluator,
    temperature: f32,
    _draw_penalty: f32,
    repetition_penalty: f32,
) -> Result<Vec<(Option<Move>, bool)>, ort::Error> {
    const BOARD_SIZE: usize = 8;
    let tensor_size = INPUT_CHANNELS * BOARD_SIZE * BOARD_SIZE;

    // Parallel move generation (tensors are f16)
    let candidates: Vec<(Vec<Move>, Vec<f16>, bool)> = states.par_iter()
        .map(|state| {
            if state.is_terminal {
                return (Vec::new(), Vec::new(), true);
            }

            let temp_sim = GameSimulator::with_state(state.clone());
            let moves = temp_sim.generate_moves(state.side_to_move);

            if moves.is_empty() {
                return (Vec::new(), Vec::new(), true);
            }

            let mut tensors: Vec<f16> = Vec::with_capacity(moves.len() * tensor_size);
            for mv in &moves {
                let mut next_sim = GameSimulator::with_state(state.clone());
                next_sim.make_move(mv);
                tensors.extend_from_slice(&TensorComputer::new(next_sim.state).compute_tensor());
            }

            (moves, tensors, false)
        })
        .collect();

    // Flatten tensors for batch evaluation
    let mut batch_tensors = Vec::new();
    let mut move_counts = Vec::new();
    let mut active_indices = Vec::new();

    for (i, (_, tensors, is_terminal)) in candidates.iter().enumerate() {
        if !*is_terminal && !tensors.is_empty() {
            batch_tensors.extend_from_slice(tensors);
            move_counts.push(tensors.len() / tensor_size);
            active_indices.push(i);
        }
    }

    // Batch evaluation
    let all_values = if !batch_tensors.is_empty() {
        let num_positions = batch_tensors.len() / tensor_size;
        let (values, _) = evaluator.evaluate(&batch_tensors, num_positions)?;
        values
    } else {
        Vec::new()
    };

    // Select moves
    let mut results = Vec::with_capacity(states.len());
    let mut val_idx = 0;

    for (i, (moves, _, is_terminal)) in candidates.iter().enumerate() {
        if *is_terminal || moves.is_empty() {
            results.push((None, true));
            continue;
        }

        let active_pos = active_indices.iter().position(|&x| x == i);
        if active_pos.is_none() {
            results.push((None, true));
            continue;
        }

        let count = move_counts[active_pos.unwrap()];
        let move_values = &all_values[val_idx..val_idx + count];
        val_idx += count;

        // Negate (opponent's perspective)
        let mut adjusted_values: Vec<f32> = move_values.iter().map(|v| -v).collect();

        // Repetition penalty
        if repetition_penalty > 0.0 {
            let state = &states[i];
            for (m_idx, mv) in moves.iter().enumerate() {
                let mut next_sim = GameSimulator::with_state(state.clone());
                next_sim.make_move(mv);
                let reps = state.position_history.iter().filter(|&&h| h == next_sim.state.hash).count();
                if reps > 0 {
                    adjusted_values[m_idx] -= reps as f32 * repetition_penalty;
                }
            }
        }

        // Selection
        let selected_idx = if temperature > 0.0 {
            let max_val = adjusted_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exps: Vec<f32> = adjusted_values.iter().map(|v| ((v - max_val) / temperature).exp()).collect();
            let sum: f32 = exps.iter().sum();

            let r = fastrand::f32();
            let mut cum = 0.0;
            let mut selected = 0;
            for (k, &e) in exps.iter().enumerate() {
                cum += e / sum;
                if r <= cum {
                    selected = k;
                    break;
                }
            }
            selected
        } else {
            adjusted_values.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        };

        results.push((Some(moves[selected_idx]), false));
    }

    Ok(results)
}
