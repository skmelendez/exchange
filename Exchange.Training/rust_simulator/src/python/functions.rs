//! Python-exposed functions.
//!
//! This module contains the #[pyfunction] wrappers for MCTS and evaluation.

use half::f16;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::PyArray4;
use ndarray::Array4;
use rayon::prelude::*;

use crate::core::BOARD_SIZE;
use crate::game::{GameSimulator, GameState, INPUT_CHANNELS, TensorComputer};
use crate::mcts::{add_dirichlet_noise, MctsTree, select_move};
use crate::eval::run_1ply_native_impl;
use super::simulator::PySimulator;
use super::evaluator::PyOnnxEvaluator;
use super::types::PyMove;

/// Convert f16 slice to f32 Vec for Python callbacks (PyTorch expects f32)
#[inline]
fn f16_to_f32(data: &[f16]) -> Vec<f32> {
    data.iter().map(|&v| v.to_f32()).collect()
}

/// Run MCTS with Python-based evaluation callback.
#[pyfunction]
pub fn run_mcts_batch(
    py: Python<'_>,
    simulators: Vec<PyRef<PySimulator>>,
    eval_fn: PyObject,
    num_simulations: usize,
    c_puct: f32,
    dirichlet_alpha: f32,
    dirichlet_epsilon: f32,
    temperature: f32,
) -> PyResult<Vec<(Option<PyMove>, Vec<f32>)>> {

    // Initialize trees
    let mut trees: Vec<MctsTree> = simulators.iter()
        .map(|sim| MctsTree::new(sim.sim.state.clone_for_mcts()))
        .collect();

    // Initial expansion
    {
        let num_trees = trees.len();
        let tensor_size = INPUT_CHANNELS * BOARD_SIZE * BOARD_SIZE;
        let mut batch_tensor_data: Vec<f16> = vec![f16::ZERO; num_trees * tensor_size];

        let valid_indices: Vec<usize> = trees.par_iter()
            .enumerate()
            .filter_map(|(i, tree)| if !tree.nodes[0].is_terminal { Some(i) } else { None })
            .collect();

        if !valid_indices.is_empty() {
            // Fill tensor buffer in parallel (f16)
            batch_tensor_data.par_chunks_mut(tensor_size)
                .zip(trees.par_iter())
                .for_each(|(chunk, tree)| {
                    if !tree.nodes[0].is_terminal {
                        let temp = TensorComputer::new(tree.nodes[0].state.clone());
                        let t = temp.compute_tensor();
                        chunk.copy_from_slice(&t);
                    }
                });

            // Compact valid tensors and convert to f32 for Python call
            let mut flat_tensors_f16: Vec<f16> = Vec::with_capacity(valid_indices.len() * tensor_size);
            for &idx in &valid_indices {
                let start = idx * tensor_size;
                flat_tensors_f16.extend_from_slice(&batch_tensor_data[start..start + tensor_size]);
            }
            let flat_tensors = f16_to_f32(&flat_tensors_f16);

            let arr = Array4::from_shape_vec((valid_indices.len(), INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), flat_tensors).unwrap();
            let py_tensor = PyArray4::from_owned_array_bound(py, arr);

            let result = eval_fn.call1(py, (py_tensor,))?;
            let (_values, policies): (Vec<f32>, Vec<Vec<f32>>) = result.extract(py)?;

            for (idx, &tree_idx) in valid_indices.iter().enumerate() {
                let policy = &policies[idx];
                trees[tree_idx].expand(0, policy);

                // Add Root Noise
                add_dirichlet_noise(&mut trees[tree_idx], dirichlet_alpha, dirichlet_epsilon);
            }
        }
    }

    // Simulation Loop
    let num_trees = trees.len();
    let tensor_size = INPUT_CHANNELS * BOARD_SIZE * BOARD_SIZE;
    let mut batch_tensor_data: Vec<f16> = vec![f16::ZERO; num_trees * tensor_size];

    for _ in 0..num_simulations {
        // Parallel Selection with Virtual Loss & Tensor Generation (f16)
        let selection_results: Vec<(usize, Vec<usize>, bool)> = trees.par_iter()
            .zip(batch_tensor_data.par_chunks_mut(tensor_size))
            .map(|(tree, chunk)| {
                if tree.nodes[0].is_terminal {
                    return (0, vec![], false);
                }

                let (leaf_idx, path) = tree.select_leaf_with_virtual_loss(0, c_puct);

                let node = &tree.nodes[leaf_idx];
                if !node.is_terminal {
                    let temp = TensorComputer::new(node.state.clone());
                    let t = temp.compute_tensor();
                    chunk.copy_from_slice(&t);
                    (leaf_idx, path, true)
                } else {
                    (leaf_idx, path, false)
                }
            })
            .collect();

        // Gather valid indices for batching
        let valid_indices: Vec<usize> = selection_results.iter()
            .enumerate()
            .filter_map(|(i, &(_, _, is_valid))| if is_valid { Some(i) } else { None })
            .collect();

        if !valid_indices.is_empty() {
            // Gather f16 tensors and convert to f32 for Python
            let mut flat_tensors_f16: Vec<f16> = Vec::with_capacity(valid_indices.len() * tensor_size);
            for &idx in &valid_indices {
                let start = idx * tensor_size;
                flat_tensors_f16.extend_from_slice(&batch_tensor_data[start..start + tensor_size]);
            }
            let flat_tensors = f16_to_f32(&flat_tensors_f16);

            let arr = Array4::from_shape_vec((valid_indices.len(), INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), flat_tensors).unwrap();
            let py_tensor = PyArray4::from_owned_array_bound(py, arr);

            let result = eval_fn.call1(py, (py_tensor,))?;
            let (values, policies): (Vec<f32>, Vec<Vec<f32>>) = result.extract(py)?;

            // Map values/policies back to full length
            let mut full_updates = vec![None; num_trees];
            for (i, &tree_idx) in valid_indices.iter().enumerate() {
                full_updates[tree_idx] = Some((values[i], &policies[i]));
            }

            trees.par_iter_mut()
                .zip(selection_results.par_iter())
                .zip(full_updates.par_iter())
                .for_each(|((tree, (leaf_idx, path, _)), update)| {
                    if let Some((value, policy)) = update {
                        tree.expand(*leaf_idx, policy);
                        tree.backup(path.clone(), *value);
                    }
                });
        }

        // Handle terminal nodes backup
        trees.par_iter_mut()
            .zip(selection_results.par_iter())
            .for_each(|(tree, (leaf_idx, path, is_valid))| {
                if !is_valid && !path.is_empty() && !tree.nodes[0].is_terminal {
                    let node = &tree.nodes[*leaf_idx];
                    if node.is_terminal {
                        let val = if let Some(winner) = node.state.winner {
                            if winner == node.state.side_to_move { 1.0 } else { -1.0 }
                        } else {
                            0.0
                        };
                        tree.backup(path.clone(), val);
                    }
                }
            });
    }

    // Results
    let results: Vec<(Option<PyMove>, Vec<f32>)> = trees.par_iter()
        .map(|tree| {
            let (best_move, move_probs) = select_move(tree, temperature);
            let py_move = best_move.map(|m| PyMove::from_move(&m));
            (py_move, move_probs)
        })
        .collect();

    Ok(results)
}

/// Run 1-ply evaluation with Python callback (optimized version).
#[pyfunction]
pub fn run_1ply_batch_optimized(
    py: Python<'_>,
    simulators: Vec<pyo3::Py<PySimulator>>,
    eval_fn: PyObject,
    temperature: f32,
    _draw_penalty: f32,
    repetition_penalty: f32,
) -> PyResult<Vec<(i32, Option<PyMove>, bool)>> {

    let tensor_size = INPUT_CHANNELS * BOARD_SIZE * BOARD_SIZE;

    // Extract states
    let states: Vec<GameState> = simulators.iter()
        .map(|s| s.borrow(py).sim.state.clone())
        .collect();

    // Parallel move generation (f16 tensors)
    let candidates: Vec<(usize, Vec<(crate::core::Move, i32)>, Vec<f16>, bool)> = states.par_iter()
        .enumerate()
        .map(|(i, state)| {
            if state.is_terminal {
                return (i, Vec::new(), Vec::new(), true);
            }

            let temp_sim = GameSimulator::with_state(state.clone());
            let moves = temp_sim.generate_moves(temp_sim.state.side_to_move);

            if moves.is_empty() {
                return (i, Vec::new(), Vec::new(), true);
            }

            let mut game_tensors: Vec<f16> = Vec::with_capacity(moves.len() * tensor_size);
            let mut moves_with_dmg = Vec::with_capacity(moves.len());

            for mv in &moves {
                let mut next_sim = GameSimulator::with_state(state.clone());
                let _ = next_sim.make_move(mv);
                let t = TensorComputer::new(next_sim.state).compute_tensor();
                game_tensors.extend_from_slice(&t);
                moves_with_dmg.push((*mv, 0));
            }

            (i, moves_with_dmg, game_tensors, false)
        })
        .collect();

    // Batch Eval (collect f16, convert to f32 for Python)
    let mut batch_tensors_f16: Vec<f16> = Vec::new();
    let mut game_move_counts = Vec::with_capacity(simulators.len());
    let mut active_indices = Vec::with_capacity(simulators.len());

    for (i, _, tensors, is_terminal) in &candidates {
        if !*is_terminal {
            batch_tensors_f16.extend_from_slice(tensors);
            let num_moves = tensors.len() / tensor_size;
            game_move_counts.push(num_moves);
            active_indices.push(*i);
        }
    }

    let mut all_values = Vec::new();
    if !batch_tensors_f16.is_empty() {
        let num_total_positions = batch_tensors_f16.len() / tensor_size;
        let batch_tensors = f16_to_f32(&batch_tensors_f16);
        let arr = Array4::from_shape_vec((num_total_positions, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), batch_tensors).unwrap();
        let py_tensor = PyArray4::from_owned_array_bound(py, arr);

        let result = eval_fn.call1(py, (py_tensor,))?;
        let (values, _): (Vec<f32>, PyObject) = result.extract(py)?;
        all_values = values;
    }

    // Select and Execute
    let mut results = Vec::with_capacity(simulators.len());
    let mut val_idx = 0;

    for (_idx, (i, moves_with_dmg, _, is_terminal)) in candidates.iter().enumerate() {
        if *is_terminal {
            results.push((0, None, true));
            continue;
        }

        let count = game_move_counts[active_indices.iter().position(|&x| x == *i).unwrap()];
        let move_values = &all_values[val_idx..val_idx + count];
        val_idx += count;

        let values: Vec<f32> = move_values.iter().map(|v| -v).collect();

        // Repetition penalty
        let mut adjusted_values = values.clone();
        if repetition_penalty > 0.0 {
             let sim_ref = simulators[*i].borrow_mut(py);
             for (m_idx, (mv, _)) in moves_with_dmg.iter().enumerate() {
                 let mut next_sim = GameSimulator::with_state(sim_ref.sim.state.clone());
                 next_sim.make_move(mv);
                 let h = next_sim.state.hash;
                 let reps = sim_ref.sim.state.position_history.iter().filter(|&&ph| ph == h).count();
                 if reps > 0 {
                     adjusted_values[m_idx] -= reps as f32 * repetition_penalty;
                 }
             }
        }

        // Selection
        let selected_idx = if temperature > 0.0 {
            let max_val = adjusted_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exps: Vec<f32> = adjusted_values.iter().map(|v| (v - max_val).exp() / temperature).collect();
            let sum: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|v| v / sum).collect();

            let r = fastrand::f32();
            let mut c = 0.0;
            let mut selected = 0;
            for (k, &p) in probs.iter().enumerate() {
                c += p;
                if r <= c {
                    selected = k;
                    break;
                }
            }
            selected
        } else {
            let mut max_val = f32::NEG_INFINITY;
            let mut selected = 0;
            for (k, &v) in adjusted_values.iter().enumerate() {
                if v > max_val {
                    max_val = v;
                    selected = k;
                }
            }
            selected
        };

        let (selected_move, _) = moves_with_dmg[selected_idx];

        // Execute on the Python object
        let mut sim_ref = simulators[*i].borrow_mut(py);
        let (damage, _) = sim_ref.sim.make_move(&selected_move);

        results.push((damage, Some(PyMove::from_move(&selected_move)), false));
    }

    Ok(results)
}

/// Run MCTS with native ONNX inference - NO PYTHON IN THE LOOP!
#[pyfunction]
#[pyo3(signature = (simulators, evaluator, num_simulations, c_puct, dirichlet_alpha, dirichlet_epsilon, temperature))]
pub fn run_mcts_native(
    py: Python<'_>,
    simulators: Vec<PyRef<PySimulator>>,
    evaluator: &PyOnnxEvaluator,
    num_simulations: usize,
    c_puct: f32,
    dirichlet_alpha: f32,
    dirichlet_epsilon: f32,
    temperature: f32,
) -> PyResult<Vec<(Option<PyMove>, Vec<f32>)>> {

    // Extract states while we have the GIL
    let states: Vec<GameState> = simulators.iter()
        .map(|s| s.sim.state.clone())
        .collect();

    // Release GIL for the heavy computation!
    py.allow_threads(|| {
        crate::mcts::run_mcts_native_impl(
            &states,
            &evaluator.evaluator,
            num_simulations,
            c_puct,
            dirichlet_alpha,
            dirichlet_epsilon,
            temperature,
        )
    })
    .map(|results| {
        results.into_iter()
            .map(|(mv, probs)| (mv.map(|m| PyMove::from_move(&m)), probs))
            .collect()
    })
    .map_err(|e| PyValueError::new_err(format!("MCTS error: {}", e)))
}

/// Run 1-ply evaluation with native ONNX - NO PYTHON IN THE LOOP!
#[pyfunction]
#[pyo3(signature = (simulators, evaluator, temperature, draw_penalty, repetition_penalty))]
pub fn run_1ply_native(
    py: Python<'_>,
    simulators: Vec<pyo3::Py<PySimulator>>,
    evaluator: &PyOnnxEvaluator,
    temperature: f32,
    draw_penalty: f32,
    repetition_penalty: f32,
) -> PyResult<Vec<(i32, Option<PyMove>, bool)>> {

    // Extract states (needs GIL)
    let states: Vec<GameState> = simulators.iter()
        .map(|s| s.borrow(py).sim.state.clone())
        .collect();

    // Heavy computation without GIL
    let move_selections = py.allow_threads(|| {
        run_1ply_native_impl(&states, &evaluator.evaluator, temperature, draw_penalty, repetition_penalty)
    }).map_err(|e| PyValueError::new_err(format!("1-ply error: {}", e)))?;

    // Apply moves (needs GIL)
    let mut results = Vec::with_capacity(simulators.len());
    for (i, (selected_move, is_terminal)) in move_selections.into_iter().enumerate() {
        if is_terminal {
            results.push((0, None, true));
        } else if let Some(mv) = selected_move {
            let mut sim_ref = simulators[i].borrow_mut(py);
            let (damage, _) = sim_ref.sim.make_move(&mv);
            results.push((damage, Some(PyMove::from_move(&mv)), false));
        } else {
            results.push((0, None, true));
        }
    }

    Ok(results)
}
