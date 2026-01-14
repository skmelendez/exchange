//! MCTS search implementations.
//!
//! This module contains the batch MCTS search functions for both
//! Python-based evaluation and native ONNX evaluation.

use half::f16;
use rayon::prelude::*;

use crate::core::Move;
use crate::game::{GameState, INPUT_CHANNELS, TensorComputer};
use crate::eval::OnnxEvaluator;
use super::tree::MctsTree;

/// Run MCTS with native ONNX inference - NO PYTHON IN THE LOOP!
pub fn run_mcts_native_impl(
    states: &[GameState],
    evaluator: &OnnxEvaluator,
    num_simulations: usize,
    c_puct: f32,
    dirichlet_alpha: f32,
    dirichlet_epsilon: f32,
    temperature: f32,
) -> Result<Vec<(Option<Move>, Vec<f32>)>, ort::Error> {
    const BOARD_SIZE: usize = 8;
    let num_trees = states.len();
    let tensor_size = INPUT_CHANNELS * BOARD_SIZE * BOARD_SIZE;

    // Initialize trees
    let mut trees: Vec<MctsTree> = states.iter()
        .map(|s| MctsTree::new(s.clone_for_mcts()))
        .collect();

    // Initial expansion with batched evaluation (f16 tensors)
    {
        let mut batch_tensors: Vec<f16> = Vec::with_capacity(num_trees * tensor_size);
        let mut valid_indices = Vec::with_capacity(num_trees);

        for (i, tree) in trees.iter().enumerate() {
            if !tree.nodes[0].state.is_terminal {
                let temp = TensorComputer::new(tree.nodes[0].state.clone());
                batch_tensors.extend_from_slice(&temp.compute_tensor());
                valid_indices.push(i);
            }
        }

        if !valid_indices.is_empty() {
            let (_, policies) = evaluator.evaluate(&batch_tensors, valid_indices.len())?;

            for (idx, &tree_idx) in valid_indices.iter().enumerate() {
                trees[tree_idx].expand(0, &policies[idx]);

                // Add Dirichlet noise at root
                if dirichlet_epsilon > 0.0 && !trees[tree_idx].nodes[0].children.is_empty() {
                    let num_children = trees[tree_idx].nodes[0].children.len();
                    let mut noise: Vec<f32> = (0..num_children)
                        .map(|_| fastrand::f32().powf(1.0 / dirichlet_alpha))
                        .collect();
                    let sum: f32 = noise.iter().sum();
                    noise.iter_mut().for_each(|n| *n /= sum);

                    let children_indices: Vec<usize> = trees[tree_idx].nodes[0].children.iter()
                        .map(|&(_, idx)| idx).collect();
                    for (i, &child_idx) in children_indices.iter().enumerate() {
                        let child = &mut trees[tree_idx].nodes[child_idx];
                        child.prior = (1.0 - dirichlet_epsilon) * child.prior + dirichlet_epsilon * noise[i];
                    }
                }
            }
        }
    }

    // Simulation loop with virtual loss
    for _ in 0..num_simulations {
        // Selection with virtual loss & tensor generation (parallel, f16 tensors)
        let selection_results: Vec<(usize, Vec<usize>, Option<Vec<f16>>)> = trees.par_iter()
            .map(|tree| {
                if tree.nodes[0].state.is_terminal {
                    return (0, vec![], None);
                }

                // Select leaf with virtual loss to prevent path collisions
                let (leaf_idx, path) = tree.select_leaf_with_virtual_loss(0, c_puct);

                let node = &tree.nodes[leaf_idx];
                if !node.is_terminal && !node.is_expanded {
                    let temp = TensorComputer::new(node.state.clone());
                    (leaf_idx, path, Some(temp.compute_tensor()))
                } else {
                    (leaf_idx, path, None)
                }
            })
            .collect();

        // Gather valid tensors for batched evaluation
        let mut batch_tensors: Vec<f16> = Vec::new();
        let mut valid_indices = Vec::new();

        for (i, (_, _, tensor_opt)) in selection_results.iter().enumerate() {
            if let Some(tensor) = tensor_opt {
                batch_tensors.extend_from_slice(tensor);
                valid_indices.push(i);
            }
        }

        // Batched neural network evaluation
        if !valid_indices.is_empty() {
            let (values, policies) = evaluator.evaluate(&batch_tensors, valid_indices.len())?;

            // Expansion and backup
            for (batch_idx, &tree_idx) in valid_indices.iter().enumerate() {
                let (leaf_idx, ref path, _) = selection_results[tree_idx];
                trees[tree_idx].expand(leaf_idx, &policies[batch_idx]);
                trees[tree_idx].backup(path.clone(), values[batch_idx]);
            }
        }

        // Handle terminal nodes
        for (tree_idx, (leaf_idx, path, tensor_opt)) in selection_results.iter().enumerate() {
            if tensor_opt.is_none() && !path.is_empty() && !trees[tree_idx].nodes[0].state.is_terminal {
                let node = &trees[tree_idx].nodes[*leaf_idx];
                if node.is_terminal {
                    let val = if let Some(winner) = node.state.winner {
                        if winner == node.state.side_to_move { 1.0 } else { -1.0 }
                    } else {
                        0.0
                    };
                    trees[tree_idx].backup(path.clone(), val);
                }
            }
        }
    }

    // Extract results
    let results: Vec<(Option<Move>, Vec<f32>)> = trees.iter()
        .map(|tree| {
            if tree.nodes[0].state.is_terminal {
                return (None, Vec::new());
            }

            let root = &tree.nodes[0];
            if root.children.is_empty() {
                return (None, Vec::new());
            }

            let total_visits: u32 = root.children.iter()
                .map(|&(_, child_idx)| tree.nodes[child_idx].visits)
                .sum();

            if total_visits == 0 {
                return (None, Vec::new());
            }

            let mut move_probs = vec![0.0f32; root.children.len()];
            let best_move;

            if temperature == 0.0 {
                // Deterministic: pick move with most visits
                let mut best_visits = 0;
                let mut best_idx = 0;
                for (idx, &(_, child_idx)) in root.children.iter().enumerate() {
                    let visits = tree.nodes[child_idx].visits;
                    if visits > best_visits {
                        best_visits = visits;
                        best_idx = idx;
                    }
                }
                move_probs[best_idx] = 1.0;
                best_move = Some(root.children[best_idx].0);
            } else {
                // Temperature-based selection
                let mut sum_probs = 0.0f32;
                for (idx, &(_, child_idx)) in root.children.iter().enumerate() {
                    let visits = tree.nodes[child_idx].visits;
                    let prob = (visits as f32).powf(1.0 / temperature);
                    move_probs[idx] = prob;
                    sum_probs += prob;
                }

                // Normalize
                for p in &mut move_probs {
                    *p /= sum_probs;
                }

                // Sample
                let sample = fastrand::f32();
                let mut cum = 0.0;
                let mut selected_idx = 0;
                for (idx, &prob) in move_probs.iter().enumerate() {
                    cum += prob;
                    if cum >= sample {
                        selected_idx = idx;
                        break;
                    }
                }
                best_move = Some(root.children[selected_idx].0);
            }

            (best_move, move_probs)
        })
        .collect();

    Ok(results)
}

/// Get the trees for external MCTS batch processing (used by Python bindings).
pub fn create_trees(states: &[GameState]) -> Vec<MctsTree> {
    states.iter()
        .map(|s| MctsTree::new(s.clone_for_mcts()))
        .collect()
}

/// Add Dirichlet noise to root children.
pub fn add_dirichlet_noise(tree: &mut MctsTree, alpha: f32, epsilon: f32) {
    if epsilon <= 0.0 || tree.nodes[0].children.is_empty() {
        return;
    }

    let num_children = tree.nodes[0].children.len();
    let mut noise: Vec<f32> = (0..num_children)
        .map(|_| fastrand::f32().powf(1.0 / alpha))
        .collect();
    let sum: f32 = noise.iter().sum();
    noise.iter_mut().for_each(|n| *n /= sum);

    let children_indices: Vec<usize> = tree.nodes[0].children.iter()
        .map(|&(_, idx)| idx).collect();
    for (i, &child_idx) in children_indices.iter().enumerate() {
        let child = &mut tree.nodes[child_idx];
        child.prior = (1.0 - epsilon) * child.prior + epsilon * noise[i];
    }
}

/// Select best move from root based on visit counts and temperature.
pub fn select_move(tree: &MctsTree, temperature: f32) -> (Option<Move>, Vec<f32>) {
    let root = &tree.nodes[0];

    if root.is_terminal || root.children.is_empty() {
        return (None, Vec::new());
    }

    let total_visits: u32 = root.children.iter()
        .map(|&(_, child_idx)| tree.nodes[child_idx].visits)
        .sum();

    if total_visits == 0 {
        return (None, Vec::new());
    }

    let mut move_probs = vec![0.0f32; root.children.len()];
    let best_move;

    if temperature == 0.0 {
        // Deterministic: pick move with most visits
        let mut best_visits = 0;
        let mut best_idx = 0;
        for (idx, &(_, child_idx)) in root.children.iter().enumerate() {
            let visits = tree.nodes[child_idx].visits;
            if visits > best_visits {
                best_visits = visits;
                best_idx = idx;
            }
        }
        move_probs[best_idx] = 1.0;
        best_move = Some(root.children[best_idx].0);
    } else {
        // Temperature-based selection
        let mut sum_probs = 0.0f32;
        for (idx, &(_, child_idx)) in root.children.iter().enumerate() {
            let visits = tree.nodes[child_idx].visits;
            let prob = (visits as f32).powf(1.0 / temperature);
            move_probs[idx] = prob;
            sum_probs += prob;
        }

        // Normalize
        for p in &mut move_probs {
            *p /= sum_probs;
        }

        // Sample
        let sample = fastrand::f32();
        let mut cum = 0.0;
        let mut selected_idx = 0;
        for (idx, &prob) in move_probs.iter().enumerate() {
            cum += prob;
            if cum >= sample {
                selected_idx = idx;
                break;
            }
        }
        best_move = Some(root.children[selected_idx].0);
    }

    (best_move, move_probs)
}
