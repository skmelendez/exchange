//! MCTS tree operations.
//!
//! This module contains the MctsTree struct which manages the arena of nodes
//! and provides tree operations like expansion, selection, and backup.

use std::sync::atomic::Ordering;

use crate::core::Move;
use crate::game::{GameSimulator, GameState};
use super::node::MctsNode;

/// Encode a move to a policy index for the neural network.
/// Policy encoding: from_square * 64 + to_square (for basic moves)
#[inline]
pub fn encode_move_rust(mv: &Move) -> usize {
    use crate::core::MoveType;

    let from_idx = (mv.from_y as usize) * 8 + (mv.from_x as usize);
    let to_idx = (mv.to_y as usize) * 8 + (mv.to_x as usize);
    let base = from_idx * 64 + to_idx;

    if mv.move_type == MoveType::MoveAndAttack && mv.attack_x >= 0 {
        let attack_idx = (mv.attack_y as usize) * 8 + (mv.attack_x as usize);
        return 4096 + from_idx * 8 + (attack_idx % 8);
    }
    base
}

/// Arena-based MCTS tree.
pub struct MctsTree {
    /// All nodes in the tree (arena allocation)
    pub nodes: Vec<MctsNode>,
}

impl MctsTree {
    /// Create a new tree with the given root state.
    pub fn new(root_state: GameState) -> Self {
        MctsTree {
            nodes: vec![MctsNode::new(root_state, 1.0)],
        }
    }

    /// Expand a node using the given policy probabilities.
    pub fn expand(&mut self, node_idx: usize, policies: &[f32]) {
        let node = &mut self.nodes[node_idx];
        if node.is_expanded { return; }

        let team = node.state.side_to_move;

        // Use simulator to generate moves for the state
        let temp_sim = GameSimulator::with_state(node.state.clone());
        let moves = temp_sim.generate_moves(team);

        node.legal_moves = moves.clone();
        node.is_expanded = true;

        if node.legal_moves.is_empty() {
            node.is_terminal = true;
            return;
        }

        // Collect children data first to avoid borrow issues
        let mut children_data = Vec::with_capacity(node.legal_moves.len());

        for mv in &node.legal_moves {
            let policy_idx = encode_move_rust(mv);
            let prior = if policy_idx < policies.len() { policies[policy_idx] } else { 0.0 };

            // Create child state
            let mut child_sim = GameSimulator::with_state(node.state.clone());
            child_sim.make_move(mv);
            children_data.push((*mv, child_sim.state, prior));
        }

        // Now create nodes
        for (mv, state, prior) in children_data {
            let child_idx = self.nodes.len();
            self.nodes.push(MctsNode::new(state, prior));
            self.nodes[node_idx].children.push((mv, child_idx));
        }
    }

    /// Select leaf node using PUCT with virtual loss.
    /// Returns (leaf_idx, path) where path includes all nodes from root to leaf.
    /// Applies virtual loss to each node on the path to discourage other threads
    /// from selecting the same path during parallel search.
    pub fn select_leaf_with_virtual_loss(&self, node_idx: usize, c_puct: f32) -> (usize, Vec<usize>) {
        let mut current = node_idx;
        let mut path = vec![current];

        loop {
            let node = &self.nodes[current];
            if !node.is_expanded || node.is_terminal || node.children.is_empty() {
                return (current, path);
            }

            // Include virtual loss in parent visit count for exploration term
            let parent_vl = node.virtual_loss.load(Ordering::Relaxed);
            let effective_parent_visits = node.visits + parent_vl;
            let sqrt_visits = (effective_parent_visits as f32).sqrt();

            let mut best_score = f32::NEG_INFINITY;
            let mut best_child_idx = 0;

            for &(_, child_idx) in &node.children {
                let child = &self.nodes[child_idx];
                let child_vl = child.virtual_loss.load(Ordering::Relaxed);
                let effective_visits = child.visits + child_vl;

                // Q-value with virtual loss penalty (treat pending evals as losses)
                let q = if effective_visits > 0 {
                    // value_sum - virtual_loss gives pessimistic estimate
                    let effective_value = child.value_sum - child_vl as f32;
                    -(effective_value / effective_visits as f32)
                } else {
                    0.0
                };

                // Exploration term uses effective visits
                let u = c_puct * child.prior * sqrt_visits / (1.0 + effective_visits as f32);
                let score = q + u;

                if score > best_score {
                    best_score = score;
                    best_child_idx = child_idx;
                }
            }

            // Apply virtual loss to selected child (makes it less attractive to other threads)
            self.nodes[best_child_idx].virtual_loss.fetch_add(1, Ordering::Relaxed);

            current = best_child_idx;
            path.push(current);
        }
    }

    /// Legacy select_leaf without virtual loss (for compatibility)
    #[allow(dead_code)]
    pub fn select_leaf(&self, node_idx: usize, c_puct: f32) -> usize {
        self.select_leaf_with_virtual_loss(node_idx, c_puct).0
    }

    /// Backup value through the tree and remove virtual loss.
    /// The path should include all nodes from root to leaf.
    pub fn backup(&mut self, node_path: Vec<usize>, mut value: f32) {
        for idx in node_path.into_iter().rev() {
            let node = &mut self.nodes[idx];
            // Remove virtual loss that was applied during selection
            // (saturating_sub to handle edge cases)
            let prev_vl = node.virtual_loss.load(Ordering::Relaxed);
            if prev_vl > 0 {
                node.virtual_loss.fetch_sub(1, Ordering::Relaxed);
            }
            // Update real statistics
            node.visits += 1;
            node.value_sum += value;
            value = -value;
        }
    }

    /// Remove virtual loss from path without updating visits/values.
    /// Used when a selection results in terminal node or other special cases.
    #[allow(dead_code)]
    pub fn clear_virtual_loss(&self, node_path: &[usize]) {
        for &idx in node_path {
            let node = &self.nodes[idx];
            let prev_vl = node.virtual_loss.load(Ordering::Relaxed);
            if prev_vl > 0 {
                node.virtual_loss.fetch_sub(1, Ordering::Relaxed);
            }
        }
    }

    /// Get the root node.
    #[inline]
    pub fn root(&self) -> &MctsNode {
        &self.nodes[0]
    }

    /// Get the root node mutably.
    #[inline]
    pub fn root_mut(&mut self) -> &mut MctsNode {
        &mut self.nodes[0]
    }
}
