//! MCTS node representation.
//!
//! This module contains the MctsNode struct used in the Monte Carlo Tree Search.

use std::sync::atomic::AtomicU32;

use crate::core::Move;
use crate::game::GameState;

/// A node in the MCTS tree.
pub struct MctsNode {
    /// The game state at this node
    pub state: GameState,
    /// Children: (Move, child_index in arena)
    pub children: Vec<(Move, usize)>,
    /// Number of times this node has been visited
    pub visits: u32,
    /// Sum of values backpropagated through this node
    pub value_sum: f32,
    /// Prior probability from neural network policy
    pub prior: f32,
    /// Whether this node has been expanded
    pub is_expanded: bool,
    /// Whether this is a terminal state
    pub is_terminal: bool,
    /// Legal moves from this state (cached during expansion)
    pub legal_moves: Vec<Move>,
    /// Virtual loss for parallel MCTS - prevents multiple threads from selecting same path
    pub virtual_loss: AtomicU32,
}

impl MctsNode {
    /// Create a new MCTS node with the given state and prior probability.
    pub fn new(state: GameState, prior: f32) -> Self {
        MctsNode {
            state,
            children: Vec::with_capacity(8),
            visits: 0,
            value_sum: 0.0,
            prior,
            is_expanded: false,
            is_terminal: false,
            legal_moves: Vec::new(),
            virtual_loss: AtomicU32::new(0),
        }
    }

    /// Get the mean value of this node.
    #[allow(dead_code)]
    pub fn value(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }

    /// Get the Q-value (mean value) for PUCT calculation.
    #[inline]
    pub fn q_value(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }

    /// Get the UCB exploration term.
    #[inline]
    pub fn ucb(&self, c_puct: f32, parent_visits: u32) -> f32 {
        c_puct * self.prior * (parent_visits as f32).sqrt() / (1.0 + self.visits as f32)
    }
}
