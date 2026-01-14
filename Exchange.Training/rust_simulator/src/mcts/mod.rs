//! Monte Carlo Tree Search module.
//!
//! This module contains all MCTS-related functionality:
//! - Configuration parameters
//! - Tree node representation
//! - Tree operations (expand, select, backup)
//! - Search implementations (batch, native)

pub mod config;
pub mod node;
pub mod search;
pub mod tree;

pub use config::MCTSConfig;
pub use node::MctsNode;
pub use search::{add_dirichlet_noise, create_trees, run_mcts_native_impl, select_move};
pub use tree::{encode_move_rust, MctsTree};
