//! Python bindings module.
//!
//! This module contains all PyO3 bindings for Python interop:
//! - PySimulator (game simulator)
//! - PyMove, PyPiece (types)
//! - PyOnnxEvaluator (ONNX evaluator)
//! - pyfunction wrappers (MCTS, 1-ply)

pub mod evaluator;
pub mod functions;
pub mod simulator;
pub mod types;

pub use evaluator::PyOnnxEvaluator;
pub use functions::{run_1ply_batch_optimized, run_1ply_native, run_mcts_batch, run_mcts_native};
pub use simulator::PySimulator;
pub use types::{PyMove, PyPiece};
