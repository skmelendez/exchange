//! Exchange Simulator - A high-performance game simulator for EXCHANGE.
//!
//! This crate provides:
//! - A fast game state representation with Zobrist hashing
//! - Move generation for all piece types and abilities
//! - Monte Carlo Tree Search (MCTS) with virtual loss for parallel search
//! - Native ONNX inference for neural network evaluation
//! - Python bindings via PyO3

// Module declarations
pub mod core;
pub mod eval;
pub mod game;
pub mod mcts;
pub mod python;

// PyO3 module registration
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

/// Python module for the Exchange game simulator.
#[pymodule]
fn exchange_simulator(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Classes
    m.add_class::<python::PySimulator>()?;
    m.add_class::<python::PyMove>()?;
    m.add_class::<python::PyPiece>()?;
    m.add_class::<python::PyOnnxEvaluator>()?;

    // Functions
    m.add_function(wrap_pyfunction!(python::run_mcts_batch, m)?)?;
    m.add_function(wrap_pyfunction!(python::run_1ply_batch_optimized, m)?)?;
    m.add_function(wrap_pyfunction!(python::run_mcts_native, m)?)?;
    m.add_function(wrap_pyfunction!(python::run_1ply_native, m)?)?;

    Ok(())
}
