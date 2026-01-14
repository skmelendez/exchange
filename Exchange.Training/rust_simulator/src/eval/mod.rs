//! Evaluation module.
//!
//! This module contains neural network evaluation functionality:
//! - ONNX Runtime inference
//! - 1-ply lookahead evaluation

pub mod one_ply;
pub mod onnx;

pub use one_ply::run_1ply_native_impl;
pub use onnx::OnnxEvaluator;
