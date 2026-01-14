//! ONNX Runtime inference for neural network evaluation.
//!
//! This module contains the OnnxEvaluator struct for native ONNX inference
//! without Python in the hot path.

use std::sync::Mutex;

use half::f16;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::execution_providers::{CoreMLExecutionProvider, CPUExecutionProvider};
use ort::value::Tensor;

use crate::game::INPUT_CHANNELS;
use crate::core::BOARD_SIZE;

/// Thread-safe ONNX session wrapper for neural network inference.
pub struct OnnxEvaluator {
    session: Mutex<Session>,
    has_policy_head: bool,
}

// Manual Sync impl since Mutex<Session> is Sync but Session might not be Send
unsafe impl Sync for OnnxEvaluator {}

impl OnnxEvaluator {
    /// Create a new ONNX evaluator from a model file.
    ///
    /// # Arguments
    /// * `model_path` - Path to the .onnx model file
    /// * `use_coreml` - Whether to use CoreML execution provider (Apple Silicon)
    pub fn new(model_path: &str, use_coreml: bool) -> Result<Self, ort::Error> {
        let mut builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?;

        if use_coreml {
            // Try to use CoreML on Apple Silicon
            builder = builder.with_execution_providers([
                CoreMLExecutionProvider::default().build(),
                CPUExecutionProvider::default().build(),
            ])?;
        }

        let session = builder.commit_from_file(model_path)?;

        // Check if model has policy head (2 outputs) or just value (1 output)
        let has_policy_head = session.outputs().len() > 1;

        Ok(Self {
            session: Mutex::new(session),
            has_policy_head,
        })
    }

    /// Evaluate a batch of positions.
    ///
    /// # Arguments
    /// * `tensors` - Flat f16 tensor data [batch, 62, 8, 8]
    /// * `batch_size` - Number of positions in the batch
    ///
    /// # Returns
    /// Tuple of (values, policies) where policies may be empty if model has no policy head.
    /// Outputs are f32 for MCTS calculations.
    pub fn evaluate(&self, tensors: &[f16], batch_size: usize) -> Result<(Vec<f32>, Vec<Vec<f32>>), ort::Error> {
        let input_shape = [batch_size, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE];

        // Convert f16 -> f32 for ONNX inference
        let tensor_vec: Vec<f32> = tensors.iter().map(|&v| v.to_f32()).collect();
        let input_tensor = Tensor::from_array((input_shape, tensor_vec))?;

        // Lock the session for inference (lock released when scope ends)
        let mut session = self.session.lock().unwrap();
        let outputs = session.run(ort::inputs!["input" => input_tensor])?;

        // Extract values (always first output)
        let (_, values_data) = outputs[0].try_extract_tensor::<f32>()?;
        let values: Vec<f32> = values_data.to_vec();

        // Extract policies if present
        let policies = if self.has_policy_head && outputs.len() > 1 {
            let (_, policy_data) = outputs[1].try_extract_tensor::<f32>()?;
            let policy_size = policy_data.len() / batch_size;

            (0..batch_size)
                .map(|i| {
                    let start = i * policy_size;
                    policy_data[start..start + policy_size].to_vec()
                })
                .collect()
        } else {
            // Generate uniform policies if no policy head
            vec![vec![1.0 / 4608.0; 4608]; batch_size]
        };

        Ok((values, policies))
    }

    /// Check if the model has a policy head.
    pub fn has_policy_head(&self) -> bool {
        self.has_policy_head
    }
}
