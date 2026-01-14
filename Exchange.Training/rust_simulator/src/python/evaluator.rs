//! Python-exposed ONNX evaluator.
//!
//! This module contains the PyOnnxEvaluator class that wraps OnnxEvaluator
//! for Python interop.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use crate::eval::OnnxEvaluator;

/// Python-exposed ONNX evaluator.
#[pyclass]
pub struct PyOnnxEvaluator {
    pub evaluator: OnnxEvaluator,
}

#[pymethods]
impl PyOnnxEvaluator {
    #[new]
    #[pyo3(signature = (model_path, use_coreml = true))]
    fn new(model_path: &str, use_coreml: bool) -> PyResult<Self> {
        let evaluator = OnnxEvaluator::new(model_path, use_coreml)
            .map_err(|e| PyValueError::new_err(format!("Failed to load ONNX model: {}", e)))?;
        Ok(Self { evaluator })
    }

    /// Check if the model has a policy head.
    fn has_policy_head(&self) -> bool {
        self.evaluator.has_policy_head()
    }
}
