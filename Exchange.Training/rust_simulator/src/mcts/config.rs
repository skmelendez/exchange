//! MCTS configuration parameters.
//!
//! This module contains the MCTSConfig struct that holds all hyperparameters
//! for Monte Carlo Tree Search.

/// MCTS hyperparameters - adjustable for training experiments.
#[derive(Clone, Debug)]
pub struct MCTSConfig {
    /// Exploration constant (higher = more exploration)
    pub c_puct: f32,
    /// Temperature for move selection (0 = deterministic)
    pub temperature: f32,
    /// Dirichlet noise alpha (lower = spikier)
    pub dirichlet_alpha: f32,
    /// Noise weight at root (0-1)
    pub dirichlet_epsilon: f32,
    /// Number of simulations per move
    pub num_simulations: usize,
}

impl Default for MCTSConfig {
    fn default() -> Self {
        Self {
            c_puct: 3.0,
            temperature: 1.0,
            dirichlet_alpha: 0.15,
            dirichlet_epsilon: 0.5,
            num_simulations: 100,
        }
    }
}

impl MCTSConfig {
    /// Create a new config with custom values.
    pub fn new(
        c_puct: f32,
        temperature: f32,
        dirichlet_alpha: f32,
        dirichlet_epsilon: f32,
        num_simulations: usize,
    ) -> Self {
        Self {
            c_puct,
            temperature,
            dirichlet_alpha,
            dirichlet_epsilon,
            num_simulations,
        }
    }

    /// Create a config for evaluation (deterministic, no noise).
    pub fn evaluation() -> Self {
        Self {
            c_puct: 3.0,
            temperature: 0.0,
            dirichlet_alpha: 0.0,
            dirichlet_epsilon: 0.0,
            num_simulations: 100,
        }
    }

    /// Create a config for training (with exploration).
    pub fn training(num_simulations: usize) -> Self {
        Self {
            c_puct: 3.0,
            temperature: 1.0,
            dirichlet_alpha: 0.15,
            dirichlet_epsilon: 0.5,
            num_simulations,
        }
    }
}
