"""
Monte Carlo Tree Search using Rust Simulator

High-performance MCTS implementation that uses the Rust game simulator
for tree exploration, dramatically reducing CPU overhead while keeping
neural network evaluation in Python/PyTorch.

Expected speedup: 50-100x compared to pure Python MCTS.
"""

from __future__ import annotations

from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .mcts import MCTSConfig
from .value_network import POLICY_SIZE

# Import Rust simulator
try:
    from exchange_simulator import PySimulator as RustSimulator, PyMove, run_mcts_batch
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    RustSimulator = None
    run_mcts_batch = None


def check_rust_available():
    """Check if Rust simulator is available."""
    if not RUST_AVAILABLE:
        raise RuntimeError(
            "Rust simulator not available. Run:\n"
            "  cd rust_simulator && ./setup.sh"
        )


def rust_state_to_tensor(sim: "RustSimulator") -> np.ndarray:
    """Convert Rust simulator state to tensor for network evaluation.

    Uses Rust's native to_tensor() for maximum performance.
    """
    return np.asarray(sim.to_tensor())


def rust_move_to_policy_index(rust_move: "PyMove") -> int:
    """
    Convert a PyMove to a policy index.

    Policy encoding matches value_network.encode_move():
    - MOVE, ATTACK, ABILITY: from_sq * 64 + to_sq
    - MOVE_AND_ATTACK: from_sq * 64 + to_sq (destination)

    Args:
        rust_move: PyMove object with from_x, from_y, to_x, to_y

    Returns:
        Policy index in range [0, POLICY_SIZE)
    """
    from_sq = rust_move.from_x + rust_move.from_y * 8
    to_sq = rust_move.to_x + rust_move.to_y * 8
    return from_sq * 64 + to_sq


class BatchedRustMCTS:
    """
    Batched MCTS using Rust simulators with native Rust tree search.

    The search loop runs entirely in Rust, calling back to Python only
    for neural network evaluation batches.
    """

    def __init__(self, network: nn.Module, device: torch.device, config: MCTSConfig):
        check_rust_available()
        self.network = network
        self.device = device
        self.config = config

    def search_batch(
        self,
        simulators: List["RustSimulator"],
        num_simulations: int,
        add_noise: bool = False,
    ) -> List[Tuple[Optional["PyMove"], np.ndarray]]:
        """
        Run MCTS search for multiple positions in parallel using Rust.

        Args:
            simulators: List of Rust simulators at current positions
            num_simulations: Simulations per position
            add_noise: Add Dirichlet noise at roots

        Returns:
            List of (best_move, visit_distribution) tuples
        """
        
        # Create callback for Rust to invoke neural network
        def eval_callback(numpy_tensor):
            with torch.no_grad():
                # Convert numpy array (from Rust) to Torch tensor on GPU
                batch_tensor = torch.as_tensor(numpy_tensor, device=self.device)
                
                if hasattr(self.network, 'forward_policy_value'):
                    values, log_policies = self.network.forward_policy_value(batch_tensor)
                    # Convert back to CPU lists for Rust
                    # Values: [batch_size] -> list of floats
                    # Policies: [batch_size, policy_size] -> list of list of floats
                    val_list = values.squeeze(1).cpu().tolist()
                    
                    # Convert log_softmax to probs
                    pol_list = torch.exp(log_policies).cpu().tolist()
                    
                    return (val_list, pol_list)
                else:
                    # Value-only network support (legacy)
                    values = self.network(batch_tensor)
                    val_list = values.squeeze(1).cpu().tolist()
                    # Uniform policy placeholder
                    pol_list = [[1.0 / POLICY_SIZE] * POLICY_SIZE] * len(val_list)
                    return (val_list, pol_list)

        # Config parameters
        epsilon = self.config.dirichlet_epsilon if add_noise else 0.0

        # DEBUG: Print MCTS params to verify they're reaching Rust
        if len(simulators) > 0 and not getattr(self, '_debug_printed', False):
            print(f"  [MCTS DEBUG] c_puct={self.config.c_puct}, temp={self.config.temperature}, "
                  f"epsilon={epsilon}, alpha={self.config.dirichlet_alpha}")
            self._debug_printed = True

        # Call Rust implementation
        results = run_mcts_batch(
            simulators, 
            eval_callback,
            num_simulations,
            self.config.c_puct,
            self.config.dirichlet_alpha,
            epsilon,
            self.config.temperature
        )
        
        # Convert results to expected format
        # Rust returns (Option<PyMove>, Vec<f32>)
        # Python expects (Optional[PyMove], np.ndarray)
        final_results = []
        for move, probs in results:
            final_results.append((move, np.array(probs)))
            
        return final_results