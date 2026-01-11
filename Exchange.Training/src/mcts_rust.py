"""
Monte Carlo Tree Search using Rust Simulator

High-performance MCTS implementation that uses the Rust game simulator
for tree exploration, dramatically reducing CPU overhead while keeping
neural network evaluation in Python/PyTorch.

Expected speedup: 50-100x compared to pure Python MCTS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .game_state import Team
from .game_simulator import Move, MoveType
from .mcts import MCTSConfig
from .value_network import encode_move, POLICY_SIZE

# Import Rust simulator
try:
    from exchange_simulator import RustSimulator, RustMove, RustPiece
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    RustSimulator = None


def check_rust_available():
    """Check if Rust simulator is available."""
    if not RUST_AVAILABLE:
        raise RuntimeError(
            "Rust simulator not available. Run:\n"
            "  cd rust_simulator && ./setup.sh"
        )


@dataclass
class RustMCTSNode:
    """
    MCTS tree node using Rust simulator.

    Stores a cloned Rust simulator instead of Python GameState.
    """
    sim: "RustSimulator"  # Cloned Rust simulator at this state
    parent: Optional[RustMCTSNode] = None
    move_idx: int = -1  # Index of move that led here (in parent's move list)
    prior: float = 1.0

    visit_count: int = 0
    value_sum: float = 0.0

    children: Dict[int, RustMCTSNode] = field(default_factory=dict)
    num_legal_moves: int = 0
    is_expanded: bool = False
    is_terminal: bool = False

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def select_child(self, c_puct: float) -> Tuple[int, 'RustMCTSNode']:
        """Select best child using PUCT formula."""
        best_score = float('-inf')
        best_idx = -1
        best_child = None

        sqrt_parent_visits = math.sqrt(self.visit_count)

        for move_idx, child in self.children.items():
            q_value = -child.value if child.visit_count > 0 else 0.0
            exploration = c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count)
            score = q_value + exploration

            if score > best_score:
                best_score = score
                best_idx = move_idx
                best_child = child

        return best_idx, best_child

    def expand(self, priors: Optional[np.ndarray] = None):
        """Expand node by creating children for all legal moves."""
        if self.is_expanded:
            return

        if self.sim.is_terminal:
            self.is_terminal = True
            self.is_expanded = True
            return

        # Get legal moves from Rust
        rust_moves = self.sim.generate_moves()
        self.num_legal_moves = len(rust_moves)

        if self.num_legal_moves == 0:
            self.is_terminal = True
            self.is_expanded = True
            return

        # Create child for each legal move
        for i, rm in enumerate(rust_moves):
            # Clone simulator and make move
            child_sim = self.sim.clone_sim()
            child_sim.make_move(rm)

            # Get prior (uniform if not provided)
            prior = 1.0 / self.num_legal_moves
            if priors is not None:
                # Map move to policy index
                move_policy_idx = rust_move_to_policy_index(rm)
                if 0 <= move_policy_idx < len(priors):
                    prior = priors[move_policy_idx]

            self.children[i] = RustMCTSNode(
                sim=child_sim,
                parent=self,
                move_idx=i,
                prior=prior,
            )

        self.is_expanded = True

    def backup(self, value: float):
        """Propagate value up the tree."""
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip for opponent
            node = node.parent


def rust_move_to_policy_index(rm: "RustMove") -> int:
    """Convert RustMove to policy index for network output."""
    # Encode as from_square * 64 + to_square for basic moves
    from_idx = rm.from_y * 8 + rm.from_x
    to_idx = rm.to_y * 8 + rm.to_x

    base_idx = from_idx * 64 + to_idx

    # Knight attacks get separate indices
    if rm.move_type == 2 and rm.attack_x is not None:  # MoveAndAttack
        attack_idx = rm.attack_y * 8 + rm.attack_x
        # Use extended policy space for knight attacks
        return 4096 + from_idx * 8 + (attack_idx % 8)

    return base_idx


def rust_state_to_tensor(sim: "RustSimulator") -> np.ndarray:
    """Convert Rust simulator state to tensor for network evaluation.

    Uses Rust's native to_tensor() for maximum performance.
    """
    # Call Rust's to_tensor directly - returns numpy array (58, 8, 8)
    return np.asarray(sim.to_tensor())


class RustMCTS:
    """
    MCTS using Rust simulator for tree exploration.

    Neural network evaluation happens in Python, but all game
    simulation (cloning, move gen, move execution) uses Rust.
    """

    def __init__(self, network: nn.Module, device: torch.device, config: MCTSConfig):
        check_rust_available()
        self.network = network
        self.device = device
        self.config = config

    def search(self, sim: "RustSimulator", num_simulations: int, add_noise: bool = False) -> Tuple[Optional["RustMove"], np.ndarray]:
        """
        Run MCTS search from current position.

        Args:
            sim: Rust simulator at current position
            num_simulations: Number of MCTS simulations to run
            add_noise: Whether to add Dirichlet noise at root

        Returns:
            (best_move, visit_distribution) or (None, []) if terminal
        """
        if sim.is_terminal:
            return None, np.array([])

        # Create root node
        root = RustMCTSNode(sim=sim.clone_sim())

        # Initial expansion with policy from network
        with torch.no_grad():
            tensor = rust_state_to_tensor(root.sim)
            x = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0).to(self.device)

            if hasattr(self.network, 'forward_policy_value'):
                value, log_policy = self.network.forward_policy_value(x)
                policy = torch.exp(log_policy).cpu().numpy()[0]
            else:
                value = self.network(x)
                policy = None

        root.expand(policy)

        # Add Dirichlet noise at root for exploration
        if add_noise and root.num_legal_moves > 0:
            noise = np.random.dirichlet([self.config.dirichlet_alpha] * root.num_legal_moves)
            for i, child in root.children.items():
                child.prior = (1 - self.config.dirichlet_epsilon) * child.prior + \
                             self.config.dirichlet_epsilon * noise[i]

        # Run simulations
        for _ in range(num_simulations):
            node = root

            # Selection: walk down tree using PUCT
            while node.is_expanded and not node.is_terminal:
                _, node = node.select_child(self.config.c_puct)

            # Expansion & Evaluation
            if not node.is_terminal:
                # Get value from network
                with torch.no_grad():
                    tensor = rust_state_to_tensor(node.sim)
                    x = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0).to(self.device)

                    if hasattr(self.network, 'forward_policy_value'):
                        value, log_policy = self.network.forward_policy_value(x)
                        policy = torch.exp(log_policy).cpu().numpy()[0]
                    else:
                        value = self.network(x)
                        policy = None

                leaf_value = value.item()

                # Expand with policy priors
                node.expand(policy)
            else:
                # Terminal node - use game result
                if node.sim.winner is not None:
                    # +1 for current player winning, -1 for losing
                    winner = node.sim.winner
                    current_player = node.sim.side_to_move
                    leaf_value = 1.0 if winner == current_player else -1.0
                else:
                    leaf_value = 0.0  # Draw

            # Backup
            node.backup(leaf_value)

        # Select best move based on visit counts
        if root.num_legal_moves == 0:
            return None, np.array([])

        visits = np.array([root.children[i].visit_count for i in range(root.num_legal_moves)])
        visit_sum = visits.sum()
        if visit_sum > 0:
            visit_dist = visits / visit_sum
        else:
            visit_dist = np.ones(root.num_legal_moves) / root.num_legal_moves

        # Select move (use temperature)
        if self.config.temperature > 0:
            probs = np.power(visits, 1.0 / self.config.temperature)
            probs = probs / probs.sum()
            best_idx = np.random.choice(len(visits), p=probs)
        else:
            best_idx = np.argmax(visits)

        # Get the actual move from Rust
        rust_moves = sim.generate_moves()
        best_move = rust_moves[best_idx] if best_idx < len(rust_moves) else None

        return best_move, visit_dist


class BatchedRustMCTS:
    """
    Batched MCTS using Rust simulators.

    Runs MCTS for multiple games in parallel, batching neural network
    evaluations for GPU efficiency.
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
    ) -> List[Tuple[Optional["RustMove"], np.ndarray]]:
        """
        Run MCTS search for multiple positions in parallel.

        Args:
            simulators: List of Rust simulators at current positions
            num_simulations: Simulations per position
            add_noise: Add Dirichlet noise at roots

        Returns:
            List of (best_move, visit_distribution) tuples
        """
        n_games = len(simulators)

        # Create root nodes
        roots: List[RustMCTSNode] = []
        active_indices: List[int] = []

        for i, sim in enumerate(simulators):
            if sim.is_terminal:
                roots.append(None)
            else:
                root = RustMCTSNode(sim=sim.clone_sim())
                roots.append(root)
                active_indices.append(i)

        if not active_indices:
            return [(None, np.array([])) for _ in range(n_games)]

        # Initial batch expansion
        active_roots = [roots[i] for i in active_indices]
        tensors = [rust_state_to_tensor(r.sim) for r in active_roots]
        batch_tensor = torch.tensor(np.stack(tensors), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            if hasattr(self.network, 'forward_policy_value'):
                values, log_policies = self.network.forward_policy_value(batch_tensor)
                policies = torch.exp(log_policies).cpu().numpy()
            else:
                values = self.network(batch_tensor)
                policies = [None] * len(active_roots)

        for j, root in enumerate(active_roots):
            policy = policies[j] if policies[j] is not None else None
            root.expand(policy)

            # Add noise at root
            if add_noise and root.num_legal_moves > 0:
                noise = np.random.dirichlet([self.config.dirichlet_alpha] * root.num_legal_moves)
                for i, child in root.children.items():
                    child.prior = (1 - self.config.dirichlet_epsilon) * child.prior + \
                                 self.config.dirichlet_epsilon * noise[i]

        # Run simulations
        for _ in range(num_simulations):
            # Selection phase: find leaf for each active game
            leaves: List[RustMCTSNode] = []
            leaf_indices: List[int] = []

            for i in active_indices:
                root = roots[i]
                if root is None or root.is_terminal:
                    continue

                node = root
                while node.is_expanded and not node.is_terminal:
                    _, node = node.select_child(self.config.c_puct)

                if not node.is_terminal:
                    leaves.append(node)
                    leaf_indices.append(i)
                else:
                    # Terminal backup
                    if node.sim.winner is not None:
                        winner = node.sim.winner
                        current = node.sim.side_to_move
                        leaf_value = 1.0 if winner == current else -1.0
                    else:
                        leaf_value = 0.0
                    node.backup(leaf_value)

            if not leaves:
                continue

            # Batch evaluate leaves
            tensors = [rust_state_to_tensor(leaf.sim) for leaf in leaves]
            batch_tensor = torch.tensor(np.stack(tensors), dtype=torch.float32).to(self.device)

            with torch.no_grad():
                if hasattr(self.network, 'forward_policy_value'):
                    values, log_policies = self.network.forward_policy_value(batch_tensor)
                    policies = torch.exp(log_policies).cpu().numpy()
                else:
                    values = self.network(batch_tensor)
                    policies = [None] * len(leaves)

            # Expand and backup
            for j, leaf in enumerate(leaves):
                policy = policies[j] if policies[j] is not None else None
                leaf.expand(policy)
                leaf.backup(values[j].item())

        # Extract results
        results: List[Tuple[Optional["RustMove"], np.ndarray]] = []

        for i, sim in enumerate(simulators):
            root = roots[i]
            if root is None or root.num_legal_moves == 0:
                results.append((None, np.array([])))
                continue

            visits = np.array([root.children[j].visit_count for j in range(root.num_legal_moves)])
            visit_sum = visits.sum()
            visit_dist = visits / visit_sum if visit_sum > 0 else np.ones(root.num_legal_moves) / root.num_legal_moves

            if self.config.temperature > 0:
                probs = np.power(visits, 1.0 / self.config.temperature)
                probs = probs / probs.sum()
                best_idx = np.random.choice(len(visits), p=probs)
            else:
                best_idx = np.argmax(visits)

            rust_moves = sim.generate_moves()
            best_move = rust_moves[best_idx] if best_idx < len(rust_moves) else None
            results.append((best_move, visit_dist))

        return results
