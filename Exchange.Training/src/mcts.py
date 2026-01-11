"""
Monte Carlo Tree Search for EXCHANGE

Implements MCTS with neural network guidance, similar to AlphaZero.
Uses the PUCT formula for node selection and supports batched evaluation.

Key components:
- MCTSNode: Tree node storing visit counts, values, and children
- MCTSConfig: Configuration for search parameters
- MCTS: Main search class with batched evaluation support
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .game_state import GameState, Team
from .game_simulator import GameSimulator, Move


@dataclass
class MCTSConfig:
    """Configuration for MCTS search."""

    # Search parameters
    num_simulations: int = 100  # Simulations per move during training
    eval_simulations: int = 200  # Simulations per move during evaluation

    # PUCT formula parameters
    c_puct: float = 1.5  # Exploration constant

    # Root exploration noise (Dirichlet)
    dirichlet_alpha: float = 0.3  # Concentration parameter
    dirichlet_epsilon: float = 0.25  # Noise weight at root

    # Move selection
    temperature: float = 1.0  # Temperature for move selection
    temperature_threshold: int = 30  # Moves before going deterministic


@dataclass
class MCTSNode:
    """
    A node in the MCTS tree.

    Stores state information, visit statistics, and links to parent/children.
    """
    state: GameState
    parent: Optional[MCTSNode] = None
    move: Optional[Move] = None  # Move that led to this node
    prior: float = 1.0  # Policy prior probability

    # Statistics
    visit_count: int = 0
    value_sum: float = 0.0

    # Children (lazily expanded)
    children: Dict[int, MCTSNode] = field(default_factory=dict)  # move_index -> node
    legal_moves: Optional[List[Move]] = None  # Cached legal moves
    is_expanded: bool = False
    is_terminal: bool = False

    @property
    def value(self) -> float:
        """Average value from this node's perspective."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def select_child(self, c_puct: float) -> Tuple[int, 'MCTSNode']:
        """
        Select best child using PUCT formula.

        PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))

        Returns:
            Tuple of (move_index, child_node)
        """
        best_score = float('-inf')
        best_idx = -1
        best_child = None

        sqrt_parent_visits = math.sqrt(self.visit_count)

        for move_idx, child in self.children.items():
            # Q-value (exploitation) - from child's perspective, negate for parent
            q_value = -child.value if child.visit_count > 0 else 0.0

            # Exploration bonus
            exploration = c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count)

            score = q_value + exploration

            if score > best_score:
                best_score = score
                best_idx = move_idx
                best_child = child

        return best_idx, best_child

    def expand(self, simulator: GameSimulator, policy_priors: Optional[np.ndarray] = None) -> None:
        """
        Expand this node by creating children for all legal moves.

        Args:
            simulator: Game simulator with state matching this node
            policy_priors: Optional policy probabilities for each move index
        """
        if self.is_expanded:
            return

        team = self.state.side_to_move
        self.legal_moves = simulator.generate_moves(team)

        if not self.legal_moves:
            self.is_terminal = True
            self.is_expanded = True
            return

        # Import here to avoid circular dependency
        from .value_network import encode_move

        # Create children for each legal move
        for move in self.legal_moves:
            move_idx = encode_move(move)

            # Get prior from policy (or uniform if not provided)
            if policy_priors is not None and move_idx < len(policy_priors):
                prior = float(policy_priors[move_idx])
            else:
                prior = 1.0 / len(self.legal_moves)

            # Create child node (state will be set during evaluation)
            child = MCTSNode(
                state=None,  # Lazily created when visited
                parent=self,
                move=move,
                prior=prior,
            )
            self.children[move_idx] = child

        self.is_expanded = True

    def add_exploration_noise(self, alpha: float, epsilon: float) -> None:
        """Add Dirichlet noise to root node priors for exploration."""
        if not self.children:
            return

        noise = np.random.dirichlet([alpha] * len(self.children))
        for i, child in enumerate(self.children.values()):
            child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]

    def backup(self, value: float) -> None:
        """
        Backpropagate value up the tree.

        Value is from the perspective of the player who just moved TO this node.
        We negate at each level since players alternate.
        """
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip for opponent
            node = node.parent


class MCTS:
    """
    Monte Carlo Tree Search with neural network guidance.

    Supports both single-game and batched multi-game search.
    """

    def __init__(
        self,
        network: nn.Module,
        device: torch.device,
        config: MCTSConfig,
    ):
        self.network = network
        self.device = device
        self.config = config

    def search(
        self,
        state: GameState,
        num_simulations: Optional[int] = None,
        add_noise: bool = True,
    ) -> Tuple[Move, np.ndarray]:
        """
        Run MCTS search from the given state.

        Args:
            state: Current game state
            num_simulations: Number of simulations (defaults to config)
            add_noise: Whether to add Dirichlet noise at root

        Returns:
            Tuple of (best_move, visit_count_distribution)
        """
        if num_simulations is None:
            num_simulations = self.config.num_simulations

        # Create root node
        root = MCTSNode(state=state.clone())
        simulator = GameSimulator(root.state)

        # Expand root and get policy priors
        with torch.no_grad():
            tensor = torch.tensor(
                root.state.to_tensor(),
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)

            # Check if network has policy head
            if hasattr(self.network, 'forward_policy_value'):
                value, log_policy = self.network.forward_policy_value(tensor)
                policy_priors = torch.exp(log_policy).cpu().numpy().flatten()
            else:
                value = self.network(tensor)
                policy_priors = None

        root.expand(simulator, policy_priors)

        if root.is_terminal:
            # No legal moves
            return None, np.array([])

        # Add exploration noise at root
        if add_noise and self.config.dirichlet_epsilon > 0:
            root.add_exploration_noise(
                self.config.dirichlet_alpha,
                self.config.dirichlet_epsilon
            )

        # Run simulations
        for _ in range(num_simulations):
            self._simulate(root)

        # Select best move based on visit counts
        best_move, visit_distribution = self._select_move(root)
        return best_move, visit_distribution

    def _simulate(self, root: MCTSNode) -> None:
        """Run one simulation from root."""
        node = root
        simulator = GameSimulator(root.state.clone())
        path = [node]

        # Selection: traverse tree using PUCT
        while node.is_expanded and not node.is_terminal:
            move_idx, child = node.select_child(self.config.c_puct)
            if child is None:
                break

            # Apply move to simulator
            move = child.move
            # Need to get piece from current simulator state
            piece = simulator.get_piece_at(move.piece.x, move.piece.y)
            if piece:
                new_move = Move(
                    piece=piece,
                    move_type=move.move_type,
                    from_pos=move.from_pos,
                    to_pos=move.to_pos,
                    attack_pos=move.attack_pos,
                    ability_id=move.ability_id,
                    ability_target=move.ability_target,
                )
                simulator.make_move(new_move)

            # Set child state if not yet set
            if child.state is None:
                child.state = simulator.state.clone()

            node = child
            path.append(node)

        # Check for terminal state
        if simulator.state.is_terminal:
            # Terminal value
            if simulator.state.winner == Team.PLAYER:
                value = 1.0 if node.state.side_to_move == Team.PLAYER else -1.0
            elif simulator.state.winner == Team.ENEMY:
                value = 1.0 if node.state.side_to_move == Team.ENEMY else -1.0
            else:
                value = 0.0
        else:
            # Expansion and evaluation
            if not node.is_expanded:
                # Get policy and value from network
                with torch.no_grad():
                    tensor = torch.tensor(
                        simulator.state.to_tensor(),
                        dtype=torch.float32,
                        device=self.device
                    ).unsqueeze(0)

                    if hasattr(self.network, 'forward_policy_value'):
                        value_tensor, log_policy = self.network.forward_policy_value(tensor)
                        policy_priors = torch.exp(log_policy).cpu().numpy().flatten()
                        value = value_tensor.item()
                    else:
                        value = self.network(tensor).item()
                        policy_priors = None

                node.state = simulator.state.clone()
                node.expand(GameSimulator(node.state), policy_priors)
            else:
                # Shouldn't happen, but handle gracefully
                with torch.no_grad():
                    tensor = torch.tensor(
                        simulator.state.to_tensor(),
                        dtype=torch.float32,
                        device=self.device
                    ).unsqueeze(0)
                    value = self.network(tensor).item()

        # Backup
        node.backup(value)

    def _select_move(
        self,
        root: MCTSNode,
        temperature: Optional[float] = None,
    ) -> Tuple[Move, np.ndarray]:
        """
        Select move from root based on visit counts.

        Args:
            root: Root node after search
            temperature: Selection temperature (0 = deterministic)

        Returns:
            Tuple of (selected_move, visit_distribution)
        """
        if temperature is None:
            temperature = self.config.temperature

        moves = list(root.children.values())
        visits = np.array([child.visit_count for child in moves])

        if len(visits) == 0:
            return None, np.array([])

        # Create visit distribution over all moves
        visit_distribution = np.zeros(len(root.legal_moves))
        for i, move in enumerate(root.legal_moves):
            from .value_network import encode_move
            move_idx = encode_move(move)
            if move_idx in root.children:
                visit_distribution[i] = root.children[move_idx].visit_count

        # Normalize
        visit_sum = visit_distribution.sum()
        if visit_sum > 0:
            visit_distribution = visit_distribution / visit_sum

        # Select move
        if temperature == 0:
            # Deterministic: pick most visited
            best_idx = np.argmax(visits)
        else:
            # Stochastic: sample proportional to visits^(1/temp)
            visits_temp = visits ** (1.0 / temperature)
            probs = visits_temp / visits_temp.sum()
            best_idx = np.random.choice(len(moves), p=probs)

        selected_move = moves[best_idx].move
        return selected_move, visit_distribution

    def get_move_priors(self, root: MCTSNode) -> Dict[int, float]:
        """Get policy priors for all legal moves from root."""
        return {
            move_idx: child.prior
            for move_idx, child in root.children.items()
        }


class BatchedMCTS:
    """
    MCTS for multiple games in parallel with batched neural network evaluation.

    Optimized for GPU efficiency by batching leaf evaluations across all games.
    """

    def __init__(
        self,
        network: nn.Module,
        device: torch.device,
        config: MCTSConfig,
    ):
        self.network = network
        self.device = device
        self.config = config

    def search_batch(
        self,
        states: List[GameState],
        num_simulations: Optional[int] = None,
        add_noise: bool = True,
    ) -> List[Tuple[Move, np.ndarray]]:
        """
        Run MCTS search for multiple games in parallel.

        Args:
            states: List of game states
            num_simulations: Simulations per game
            add_noise: Whether to add Dirichlet noise

        Returns:
            List of (best_move, visit_distribution) for each game
        """
        if num_simulations is None:
            num_simulations = self.config.num_simulations

        num_games = len(states)

        # Initialize roots
        roots = [MCTSNode(state=state.clone()) for state in states]
        simulators = [GameSimulator(root.state) for root in roots]

        # Expand all roots with batched evaluation
        self._batch_expand_roots(roots, simulators, add_noise)

        # Run simulations
        for _ in range(num_simulations):
            self._batch_simulate(roots, simulators)

        # Select moves
        results = []
        for root in roots:
            if root.is_terminal or not root.children:
                results.append((None, np.array([])))
            else:
                move, dist = self._select_move(root)
                results.append((move, dist))

        return results

    def _batch_expand_roots(
        self,
        roots: List[MCTSNode],
        simulators: List[GameSimulator],
        add_noise: bool,
    ) -> None:
        """Expand all root nodes with batched network evaluation."""
        # Collect tensors for all roots
        tensors = []
        for root in roots:
            tensors.append(root.state.to_tensor())

        batch = torch.tensor(
            np.array(tensors),
            dtype=torch.float32,
            device=self.device
        )

        # Batched network evaluation
        with torch.no_grad():
            if hasattr(self.network, 'forward_policy_value'):
                values, log_policies = self.network.forward_policy_value(batch)
                policies = torch.exp(log_policies).cpu().numpy()
            else:
                values = self.network(batch)
                policies = None

        # Expand each root
        for i, (root, sim) in enumerate(zip(roots, simulators)):
            policy = policies[i] if policies is not None else None
            root.expand(sim, policy)

            if add_noise and self.config.dirichlet_epsilon > 0 and not root.is_terminal:
                root.add_exploration_noise(
                    self.config.dirichlet_alpha,
                    self.config.dirichlet_epsilon
                )

    def _batch_simulate(
        self,
        roots: List[MCTSNode],
        simulators: List[GameSimulator],
    ) -> None:
        """Run one simulation for each game, batching leaf evaluations."""
        leaves = []
        leaf_states = []
        leaf_indices = []  # Which game each leaf belongs to

        # Selection phase: find leaf for each game
        for game_idx, root in enumerate(roots):
            if root.is_terminal:
                continue

            # Reset simulator to root state
            sim = GameSimulator(root.state.clone())
            node = root

            # Selection
            while node.is_expanded and not node.is_terminal and node.children:
                move_idx, child = node.select_child(self.config.c_puct)
                if child is None:
                    break

                # Apply move
                move = child.move
                piece = sim.get_piece_at(move.piece.x, move.piece.y)
                if piece:
                    new_move = Move(
                        piece=piece,
                        move_type=move.move_type,
                        from_pos=move.from_pos,
                        to_pos=move.to_pos,
                        attack_pos=move.attack_pos,
                        ability_id=move.ability_id,
                        ability_target=move.ability_target,
                    )
                    sim.make_move(new_move)

                if child.state is None:
                    child.state = sim.state.clone()

                node = child

            # Handle terminal or unexpanded node
            if sim.state.is_terminal:
                # Terminal - immediate backup
                if sim.state.winner == Team.PLAYER:
                    value = 1.0 if node.state.side_to_move == Team.PLAYER else -1.0
                elif sim.state.winner == Team.ENEMY:
                    value = 1.0 if node.state.side_to_move == Team.ENEMY else -1.0
                else:
                    value = 0.0
                node.backup(value)
            elif not node.is_expanded:
                # Need to evaluate this leaf
                leaves.append(node)
                leaf_states.append(sim.state.clone())
                leaf_indices.append(game_idx)

        if not leaves:
            return

        # Batched evaluation of all leaves
        tensors = [state.to_tensor() for state in leaf_states]
        batch = torch.tensor(
            np.array(tensors),
            dtype=torch.float32,
            device=self.device
        )

        with torch.no_grad():
            if hasattr(self.network, 'forward_policy_value'):
                values, log_policies = self.network.forward_policy_value(batch)
                values = values.cpu().numpy().flatten()
                policies = torch.exp(log_policies).cpu().numpy()
            else:
                values = self.network(batch).cpu().numpy().flatten()
                policies = None

        # Expand leaves and backup
        for i, (node, state) in enumerate(zip(leaves, leaf_states)):
            node.state = state
            policy = policies[i] if policies is not None else None
            node.expand(GameSimulator(state), policy)
            node.backup(values[i])

    def _select_move(
        self,
        root: MCTSNode,
        temperature: Optional[float] = None,
    ) -> Tuple[Move, np.ndarray]:
        """Select move from root based on visit counts."""
        if temperature is None:
            temperature = self.config.temperature

        moves = list(root.children.values())
        visits = np.array([child.visit_count for child in moves])

        if len(visits) == 0:
            return None, np.array([])

        # Create visit distribution
        visit_distribution = np.zeros(len(root.legal_moves))
        for i, move in enumerate(root.legal_moves):
            from .value_network import encode_move
            move_idx = encode_move(move)
            if move_idx in root.children:
                visit_distribution[i] = root.children[move_idx].visit_count

        visit_sum = visit_distribution.sum()
        if visit_sum > 0:
            visit_distribution = visit_distribution / visit_sum

        # Select
        if temperature == 0:
            best_idx = np.argmax(visits)
        else:
            visits_temp = visits ** (1.0 / temperature)
            probs = visits_temp / visits_temp.sum()
            best_idx = np.random.choice(len(moves), p=probs)

        return moves[best_idx].move, visit_distribution
