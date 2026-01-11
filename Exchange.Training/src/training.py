"""
Training Loop for EXCHANGE Value Network

Implements self-play training with the following phases:
1. Initial bootstrap: Random games to seed the network
2. Self-play generation: Use current network to play games
3. Network training: Update weights from game outcomes
4. Repeat with improved network

Supports:
- Multi-process game generation for M4 Max efficiency
- TensorBoard logging for training visualization
- Checkpoint saving and resumption
- Temperature annealing for exploration
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .game_state import GameState, Team, INPUT_CHANNELS, BOARD_SIZE, CHANNEL_NAMES
from .game_simulator import GameSimulator
from .value_network import (
    ExchangeValueNetwork,
    PolicyValueNetwork,
    create_from_preset,
    create_policy_value_from_preset,
    encode_move,
    POLICY_SIZE,
)
from .mcts import MCTSConfig, MCTS, BatchedMCTS
from .mcts_rust import RustMCTS, BatchedRustMCTS, rust_state_to_tensor, RUST_AVAILABLE


@dataclass
class TrainingConfig:
    """Configuration for training run."""

    # Network architecture
    network_preset: str = "medium"  # tiny, small, medium, large

    # Training hyperparameters
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs_per_iteration: int = 10
    gradient_clip: float = 1.0

    # Self-play settings
    games_per_iteration: int = 200
    num_workers: int = 0  # 0 = auto-detect
    temperature_start: float = 1.0
    temperature_end: float = 0.3  # Keep some exploration, don't go too deterministic
    temperature_decay_iterations: int = 50

    # Bootstrap settings
    bootstrap_games: int = 1000
    bootstrap_epochs: int = 20

    # Training loop
    num_iterations: int = 100
    eval_interval: int = 5
    checkpoint_interval: int = 10

    # Game logging - save sample games to review AI behavior
    log_games: bool = True
    log_games_interval: int = 5  # Save sample games every N iterations
    log_games_count: int = 3  # Number of games to save per interval

    # Champion system - evaluate vs previous best, not random
    champion_eval_games: int = 50  # Games to play vs champion
    champion_win_threshold: float = 0.55  # Must win >55% of decisive games to become champion
    champion_min_decisive: int = 5  # Minimum decisive games required

    # MCTS settings
    use_mcts: bool = False  # Enable MCTS for move selection (Python)
    use_mcts_rust: bool = False  # Enable MCTS with Rust simulator (faster)
    mcts_simulations: int = 100  # Simulations per move during self-play
    mcts_eval_simulations: int = 200  # Simulations per move during evaluation
    mcts_c_puct: float = 1.5  # Exploration constant
    mcts_dirichlet_alpha: float = 0.3  # Root noise concentration
    mcts_dirichlet_epsilon: float = 0.25  # Root noise weight
    policy_loss_weight: float = 1.0  # Weight of policy loss vs value loss

    # Paths
    output_dir: str = "runs/experiment"
    checkpoint_dir: str = "checkpoints"
    tensorboard_dir: str = "logs"
    replays_dir: str = "replays"

    def get_temperature(self, iteration: int) -> float:
        """Get temperature for current iteration (annealed)."""
        if iteration >= self.temperature_decay_iterations:
            return self.temperature_end

        ratio = iteration / self.temperature_decay_iterations
        return self.temperature_start - ratio * (self.temperature_start - self.temperature_end)


@dataclass
class TrainingState:
    """Mutable training state for checkpointing."""
    iteration: int = 0
    best_eval_score: float = float('-inf')
    training_history: list[dict] = field(default_factory=list)


def _play_random_game(seed: int) -> list[tuple[np.ndarray, float]]:
    """Worker function for parallel random game generation."""
    sim = GameSimulator()
    sim.set_seed(seed)
    # Alternate color perspective based on seed for balanced training
    sim.state.playing_as = Team.PLAYER if seed % 2 == 0 else Team.ENEMY
    _, data = sim.play_random_game()
    return data


def _play_self_play_game(args: tuple[int, str, float]) -> tuple[list[tuple[np.ndarray, float]], dict]:
    """Worker function for parallel self-play game generation. Returns training data AND game replay."""
    seed, model_path, temperature = args

    # Load model in worker process (weights_only=False for our own trusted models)
    network = torch.load(model_path, map_location='cpu', weights_only=False)
    network.eval()

    sim = GameSimulator()
    sim.set_seed(seed)
    # Alternate color perspective based on seed for balanced training
    sim.state.playing_as = Team.PLAYER if seed % 2 == 0 else Team.ENEMY

    from .game_simulator import Move, MoveType
    from .game_state import Team

    history: list[tuple[np.ndarray, int]] = []  # (tensor, team)
    moves_record: list[dict] = []
    initial_state = sim.state.to_json()

    # Track damage for reward shaping
    player_damage_dealt = 0
    enemy_damage_dealt = 0

    while not sim.state.is_terminal:
        # Record state
        tensor = sim.state.to_tensor()
        team = sim.state.side_to_move
        history.append((tensor, int(team)))

        # Generate moves
        moves = sim.generate_moves(team)
        if not moves:
            break

        # Evaluate each move with network
        move_values: list[float] = []
        for move in moves:
            sim_copy = GameSimulator(sim.state.clone())
            piece = sim_copy.get_piece_at(move.piece.x, move.piece.y)
            if piece:
                new_move = Move(
                    piece=piece,
                    move_type=move.move_type,
                    from_pos=move.from_pos,
                    to_pos=move.to_pos,
                    attack_pos=move.attack_pos,
                )
                sim_copy.make_move(new_move)
                with torch.no_grad():
                    state_tensor = torch.tensor(sim_copy.state.to_tensor()).unsqueeze(0)
                    value = network(state_tensor).item()
                move_values.append(-value)
            else:
                move_values.append(0.0)

        # Select move with temperature
        if temperature > 0:
            values = np.array(move_values)
            exp_values = np.exp(values / temperature)
            probs = exp_values / exp_values.sum()
            move_idx = np.random.choice(len(moves), p=probs)
        else:
            move_idx = np.argmax(move_values)

        selected_move = moves[move_idx]

        # Record the move with AI evaluation
        move_record = {
            "turn": sim.state.turn_number,
            "team": int(team),
            "piece_type": int(selected_move.piece.piece_type),
            "move_type": int(selected_move.move_type),
            "from": list(selected_move.from_pos),
            "to": list(selected_move.to_pos),
            "attack": list(selected_move.attack_pos) if selected_move.attack_pos else None,
            "ability_id": int(selected_move.ability_id) if selected_move.ability_id is not None else None,
            "ability_target": list(selected_move.ability_target) if selected_move.ability_target else None,
            "value": float(move_values[move_idx]),
            "top_moves": len(moves),  # How many moves were considered
        }

        # Execute move and track damage
        damage = sim.make_move(selected_move)
        move_record["damage"] = damage
        moves_record.append(move_record)

        # Track damage by team
        if team == Team.PLAYER:
            player_damage_dealt += damage
        else:
            enemy_damage_dealt += damage

    # Determine outcome with damage-weighted rewards
    if sim.state.winner == Team.PLAYER:
        player_value = 1.0
    elif sim.state.winner == Team.ENEMY:
        player_value = -1.0
    else:
        # Draw: always negative (a soft loss), but less bad with more damage
        # Maps damage_ratio [-1, +1] → [-0.8, -0.5]
        total_damage = player_damage_dealt + enemy_damage_dealt
        if total_damage > 0:
            damage_ratio = (player_damage_dealt - enemy_damage_dealt) / total_damage
            player_value = -0.65 + (damage_ratio * 0.15)  # Maps [-1,+1] → [-0.8, -0.5]
        else:
            player_value = -0.65  # Draw with no damage = middle of range

    # Convert to training data
    training_data: list[tuple[np.ndarray, float]] = []
    for tensor, team_int in history:
        value = player_value if team_int == 0 else -player_value
        training_data.append((tensor, value))

    # Build replay dict
    replay_data = {
        "moves": moves_record,
        "initial_state": initial_state,
        "final_state": sim.state.to_json(),
        "winner": int(sim.state.winner) if sim.state.winner is not None else None,
        "total_turns": sim.state.turn_number,
        "seed": seed,
    }

    return training_data, replay_data


def play_batched_self_play_games(
    network: nn.Module,
    device: torch.device,
    num_games: int,
    temperature: float,
    base_seed: int = 0,
) -> tuple[list[tuple[np.ndarray, float]], list[dict]]:
    """
    Play multiple self-play games with batched GPU evaluation.

    Instead of running games in separate processes with CPU inference,
    this runs all games in a single process and batches position evaluations
    on GPU for much better performance on Apple Silicon.

    Returns:
        Tuple of (all_training_data, all_replay_data)
    """
    from .game_simulator import GameSimulator, Move, MoveType

    network.eval()

    # Game state for each active game
    @dataclass
    class GameInstance:
        sim: GameSimulator
        history: list[tuple[np.ndarray, int]]  # (tensor, team)
        moves_record: list[dict]
        initial_state: dict
        player_damage: int = 0
        enemy_damage: int = 0
        seed: int = 0
        finished: bool = False

    # Initialize all games
    # Alternate playing_as between PLAYER (White) and ENEMY (Black) for balanced training
    # This lets the network learn color-specific strategies (e.g., White needs to be more aggressive)
    games: list[GameInstance] = []
    for i in range(num_games):
        sim = GameSimulator()
        sim.set_seed(base_seed + i)
        # Alternate color perspective for training
        sim.state.playing_as = Team.PLAYER if i % 2 == 0 else Team.ENEMY
        games.append(GameInstance(
            sim=sim,
            history=[],
            moves_record=[],
            initial_state=sim.state.to_json(),
            seed=base_seed + i,
        ))

    # Play until all games are done
    move_count = 0
    games_completed = 0
    pbar = tqdm(total=num_games, desc="  Playing games", leave=False, ncols=80)

    with torch.no_grad():
        while any(not g.finished for g in games):
            # Collect all positions that need evaluation from all active games
            # Structure: (game_idx, move_idx, tensor, move_obj)
            all_evaluations: list[tuple[int, int, np.ndarray, Move]] = []
            game_move_counts: list[int] = []  # How many moves each game has

            for game_idx, game in enumerate(games):
                if game.finished:
                    game_move_counts.append(0)
                    continue

                # Record current state for training
                tensor = game.sim.state.to_tensor()
                team = game.sim.state.side_to_move
                game.history.append((tensor, int(team)))

                # Generate moves
                moves = game.sim.generate_moves(team)
                if not moves or game.sim.state.is_terminal:
                    if not game.finished:
                        game.finished = True
                        games_completed += 1
                        pbar.update(1)
                    game_move_counts.append(0)
                    continue

                game_move_counts.append(len(moves))

                # Generate positions for each candidate move
                for move_idx, move in enumerate(moves):
                    sim_copy = GameSimulator(game.sim.state.clone())
                    piece = sim_copy.get_piece_at(move.piece.x, move.piece.y)
                    if piece:
                        new_move = Move(
                            piece=piece,
                            move_type=move.move_type,
                            from_pos=move.from_pos,
                            to_pos=move.to_pos,
                            attack_pos=move.attack_pos,
                        )
                        sim_copy.make_move(new_move)
                        all_evaluations.append((game_idx, move_idx, sim_copy.state.to_tensor(), move))
                    else:
                        # Shouldn't happen, but handle gracefully
                        all_evaluations.append((game_idx, move_idx, game.sim.state.to_tensor(), move))

            if not all_evaluations:
                break

            move_count += 1

            # Batch evaluate ALL positions on GPU
            batch_tensors = torch.tensor(
                np.array([e[2] for e in all_evaluations]),
                dtype=torch.float32,
                device=device
            )
            batch_values = network(batch_tensors).cpu().numpy().flatten()

            # Distribute values back to games and make moves
            eval_idx = 0
            for game_idx, game in enumerate(games):
                if game.finished or game_move_counts[game_idx] == 0:
                    continue

                num_moves = game_move_counts[game_idx]
                move_values = [-batch_values[eval_idx + i] for i in range(num_moves)]
                moves = [all_evaluations[eval_idx + i][3] for i in range(num_moves)]
                eval_idx += num_moves

                # Select move with temperature
                if temperature > 0:
                    values = np.array(move_values)
                    # Numerical stability
                    values = values - values.max()
                    exp_values = np.exp(values / temperature)
                    probs = exp_values / (exp_values.sum() + 1e-10)
                    move_idx = np.random.choice(len(moves), p=probs)
                else:
                    move_idx = int(np.argmax(move_values))

                selected_move = moves[move_idx]
                team = game.sim.state.side_to_move

                # Record the move
                move_record = {
                    "turn": game.sim.state.turn_number,
                    "team": int(team),
                    "piece_type": int(selected_move.piece.piece_type),
                    "move_type": int(selected_move.move_type),
                    "from": list(selected_move.from_pos),
                    "to": list(selected_move.to_pos),
                    "attack": list(selected_move.attack_pos) if selected_move.attack_pos else None,
                    "ability_id": int(selected_move.ability_id) if selected_move.ability_id is not None else None,
                    "ability_target": list(selected_move.ability_target) if selected_move.ability_target else None,
                    "value": float(move_values[move_idx]),
                    "top_moves": len(moves),
                }

                # Execute move
                damage = game.sim.make_move(selected_move)
                move_record["damage"] = damage
                game.moves_record.append(move_record)

                # Track damage
                if team == Team.PLAYER:
                    game.player_damage += damage
                else:
                    game.enemy_damage += damage

                # Check if game ended
                if game.sim.state.is_terminal:
                    game.finished = True
                    games_completed += 1
                    pbar.update(1)

    pbar.close()

    # Convert finished games to training data and replays
    all_training_data: list[tuple[np.ndarray, float]] = []
    all_replays: list[dict] = []

    for game in games:
        # Determine outcome
        if game.sim.state.winner == Team.PLAYER:
            player_value = 1.0
        elif game.sim.state.winner == Team.ENEMY:
            player_value = -1.0
        else:
            # Draw: always negative (a soft loss), but less bad with more damage
            # Maps damage_ratio [-1, +1] → [-0.8, -0.5]
            total_damage = game.player_damage + game.enemy_damage
            if total_damage > 0:
                damage_ratio = (game.player_damage - game.enemy_damage) / total_damage
                player_value = -0.65 + (damage_ratio * 0.15)  # Maps [-1,+1] → [-0.8, -0.5]
            else:
                player_value = -0.65  # Draw with no damage = middle of range

        # Convert to training data
        for tensor, team_int in game.history:
            value = player_value if team_int == 0 else -player_value
            all_training_data.append((tensor, value))

        # Build replay
        replay = {
            "moves": game.moves_record,
            "initial_state": game.initial_state,
            "final_state": game.sim.state.to_json(),
            "winner": int(game.sim.state.winner) if game.sim.state.winner is not None else None,
            "total_turns": game.sim.state.turn_number,
            "seed": game.seed,
        }
        all_replays.append(replay)

    return all_training_data, all_replays


def play_mcts_self_play_games(
    network: nn.Module,
    device: torch.device,
    mcts_config: MCTSConfig,
    num_games: int,
    base_seed: int = 0,
) -> tuple[list[tuple[np.ndarray, float, np.ndarray]], list[dict]]:
    """
    Play self-play games using MCTS for move selection.

    Returns training data with policy targets from MCTS visit counts.

    Returns:
        Tuple of (all_training_data, all_replay_data)
        Training data is list of (state_tensor, value_target, policy_target)
    """
    from .game_simulator import GameSimulator, Move

    network.eval()

    @dataclass
    class MCTSGameInstance:
        sim: GameSimulator
        history: list[tuple[np.ndarray, int, np.ndarray]]  # (tensor, team, policy_target)
        moves_record: list[dict]
        initial_state: dict
        player_damage: int = 0
        enemy_damage: int = 0
        seed: int = 0
        finished: bool = False

    # Initialize games
    games: list[MCTSGameInstance] = []
    for i in range(num_games):
        sim = GameSimulator()
        sim.set_seed(base_seed + i)
        sim.state.playing_as = Team.PLAYER if i % 2 == 0 else Team.ENEMY
        games.append(MCTSGameInstance(
            sim=sim,
            history=[],
            moves_record=[],
            initial_state=sim.state.to_json(),
            seed=base_seed + i,
        ))

    # Create MCTS searcher
    mcts = BatchedMCTS(network, device, mcts_config)

    # Play games (process in batches for efficiency)
    games_completed = 0
    pbar = tqdm(total=num_games, desc="  Playing MCTS games", leave=False, ncols=80)

    while any(not g.finished for g in games):
        # Get active games
        active_indices = [i for i, g in enumerate(games) if not g.finished]
        if not active_indices:
            break

        active_games = [games[i] for i in active_indices]
        active_states = [g.sim.state for g in active_games]

        # Run batched MCTS search
        search_results = mcts.search_batch(
            active_states,
            num_simulations=mcts_config.num_simulations,
            add_noise=True,
        )

        # Process results and execute moves
        for game_idx, (game, (best_move, visit_dist)) in zip(active_indices, zip(active_games, search_results)):
            # Record current state and policy target
            tensor = game.sim.state.to_tensor()
            team = game.sim.state.side_to_move

            # Create policy target from visit distribution
            # visit_dist is normalized over legal moves, need to expand to full policy size
            policy_target = np.zeros(POLICY_SIZE, dtype=np.float32)
            if len(visit_dist) > 0:
                legal_moves = game.sim.generate_moves(team)
                for i, move in enumerate(legal_moves):
                    if i < len(visit_dist):
                        move_idx = encode_move(move)
                        if 0 <= move_idx < POLICY_SIZE:
                            policy_target[move_idx] = visit_dist[i]

            game.history.append((tensor, int(team), policy_target))

            if best_move is None:
                # No legal moves - game over
                game.finished = True
                games_completed += 1
                pbar.update(1)
                continue

            # Record move
            move_record = {
                "turn": game.sim.state.turn_number,
                "team": int(team),
                "piece_type": int(best_move.piece.piece_type),
                "move_type": int(best_move.move_type),
                "from": list(best_move.from_pos),
                "to": list(best_move.to_pos),
                "attack": list(best_move.attack_pos) if best_move.attack_pos else None,
                "ability_id": int(best_move.ability_id) if best_move.ability_id is not None else None,
                "ability_target": list(best_move.ability_target) if best_move.ability_target else None,
                "mcts_visits": int(visit_dist.max() * mcts_config.num_simulations) if len(visit_dist) > 0 else 0,
            }

            # Execute move - need to get piece from current sim state
            from .game_simulator import Move
            piece = game.sim.get_piece_at(best_move.from_pos[0], best_move.from_pos[1])
            if piece is None:
                # Piece not found - shouldn't happen, mark game as finished
                game.finished = True
                games_completed += 1
                pbar.update(1)
                continue

            actual_move = Move(
                piece=piece,
                move_type=best_move.move_type,
                from_pos=best_move.from_pos,
                to_pos=best_move.to_pos,
                attack_pos=best_move.attack_pos,
                ability_id=best_move.ability_id,
                ability_target=best_move.ability_target,
            )
            damage = game.sim.make_move(actual_move)
            move_record["damage"] = damage
            game.moves_record.append(move_record)

            # Track damage
            if team == Team.PLAYER:
                game.player_damage += damage
            else:
                game.enemy_damage += damage

            # Check if game ended
            if game.sim.state.is_terminal:
                game.finished = True
                games_completed += 1
                pbar.update(1)

    pbar.close()

    # Convert to training data with policy targets
    all_training_data: list[tuple[np.ndarray, float, np.ndarray]] = []
    all_replays: list[dict] = []

    for game in games:
        # Determine outcome (same as non-MCTS version)
        if game.sim.state.winner == Team.PLAYER:
            player_value = 1.0
        elif game.sim.state.winner == Team.ENEMY:
            player_value = -1.0
        else:
            total_damage = game.player_damage + game.enemy_damage
            if total_damage > 0:
                damage_ratio = (game.player_damage - game.enemy_damage) / total_damage
                player_value = -0.65 + (damage_ratio * 0.15)
            else:
                player_value = -0.65

        # Convert to training data with policy
        for tensor, team_int, policy_target in game.history:
            value = player_value if team_int == 0 else -player_value
            all_training_data.append((tensor, value, policy_target))

        # Build replay
        replay = {
            "moves": game.moves_record,
            "initial_state": game.initial_state,
            "final_state": game.sim.state.to_json(),
            "winner": int(game.sim.state.winner) if game.sim.state.winner is not None else None,
            "total_turns": game.sim.state.turn_number,
            "seed": game.seed,
            "mcts_simulations": mcts_config.num_simulations,
        }
        all_replays.append(replay)

    return all_training_data, all_replays


def play_mcts_rust_self_play_games(
    network: nn.Module,
    device: torch.device,
    mcts_config: MCTSConfig,
    num_games: int,
    base_seed: int = 0,
) -> tuple[list[tuple[np.ndarray, float, np.ndarray]], list[dict]]:
    """
    Play self-play games using MCTS with Rust simulator.

    Much faster than Python MCTS due to native game simulation.

    Returns:
        Tuple of (all_training_data, all_replay_data)
        Training data is list of (state_tensor, value_target, policy_target)
    """
    from exchange_simulator import RustSimulator

    network.eval()

    @dataclass
    class RustMCTSGameInstance:
        sim: RustSimulator
        history: list[tuple[np.ndarray, int, np.ndarray]]  # (tensor, team, policy_target)
        moves_record: list[dict]
        initial_state: dict
        player_damage: int = 0
        enemy_damage: int = 0
        seed: int = 0
        playing_as: int = 0  # 0 = Player, 1 = Enemy
        finished: bool = False

    # Initialize games
    games: list[RustMCTSGameInstance] = []
    for i in range(num_games):
        sim = RustSimulator()
        sim.set_seed(base_seed + i)
        playing_as = 0 if i % 2 == 0 else 1

        # Capture initial state
        initial_state = _rust_sim_to_json(sim)

        games.append(RustMCTSGameInstance(
            sim=sim,
            history=[],
            moves_record=[],
            initial_state=initial_state,
            seed=base_seed + i,
            playing_as=playing_as,
        ))

    # Create Rust MCTS searcher
    mcts = BatchedRustMCTS(network, device, mcts_config)

    # Play games
    games_completed = 0
    pbar = tqdm(total=num_games, desc="  Playing Rust MCTS games", leave=False, ncols=80)

    while any(not g.finished for g in games):
        # Get active games
        active_indices = [i for i, g in enumerate(games) if not g.finished]
        if not active_indices:
            break

        active_games = [games[i] for i in active_indices]
        active_sims = [g.sim for g in active_games]

        # Run batched MCTS search
        search_results = mcts.search_batch(
            active_sims,
            num_simulations=mcts_config.num_simulations,
            add_noise=True,
        )

        # Process results and execute moves
        for game_idx, (game, (best_move, visit_dist)) in zip(active_indices, zip(active_games, search_results)):
            # Record current state and policy target
            tensor = rust_state_to_tensor(game.sim)
            team = game.sim.side_to_move

            # Create policy target from visit distribution
            policy_target = np.zeros(POLICY_SIZE, dtype=np.float32)
            if len(visit_dist) > 0:
                rust_moves = game.sim.generate_moves()
                for i, rm in enumerate(rust_moves):
                    if i < len(visit_dist):
                        from .mcts_rust import rust_move_to_policy_index
                        move_idx = rust_move_to_policy_index(rm)
                        if 0 <= move_idx < POLICY_SIZE:
                            policy_target[move_idx] = visit_dist[i]

            game.history.append((tensor, int(team), policy_target))

            if best_move is None:
                # No legal moves - game over
                game.finished = True
                games_completed += 1
                pbar.update(1)
                continue

            # Record move
            move_record = {
                "turn": game.sim.turn_number,
                "team": int(team),
                "piece_type": int(best_move.piece_idx),  # Approximate - using piece index
                "move_type": int(best_move.move_type),
                "from": [best_move.from_x, best_move.from_y],
                "to": [best_move.to_x, best_move.to_y],
                "attack": [best_move.attack_x, best_move.attack_y] if best_move.attack_x is not None else None,
                "ability_id": int(best_move.ability_id) if best_move.ability_id is not None else None,
                "ability_target": [best_move.ability_target_x, best_move.ability_target_y] if best_move.ability_target_x is not None else None,
                "mcts_visits": int(visit_dist.max() * mcts_config.num_simulations) if len(visit_dist) > 0 else 0,
            }

            # Execute move
            damage = game.sim.make_move(best_move)
            move_record["damage"] = damage
            game.moves_record.append(move_record)

            # Track damage
            if team == 0:  # Player
                game.player_damage += damage
            else:
                game.enemy_damage += damage

            # Check if game ended
            if game.sim.is_terminal:
                game.finished = True
                games_completed += 1
                pbar.update(1)

    pbar.close()

    # Convert to training data with policy targets
    all_training_data: list[tuple[np.ndarray, float, np.ndarray]] = []
    all_replays: list[dict] = []

    for game in games:
        # Determine outcome
        if game.sim.winner == 0:  # Player wins
            player_value = 1.0
        elif game.sim.winner == 1:  # Enemy wins
            player_value = -1.0
        else:
            # Draw - use damage ratio
            total_damage = game.player_damage + game.enemy_damage
            if total_damage > 0:
                damage_ratio = (game.player_damage - game.enemy_damage) / total_damage
                player_value = -0.65 + (damage_ratio * 0.15)
            else:
                player_value = -0.65

        # Convert to training data with policy
        for tensor, team_int, policy_target in game.history:
            value = player_value if team_int == 0 else -player_value
            all_training_data.append((tensor, value, policy_target))

        # Build replay
        replay = {
            "moves": game.moves_record,
            "initial_state": game.initial_state,
            "final_state": _rust_sim_to_json(game.sim),
            "winner": game.sim.winner,
            "total_turns": game.sim.turn_number,
            "seed": game.seed,
            "mcts_simulations": mcts_config.num_simulations,
            "rust_accelerated": True,
        }
        all_replays.append(replay)

    return all_training_data, all_replays


def _rust_sim_to_json(sim) -> dict:
    """Convert Rust simulator state to JSON dict."""
    pieces = []
    for rp in sim.get_pieces():
        pieces.append({
            "pieceType": rp.piece_type,
            "team": rp.team,
            "x": rp.x,
            "y": rp.y,
            "currentHp": rp.current_hp,
            "maxHp": rp.max_hp,
            "baseDamage": rp.base_damage,
            "abilityCooldown": rp.ability_cooldown,
            "abilityUsedThisMatch": rp.ability_used_this_match,
            "interposeActive": rp.interpose_active,
            "wasPawn": rp.was_pawn,
        })
    return {
        "pieces": pieces,
        "sideToMove": sim.side_to_move,
        "turnNumber": sim.turn_number,
        "isTerminal": sim.is_terminal,
        "isDraw": sim.is_draw,
        "winner": sim.winner,
    }


class Trainer:
    """
    Main training class for EXCHANGE value network.

    Handles the complete training pipeline including:
    - Network initialization and checkpointing
    - Self-play game generation (parallel)
    - Training data management
    - Network optimization
    - Logging and evaluation
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.state = TrainingState()

        # Setup directories
        self.output_path = Path(config.output_dir)
        self.checkpoint_path = self.output_path / config.checkpoint_dir
        self.tensorboard_path = self.output_path / config.tensorboard_dir
        self.replays_path = self.output_path / config.replays_dir

        for path in [self.output_path, self.checkpoint_path, self.tensorboard_path, self.replays_path]:
            path.mkdir(parents=True, exist_ok=True)

        # Determine device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")  # M4 Mac
            print("Using MPS (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        # Initialize network (PolicyValueNetwork if MCTS enabled)
        if config.use_mcts_rust:
            self.network = create_policy_value_from_preset(config.network_preset)
            print(f"Using PolicyValueNetwork with Rust MCTS ({config.mcts_simulations} simulations/move)")
        elif config.use_mcts:
            self.network = create_policy_value_from_preset(config.network_preset)
            print(f"Using PolicyValueNetwork with MCTS ({config.mcts_simulations} simulations/move)")
        else:
            self.network = create_from_preset(config.network_preset)
        self.network.to(self.device)
        print(f"Network parameters: {self.network.count_parameters():,}")

        # MCTS config (if enabled)
        if config.use_mcts or config.use_mcts_rust:
            self.mcts_config = MCTSConfig(
                num_simulations=config.mcts_simulations,
                eval_simulations=config.mcts_eval_simulations,
                c_puct=config.mcts_c_puct,
                dirichlet_alpha=config.mcts_dirichlet_alpha,
                dirichlet_epsilon=config.mcts_dirichlet_epsilon,
            )
        else:
            self.mcts_config = None

        # Optimizer
        self.optimizer = optim.AdamW(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.tensorboard_path))

        # Number of workers for parallel bootstrap game generation
        self.num_workers = config.num_workers if config.num_workers > 0 else cpu_count()
        print(f"Using {self.num_workers} workers for bootstrap games (self-play uses batched GPU)")

        # Data buffer for replay
        self.data_buffer: list[tuple[np.ndarray, float]] = []
        self.buffer_max_size = 100000  # Keep last 100k positions

        # Champion network for evaluation (initialized after bootstrap)
        self.champion_network: Optional[nn.Module] = None

    def save_checkpoint(self, name: str = "latest") -> None:
        """Save training checkpoint."""
        checkpoint = {
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "training_state": {
                "iteration": self.state.iteration,
                "best_eval_score": self.state.best_eval_score,
            },
            "config": self.config.__dict__,
            "network_config": self.network.get_config(),
        }
        path = self.checkpoint_path / f"{name}.pt"
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.state.iteration = checkpoint["training_state"]["iteration"]
        self.state.best_eval_score = checkpoint["training_state"]["best_eval_score"]
        print(f"Loaded checkpoint: {path} (iteration {self.state.iteration})")

    def _init_champion(self) -> None:
        """Initialize champion network as a copy of current network."""
        if self.config.use_mcts or self.config.use_mcts_rust:
            self.champion_network = create_policy_value_from_preset(self.config.network_preset)
        else:
            self.champion_network = create_from_preset(self.config.network_preset)
        self.champion_network.load_state_dict(self.network.state_dict())
        self.champion_network.to(self.device)
        self.champion_network.eval()
        print("Champion network initialized")

    def _update_champion(self) -> None:
        """Update champion network to current network weights."""
        if self.champion_network is None:
            self._init_champion()
        else:
            self.champion_network.load_state_dict(self.network.state_dict())
        self.save_checkpoint("champion")
        print("  -> NEW CHAMPION!")

    def bootstrap(self) -> None:
        """
        Bootstrap training with random games.

        This provides initial signal before self-play can be effective.
        """
        print(f"\n{'='*60}")
        print(f"BOOTSTRAP PHASE: {self.config.bootstrap_games} random games")
        print(f"{'='*60}\n")

        # Generate random games in parallel
        seeds = list(range(self.config.bootstrap_games))

        all_data: list[tuple[np.ndarray, float]] = []

        with Pool(self.num_workers) as pool:
            results = list(tqdm(
                pool.imap(_play_random_game, seeds),
                total=len(seeds),
                desc="Generating bootstrap games"
            ))

        for game_data in results:
            all_data.extend(game_data)

        print(f"Generated {len(all_data):,} training positions")

        # Add to buffer
        self.data_buffer = all_data[-self.buffer_max_size:]

        # Train on bootstrap data
        self._train_on_buffer(epochs=self.config.bootstrap_epochs, phase="bootstrap")

        # Save initial checkpoint
        self.save_checkpoint("post_bootstrap")

    def generate_self_play_games(self, num_games: int, temperature: float) -> list:
        """Generate games using current network with batched GPU evaluation. Saves ALL game replays.

        Returns:
            List of training tuples. Format depends on use_mcts:
            - Without MCTS: (state_tensor, value)
            - With MCTS: (state_tensor, value, policy_target)
        """
        base_seed = self.state.iteration * 10000

        if self.config.use_mcts_rust:
            print(f"  Generating {num_games} games with Rust MCTS ({self.config.mcts_simulations} sims/move)...")
            all_data, all_replays = play_mcts_rust_self_play_games(
                network=self.network,
                device=self.device,
                mcts_config=self.mcts_config,
                num_games=num_games,
                base_seed=base_seed,
            )
        elif self.config.use_mcts:
            print(f"  Generating {num_games} games with MCTS ({self.config.mcts_simulations} sims/move)...")
            all_data, all_replays = play_mcts_self_play_games(
                network=self.network,
                device=self.device,
                mcts_config=self.mcts_config,
                num_games=num_games,
                base_seed=base_seed,
            )
        else:
            print(f"  Generating {num_games} games with batched GPU evaluation...")
            all_data, all_replays = play_batched_self_play_games(
                network=self.network,
                device=self.device,
                num_games=num_games,
                temperature=temperature,
                base_seed=base_seed,
            )

        # Save replays to disk
        iteration = self.state.iteration
        iter_replays_dir = self.replays_path / f"iter_{iteration:04d}"
        iter_replays_dir.mkdir(parents=True, exist_ok=True)

        wins, losses, draws = 0, 0, 0
        for game_idx, replay_data in enumerate(all_replays):
            # Save replay
            replay_path = iter_replays_dir / f"game_{game_idx:04d}.json"
            with open(replay_path, 'w') as f:
                json.dump(replay_data, f)

            # Track stats
            if replay_data["winner"] == 0:
                wins += 1
            elif replay_data["winner"] == 1:
                losses += 1
            else:
                draws += 1

        print(f"  Games saved to {iter_replays_dir}/ (W:{wins} B:{losses} D:{draws})")

        return all_data

    def _train_on_buffer(self, epochs: int, phase: str = "training") -> dict:
        """Train network on current data buffer.

        Handles both value-only training and combined policy+value training (MCTS).
        """
        if not self.data_buffer:
            return {"loss": 0.0}

        # Check if we have policy targets (MCTS mode)
        has_policy = (self.config.use_mcts or self.config.use_mcts_rust) and len(self.data_buffer[0]) == 3

        # Prepare data
        states = np.stack([d[0] for d in self.data_buffer])
        values = np.array([d[1] for d in self.data_buffer], dtype=np.float32)

        if has_policy:
            policies = np.stack([d[2] for d in self.data_buffer])
            dataset = TensorDataset(
                torch.tensor(states, dtype=torch.float32),
                torch.tensor(values, dtype=torch.float32).unsqueeze(1),
                torch.tensor(policies, dtype=torch.float32),
            )
        else:
            dataset = TensorDataset(
                torch.tensor(states, dtype=torch.float32),
                torch.tensor(values, dtype=torch.float32).unsqueeze(1),
            )

        # pin_memory only works with CUDA, not MPS or CPU
        use_pin_memory = self.device.type == "cuda"
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Data already in memory
            pin_memory=use_pin_memory,
        )

        self.network.train()
        total_loss = 0.0
        total_value_loss = 0.0
        total_policy_loss = 0.0
        num_batches = 0

        pbar = tqdm(range(epochs), desc="  Training", leave=False, ncols=80)
        for epoch in pbar:
            epoch_loss = 0.0
            batch_count = 0

            for batch_data in dataloader:
                if has_policy:
                    batch_states, batch_values, batch_policies = batch_data
                    batch_policies = batch_policies.to(self.device)
                else:
                    batch_states, batch_values = batch_data

                batch_states = batch_states.to(self.device)
                batch_values = batch_values.to(self.device)

                if has_policy and hasattr(self.network, 'forward_policy_value'):
                    # Combined policy + value loss
                    pred_values, pred_log_policies = self.network.forward_policy_value(batch_states)

                    # Value loss (MSE)
                    value_loss = nn.functional.mse_loss(pred_values, batch_values)

                    # Policy loss (cross entropy with soft targets)
                    # We have soft targets from MCTS visit counts
                    # Use KL divergence: sum(p * log(p/q)) = sum(p * log(p)) - sum(p * log(q))
                    # The first term is entropy of targets (constant), so minimize -sum(p * log(q))
                    policy_loss = -(batch_policies * pred_log_policies).sum(dim=-1).mean()

                    loss = value_loss + self.config.policy_loss_weight * policy_loss
                    total_value_loss += value_loss.item()
                    total_policy_loss += policy_loss.item()
                else:
                    # Value-only training
                    predictions = self.network(batch_states)
                    loss = nn.functional.mse_loss(predictions, batch_values)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.config.gradient_clip
                )

                self.optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1
                num_batches += 1

            avg_epoch_loss = epoch_loss / max(batch_count, 1)
            total_loss += epoch_loss
            pbar.set_postfix({"loss": f"{avg_epoch_loss:.4f}"})

        pbar.close()
        self.scheduler.step()

        avg_loss = total_loss / max(num_batches, 1)
        result = {"loss": avg_loss, "num_positions": len(self.data_buffer)}

        if has_policy:
            result["value_loss"] = total_value_loss / max(num_batches, 1)
            result["policy_loss"] = total_policy_loss / max(num_batches, 1)

        return result

    def evaluate(self) -> dict:
        """
        Evaluate current network strength.

        Plays games against random opponents and measures win rate.
        Uses batched GPU evaluation for efficiency.
        """
        from .game_simulator import Move, MoveType

        self.network.eval()
        num_games = 50

        # Initialize all games at once
        @dataclass
        class EvalGame:
            sim: GameSimulator
            finished: bool = False
            winner: Optional[Team] = None

        games = [EvalGame(sim=GameSimulator()) for _ in range(num_games)]
        for i, game in enumerate(games):
            game.sim.set_seed(i + 999999)

        move_count = 0

        with torch.no_grad():
            while any(not g.finished for g in games):
                # Collect all positions needing network evaluation
                # Structure: (game_idx, moves_list, position_tensors)
                network_evals: list[tuple[int, list, list[np.ndarray]]] = []

                for game_idx, game in enumerate(games):
                    if game.finished:
                        continue

                    sim = game.sim
                    current_team = sim.state.side_to_move
                    moves = sim.generate_moves(current_team)

                    if not moves or sim.state.is_terminal:
                        game.finished = True
                        game.winner = sim.state.winner
                        continue

                    if current_team == Team.ENEMY:
                        # Network plays - collect all candidate positions
                        position_tensors = []
                        for move in moves:
                            sim_copy = GameSimulator(sim.state.clone())
                            piece = sim_copy.get_piece_at(move.piece.x, move.piece.y)
                            if piece:
                                new_move = Move(
                                    piece=piece,
                                    move_type=move.move_type,
                                    from_pos=move.from_pos,
                                    to_pos=move.to_pos,
                                    attack_pos=move.attack_pos,
                                )
                                sim_copy.make_move(new_move)
                                position_tensors.append(sim_copy.state.to_tensor())
                            else:
                                position_tensors.append(sim.state.to_tensor())
                        network_evals.append((game_idx, moves, position_tensors))
                    else:
                        # Random player - execute immediately
                        move = sim._rng.choice(moves)
                        sim.make_move(move)

                if not network_evals:
                    continue

                move_count += 1

                # Batch evaluate ALL network positions at once
                all_tensors = []
                eval_indices = []  # (game_idx, num_moves) to reconstruct results
                for game_idx, moves, tensors in network_evals:
                    eval_indices.append((game_idx, len(moves)))
                    all_tensors.extend(tensors)

                if all_tensors:
                    batch = torch.tensor(
                        np.array(all_tensors),
                        dtype=torch.float32,
                        device=self.device
                    )
                    batch_values = self.network(batch).cpu().numpy().flatten()

                    # Distribute values back and make moves
                    value_idx = 0
                    for (game_idx, num_moves), (_, moves, _) in zip(eval_indices, network_evals):
                        move_values = [-batch_values[value_idx + i] for i in range(num_moves)]
                        value_idx += num_moves

                        best_idx = int(np.argmax(move_values))
                        games[game_idx].sim.make_move(moves[best_idx])

                        # Check if game ended
                        if games[game_idx].sim.state.is_terminal:
                            games[game_idx].finished = True
                            games[game_idx].winner = games[game_idx].sim.state.winner

        # Count results
        wins = sum(1 for g in games if g.winner == Team.ENEMY)
        losses = sum(1 for g in games if g.winner == Team.PLAYER)
        draws = sum(1 for g in games if g.winner is None or g.winner not in (Team.ENEMY, Team.PLAYER))

        win_rate = wins / num_games
        return {
            "win_rate": win_rate,
            "wins": wins,
            "losses": losses,
            "draws": draws,
        }

    def evaluate_vs_champion(self) -> dict:
        """
        Evaluate current network against the champion network.

        Plays games with current network vs champion, splitting sides evenly.
        Win rate is calculated from DECISIVE games only (draws excluded).

        Returns:
            Dict with wins, losses, draws, decisive_win_rate, should_promote
        """
        from .game_simulator import Move, MoveType

        if self.champion_network is None:
            # No champion yet - auto-promote
            return {
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "decisive_win_rate": 1.0,
                "should_promote": True,
                "reason": "No champion yet",
            }

        self.network.eval()
        self.champion_network.eval()
        num_games = self.config.champion_eval_games

        @dataclass
        class ChampionGame:
            sim: GameSimulator
            current_plays_as: Team  # Which side the current network plays
            finished: bool = False
            winner: Optional[Team] = None

        # Half games as Player (White), half as Enemy (Black)
        games: list[ChampionGame] = []
        for i in range(num_games):
            sim = GameSimulator()
            sim.set_seed(i + 888888)
            # Alternate sides for fairness
            current_plays_as = Team.PLAYER if i % 2 == 0 else Team.ENEMY
            games.append(ChampionGame(sim=sim, current_plays_as=current_plays_as))

        move_count = 0

        with torch.no_grad():
            while any(not g.finished for g in games):
                # Collect positions for current network evaluation
                current_evals: list[tuple[int, list, list[np.ndarray]]] = []
                # Collect positions for champion network evaluation
                champion_evals: list[tuple[int, list, list[np.ndarray]]] = []

                for game_idx, game in enumerate(games):
                    if game.finished:
                        continue

                    sim = game.sim
                    if sim.state.is_terminal:
                        game.finished = True
                        game.winner = sim.state.winner
                        continue

                    current_team = sim.state.side_to_move
                    moves = sim.generate_moves(current_team)

                    if not moves:
                        game.finished = True
                        game.winner = sim.state.winner
                        continue

                    # Determine which network plays this side
                    use_current_network = (current_team == game.current_plays_as)

                    # Collect candidate positions
                    position_tensors = []
                    for move in moves:
                        sim_copy = GameSimulator(sim.state.clone())
                        piece = sim_copy.get_piece_at(move.piece.x, move.piece.y)
                        if piece:
                            new_move = Move(
                                piece=piece,
                                move_type=move.move_type,
                                from_pos=move.from_pos,
                                to_pos=move.to_pos,
                                attack_pos=move.attack_pos,
                            )
                            sim_copy.make_move(new_move)
                            position_tensors.append(sim_copy.state.to_tensor())
                        else:
                            position_tensors.append(sim.state.to_tensor())

                    if use_current_network:
                        current_evals.append((game_idx, moves, position_tensors))
                    else:
                        champion_evals.append((game_idx, moves, position_tensors))

                move_count += 1

                # Batch evaluate with current network
                if current_evals:
                    all_tensors = []
                    eval_meta = []
                    for game_idx, moves, tensors in current_evals:
                        eval_meta.append((game_idx, len(moves)))
                        all_tensors.extend(tensors)

                    batch = torch.tensor(np.array(all_tensors), dtype=torch.float32, device=self.device)
                    batch_values = self.network(batch).cpu().numpy().flatten()

                    value_idx = 0
                    for (game_idx, num_moves), (_, moves, _) in zip(eval_meta, current_evals):
                        move_values = [-batch_values[value_idx + i] for i in range(num_moves)]
                        value_idx += num_moves
                        best_idx = int(np.argmax(move_values))
                        games[game_idx].sim.make_move(moves[best_idx])

                        if games[game_idx].sim.state.is_terminal:
                            games[game_idx].finished = True
                            games[game_idx].winner = games[game_idx].sim.state.winner

                # Batch evaluate with champion network
                if champion_evals:
                    all_tensors = []
                    eval_meta = []
                    for game_idx, moves, tensors in champion_evals:
                        eval_meta.append((game_idx, len(moves)))
                        all_tensors.extend(tensors)

                    batch = torch.tensor(np.array(all_tensors), dtype=torch.float32, device=self.device)
                    batch_values = self.champion_network(batch).cpu().numpy().flatten()

                    value_idx = 0
                    for (game_idx, num_moves), (_, moves, _) in zip(eval_meta, champion_evals):
                        move_values = [-batch_values[value_idx + i] for i in range(num_moves)]
                        value_idx += num_moves
                        best_idx = int(np.argmax(move_values))
                        games[game_idx].sim.make_move(moves[best_idx])

                        if games[game_idx].sim.state.is_terminal:
                            games[game_idx].finished = True
                            games[game_idx].winner = games[game_idx].sim.state.winner

        # Count results from current network's perspective
        wins = 0
        losses = 0
        draws = 0

        for game in games:
            if game.winner is None:
                draws += 1
            elif game.winner == game.current_plays_as:
                wins += 1
            else:
                losses += 1

        # Calculate decisive win rate (excluding draws)
        decisive_games = wins + losses
        if decisive_games >= self.config.champion_min_decisive:
            decisive_win_rate = wins / decisive_games
            should_promote = decisive_win_rate >= self.config.champion_win_threshold
            reason = f"{decisive_win_rate:.1%} of decisive games"
        else:
            decisive_win_rate = 0.0
            should_promote = False
            reason = f"Not enough decisive games ({decisive_games} < {self.config.champion_min_decisive})"

        return {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "decisive_games": decisive_games,
            "decisive_win_rate": decisive_win_rate,
            "should_promote": should_promote,
            "reason": reason,
        }

    def _log_sample_games(self, iteration: int) -> None:
        """
        Save sample games played by current network for review.
        Games are saved to replays/ directory and can be viewed in the dashboard.
        Uses batched GPU evaluation for efficiency.
        """
        from .game_viewer import GameReplay
        from .game_simulator import Move, MoveType

        self.network.eval()
        num_games = self.config.log_games_count
        print(f"  Logging {num_games} sample games...")

        # Game state tracking
        @dataclass
        class LogGame:
            sim: GameSimulator
            moves_record: list
            initial_state: dict
            finished: bool = False

        games = []
        for game_idx in range(num_games):
            sim = GameSimulator()
            sim.set_seed(iteration * 1000 + game_idx)
            games.append(LogGame(
                sim=sim,
                moves_record=[],
                initial_state=sim.state.to_json(),
            ))

        move_count = 0

        with torch.no_grad():
            while any(not g.finished for g in games):
                # Collect all positions needing evaluation
                # Structure: (game_idx, moves_list, position_tensors, team)
                evals_needed: list[tuple[int, list, list[np.ndarray], Team]] = []

                for game_idx, game in enumerate(games):
                    if game.finished:
                        continue

                    sim = game.sim
                    if sim.state.is_terminal:
                        game.finished = True
                        continue

                    team = sim.state.side_to_move
                    moves = sim.generate_moves(team)

                    if not moves:
                        game.finished = True
                        continue

                    # Collect candidate positions for batched eval
                    position_tensors = []
                    for move in moves:
                        sim_copy = GameSimulator(sim.state.clone())
                        piece = sim_copy.get_piece_at(move.piece.x, move.piece.y)
                        if piece:
                            new_move = Move(
                                piece=piece,
                                move_type=move.move_type,
                                from_pos=move.from_pos,
                                to_pos=move.to_pos,
                                attack_pos=move.attack_pos,
                            )
                            sim_copy.make_move(new_move)
                            position_tensors.append(sim_copy.state.to_tensor())
                        else:
                            position_tensors.append(sim.state.to_tensor())

                    evals_needed.append((game_idx, moves, position_tensors, team))

                if not evals_needed:
                    break

                move_count += 1

                # Batch evaluate ALL positions
                all_tensors = []
                eval_meta = []  # (game_idx, num_moves, team)
                for game_idx, moves, tensors, team in evals_needed:
                    eval_meta.append((game_idx, len(moves), team))
                    all_tensors.extend(tensors)

                batch = torch.tensor(
                    np.array(all_tensors),
                    dtype=torch.float32,
                    device=self.device
                )
                batch_values = self.network(batch).cpu().numpy().flatten()

                # Distribute values and make moves
                value_idx = 0
                for (game_idx, num_moves, team), (_, moves, _, _) in zip(eval_meta, evals_needed):
                    move_values = [-batch_values[value_idx + i] for i in range(num_moves)]
                    value_idx += num_moves

                    best_idx = int(np.argmax(move_values))
                    selected_move = moves[best_idx]
                    game = games[game_idx]

                    # Record the move
                    move_record = {
                        "turn": game.sim.state.turn_number,
                        "team": int(team),
                        "piece_type": int(selected_move.piece.piece_type),
                        "move_type": int(selected_move.move_type),
                        "from": list(selected_move.from_pos),
                        "to": list(selected_move.to_pos),
                        "attack": list(selected_move.attack_pos) if selected_move.attack_pos else None,
                        "ability_id": int(selected_move.ability_id) if selected_move.ability_id is not None else None,
                        "ability_target": list(selected_move.ability_target) if selected_move.ability_target else None,
                        "value": float(move_values[best_idx]),
                    }

                    # Execute move
                    damage = game.sim.make_move(selected_move)
                    move_record["damage"] = damage
                    game.moves_record.append(move_record)

                    if game.sim.state.is_terminal:
                        game.finished = True

        # Save all replays
        for game_idx, game in enumerate(games):
            replay = GameReplay(
                moves=game.moves_record,
                initial_state=game.initial_state,
                final_state=game.sim.state.to_json(),
                winner=int(game.sim.state.winner) if game.sim.state.winner is not None else None,
                total_turns=game.sim.state.turn_number,
            )
            replay_path = self.replays_path / f"iter{iteration:04d}_game{game_idx:02d}.json"
            replay.save(str(replay_path))

        print(f"  Saved to {self.replays_path}/")

    def _log_channel_stats(self, iteration: int, sample_size: int = 100) -> None:
        """
        Log channel statistics to TensorBoard for visualization.

        Samples random positions from the buffer and logs:
        - Average activation for each channel
        - Heatmap images for spatial channels
        - Key strategic indicators
        """
        if len(self.data_buffer) < sample_size:
            return

        # Sample random positions from buffer
        indices = np.random.choice(len(self.data_buffer), sample_size, replace=False)
        tensors = np.stack([self.data_buffer[i][0] for i in indices])

        # Log average activation for each channel
        for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
            avg_activation = tensors[:, ch_idx].mean()
            self.writer.add_scalar(f"channels/{ch_name}", avg_activation, iteration)

        # Log key strategic indicators as a group
        key_channels = {
            "draw_urgency": 29,      # moves_without_damage
            "hp_advantage": 31,      # hp_ratio (0.5 = equal)
            "material_balance": 33,  # material_balance
            "white_attacking": 51,   # black_king_attackers (white attacking black)
            "black_attacking": 50,   # white_king_attackers (black attacking white)
            "hanging_white": 46,
            "hanging_black": 47,
        }
        for name, ch_idx in key_channels.items():
            self.writer.add_scalar(f"strategy/{name}", tensors[:, ch_idx].mean(), iteration)

        # Log heatmap images every 10 iterations (these are larger)
        if iteration % 10 == 0:
            # Average the spatial channels across samples for visualization
            avg_tensor = tensors.mean(axis=0)

            # Log attack maps as images
            self.writer.add_image("heatmaps/white_attacks",
                                  avg_tensor[34:35], iteration, dataformats='CHW')
            self.writer.add_image("heatmaps/black_attacks",
                                  avg_tensor[35:36], iteration, dataformats='CHW')
            self.writer.add_image("heatmaps/contested",
                                  avg_tensor[36:37], iteration, dataformats='CHW')
            self.writer.add_image("heatmaps/damage_potential",
                                  avg_tensor[57:58], iteration, dataformats='CHW')
            self.writer.add_image("heatmaps/center_control",
                                  avg_tensor[44:45], iteration, dataformats='CHW')

    def train(self, resume_from: Optional[str] = None) -> None:
        """
        Main training loop.

        Args:
            resume_from: Optional checkpoint path to resume from
        """
        if resume_from:
            self.load_checkpoint(resume_from)
            # Load champion if it exists
            champion_path = self.checkpoint_path / "champion.pt"
            if champion_path.exists():
                self._init_champion()
                checkpoint = torch.load(champion_path, map_location=self.device, weights_only=False)
                self.champion_network.load_state_dict(checkpoint["network_state_dict"])
                print(f"Loaded champion from {champion_path}")
        elif self.state.iteration == 0:
            # Bootstrap if starting fresh
            self.bootstrap()
            # Initialize champion as the bootstrapped model
            self._init_champion()

        print(f"\n{'='*60}")
        print(f"SELF-PLAY TRAINING: {self.config.num_iterations} iterations")
        print(f"{'='*60}\n")

        for iteration in range(self.state.iteration, self.config.num_iterations):
            self.state.iteration = iteration
            start_time = time.time()

            # Get temperature for this iteration
            temperature = self.config.get_temperature(iteration)

            # Generate self-play games
            new_data = self.generate_self_play_games(
                self.config.games_per_iteration,
                temperature=temperature,
            )

            # Add to buffer (with size limit)
            self.data_buffer.extend(new_data)
            if len(self.data_buffer) > self.buffer_max_size:
                self.data_buffer = self.data_buffer[-self.buffer_max_size:]

            # Train on accumulated data
            train_metrics = self._train_on_buffer(
                epochs=self.config.epochs_per_iteration,
                phase="self_play"
            )

            elapsed = time.time() - start_time

            # Log metrics
            self.writer.add_scalar("training/loss", train_metrics["loss"], iteration)
            self.writer.add_scalar("training/temperature", temperature, iteration)
            self.writer.add_scalar("training/buffer_size", len(self.data_buffer), iteration)
            self.writer.add_scalar("training/learning_rate", self.scheduler.get_last_lr()[0], iteration)

            # Log channel statistics for strategic awareness monitoring
            self._log_channel_stats(iteration)

            print(f"Iteration {iteration + 1}/{self.config.num_iterations} | "
                  f"Loss: {train_metrics['loss']:.4f} | "
                  f"Temp: {temperature:.2f} | "
                  f"Buffer: {len(self.data_buffer):,} | "
                  f"Time: {elapsed:.1f}s")

            # Champion evaluation every iteration
            eval_metrics = self.evaluate_vs_champion()
            self.writer.add_scalar("eval/decisive_win_rate", eval_metrics["decisive_win_rate"], iteration)
            self.writer.add_scalar("eval/wins", eval_metrics["wins"], iteration)
            self.writer.add_scalar("eval/losses", eval_metrics["losses"], iteration)
            self.writer.add_scalar("eval/draws", eval_metrics["draws"], iteration)

            decisive_str = f"{eval_metrics['decisive_win_rate']:.0%}" if eval_metrics.get('decisive_games', 0) > 0 else "N/A"
            print(f"  vs Champion: W:{eval_metrics['wins']} L:{eval_metrics['losses']} D:{eval_metrics['draws']} "
                  f"| Decisive: {decisive_str} ({eval_metrics.get('reason', '')})")

            if eval_metrics["should_promote"]:
                self._update_champion()
                self.state.best_eval_score = eval_metrics["decisive_win_rate"]

            # Periodic checkpointing
            if (iteration + 1) % self.config.checkpoint_interval == 0:
                self.save_checkpoint("latest")
                self.save_checkpoint(f"iter_{iteration + 1}")

        # Final save
        self.save_checkpoint("final")
        self.writer.close()
        print(f"\nTraining complete! Best eval score: {self.state.best_eval_score:.1%}")


def train_personality(
    base_checkpoint: str,
    personality: str,
    output_dir: str,
    reward_shaping: dict,
) -> None:
    """
    Fine-tune a base model for a specific personality.

    Args:
        base_checkpoint: Path to trained base model
        personality: Name of personality (e.g., "aggressive", "defensive")
        output_dir: Output directory for personality model
        reward_shaping: Dict of reward modifications for this personality
    """
    # TODO: Implement personality training with reward shaping
    raise NotImplementedError("Personality training coming soon!")


def main():
    """CLI entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(
        description="EXCHANGE Neural Network Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training settings
    parser.add_argument("--iterations", type=int, default=1000,
                        help="Number of training iterations")
    parser.add_argument("--games", type=int, default=100,
                        help="Games per iteration")
    parser.add_argument("--preset", type=str, default="medium",
                        choices=["tiny", "small", "medium", "large"],
                        help="Network size preset")

    # MCTS settings
    parser.add_argument("--mcts", action="store_true",
                        help="Enable MCTS for move selection (Python, slower)")
    parser.add_argument("--mcts-rust", action="store_true",
                        help="Enable MCTS with Rust simulator (much faster)")
    parser.add_argument("--mcts-sims", type=int, default=100,
                        help="MCTS simulations per move")
    parser.add_argument("--mcts-eval-sims", type=int, default=200,
                        help="MCTS simulations during evaluation")

    # Paths
    parser.add_argument("--output", type=str, default="runs/experiment",
                        help="Output directory")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")

    # Other settings
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")

    args = parser.parse_args()

    # Build config from args
    config = TrainingConfig(
        num_iterations=args.iterations,
        games_per_iteration=args.games,
        network_preset=args.preset,
        output_dir=args.output,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        # MCTS settings
        use_mcts=args.mcts,
        use_mcts_rust=args.mcts_rust,
        mcts_simulations=args.mcts_sims,
        mcts_eval_simulations=args.mcts_eval_sims,
    )

    # Print config summary
    print("=" * 60)
    print("EXCHANGE Training Configuration")
    print("=" * 60)
    print(f"  Network preset: {config.network_preset}")
    print(f"  Iterations: {config.num_iterations}")
    print(f"  Games/iteration: {config.games_per_iteration}")
    if config.use_mcts_rust:
        print(f"  MCTS: Rust (fast)")
        print(f"    Simulations/move: {config.mcts_simulations}")
        print(f"    Eval simulations: {config.mcts_eval_simulations}")
    elif config.use_mcts:
        print(f"  MCTS: Python")
        print(f"    Simulations/move: {config.mcts_simulations}")
        print(f"    Eval simulations: {config.mcts_eval_simulations}")
    else:
        print(f"  MCTS: disabled (1-ply evaluation)")
    print(f"  Output: {config.output_dir}")
    print("=" * 60)
    print()

    # Create trainer and run
    trainer = Trainer(config)
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
