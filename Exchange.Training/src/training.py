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
from typing import Any, Optional

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
from .mcts_rust import BatchedRustMCTS, check_rust_available, rust_state_to_tensor, RUST_AVAILABLE
from datetime import datetime


# Helper functions to convert Rust string types to integers for training data
def team_to_int(team: str) -> int:
    """Convert team string to integer (0=White, 1=Black)."""
    return 0 if team == "White" else 1

def piece_type_to_int(piece_type: str) -> int:
    """Convert piece type string to integer."""
    mapping = {"King": 0, "Queen": 1, "Rook": 2, "Bishop": 3, "Knight": 4, "Pawn": 5}
    return mapping.get(piece_type, 0)

def move_type_to_int(move_type: str) -> int:
    """Convert move type string to integer."""
    mapping = {"Move": 0, "Attack": 1, "MoveAndAttack": 2, "Ability": 3}
    return mapping.get(move_type, 0)

def ability_id_to_int(ability_id: str) -> int:
    """Convert ability ID string to integer."""
    mapping = {"RoyalDecree": 0, "Overextend": 1, "Interpose": 2, "Consecration": 3, "Skirmish": 4, "Advance": 5}
    return mapping.get(ability_id, 0)


def tprint(msg: str) -> None:
    """Print with timestamp prefix."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


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
    temperature_start: float = 2.0  # Higher = more random early on (was 1.0)
    temperature_end: float = 0.5  # Keep some exploration (was 0.3)
    temperature_decay_iterations: int = 100  # Doubled from 50 for slower decay (~0.007/iter)

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

    # Rust simulator (non-MCTS, fast 1-ply evaluation)
    use_rust: bool = False  # Enable Rust simulator for fast 1-ply self-play (no MCTS)

    # MCTS settings
    use_mcts: bool = False  # Enable MCTS for move selection (Python)
    use_mcts_rust: bool = False  # Enable MCTS with Rust simulator (faster)
    mcts_simulations: int = 100  # Simulations per move during self-play
    mcts_eval_simulations: int = 200  # Simulations per move during evaluation
    mcts_c_puct: float = 3.0  # Exploration constant (higher = more exploration, was 1.5)
    mcts_dirichlet_alpha: float = 0.15  # Root noise concentration (lower = spikier, more aggressive)
    mcts_dirichlet_epsilon: float = 0.5  # Root noise weight (50% noise! Force exploration)
    mcts_noise_decay: bool = True  # Decay noise over iterations (exploration -> exploitation)
    mcts_noise_min_epsilon: float = 0.15  # Minimum epsilon after decay (still some exploration)
    mcts_noise_decay_iterations: int = 1000  # Iterations to decay (effectively disabled for typical runs)
    policy_loss_weight: float = 1.0  # Weight of policy loss vs value loss

    # Hybrid training: mix fast 1-ply with quality MCTS games
    use_hybrid: bool = False  # Enable hybrid 1-ply + MCTS training
    hybrid_mcts_ratio: float = 0.2  # Fraction of games to use MCTS (0.2 = 20% MCTS, 80% 1-ply)
    hybrid_mcts_simulations: int = 50  # Simulations for hybrid MCTS games (lower for speed)

    # Asymmetric training: MCTS (teacher) vs 1-ply network (student) in same game
    # This prevents self-play collapse by having a stable teacher that doesn't degrade
    use_asymmetric: bool = False  # Enable asymmetric teacher-student training
    asymmetric_mcts_simulations: int = 50  # Simulations for the MCTS (teacher) side

    # Outcome value settings
    # Win = +1.0, Draw = -draw_penalty, Loss = -loss_penalty
    # Reasonable values that don't distort learning
    draw_penalty: float = 0.8  # Draws are mildly bad
    loss_penalty: float = 1.0  # Symmetric with wins
    repetition_penalty: float = 0.50  # Strong penalty per repeat (2 repeats = loss!)

    # Stalemate penalty - DISABLED by default (set stalemate_moves very high)
    # The asymmetric training mode handles draw prevention better
    stalemate_moves: int = 9999  # Effectively disabled
    stalemate_penalty: float = 1.0  # Same as loss if triggered

    # Combo bonus for attacks during active abilities (Royal Decree)
    # Incentivizes the AI to attack while buffs are active, not waste them
    combo_bonus_enabled: bool = True  # Enable combo attack bonus
    combo_bonus_per_attack: float = 0.15  # TRIPLED: Bonus per attack during Royal Decree
    combo_bonus_max: float = 0.60  # DOUBLED: Maximum combo bonus cap

    # New reward system (see RewardConfig for all values)
    use_new_rewards: bool = True  # Enable new aggression/shuffle/ability rewards

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

    def get_noise_epsilon(self, iteration: int) -> float:
        """Get MCTS noise epsilon for current iteration (decayed if enabled).

        Starts high (mcts_dirichlet_epsilon) for exploration, decays to
        mcts_noise_min_epsilon over mcts_noise_decay_iterations.
        """
        if not self.mcts_noise_decay:
            return self.mcts_dirichlet_epsilon

        if iteration >= self.mcts_noise_decay_iterations:
            return self.mcts_noise_min_epsilon

        ratio = iteration / self.mcts_noise_decay_iterations
        return self.mcts_dirichlet_epsilon - ratio * (self.mcts_dirichlet_epsilon - self.mcts_noise_min_epsilon)


@dataclass
class TrainingState:
    """Mutable training state for checkpointing."""
    iteration: int = 0
    best_eval_score: float = float('-inf')
    training_history: list[dict] = field(default_factory=list)


@dataclass
class RewardConfig:
    """
    All tunable reward values centralized for easy experimentation.

    This config shapes the AI's behavior during self-play training.
    All values are designed to work within the +/-1.0 outcome scale.
    """
    # Aggression rewards
    damage_reward_base: float = 0.01  # Per HP dealt
    damage_reward_late_multiplier: float = 2.0  # Scale at turn 100
    damage_reward_scale_turn: int = 100  # Turn at which multiplier maxes out
    attack_action_reward: float = 0.02  # Per attack action chosen

    # Shuffle penalties (exponential: base * mult^moves)
    shuffle_penalty_base: float = 0.01
    shuffle_penalty_multiplier: float = 2.0  # Doubles each consecutive move
    shuffle_penalty_max: float = 1.0  # Capped at loss equivalent

    # King hunt rewards
    king_proximity_reward: float = 0.02  # Per square closer to enemy king
    king_attack_opportunity_reward: float = 0.05  # Per piece that can attack king

    # Royal Decree (King ability)
    rd_activation_reward: float = 0.05  # Small reward for using RD
    rd_combo_bonus_per_attack: float = 0.15  # TRIPLED from 0.05
    rd_combo_bonus_max: float = 0.60  # DOUBLED from 0.30
    rd_waste_penalty: float = 0.10  # Penalty if RD expires with no attacks

    # Ability rewards
    interpose_damage_blocked_reward: float = 0.02  # Per damage blocked
    consecration_efficiency_reward: float = 0.10  # Max reward for efficient heal
    skirmish_reward: float = 0.03  # Per Skirmish use
    overextend_net_damage_reward: float = 0.05  # Per net damage (dealt - 2)
    pawn_advance_reward: float = 0.02  # Per advance
    pawn_promotion_reward: float = 0.20  # Big bonus for promotion
    pawn_threat_reward: float = 0.01  # Per threat square created

    # Game outcomes
    win_value: float = 1.0
    early_win_bonus: float = 0.5  # Added for wins before turn 50
    early_win_threshold: int = 50
    early_win_decay_end: int = 100
    loss_penalty: float = 1.0
    draw_penalty: float = 0.5  # Only for insufficient material draws now

    def calculate_damage_reward(self, damage: int, turn_number: int) -> float:
        """Calculate damage reward with late-game scaling."""
        multiplier = min(
            self.damage_reward_late_multiplier,
            1.0 + turn_number / self.damage_reward_scale_turn
        )
        return damage * self.damage_reward_base * multiplier

    def calculate_shuffle_penalty(self, consecutive_no_damage_moves: int) -> float:
        """Calculate exponential shuffle penalty."""
        if consecutive_no_damage_moves <= 0:
            return 0.0
        penalty = self.shuffle_penalty_base * (
            self.shuffle_penalty_multiplier ** consecutive_no_damage_moves
        )
        return min(penalty, self.shuffle_penalty_max)

    def calculate_win_value(self, turn_number: int) -> float:
        """Calculate win value with early-win bonus."""
        if turn_number <= self.early_win_threshold:
            return self.win_value + self.early_win_bonus
        elif turn_number >= self.early_win_decay_end:
            return self.win_value
        else:
            progress = (turn_number - self.early_win_threshold) / (
                self.early_win_decay_end - self.early_win_threshold
            )
            return self.win_value + self.early_win_bonus * (1.0 - progress)


# Default reward config instance
DEFAULT_REWARDS = RewardConfig()


@dataclass
class AbilityTracker:
    """
    Tracks ability usage and rewards for one side during a game.

    Used to calculate ability-specific rewards at game end.
    """
    # Royal Decree tracking
    rd_activations: int = 0  # Times RD was activated
    rd_attacks_during: int = 0  # Attacks made while RD active
    rd_wasted: int = 0  # Times RD expired with 0 attacks

    # Interpose tracking (reactive)
    interpose_triggers: int = 0  # Times Interpose triggered
    interpose_damage_blocked: int = 0  # Total damage absorbed by Rook

    # Consecration tracking
    consecration_uses: int = 0  # Times Consecration used
    consecration_hp_healed: int = 0  # Total HP healed
    consecration_efficiency_sum: float = 0.0  # Sum of (healed / missing) ratios

    # Skirmish tracking
    skirmish_uses: int = 0  # Times Skirmish used

    # Overextend tracking
    overextend_uses: int = 0  # Times Overextend used
    overextend_net_damage: int = 0  # Total (dealt - 2) across uses

    # Pawn tracking
    pawn_advances: int = 0  # Times Advance ability used
    pawn_promotions: int = 0  # Pawns promoted to Queen

    # Aggression tracking
    total_damage_dealt: int = 0  # All damage dealt
    attack_actions: int = 0  # Number of attack actions
    consecutive_no_damage: int = 0  # Current streak without damage
    max_consecutive_no_damage: int = 0  # Worst shuffle streak

    # King hunt tracking
    king_proximity_delta: float = 0.0  # Net change in distance to enemy king
    king_attack_opportunities: int = 0  # Moves where could attack king

    def calculate_total_reward(self, config: RewardConfig, turn_number: int) -> float:
        """Calculate total reward bonus from ability usage."""
        reward = 0.0

        # Royal Decree rewards
        reward += self.rd_activations * config.rd_activation_reward
        rd_combo = min(self.rd_attacks_during * config.rd_combo_bonus_per_attack, config.rd_combo_bonus_max)
        reward += rd_combo
        reward -= self.rd_wasted * config.rd_waste_penalty

        # Interpose rewards
        reward += self.interpose_damage_blocked * config.interpose_damage_blocked_reward

        # Consecration rewards (average efficiency)
        if self.consecration_uses > 0:
            avg_efficiency = self.consecration_efficiency_sum / self.consecration_uses
            reward += avg_efficiency * config.consecration_efficiency_reward

        # Skirmish rewards
        reward += self.skirmish_uses * config.skirmish_reward

        # Overextend rewards
        reward += max(0, self.overextend_net_damage) * config.overextend_net_damage_reward

        # Pawn rewards
        reward += self.pawn_advances * config.pawn_advance_reward
        reward += self.pawn_promotions * config.pawn_promotion_reward

        # Damage rewards (with late-game scaling)
        reward += config.calculate_damage_reward(self.total_damage_dealt, turn_number)

        # Attack action rewards
        reward += self.attack_actions * config.attack_action_reward

        # King hunt rewards
        # Proximity: reward for getting closer to enemy king (positive delta = closer)
        reward += self.king_proximity_delta * config.king_proximity_reward
        # Attack opportunities: reward for attacking the enemy king directly
        reward += self.king_attack_opportunities * config.king_attack_opportunity_reward

        # Shuffle penalty (exponential based on worst streak)
        reward -= config.calculate_shuffle_penalty(self.max_consecutive_no_damage)

        return reward


def _calculate_win_value(turn_number: int) -> float:
    """
    Calculate win value with early-win bonus.

    Winning early is rewarded more:
    - Win at move 50 or earlier: +1.5 (50% bonus)
    - Win at move 100 or later: +1.0 (base)
    - Linear scaling between

    This incentivizes decisive, aggressive play.
    """
    early_threshold = 50   # Full bonus before this
    late_threshold = 100   # No bonus after this
    max_bonus = 0.5        # Extra 50% for early wins

    if turn_number <= early_threshold:
        return 1.0 + max_bonus
    elif turn_number >= late_threshold:
        return 1.0
    else:
        # Linear decay from 1.5 to 1.0 over moves 50-100
        progress = (turn_number - early_threshold) / (late_threshold - early_threshold)
        return 1.0 + max_bonus * (1.0 - progress)


def _calculate_early_draw_penalty(turn_number: int, base_penalty: float = 0.8) -> float:
    """
    Calculate draw penalty with HARSH early-draw punishment.

    Drawing early is catastrophic - the AI should fight to the death!
    - Draw at turn 50 or earlier: -100.0 (catastrophic!)
    - Draw at turn 150 or later: -base_penalty (typically -0.8)
    - Linear decay between

    This strongly incentivizes decisive play and punishes early stalemates.
    """
    early_threshold = 50    # Maximum penalty before this
    late_threshold = 150    # Standard penalty after this
    max_penalty = 100.0     # Catastrophic early draw penalty

    if turn_number <= early_threshold:
        return -max_penalty
    elif turn_number >= late_threshold:
        return -base_penalty
    else:
        # Linear decay from -100 to -base_penalty over turns 50-150
        progress = (turn_number - early_threshold) / (late_threshold - early_threshold)
        return -max_penalty + progress * (max_penalty - base_penalty)


def _build_score_summary(
    moves_record: list[dict],
    winner: int | None,
    turn_number: int,
    stalemate: bool,
    draw_penalty: float,
    loss_penalty: float,
    white_combo_attacks: int = 0,
    black_combo_attacks: int = 0,
    combo_bonus_per_attack: float = 0.05,
    combo_bonus_max: float = 0.3,
    combo_bonus_enabled: bool = True,
) -> dict:
    """
    Build comprehensive score summary for debugging training signals.

    Returns dict with:
    - white_training_value: Final training value for White positions
    - black_training_value: Final training value for Black positions
    - outcome_breakdown: Detailed breakdown of how values were calculated
    - prediction_stats: Network prediction statistics per side
    """
    # Calculate base outcome value (from White's perspective)
    win_value = _calculate_win_value(turn_number)
    early_draw_penalty = _calculate_early_draw_penalty(turn_number, draw_penalty)

    if stalemate:
        base_outcome = -loss_penalty
        outcome_type = "stalemate"
    elif winner == 0:  # White wins
        base_outcome = win_value
        outcome_type = "white_win"
    elif winner == 1:  # Black wins
        base_outcome = -loss_penalty
        outcome_type = "black_win"
    else:
        base_outcome = early_draw_penalty
        outcome_type = "draw"

    # Calculate combo bonuses
    white_combo_bonus = 0.0
    black_combo_bonus = 0.0
    if combo_bonus_enabled:
        white_combo_bonus = min(white_combo_attacks * combo_bonus_per_attack, combo_bonus_max)
        black_combo_bonus = min(black_combo_attacks * combo_bonus_per_attack, combo_bonus_max)

    # Net combo from White's perspective
    combo_net = white_combo_bonus - black_combo_bonus

    # Final training values
    white_training_value = base_outcome + combo_net
    black_training_value = -white_training_value

    # Collect prediction stats from moves
    white_predictions = []
    black_predictions = []
    for move in moves_record:
        if "value" in move and move["value"] is not None:
            if move.get("team") == 0 or move.get("team") == "White":
                white_predictions.append(move["value"])
            else:
                black_predictions.append(move["value"])

    return {
        # Final training values
        "white_training_value": round(white_training_value, 4),
        "black_training_value": round(black_training_value, 4),

        # Outcome breakdown
        "outcome_breakdown": {
            "outcome_type": outcome_type,
            "base_outcome_white": round(base_outcome, 4),
            "win_bonus_applied": round(win_value - 1.0, 4) if winner == 0 else 0.0,
            "early_draw_penalty": round(early_draw_penalty, 4) if winner is None else None,
            "combo_bonus_white": round(white_combo_bonus, 4),
            "combo_bonus_black": round(black_combo_bonus, 4),
            "combo_net": round(combo_net, 4),
        },

        # Network prediction stats (what the engine "thought")
        "prediction_stats": {
            "white_moves": len(white_predictions),
            "black_moves": len(black_predictions),
            "white_avg_prediction": round(sum(white_predictions) / len(white_predictions), 4) if white_predictions else None,
            "black_avg_prediction": round(sum(black_predictions) / len(black_predictions), 4) if black_predictions else None,
            "white_min_prediction": round(min(white_predictions), 4) if white_predictions else None,
            "white_max_prediction": round(max(white_predictions), 4) if white_predictions else None,
            "black_min_prediction": round(min(black_predictions), 4) if black_predictions else None,
            "black_max_prediction": round(max(black_predictions), 4) if black_predictions else None,
        },
    }


def _calculate_draw_value(
    player_hp: float,
    enemy_hp: float,
    player_damage: float = 0,
    enemy_damage: float = 0,
    draw_penalty: float = 2.0,
) -> float:
    """
    Calculate the value for a drawn game using Dynamic Contempt.

    Uses material balance (HP difference) to adjust draw penalty:
    - If we were winning (more HP): harsh penalty (threw away a win)
    - If we were losing (less HP): mild penalty (good save)
    - If even: moderate penalty

    Args:
        player_hp: Total HP remaining for player's pieces
        enemy_hp: Total HP remaining for enemy's pieces
        player_damage: Total damage dealt by player (unused, kept for compatibility)
        enemy_damage: Total damage dealt by enemy (unused, kept for compatibility)
        draw_penalty: Base penalty for draws (default 0.7)

    Returns:
        Value based on material advantage:
        - Big advantage (>30 HP): -0.95 (almost a loss - "you threw away a win!")
        - Winning (>15 HP): -0.8 ("should have closed it out")
        - Slight advantage (>5 HP): -0.7 (base penalty)
        - Even: -0.6 ("fight harder")
        - Slight disadvantage (<-5 HP): -0.5 ("decent save")
        - Losing (<-15 HP): -0.4 ("good save")
        - Big disadvantage (<-30 HP): -0.3 ("great save!")
    """
    material_advantage = player_hp - enemy_hp

    if material_advantage > 30:
        # Was winning big - this is almost as bad as losing!
        return -0.95
    elif material_advantage > 15:
        # Was winning - should have closed it out
        return -0.8
    elif material_advantage > 5:
        # Slight advantage - base penalty
        return -draw_penalty
    elif material_advantage > -5:
        # Even position - moderate penalty
        return -0.6
    elif material_advantage > -15:
        # Slight disadvantage - decent save
        return -0.5
    elif material_advantage > -30:
        # Was losing - good save
        return -0.4
    else:
        # Was losing big - great save!
        return -0.3


def _get_hp_totals_rust(sim) -> tuple[float, float]:
    """Get total HP for each team from Rust simulator.

    Returns:
        Tuple of (player_hp, enemy_hp) where player = White (team 0)
    """
    player_hp = 0.0
    enemy_hp = 0.0
    for piece in sim.get_pieces():
        if piece.current_hp > 0:
            if piece.team == 0:  # White/Player
                player_hp += piece.current_hp
            else:  # Black/Enemy
                enemy_hp += piece.current_hp
    return player_hp, enemy_hp


def _get_hp_totals_python(state) -> tuple[float, float]:
    """Get total HP for each team from Python GameState.

    Returns:
        Tuple of (player_hp, enemy_hp) where player = Team.PLAYER
    """
    from .game_state import Team
    player_hp = 0.0
    enemy_hp = 0.0
    for piece in state.pieces:
        if piece.current_hp > 0:
            if piece.team == Team.PLAYER:
                player_hp += piece.current_hp
            else:
                enemy_hp += piece.current_hp
    return player_hp, enemy_hp


def _play_random_game(seed: int) -> list[tuple[np.ndarray, float]]:
    """Worker function for parallel random game generation."""
    sim = GameSimulator()
    sim.set_seed(seed)
    # Alternate color perspective based on seed for balanced training
    sim.state.playing_as = Team.PLAYER if seed % 2 == 0 else Team.ENEMY
    _, data = sim.play_random_game()
    return data


def _play_self_play_game(args: tuple[int, str, float, float, float]) -> tuple[list[tuple[np.ndarray, float]], dict]:
    """Worker function for parallel self-play game generation. Returns training data AND game replay."""
    seed, model_path, temperature, draw_penalty, loss_penalty = args

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

        # Execute move and track results
        move_result = sim.make_move(selected_move)
        move_record["damage"] = move_result["damage"]
        move_record["interpose_blocked"] = move_result.get("interpose_blocked", 0)
        move_record["consecration_heal"] = move_result.get("consecration_heal", 0)
        move_record["promotion"] = move_result.get("promotion", False)
        moves_record.append(move_record)

        # Track damage by team
        damage = move_result["damage"]
        if team == Team.PLAYER:
            player_damage_dealt += damage
        else:
            enemy_damage_dealt += damage

    # Determine outcome with early-win bonus
    win_value = _calculate_win_value(sim.state.turn_number)
    if sim.state.winner == Team.PLAYER:
        player_value = win_value
    elif sim.state.winner == Team.ENEMY:
        player_value = -loss_penalty
    else:
        # Draw: use early-draw penalty (catastrophic early, mild late)
        player_value = _calculate_early_draw_penalty(sim.state.turn_number, draw_penalty)

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
    draw_penalty: float = 1.5,
    loss_penalty: float = 2.0,
    repetition_penalty: float = 0.30,
    stalemate_moves: int = 10,
    stalemate_penalty: float = 10.0,
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
        stalemate: bool = False  # True if game ended due to stalemate

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

                # Apply boredom penalty for moves that lead to repeated positions
                if repetition_penalty > 0:
                    current_history = game.sim.state.position_history.copy()
                    for i, move in enumerate(moves):
                        # Simulate the move to get resulting position hash
                        sim_copy = GameSimulator(game.sim.state.clone())
                        sim_copy.make_move(move)
                        new_hash = sim_copy.state.position_hash()
                        rep_count = current_history.count(new_hash)
                        if rep_count > 0:
                            move_values[i] -= rep_count * repetition_penalty

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
                move_result = game.sim.make_move(selected_move)
                move_record["damage"] = move_result["damage"]
                move_record["interpose_blocked"] = move_result.get("interpose_blocked", 0)
                move_record["consecration_heal"] = move_result.get("consecration_heal", 0)
                move_record["promotion"] = move_result.get("promotion", False)
                game.moves_record.append(move_record)

                # Track damage
                damage = move_result["damage"]
                if team == Team.PLAYER:
                    game.player_damage += damage
                else:
                    game.enemy_damage += damage

                # Check if game ended
                if game.sim.state.is_terminal:
                    game.finished = True
                    games_completed += 1
                    pbar.update(1)
                # Check stalemate (N moves without damage = cowardice)
                elif game.sim.state.moves_without_damage >= stalemate_moves:
                    game.finished = True
                    game.stalemate = True
                    games_completed += 1
                    pbar.update(1)

    pbar.close()

    # Convert finished games to training data and replays
    all_training_data: list[tuple[np.ndarray, float]] = []
    all_replays: list[dict] = []

    for game in games:
        # Determine outcome
        # Determine outcome with early-win bonus
        win_value = _calculate_win_value(game.sim.state.turn_number)

        if game.stalemate:
            player_value = -stalemate_penalty
        elif game.sim.state.winner == Team.PLAYER:
            player_value = win_value
        elif game.sim.state.winner == Team.ENEMY:
            player_value = -loss_penalty
        else:
            # Draw: use early-draw penalty (catastrophic early, mild late)
            player_value = _calculate_early_draw_penalty(game.sim.state.turn_number, draw_penalty)

        # Convert to training data
        for tensor, team_int in game.history:
            value = player_value if team_int == 0 else -player_value
            all_training_data.append((tensor, value))

        # Build score summary
        winner_int = int(game.sim.state.winner) if game.sim.state.winner is not None else None
        score_summary = _build_score_summary(
            moves_record=game.moves_record,
            winner=winner_int,
            turn_number=game.sim.state.turn_number,
            stalemate=game.stalemate,
            draw_penalty=draw_penalty,
            loss_penalty=loss_penalty,
            combo_bonus_enabled=False,  # Not tracked in this mode
        )

        # Build replay
        replay = {
            "moves": game.moves_record,
            "initial_state": game.initial_state,
            "final_state": game.sim.state.to_json(),
            "winner": winner_int,
            "total_turns": game.sim.state.turn_number,
            "seed": game.seed,
            "stalemate": game.stalemate,
            "score_summary": score_summary,
        }
        all_replays.append(replay)

    return all_training_data, all_replays


def play_rust_self_play_games(
    network: nn.Module,
    device: torch.device,
    num_games: int,
    temperature: float,
    base_seed: int = 0,
    draw_penalty: float = 1.5,
    loss_penalty: float = 2.0,
    repetition_penalty: float = 0.30,
    stalemate_moves: int = 10,
    stalemate_penalty: float = 10.0,
) -> tuple[list[tuple[np.ndarray, float]], list[dict]]:
    """
    Play multiple self-play games using Rust simulator with 1-ply network evaluation.

    This is the fastest training mode - uses Rust for all game simulation
    and batches network evaluation on GPU. No MCTS search, just direct
    1-ply evaluation of all legal moves.

    Great for training a baseline "easy" mode AI using heuristic-guided play.

    Returns:
        Tuple of (all_training_data, all_replay_data)
    """
    from .mcts_rust import check_rust_available, rust_state_to_tensor
    check_rust_available()
    from exchange_simulator import PySimulator as RustSimulator

    network.eval()

    @dataclass
    class RustGameInstance:
        sim: RustSimulator
        history: list[tuple[np.ndarray, int]]  # (tensor, team)
        moves_record: list[dict]
        initial_state: dict
        player_damage: int = 0
        enemy_damage: int = 0
        seed: int = 0
        playing_as: int = 0
        finished: bool = False
        stalemate: bool = False  # True if game ended due to stalemate (no damage for N moves)

    # Initialize all games
    games: list[RustGameInstance] = []
    for i in range(num_games):
        sim = RustSimulator()
        sim.set_seed(base_seed + i)
        playing_as = 0 if i % 2 == 0 else 1
        sim.set_playing_as("White" if playing_as == 0 else "Black")

        games.append(RustGameInstance(
            sim=sim,
            history=[],
            moves_record=[],
            initial_state=_rust_sim_to_json(sim),
            seed=base_seed + i,
            playing_as=playing_as,
        ))

    # Play until all games are done
    games_completed = 0
    pbar = tqdm(total=num_games, desc="  Playing Rust games (1-ply)", leave=False, ncols=80)

    with torch.no_grad():
        while any(not g.finished for g in games):
            # Collect all positions that need evaluation from all active games
            # Structure: (game_idx, move_idx, tensor, rust_move)
            all_evaluations: list[tuple[int, int, np.ndarray, Any]] = []
            game_move_counts: list[int] = []

            for game_idx, game in enumerate(games):
                if game.finished:
                    game_move_counts.append(0)
                    continue

                # Check terminal
                if game.sim.is_terminal():
                    game.finished = True
                    games_completed += 1
                    pbar.update(1)
                    game_move_counts.append(0)
                    continue

                # Check stalemate (N moves without damage = cowardice)
                if game.sim.moves_without_damage() >= stalemate_moves:
                    game.finished = True
                    game.stalemate = True
                    games_completed += 1
                    pbar.update(1)
                    game_move_counts.append(0)
                    continue

                # Record current state for training
                tensor = rust_state_to_tensor(game.sim)
                team = game.sim.side_to_move()
                game.history.append((tensor, team_to_int(team)))

                # Generate moves using Rust
                rust_moves = game.sim.get_legal_moves(game.sim.side_to_move())
                if not rust_moves:
                    game.finished = True
                    games_completed += 1
                    pbar.update(1)
                    game_move_counts.append(0)
                    continue

                game_move_counts.append(len(rust_moves))

                # Simulate each move to get resulting position
                for move_idx, rust_move in enumerate(rust_moves):
                    sim_copy = game.sim.clone()
                    sim_copy.make_move(rust_move)
                    all_evaluations.append((game_idx, move_idx, rust_state_to_tensor(sim_copy), rust_move))

            if not all_evaluations:
                break

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
                # Negate because we evaluated from opponent's perspective
                move_values = [-batch_values[eval_idx + i] for i in range(num_moves)]
                rust_moves = [all_evaluations[eval_idx + i][3] for i in range(num_moves)]
                eval_idx += num_moves

                # Apply boredom penalty for moves that lead to repeated positions
                if repetition_penalty > 0:
                    for i in range(num_moves):
                        rep_count = game.sim.get_repetition_count(i)
                        if rep_count > 0:
                            # Penalize based on how many times this position was seen
                            move_values[i] -= rep_count * repetition_penalty

                # Select move with temperature
                if temperature > 0:
                    values = np.array(move_values)
                    values = values - values.max()
                    exp_values = np.exp(values / temperature)
                    probs = exp_values / (exp_values.sum() + 1e-10)
                    move_idx = np.random.choice(len(rust_moves), p=probs)
                else:
                    move_idx = int(np.argmax(move_values))

                best_move = rust_moves[move_idx]
                team = game.sim.side_to_move()

                # Get piece type from simulator
                pieces = game.sim.get_pieces()
                actual_piece_type = pieces[best_move.piece_idx].piece_type if best_move.piece_idx < len(pieces) else 0

                # Record the move
                move_record = {
                    "turn": game.sim.turn_number(),
                    "team": team_to_int(team),
                    "piece_type": piece_type_to_int(actual_piece_type),
                    "move_type": move_type_to_int(best_move.move_type),
                    "from": [best_move.from_x, best_move.from_y],
                    "to": [best_move.to_x, best_move.to_y],
                    "attack": [best_move.attack_x, best_move.attack_y] if best_move.attack_x is not None else None,
                    "ability_id": ability_id_to_int(best_move.ability_id) if best_move.ability_id is not None else None,
                    "ability_target": [best_move.ability_target_x, best_move.ability_target_y] if best_move.ability_target_x is not None else None,
                    "value": float(move_values[move_idx]),
                    "top_moves": len(rust_moves),
                }

                # Execute move
                damage, _ = game.sim.make_move(best_move)
                move_record["damage"] = damage
                game.moves_record.append(move_record)

                # Track damage
                if team == "White":
                    game.player_damage += damage
                else:
                    game.enemy_damage += damage

                # Check if game ended
                if game.sim.is_terminal():
                    game.finished = True
                    games_completed += 1
                    pbar.update(1)

    pbar.close()

    # Convert finished games to training data and replays
    all_training_data: list[tuple[np.ndarray, float]] = []
    all_replays: list[dict] = []

    for game in games:
        # Determine outcome with early-win bonus
        win_value = _calculate_win_value(game.sim.turn_number())

        if game.stalemate:
            player_value = -stalemate_penalty
        elif game.sim.winner() == "White":  # Player/White wins
            player_value = win_value
        elif game.sim.winner() == "Black":  # Enemy/Black wins
            player_value = -loss_penalty
        else:
            # Draw: use early-draw penalty (catastrophic early, mild late)
            player_value = _calculate_early_draw_penalty(game.sim.turn_number(), draw_penalty)

        # Convert to training data
        for tensor, team_int in game.history:
            value = player_value if team_int == 0 else -player_value
            all_training_data.append((tensor, value))

        # Build score summary
        score_summary = _build_score_summary(
            moves_record=game.moves_record,
            winner=game.sim.winner(),
            turn_number=game.sim.turn_number(),
            stalemate=game.stalemate,
            draw_penalty=draw_penalty,
            loss_penalty=loss_penalty,
            combo_bonus_enabled=False,  # Not tracked in this mode
        )

        # Build replay
        replay = {
            "moves": game.moves_record,
            "initial_state": game.initial_state,
            "final_state": _rust_sim_to_json(game.sim),
            "winner": game.sim.winner(),
            "total_turns": game.sim.turn_number(),
            "seed": game.seed,
            "rust_accelerated": True,
            "mode": "1-ply",
            "stalemate": game.stalemate,
            "score_summary": score_summary,
        }
        all_replays.append(replay)

    return all_training_data, all_replays


def play_mcts_self_play_games(
    network: nn.Module,
    device: torch.device,
    mcts_config: MCTSConfig,
    num_games: int,
    base_seed: int = 0,
    draw_penalty: float = 1.5,
    loss_penalty: float = 2.0,
    stalemate_moves: int = 10,
    stalemate_penalty: float = 10.0,
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
        stalemate: bool = False  # True if game ended due to stalemate

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
            move_result = game.sim.make_move(actual_move)
            move_record["damage"] = move_result["damage"]
            move_record["interpose_blocked"] = move_result.get("interpose_blocked", 0)
            move_record["consecration_heal"] = move_result.get("consecration_heal", 0)
            move_record["promotion"] = move_result.get("promotion", False)
            game.moves_record.append(move_record)

            # Track damage
            damage = move_result["damage"]
            if team == Team.PLAYER:
                game.player_damage += damage
            else:
                game.enemy_damage += damage

            # Check if game ended
            if game.sim.state.is_terminal:
                game.finished = True
                games_completed += 1
                pbar.update(1)
            # Check stalemate (N moves without damage = cowardice)
            elif game.sim.state.moves_without_damage >= stalemate_moves:
                game.finished = True
                game.stalemate = True
                games_completed += 1
                pbar.update(1)

    pbar.close()

    # Convert to training data with policy targets
    all_training_data: list[tuple[np.ndarray, float, np.ndarray]] = []
    all_replays: list[dict] = []

    for game in games:
        # Determine outcome with early-win bonus
        win_value = _calculate_win_value(game.sim.state.turn_number)

        if game.stalemate:
            player_value = -stalemate_penalty
        elif game.sim.state.winner == Team.PLAYER:
            player_value = win_value
        elif game.sim.state.winner == Team.ENEMY:
            player_value = -loss_penalty
        else:
            # Draw: use early-draw penalty (catastrophic early, mild late)
            player_value = _calculate_early_draw_penalty(game.sim.state.turn_number, draw_penalty)

        # Convert to training data with policy
        for tensor, team_int, policy_target in game.history:
            value = player_value if team_int == 0 else -player_value
            all_training_data.append((tensor, value, policy_target))

        # Build score summary
        winner_int = int(game.sim.state.winner) if game.sim.state.winner is not None else None
        score_summary = _build_score_summary(
            moves_record=game.moves_record,
            winner=winner_int,
            turn_number=game.sim.state.turn_number,
            stalemate=game.stalemate,
            draw_penalty=draw_penalty,
            loss_penalty=loss_penalty,
            combo_bonus_enabled=False,  # Not tracked in this mode
        )

        # Build replay
        replay = {
            "moves": game.moves_record,
            "initial_state": game.initial_state,
            "final_state": game.sim.state.to_json(),
            "winner": winner_int,
            "total_turns": game.sim.state.turn_number,
            "seed": game.seed,
            "mcts_simulations": mcts_config.num_simulations,
            "stalemate": game.stalemate,
            "score_summary": score_summary,
        }
        all_replays.append(replay)

    return all_training_data, all_replays


def play_mcts_rust_self_play_games(
    network: nn.Module,
    device: torch.device,
    mcts_config: MCTSConfig,
    num_games: int,
    base_seed: int = 0,
    draw_penalty: float = 1.5,
    loss_penalty: float = 2.0,
    stalemate_moves: int = 10,
    stalemate_penalty: float = 10.0,
) -> tuple[list[tuple[np.ndarray, float, np.ndarray]], list[dict]]:
    """
    Play self-play games using MCTS with Rust simulator.

    Much faster than Python MCTS due to native game simulation.

    Returns:
        Tuple of (all_training_data, all_replay_data)
        Training data is list of (state_tensor, value_target, policy_target)
    """
    from exchange_simulator import PySimulator as RustSimulator

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
        stalemate: bool = False  # True if game ended due to stalemate

    # Initialize games
    games: list[RustMCTSGameInstance] = []
    for i in range(num_games):
        sim = RustSimulator()
        sim.set_seed(base_seed + i)
        playing_as = 0 if i % 2 == 0 else 1
        sim.set_playing_as("White" if playing_as == 0 else "Black")

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
    pbar = tqdm(total=num_games, desc="  Playing Rust MCTS games", leave=True, ncols=80)

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
            team = game.sim.side_to_move()

            # Get legal moves for policy target creation
            rust_moves = game.sim.get_legal_moves(game.sim.side_to_move())

            # Apply repetition penalty and aggression bonus BEFORE recording policy target
            # This ensures the network learns that attacks are good, not just that we select them
            # rep_count includes the move we'd make, so:
            #   rep_count=1: first occurrence (fine)
            #   rep_count=2: second occurrence (penalize)
            #   rep_count=3: third occurrence = DRAW (block!)
            adjusted_dist = None
            if len(visit_dist) > 0 and len(rust_moves) > 0:
                adjusted_dist = visit_dist.copy()

                for i in range(min(len(rust_moves), len(adjusted_dist))):
                    move = rust_moves[i]

                    # Repetition penalty
                    rep_count = game.sim.get_repetition_count(i)
                    if rep_count >= 3:
                        # Would cause threefold repetition (draw) - BLOCK completely
                        adjusted_dist[i] = 0.0
                        continue
                    elif rep_count == 2:
                        # Second occurrence - heavy penalty (10% of original)
                        adjusted_dist[i] *= 0.1

                    # Note: No artificial attack bonus - let MCTS find attacks naturally
                    # The asymmetric training mode (MCTS vs 1-ply) handles exploration

                # Re-normalize
                if adjusted_dist.sum() > 0:
                    adjusted_dist /= adjusted_dist.sum()
                    best_move_idx = np.argmax(adjusted_dist)
                    if best_move_idx < len(rust_moves):
                        best_move = rust_moves[best_move_idx]
                # If all moves lead to repetition, fall back to original best_move

            # Create policy target from ADJUSTED distribution (not raw MCTS visits)
            # This teaches the network that attacks are good, enabling the feedback loop
            policy_target = np.zeros(POLICY_SIZE, dtype=np.float32)
            if len(visit_dist) > 0 and len(rust_moves) > 0:
                # Use adjusted_dist if we computed it, otherwise fall back to visit_dist
                dist_to_use = adjusted_dist if adjusted_dist is not None and adjusted_dist.sum() > 0 else visit_dist
                for i, rm in enumerate(rust_moves):
                    if i < len(dist_to_use):
                        from .mcts_rust import rust_move_to_policy_index
                        move_idx = rust_move_to_policy_index(rm)
                        if 0 <= move_idx < POLICY_SIZE:
                            policy_target[move_idx] = dist_to_use[i]

            game.history.append((tensor, team_to_int(team), policy_target))

            if best_move is None:
                # No legal moves - game over
                game.finished = True
                games_completed += 1
                pbar.update(1)
                continue

            # Record move - get actual piece type from simulator
            pieces = game.sim.get_pieces()
            actual_piece_type = pieces[best_move.piece_idx].piece_type if best_move.piece_idx < len(pieces) else "King"

            move_record = {
                "turn": game.sim.turn_number(),
                "team": team_to_int(team),
                "piece_type": piece_type_to_int(actual_piece_type),
                "move_type": move_type_to_int(best_move.move_type),
                "from": [best_move.from_x, best_move.from_y],
                "to": [best_move.to_x, best_move.to_y],
                "attack": [best_move.attack_x, best_move.attack_y] if best_move.attack_x is not None else None,
                "ability_id": ability_id_to_int(best_move.ability_id) if best_move.ability_id is not None else None,
                "ability_target": [best_move.ability_target_x, best_move.ability_target_y] if best_move.ability_target_x is not None else None,
                "mcts_visits": int(visit_dist.max() * mcts_config.num_simulations) if len(visit_dist) > 0 else 0,
            }

            # Execute move
            damage, _ = game.sim.make_move(best_move)
            move_record["damage"] = damage
            game.moves_record.append(move_record)

            # Track damage
            if team == "White":
                game.player_damage += damage
            else:
                game.enemy_damage += damage

            # Check if game ended
            if game.sim.is_terminal():
                game.finished = True
                games_completed += 1
                pbar.update(1)
            # Check stalemate (N moves without damage = cowardice)
            elif game.sim.moves_without_damage() >= stalemate_moves:
                game.finished = True
                game.stalemate = True
                games_completed += 1
                pbar.update(1)

    pbar.close()

    # Convert to training data with policy targets
    all_training_data: list[tuple[np.ndarray, float, np.ndarray]] = []
    all_replays: list[dict] = []

    for game in games:
        # Determine outcome with early-win bonus
        win_value = _calculate_win_value(game.sim.turn_number())

        if game.stalemate:
            player_value = -stalemate_penalty
        elif game.sim.winner() == "White":  # Player wins
            player_value = win_value
        elif game.sim.winner() == "Black":  # Enemy wins
            player_value = -loss_penalty
        else:
            # Draw: use early-draw penalty (catastrophic early, mild late)
            player_value = _calculate_early_draw_penalty(game.sim.turn_number(), draw_penalty)

        # Convert to training data with policy
        for tensor, team_int, policy_target in game.history:
            value = player_value if team_int == 0 else -player_value
            all_training_data.append((tensor, value, policy_target))

        # Build score summary
        score_summary = _build_score_summary(
            moves_record=game.moves_record,
            winner=game.sim.winner(),
            turn_number=game.sim.turn_number(),
            stalemate=game.stalemate,
            draw_penalty=draw_penalty,
            loss_penalty=loss_penalty,
            combo_bonus_enabled=False,  # Not tracked in this mode
        )

        # Build replay
        replay = {
            "moves": game.moves_record,
            "initial_state": game.initial_state,
            "final_state": _rust_sim_to_json(game.sim),
            "winner": game.sim.winner(),
            "total_turns": game.sim.turn_number(),
            "seed": game.seed,
            "mcts_simulations": mcts_config.num_simulations,
            "rust_accelerated": True,
            "stalemate": game.stalemate,
            "score_summary": score_summary,
        }
        all_replays.append(replay)

    return all_training_data, all_replays


def _execute_asymmetric_move(game, tensor, team, best_move, rust_moves, policy_target, combo_bonus_enabled, use_new_rewards=True):
    """Helper to execute a move and record training data for asymmetric games."""
    # Record state for training
    game.history.append((tensor, team_to_int(team), policy_target))

    # Get piece info BEFORE move for tracking
    pieces_before = game.sim.get_pieces()
    actual_piece_type = pieces_before[best_move.piece_idx].piece_type if best_move.piece_idx < len(pieces_before) else "King"

    # For king hunt tracking: find enemy king and compute distance before move
    enemy_king_pos = None
    piece_pos_before = None
    consecration_target_hp_before = None

    if use_new_rewards:
        enemy_team = "Black" if team == "White" else "White"
        for p in pieces_before:
            if p.piece_type == "King" and p.team == enemy_team and p.current_hp > 0:
                enemy_king_pos = (p.x, p.y)
                break

        # Get moving piece position
        moving_piece = pieces_before[best_move.piece_idx] if best_move.piece_idx < len(pieces_before) else None
        if moving_piece:
            piece_pos_before = (moving_piece.x, moving_piece.y)

        # For consecration: track target HP before
        if best_move.ability_id == "Consecration" and best_move.ability_target_x is not None:
            for p in pieces_before:
                if p.x == best_move.ability_target_x and p.y == best_move.ability_target_y:
                    consecration_target_hp_before = (p.current_hp, p.max_hp)
                    break

    # Record the move
    move_record = {
        "turn": game.sim.turn_number(),
        "team": team_to_int(team),
        "piece_type": piece_type_to_int(actual_piece_type),
        "move_type": move_type_to_int(best_move.move_type),
        "from": [best_move.from_x, best_move.from_y],
        "to": [best_move.to_x, best_move.to_y],
        "attack": [best_move.attack_x, best_move.attack_y] if best_move.attack_x is not None else None,
        "ability_id": ability_id_to_int(best_move.ability_id) if best_move.ability_id is not None else None,
        "ability_target": [best_move.ability_target_x, best_move.ability_target_y] if best_move.ability_target_x is not None else None,
        "mcts_side": team == ("White" if game.mcts_side == 0 else "Black"),
    }

    # Get tracker for this team
    tracker = game.white_tracker if team == "White" else game.black_tracker

    # Check if Royal Decree is active BEFORE the move (for combo tracking)
    white_rd_before = game.sim.white_royal_decree_turns()
    black_rd_before = game.sim.black_royal_decree_turns()
    rd_active_before = (white_rd_before > 0 if team == "White" else black_rd_before > 0) if combo_bonus_enabled else False

    # Track ability activation BEFORE move executes
    if use_new_rewards and best_move.ability_id is not None:
        ability_id = best_move.ability_id
        if ability_id == "RoyalDecree":
            tracker.rd_activations += 1
            # Reset attack counter for this activation
            if team == "White":
                game.white_rd_attacks_this_activation = 0
            else:
                game.black_rd_attacks_this_activation = 0
        elif ability_id == "Skirmish":
            tracker.skirmish_uses += 1
        elif ability_id == "Advance":
            tracker.pawn_advances += 1
        elif ability_id == "Consecration":
            tracker.consecration_uses += 1
        elif ability_id == "Overextend":
            tracker.overextend_uses += 1

    # Execute move and get result
    damage, interpose_blocked = game.sim.make_move(best_move)
    move_record["damage"] = damage
    move_record["interpose_blocked"] = interpose_blocked
    game.moves_record.append(move_record)

    # Get pieces AFTER move for comparison
    pieces_after = game.sim.get_pieces() if use_new_rewards else None

    # Track damage and abilities for new reward system
    if use_new_rewards:
        # Track total damage
        tracker.total_damage_dealt += damage

        # Track interpose damage blocked (for defender's tracker)
        if interpose_blocked > 0:
            defender_tracker = game.black_tracker if team == "White" else game.white_tracker
            defender_tracker.interpose_triggers += 1
            defender_tracker.interpose_damage_blocked += interpose_blocked

        # Track attack actions
        is_attack = best_move.move_type in ("Attack", "MoveAndAttack") or (
            best_move.ability_id in ("Overextend", "Skirmish") and damage > 0
        )
        if is_attack:
            tracker.attack_actions += 1

        # Track shuffle streaks (moves without damage)
        if damage > 0:
            tracker.consecutive_no_damage = 0
        else:
            tracker.consecutive_no_damage += 1
            tracker.max_consecutive_no_damage = max(
                tracker.max_consecutive_no_damage,
                tracker.consecutive_no_damage
            )

        # Track Overextend net damage
        if best_move.ability_id == "Overextend":
            tracker.overextend_net_damage += (damage - 2)

        # === KING HUNT TRACKING ===
        # Track how much closer we got to the enemy king
        if enemy_king_pos and piece_pos_before:
            dist_before = abs(piece_pos_before[0] - enemy_king_pos[0]) + abs(piece_pos_before[1] - enemy_king_pos[1])
            # Get piece position after move
            moving_piece_after = pieces_after[best_move.piece_idx] if best_move.piece_idx < len(pieces_after) else None
            if moving_piece_after and moving_piece_after.current_hp > 0:
                piece_pos_after = (moving_piece_after.x, moving_piece_after.y)
                dist_after = abs(piece_pos_after[0] - enemy_king_pos[0]) + abs(piece_pos_after[1] - enemy_king_pos[1])
                # Positive delta = got closer to enemy king
                tracker.king_proximity_delta += (dist_before - dist_after)

        # Track attack opportunities on enemy king (if we CAN attack king next turn)
        if is_attack and enemy_king_pos:
            # Check if this attack was on or near the king
            attack_pos = None
            if best_move.move_type == "Attack":
                attack_pos = (best_move.to_x, best_move.to_y)
            elif best_move.attack_x is not None:
                attack_pos = (best_move.attack_x, best_move.attack_y)
            if attack_pos and attack_pos == enemy_king_pos:
                tracker.king_attack_opportunities += 1

        # === PAWN PROMOTION TRACKING ===
        if actual_piece_type == "Pawn" and pieces_after:
            moving_piece_after = pieces_after[best_move.piece_idx] if best_move.piece_idx < len(pieces_after) else None
            if moving_piece_after and moving_piece_after.piece_type == "Queen":
                tracker.pawn_promotions += 1

        # === CONSECRATION EFFICIENCY TRACKING ===
        if best_move.ability_id == "Consecration" and consecration_target_hp_before:
            old_hp, max_hp = consecration_target_hp_before
            missing_before = max_hp - old_hp
            if missing_before > 0:
                # Find target piece after
                for p in pieces_after:
                    if p.x == best_move.ability_target_x and p.y == best_move.ability_target_y:
                        heal_amount = p.current_hp - old_hp
                        efficiency = min(1.0, heal_amount / missing_before) if missing_before > 0 else 0.0
                        tracker.consecration_efficiency_sum += efficiency
                        tracker.consecration_hp_healed += heal_amount
                        break

        # Track RD attacks for waste penalty
        if rd_active_before and damage > 0:
            if team == "White":
                game.white_rd_attacks_this_activation += 1
                tracker.rd_attacks_during += 1
            else:
                game.black_rd_attacks_this_activation += 1
                tracker.rd_attacks_during += 1

        # Check if RD just expired (was active, now 0)
        white_rd_after = game.sim.white_royal_decree_turns()
        black_rd_after = game.sim.black_royal_decree_turns()

        # White RD expired after this move
        if white_rd_before > 0 and white_rd_after == 0:
            if game.white_rd_attacks_this_activation == 0:
                game.white_tracker.rd_wasted += 1

        # Black RD expired after this move
        if black_rd_before > 0 and black_rd_after == 0:
            if game.black_rd_attacks_this_activation == 0:
                game.black_tracker.rd_wasted += 1

    # Track damage and combo attacks (legacy)
    if team == "White":
        game.player_damage += damage
        if rd_active_before and damage > 0:
            game.white_combo_attacks += 1
    else:
        game.enemy_damage += damage
        if rd_active_before and damage > 0:
            game.black_combo_attacks += 1


def play_asymmetric_self_play_games(
    network: nn.Module,
    device: torch.device,
    mcts_config: MCTSConfig,
    num_games: int,
    base_seed: int = 0,
    draw_penalty: float = 0.8,
    loss_penalty: float = 1.0,
    combo_bonus_enabled: bool = True,
    combo_bonus_per_attack: float = 0.05,
    combo_bonus_max: float = 0.3,
    reward_config: Optional[RewardConfig] = None,
    use_new_rewards: bool = True,
) -> tuple[list[tuple[np.ndarray, float, np.ndarray]], list[dict]]:
    """
    Play self-play games with asymmetric move selection:
    - One side uses MCTS (teacher) - strong, principled play
    - Other side uses 1-ply network eval (student) - learning from teacher

    This prevents self-play collapse by having a stable teacher that doesn't
    degrade with training. The MCTS side consistently wins, providing clear
    learning signal.

    Games alternate which side gets MCTS to ensure both colors learn.

    Returns:
        Tuple of (all_training_data, all_replay_data)
        Training data is list of (state_tensor, value_target, policy_target)
    """
    from .mcts_rust import BatchedRustMCTS, rust_state_to_tensor
    from exchange_simulator import PySimulator as RustSimulator

    network.eval()

    # Use default reward config if not provided
    rewards = reward_config or DEFAULT_REWARDS

    @dataclass
    class AsymmetricGameInstance:
        sim: RustSimulator
        history: list[tuple[np.ndarray, int, np.ndarray]]  # (tensor, team, policy_target)
        moves_record: list[dict]
        initial_state: dict
        mcts_side: int  # Which side uses MCTS (0=White, 1=Black)
        player_damage: int = 0
        enemy_damage: int = 0
        seed: int = 0
        playing_as: int = 0
        finished: bool = False
        stalemate: bool = False
        # Combo tracking: attacks made while Royal Decree is active
        white_combo_attacks: int = 0
        black_combo_attacks: int = 0
        # NEW: Ability tracking for reward calculation
        white_tracker: AbilityTracker = field(default_factory=AbilityTracker)
        black_tracker: AbilityTracker = field(default_factory=AbilityTracker)
        # RD waste tracking (turns active without attacks)
        white_rd_attacks_this_activation: int = 0
        black_rd_attacks_this_activation: int = 0

    # Initialize games - alternate which side gets MCTS
    games: list[AsymmetricGameInstance] = []
    for i in range(num_games):
        sim = RustSimulator()
        sim.set_seed(base_seed + i)
        playing_as = 0 if i % 2 == 0 else 1
        sim.set_playing_as("White" if playing_as == 0 else "Black")

        # Alternate MCTS side: even games = White MCTS, odd games = Black MCTS
        mcts_side = i % 2

        games.append(AsymmetricGameInstance(
            sim=sim,
            history=[],
            moves_record=[],
            initial_state=_rust_sim_to_json(sim),
            mcts_side=mcts_side,
            seed=base_seed + i,
            playing_as=playing_as,
            white_tracker=AbilityTracker(),
            black_tracker=AbilityTracker(),
        ))

    # Create MCTS searcher
    mcts = BatchedRustMCTS(network, device, mcts_config)

    # Create policy size constant
    from .value_network import POLICY_SIZE

    from .mcts_rust import rust_move_to_policy_index

    # Play games with batched evaluation
    games_completed = 0
    pbar = tqdm(total=num_games, desc="  Playing asymmetric games", leave=False, ncols=80)

    with torch.no_grad():
        while any(not g.finished for g in games):
            # Separate games by which evaluation method they need
            mcts_games = []  # (game_idx, game, tensor, rust_moves)
            oneply_games = []  # (game_idx, game, tensor, rust_moves)

            for game_idx, game in enumerate(games):
                if game.finished:
                    continue

                if game.sim.is_terminal():
                    game.finished = True
                    games_completed += 1
                    pbar.update(1)
                    continue

                rust_moves = game.sim.get_legal_moves(game.sim.side_to_move())
                if not rust_moves:
                    game.finished = True
                    games_completed += 1
                    pbar.update(1)
                    continue

                tensor = rust_state_to_tensor(game.sim)
                team = game.sim.side_to_move()

                if team == ("White" if game.mcts_side == 0 else "Black"):
                    mcts_games.append((game_idx, game, tensor, rust_moves))
                else:
                    oneply_games.append((game_idx, game, tensor, rust_moves))

            # --- BATCH MCTS EVALUATION ---
            if mcts_games:
                # Batch all MCTS searches together
                mcts_sims = [g[1].sim for g in mcts_games]
                mcts_results = mcts.search_batch(mcts_sims, mcts_config.num_simulations, add_noise=True)

                for i, (game_idx, game, tensor, rust_moves) in enumerate(mcts_games):
                    best_move, visit_dist = mcts_results[i]
                    team = game.sim.side_to_move()

                    # Create policy target from MCTS visit distribution
                    policy_target = np.zeros(POLICY_SIZE, dtype=np.float32)
                    if visit_dist is not None and len(visit_dist) > 0 and len(rust_moves) > 0:
                        for j, rm in enumerate(rust_moves):
                            if j < len(visit_dist):
                                move_idx = rust_move_to_policy_index(rm)
                                if 0 <= move_idx < POLICY_SIZE:
                                    policy_target[move_idx] = visit_dist[j]

                    # Record and execute move
                    _execute_asymmetric_move(game, tensor, team, best_move, rust_moves, policy_target, combo_bonus_enabled, use_new_rewards)

            # --- BATCH 1-PLY EVALUATION ---
            if oneply_games:
                # Collect all positions for batched evaluation
                all_evals = []  # (oneply_idx, move_idx, child_tensor, rust_move)
                game_move_counts = []

                for oneply_idx, (game_idx, game, tensor, rust_moves) in enumerate(oneply_games):
                    game_move_counts.append(len(rust_moves))
                    for move_idx, rust_move in enumerate(rust_moves):
                        sim_copy = game.sim.clone()
                        sim_copy.make_move(rust_move)
                        child_tensor = rust_state_to_tensor(sim_copy)
                        all_evals.append((oneply_idx, move_idx, child_tensor, rust_move))

                # Batch evaluate ALL positions on GPU
                if all_evals:
                    batch_tensors = torch.tensor(
                        np.array([e[2] for e in all_evals]),
                        dtype=torch.float32,
                        device=device
                    )
                    if hasattr(network, 'forward_value'):
                        batch_values = network.forward_value(batch_tensors).cpu().numpy().flatten()
                    else:
                        batch_values = network(batch_tensors).cpu().numpy().flatten()

                    # Distribute values back to games
                    eval_idx = 0
                    for oneply_idx, (game_idx, game, tensor, rust_moves) in enumerate(oneply_games):
                        num_moves = game_move_counts[oneply_idx]
                        team = game.sim.side_to_move()

                        # Negate because we evaluated from opponent's perspective
                        move_values = [-batch_values[eval_idx + i] for i in range(num_moves)]
                        eval_idx += num_moves

                        # Select best move (greedy for student)
                        best_idx = int(np.argmax(move_values))
                        best_move = rust_moves[best_idx]

                        # Create policy target from 1-ply evaluation (softmax of values)
                        policy_target = np.zeros(POLICY_SIZE, dtype=np.float32)
                        values_array = np.array(move_values, dtype=np.float32)
                        exp_values = np.exp((values_array - values_array.max()) * 2.0)  # temp=0.5
                        probs = exp_values / exp_values.sum()
                        for i, rm in enumerate(rust_moves):
                            move_idx = rust_move_to_policy_index(rm)
                            if 0 <= move_idx < POLICY_SIZE:
                                policy_target[move_idx] = probs[i]

                        # Record and execute move
                        _execute_asymmetric_move(game, tensor, team, best_move, rust_moves, policy_target, combo_bonus_enabled, use_new_rewards)

    pbar.close()

    # Convert to training data
    all_training_data: list[tuple[np.ndarray, float, np.ndarray]] = []
    all_replays: list[dict] = []

    for game in games:
        turn_number = game.sim.turn_number()

        # Determine base outcome value
        if use_new_rewards:
            # Use new reward config
            if game.stalemate:
                player_value = -rewards.loss_penalty
            elif game.sim.winner() == "White":
                player_value = rewards.calculate_win_value(turn_number)
            elif game.sim.winner() == "Black":
                player_value = -rewards.loss_penalty
            else:
                # Draw: mild penalty (only for insufficient material now)
                player_value = -rewards.draw_penalty

            # Apply ability-based rewards from trackers
            white_ability_reward = game.white_tracker.calculate_total_reward(rewards, turn_number)
            black_ability_reward = game.black_tracker.calculate_total_reward(rewards, turn_number)

            # Net reward from White's perspective
            player_value += (white_ability_reward - black_ability_reward)
        else:
            # Legacy reward calculation
            win_value = _calculate_win_value(turn_number)

            if game.stalemate:
                player_value = -loss_penalty
            elif game.sim.winner() == "White":
                player_value = win_value
            elif game.sim.winner() == "Black":
                player_value = -loss_penalty
            else:
                player_value = _calculate_early_draw_penalty(turn_number, draw_penalty)

            # Apply legacy combo bonus
            if combo_bonus_enabled:
                white_combo_bonus = min(game.white_combo_attacks * combo_bonus_per_attack, combo_bonus_max)
                black_combo_bonus = min(game.black_combo_attacks * combo_bonus_per_attack, combo_bonus_max)
                combo_net = white_combo_bonus - black_combo_bonus
                player_value += combo_net

        # Convert to training data
        for tensor, team_int, policy_target in game.history:
            value = player_value if team_int == 0 else -player_value
            all_training_data.append((tensor, value, policy_target))

        # Build score summary for debugging
        score_summary = _build_score_summary(
            moves_record=game.moves_record,
            winner=game.sim.winner(),
            turn_number=turn_number,
            stalemate=game.stalemate,
            draw_penalty=draw_penalty,
            loss_penalty=loss_penalty,
            white_combo_attacks=game.white_combo_attacks,
            black_combo_attacks=game.black_combo_attacks,
            combo_bonus_per_attack=combo_bonus_per_attack,
            combo_bonus_max=combo_bonus_max,
            combo_bonus_enabled=combo_bonus_enabled,
        )

        # Build replay with new ability tracking stats
        replay = {
            "moves": game.moves_record,
            "initial_state": game.initial_state,
            "final_state": _rust_sim_to_json(game.sim),
            "winner": game.sim.winner(),
            "total_turns": turn_number,
            "seed": game.seed,
            "mcts_side": game.mcts_side,
            "asymmetric": True,
            "mcts_simulations": mcts_config.num_simulations,
            "white_combo_attacks": game.white_combo_attacks,
            "black_combo_attacks": game.black_combo_attacks,
            # Score breakdown for debugging
            "score_summary": score_summary,
            # NEW: Ability tracking stats
            "white_tracker": {
                "rd_activations": game.white_tracker.rd_activations,
                "rd_attacks_during": game.white_tracker.rd_attacks_during,
                "rd_wasted": game.white_tracker.rd_wasted,
                "interpose_triggers": game.white_tracker.interpose_triggers,
                "interpose_damage_blocked": game.white_tracker.interpose_damage_blocked,
                "consecration_uses": game.white_tracker.consecration_uses,
                "consecration_hp_healed": game.white_tracker.consecration_hp_healed,
                "skirmish_uses": game.white_tracker.skirmish_uses,
                "overextend_uses": game.white_tracker.overextend_uses,
                "overextend_net_damage": game.white_tracker.overextend_net_damage,
                "pawn_advances": game.white_tracker.pawn_advances,
                "pawn_promotions": game.white_tracker.pawn_promotions,
                "total_damage": game.white_tracker.total_damage_dealt,
                "attack_actions": game.white_tracker.attack_actions,
                "max_shuffle_streak": game.white_tracker.max_consecutive_no_damage,
                "king_proximity_delta": game.white_tracker.king_proximity_delta,
                "king_attack_opportunities": game.white_tracker.king_attack_opportunities,
            },
            "black_tracker": {
                "rd_activations": game.black_tracker.rd_activations,
                "rd_attacks_during": game.black_tracker.rd_attacks_during,
                "rd_wasted": game.black_tracker.rd_wasted,
                "interpose_triggers": game.black_tracker.interpose_triggers,
                "interpose_damage_blocked": game.black_tracker.interpose_damage_blocked,
                "consecration_uses": game.black_tracker.consecration_uses,
                "consecration_hp_healed": game.black_tracker.consecration_hp_healed,
                "skirmish_uses": game.black_tracker.skirmish_uses,
                "overextend_uses": game.black_tracker.overextend_uses,
                "overextend_net_damage": game.black_tracker.overextend_net_damage,
                "pawn_advances": game.black_tracker.pawn_advances,
                "pawn_promotions": game.black_tracker.pawn_promotions,
                "total_damage": game.black_tracker.total_damage_dealt,
                "attack_actions": game.black_tracker.attack_actions,
                "max_shuffle_streak": game.black_tracker.max_consecutive_no_damage,
                "king_proximity_delta": game.black_tracker.king_proximity_delta,
                "king_attack_opportunities": game.black_tracker.king_attack_opportunities,
            },
        }
        all_replays.append(replay)

    # === MCTS vs Network Stats ===
    mcts_wins = 0
    network_wins = 0
    draws = 0
    for game in games:
        if game.sim.winner() is None:
            draws += 1
        elif game.sim.winner() == ("White" if game.mcts_side == 0 else "Black"):
            mcts_wins += 1
        else:
            network_wins += 1

    total = len(games)
    tprint(f"  [ASYMMETRIC STATS] MCTS: {mcts_wins}W ({100*mcts_wins/total:.0f}%) | "
           f"Network: {network_wins}W ({100*network_wins/total:.0f}%) | "
           f"Draw: {draws} ({100*draws/total:.0f}%)")

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
        "sideToMove": sim.side_to_move(),
        "turnNumber": sim.turn_number(),
        "isTerminal": sim.is_terminal(),
        "isDraw": sim.is_draw(),
        "winner": sim.winner(),
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

        # Initialize network (PolicyValueNetwork if MCTS enabled, including hybrid/asymmetric mode)
        if config.use_mcts_rust or config.use_hybrid or config.use_asymmetric:
            self.network = create_policy_value_from_preset(config.network_preset)
            if config.use_asymmetric:
                print(f"Using PolicyValueNetwork for Asymmetric mode ({config.asymmetric_mcts_simulations} sims/move)")
            elif config.use_hybrid:
                print(f"Using PolicyValueNetwork for Hybrid mode ({config.hybrid_mcts_simulations} sims/move for MCTS games)")
            else:
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
        tprint(f"Saved checkpoint: {path}")

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
        if self.config.use_mcts or self.config.use_mcts_rust or self.config.use_hybrid or self.config.use_asymmetric:
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
        tprint("-> NEW CHAMPION!")

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

        # Asymmetric mode: MCTS teacher vs 1-ply student in same game
        if self.config.use_asymmetric:
            # Get decayed noise epsilon for this iteration
            current_epsilon = self.config.get_noise_epsilon(self.state.iteration)
            tprint(f"Generating {num_games} games with ASYMMETRIC mode ({self.config.asymmetric_mcts_simulations} sims/move, noise ε={current_epsilon:.2f})...")
            asymmetric_mcts_config = MCTSConfig(
                num_simulations=self.config.asymmetric_mcts_simulations,
                c_puct=self.config.mcts_c_puct,
                dirichlet_alpha=self.config.mcts_dirichlet_alpha,
                dirichlet_epsilon=current_epsilon,  # Decayed noise
                temperature=temperature,
            )
            all_data, all_replays = play_asymmetric_self_play_games(
                network=self.network,
                device=self.device,
                mcts_config=asymmetric_mcts_config,
                num_games=num_games,
                base_seed=base_seed,
                draw_penalty=self.config.draw_penalty,
                loss_penalty=self.config.loss_penalty,
                combo_bonus_enabled=self.config.combo_bonus_enabled,
                combo_bonus_per_attack=self.config.combo_bonus_per_attack,
                combo_bonus_max=self.config.combo_bonus_max,
                reward_config=DEFAULT_REWARDS,
                use_new_rewards=self.config.use_new_rewards,
            )

        # Hybrid mode: mix fast 1-ply with quality MCTS games
        elif self.config.use_hybrid:
            mcts_games = int(num_games * self.config.hybrid_mcts_ratio)
            oneply_games = num_games - mcts_games

            tprint(f"Generating {oneply_games} games with Rust 1-ply + {mcts_games} games with MCTS (hybrid)...")

            # Generate fast 1-ply games
            all_data = []
            all_replays = []

            if oneply_games > 0:
                tprint(f"  [1-ply START] Generating {oneply_games} games...")
                oneply_data, oneply_replays = play_rust_self_play_games(
                    network=self.network,
                    device=self.device,
                    num_games=oneply_games,
                    temperature=temperature,
                    base_seed=base_seed,
                    draw_penalty=self.config.draw_penalty,
                    loss_penalty=self.config.loss_penalty,
                    repetition_penalty=self.config.repetition_penalty,
                    stalemate_moves=self.config.stalemate_moves,
                    stalemate_penalty=self.config.stalemate_penalty,
                )
                tprint(f"  [1-ply COMPLETE] {len(oneply_replays)} games, {len(oneply_data)} positions")
                all_data.extend(oneply_data)
                all_replays.extend(oneply_replays)

            # Generate quality MCTS games
            if mcts_games > 0:
                # Get decayed noise epsilon for this iteration
                current_epsilon = self.config.get_noise_epsilon(self.state.iteration)
                tprint(f"  [MCTS START] Generating {mcts_games} games ({self.config.hybrid_mcts_simulations} sims/move, noise ε={current_epsilon:.2f})...")
                # Create a temporary MCTS config with hybrid settings
                hybrid_mcts_config = MCTSConfig(
                    num_simulations=self.config.hybrid_mcts_simulations,
                    c_puct=self.config.mcts_c_puct,
                    dirichlet_alpha=self.config.mcts_dirichlet_alpha,
                    dirichlet_epsilon=current_epsilon,  # Decayed noise
                    temperature=temperature,
                )
                mcts_data, mcts_replays = play_mcts_rust_self_play_games(
                    network=self.network,
                    device=self.device,
                    mcts_config=hybrid_mcts_config,
                    num_games=mcts_games,
                    base_seed=base_seed + oneply_games,  # Different seeds
                    draw_penalty=self.config.draw_penalty,
                    loss_penalty=self.config.loss_penalty,
                    stalemate_moves=self.config.stalemate_moves,
                    stalemate_penalty=self.config.stalemate_penalty,
                )
                tprint(f"  [MCTS COMPLETE] {len(mcts_replays)} games, {len(mcts_data)} positions")
                # MCTS returns (state, value, policy) tuples - we only use (state, value) for consistency
                # But we keep all data for potential policy learning later
                all_data.extend(mcts_data)
                all_replays.extend(mcts_replays)

            tprint(f"Hybrid generation complete: {len(all_replays)} total games ({len(all_data)} positions)")

        elif self.config.use_mcts_rust:
            # Get decayed noise epsilon for this iteration
            current_epsilon = self.config.get_noise_epsilon(self.state.iteration)
            tprint(f"Generating {num_games} games with Rust MCTS ({self.config.mcts_simulations} sims/move, noise ε={current_epsilon:.2f})...")
            # Create fresh config with decayed noise
            mcts_config = MCTSConfig(
                num_simulations=self.config.mcts_simulations,
                eval_simulations=self.config.mcts_eval_simulations,
                c_puct=self.config.mcts_c_puct,
                dirichlet_alpha=self.config.mcts_dirichlet_alpha,
                dirichlet_epsilon=current_epsilon,  # Decayed noise
                temperature=temperature,
            )
            all_data, all_replays = play_mcts_rust_self_play_games(
                network=self.network,
                device=self.device,
                mcts_config=mcts_config,
                num_games=num_games,
                base_seed=base_seed,
                draw_penalty=self.config.draw_penalty,
                loss_penalty=self.config.loss_penalty,
                stalemate_moves=self.config.stalemate_moves,
                stalemate_penalty=self.config.stalemate_penalty,
            )
        elif self.config.use_mcts:
            # Get decayed noise epsilon for this iteration
            current_epsilon = self.config.get_noise_epsilon(self.state.iteration)
            tprint(f"Generating {num_games} games with MCTS ({self.config.mcts_simulations} sims/move, noise ε={current_epsilon:.2f})...")
            # Create fresh config with decayed noise
            mcts_config = MCTSConfig(
                num_simulations=self.config.mcts_simulations,
                eval_simulations=self.config.mcts_eval_simulations,
                c_puct=self.config.mcts_c_puct,
                dirichlet_alpha=self.config.mcts_dirichlet_alpha,
                dirichlet_epsilon=current_epsilon,  # Decayed noise
                temperature=temperature,
            )
            all_data, all_replays = play_mcts_self_play_games(
                network=self.network,
                device=self.device,
                mcts_config=mcts_config,
                num_games=num_games,
                base_seed=base_seed,
                draw_penalty=self.config.draw_penalty,
                loss_penalty=self.config.loss_penalty,
                stalemate_moves=self.config.stalemate_moves,
                stalemate_penalty=self.config.stalemate_penalty,
            )
        elif self.config.use_rust:
            tprint(f"Generating {num_games} games with Rust 1-ply evaluation...")
            all_data, all_replays = play_rust_self_play_games(
                network=self.network,
                device=self.device,
                num_games=num_games,
                temperature=temperature,
                base_seed=base_seed,
                draw_penalty=self.config.draw_penalty,
                loss_penalty=self.config.loss_penalty,
                repetition_penalty=self.config.repetition_penalty,
                stalemate_moves=self.config.stalemate_moves,
                stalemate_penalty=self.config.stalemate_penalty,
            )
        else:
            tprint(f"Generating {num_games} games with batched GPU evaluation...")
            all_data, all_replays = play_batched_self_play_games(
                network=self.network,
                device=self.device,
                num_games=num_games,
                temperature=temperature,
                base_seed=base_seed,
                draw_penalty=self.config.draw_penalty,
                loss_penalty=self.config.loss_penalty,
                repetition_penalty=self.config.repetition_penalty,
                stalemate_moves=self.config.stalemate_moves,
                stalemate_penalty=self.config.stalemate_penalty,
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

            # Track stats - winner is a string ("White", "Black") or None
            if replay_data["winner"] == "White":
                wins += 1
            elif replay_data["winner"] == "Black":
                losses += 1
            else:
                draws += 1

        tprint(f"Games saved to {iter_replays_dir}/ (W:{wins} B:{losses} D:{draws})")

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

    def evaluate_vs_champion_rust(self) -> dict:
        """
        Evaluate current network against champion using Rust simulator (faster).
        """
        from .mcts_rust import check_rust_available, rust_state_to_tensor
        check_rust_available()
        from exchange_simulator import PySimulator as RustSimulator

        if self.champion_network is None:
            return {
                "wins": 0, "losses": 0, "draws": 0,
                "decisive_win_rate": 1.0, "should_promote": True,
                "reason": "No champion yet",
            }

        self.network.eval()
        self.champion_network.eval()
        num_games = self.config.champion_eval_games

        tprint(f"Evaluating vs champion ({num_games} games)...")

        @dataclass
        class RustChampionGame:
            sim: Any  # RustSimulator
            current_plays_as: int  # 0=White, 1=Black
            finished: bool = False
            winner: Optional[int] = None

        # Initialize games - alternate sides
        games: list[RustChampionGame] = []
        for i in range(num_games):
            sim = RustSimulator()
            sim.set_seed(i + 888888)
            current_plays_as = 0 if i % 2 == 0 else 1  # 0=White, 1=Black
            sim.set_playing_as("White" if current_plays_as == 0 else "Black")
            games.append(RustChampionGame(sim=sim, current_plays_as=current_plays_as))

        games_completed = 0
        pbar = tqdm(total=num_games, desc="  vs Champion", leave=False, ncols=80)

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
                    if sim.is_terminal:
                        game.finished = True
                        game.winner = sim.winner
                        games_completed += 1
                        pbar.update(1)
                        continue

                    current_team = sim.side_to_move
                    rust_moves = sim.generate_moves()

                    if not rust_moves:
                        game.finished = True
                        game.winner = sim.winner
                        games_completed += 1
                        pbar.update(1)
                        continue

                    # Determine which network plays this side
                    use_current_network = (current_team == game.current_plays_as)

                    # Collect candidate positions using Rust cloning
                    position_tensors = []
                    for rust_move in rust_moves:
                        sim_copy = sim.clone_sim()
                        sim_copy.make_move(rust_move)
                        position_tensors.append(rust_state_to_tensor(sim_copy))

                    if use_current_network:
                        current_evals.append((game_idx, rust_moves, position_tensors))
                    else:
                        champion_evals.append((game_idx, rust_moves, position_tensors))

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

                        if games[game_idx].sim.is_terminal:
                            games[game_idx].finished = True
                            games[game_idx].winner = games[game_idx].sim.winner
                            games_completed += 1
                            pbar.update(1)

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

                        if games[game_idx].sim.is_terminal:
                            games[game_idx].finished = True
                            games[game_idx].winner = games[game_idx].sim.winner
                            games_completed += 1
                            pbar.update(1)

        pbar.close()

        # Count results from current network's perspective
        wins = losses = draws = 0
        for game in games:
            if game.winner is None:
                draws += 1
            elif game.winner == game.current_plays_as:
                wins += 1
            else:
                losses += 1

        # Calculate decisive win rate
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
                    move_result = game.sim.make_move(selected_move)
                    move_record["damage"] = move_result["damage"]
                    move_record["interpose_blocked"] = move_result.get("interpose_blocked", 0)
                    move_record["consecration_heal"] = move_result.get("consecration_heal", 0)
                    move_record["promotion"] = move_result.get("promotion", False)
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

            tprint(f"Iteration {iteration + 1}/{self.config.num_iterations} | "
                   f"Loss: {train_metrics['loss']:.4f} | "
                   f"Temp: {temperature:.2f} | "
                   f"Buffer: {len(self.data_buffer):,} | "
                   f"Time: {elapsed:.1f}s")

            # Champion evaluation every iteration
            # Use Rust eval for any mode that uses Rust (including hybrid/asymmetric)
            tprint(f"[EVAL START] Evaluating vs champion...")
            if self.config.use_rust or self.config.use_mcts_rust or self.config.use_hybrid or self.config.use_asymmetric:
                eval_metrics = self.evaluate_vs_champion_rust()
            else:
                eval_metrics = self.evaluate_vs_champion()
            tprint(f"[EVAL COMPLETE] W:{eval_metrics['wins']} L:{eval_metrics['losses']} D:{eval_metrics['draws']}")
            self.writer.add_scalar("eval/decisive_win_rate", eval_metrics["decisive_win_rate"], iteration)
            self.writer.add_scalar("eval/wins", eval_metrics["wins"], iteration)
            self.writer.add_scalar("eval/losses", eval_metrics["losses"], iteration)
            self.writer.add_scalar("eval/draws", eval_metrics["draws"], iteration)

            decisive_str = f"{eval_metrics['decisive_win_rate']:.0%}" if eval_metrics.get('decisive_games', 0) > 0 else "N/A"
            tprint(f"vs Champion: W:{eval_metrics['wins']} L:{eval_metrics['losses']} D:{eval_metrics['draws']} "
                   f"| Decisive: {decisive_str} ({eval_metrics.get('reason', '')})")

            if eval_metrics["should_promote"]:
                self._update_champion()
                self.state.best_eval_score = eval_metrics["decisive_win_rate"]

            # Save latest checkpoint every iteration (for resume)
            self.save_checkpoint("latest")

            # Milestone checkpoints at intervals
            if (iteration + 1) % self.config.checkpoint_interval == 0:
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


# CLI entry point is in scripts/train.py - use that instead of running this module directly
