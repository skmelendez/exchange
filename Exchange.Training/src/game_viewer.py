"""
Game Viewer and Logger for EXCHANGE Training

Saves game replays during training and provides tools to view them.
Run standalone to watch a game or review saved replays.

Usage:
    # Watch a random game live
    python -m src.game_viewer --watch

    # Watch AI vs Random
    python -m src.game_viewer --watch --model runs/experiment/checkpoints/best.pt

    # Review saved replays
    python -m src.game_viewer --replay data/replays/game_001.json

    # Save N games during next training run
    python -m src.game_viewer --record 10
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from .game_state import GameState, Piece, PieceType, Team, create_initial_state
from .game_simulator import GameSimulator, Move, MoveType


# Piece symbols for display
PIECE_SYMBOLS = {
    (PieceType.KING, Team.PLAYER): "♔",
    (PieceType.QUEEN, Team.PLAYER): "♕",
    (PieceType.ROOK, Team.PLAYER): "♖",
    (PieceType.BISHOP, Team.PLAYER): "♗",
    (PieceType.KNIGHT, Team.PLAYER): "♘",
    (PieceType.PAWN, Team.PLAYER): "♙",
    (PieceType.KING, Team.ENEMY): "♚",
    (PieceType.QUEEN, Team.ENEMY): "♛",
    (PieceType.ROOK, Team.ENEMY): "♜",
    (PieceType.BISHOP, Team.ENEMY): "♝",
    (PieceType.KNIGHT, Team.ENEMY): "♞",
    (PieceType.PAWN, Team.ENEMY): "♟",
}

# Letter symbols fallback
PIECE_LETTERS = {
    (PieceType.KING, Team.PLAYER): "K",
    (PieceType.QUEEN, Team.PLAYER): "Q",
    (PieceType.ROOK, Team.PLAYER): "R",
    (PieceType.BISHOP, Team.PLAYER): "B",
    (PieceType.KNIGHT, Team.PLAYER): "N",
    (PieceType.PAWN, Team.PLAYER): "P",
    (PieceType.KING, Team.ENEMY): "k",
    (PieceType.QUEEN, Team.ENEMY): "q",
    (PieceType.ROOK, Team.ENEMY): "r",
    (PieceType.BISHOP, Team.ENEMY): "b",
    (PieceType.KNIGHT, Team.ENEMY): "n",
    (PieceType.PAWN, Team.ENEMY): "p",
}


def pos_to_chess(x: int, y: int) -> str:
    """Convert (x, y) to chess notation."""
    return f"{chr(ord('a') + x)}{y + 1}"


def render_board(state: GameState, use_unicode: bool = True) -> str:
    """Render board state as ASCII/Unicode art."""
    symbols = PIECE_SYMBOLS if use_unicode else PIECE_LETTERS
    lines = []

    lines.append("  ┌───┬───┬───┬───┬───┬───┬───┬───┐")

    for y in range(7, -1, -1):
        row = f"{y + 1} │"
        for x in range(8):
            piece = state.get_piece_at(x, y)
            if piece and piece.is_alive:
                symbol = symbols.get((piece.piece_type, piece.team), "?")
                # Add HP indicator
                hp_pct = piece.current_hp / piece.max_hp
                if hp_pct <= 0.3:
                    hp_mark = "!"  # Critical
                elif hp_pct <= 0.6:
                    hp_mark = "·"  # Damaged
                else:
                    hp_mark = " "  # Healthy
                row += f"{symbol}{hp_mark}│"
            else:
                # Empty square with checkerboard pattern
                if (x + y) % 2 == 0:
                    row += "  │"
                else:
                    row += "░░│"
        lines.append(row)
        if y > 0:
            lines.append("  ├───┼───┼───┼───┼───┼───┼───┼───┤")

    lines.append("  └───┴───┴───┴───┴───┴───┴───┴───┘")
    lines.append("    a   b   c   d   e   f   g   h")

    return "\n".join(lines)


def render_move(move: Move) -> str:
    """Format a move for display."""
    piece_name = move.piece.piece_type.name.capitalize()
    from_pos = pos_to_chess(move.from_pos[0], move.from_pos[1])
    to_pos = pos_to_chess(move.to_pos[0], move.to_pos[1])

    if move.move_type == MoveType.MOVE:
        return f"{piece_name} {from_pos} → {to_pos}"
    elif move.move_type == MoveType.ATTACK:
        return f"{piece_name} {from_pos} ⚔ {to_pos}"
    elif move.move_type == MoveType.MOVE_AND_ATTACK:
        attack_pos = pos_to_chess(move.attack_pos[0], move.attack_pos[1])
        return f"{piece_name} {from_pos} → {to_pos} ⚔ {attack_pos}"
    else:
        return f"{piece_name} ABILITY"


def render_game_state(state: GameState) -> str:
    """Render full game state with piece lists."""
    lines = [render_board(state)]
    lines.append("")

    # Side to move
    side = "WHITE (Player)" if state.side_to_move == Team.PLAYER else "BLACK (Enemy)"
    lines.append(f"Turn {state.turn_number} - {side} to move")
    lines.append("")

    # Piece lists with HP
    player_pieces = state.get_pieces(Team.PLAYER)
    enemy_pieces = state.get_pieces(Team.ENEMY)

    lines.append("WHITE: " + ", ".join(
        f"{p.piece_type.name[0]}({p.current_hp}/{p.max_hp})" for p in player_pieces
    ))
    lines.append("BLACK: " + ", ".join(
        f"{p.piece_type.name[0]}({p.current_hp}/{p.max_hp})" for p in enemy_pieces
    ))

    return "\n".join(lines)


@dataclass
class GameReplay:
    """Recorded game for replay."""
    moves: list[dict]
    initial_state: dict
    final_state: dict
    winner: Optional[int]
    total_turns: int

    def save(self, path: str) -> None:
        """Save replay to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "GameReplay":
        """Load replay from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


def play_and_record(
    simulator: Optional[GameSimulator] = None,
    network = None,
    temperature: float = 0.5,
    verbose: bool = True,
    delay: float = 0.5,
) -> GameReplay:
    """
    Play a game and optionally display it live.

    Args:
        simulator: Game simulator (creates new if None)
        network: Optional neural network for move selection
        temperature: Move selection temperature
        verbose: Print board after each move
        delay: Seconds between moves when verbose

    Returns:
        GameReplay object with full game record
    """
    import random

    if simulator is None:
        simulator = GameSimulator()
        simulator.reset()

    initial_state = simulator.state.to_json()
    moves_record: list[dict] = []

    if verbose:
        print("\n" + "=" * 50)
        print("GAME START")
        print("=" * 50)
        print(render_game_state(simulator.state))
        print()

    while not simulator.state.is_terminal:
        team = simulator.state.side_to_move
        moves = simulator.generate_moves(team)

        if not moves:
            break

        # Select move
        if network is not None:
            # Use network for move selection
            import torch
            import numpy as np

            move_values = []
            for move in moves:
                sim_copy = GameSimulator(simulator.state.clone())
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
                        tensor = torch.tensor(sim_copy.state.to_tensor()).unsqueeze(0)
                        value = network(tensor).item()
                    move_values.append(-value)
                else:
                    move_values.append(0.0)

            if temperature > 0:
                values = np.array(move_values)
                exp_values = np.exp(values / temperature)
                probs = exp_values / exp_values.sum()
                move_idx = np.random.choice(len(moves), p=probs)
            else:
                move_idx = np.argmax(move_values)

            selected_move = moves[move_idx]
        else:
            # Random move
            selected_move = random.choice(moves)

        # Record move
        move_record = {
            "turn": simulator.state.turn_number,
            "team": int(team),
            "piece_type": int(selected_move.piece.piece_type),
            "move_type": int(selected_move.move_type),
            "from": list(selected_move.from_pos),
            "to": list(selected_move.to_pos),
            "attack": list(selected_move.attack_pos) if selected_move.attack_pos else None,
        }

        # Execute move
        damage = simulator.make_move(selected_move)
        move_record["damage"] = damage
        moves_record.append(move_record)

        if verbose:
            team_name = "WHITE" if team == Team.PLAYER else "BLACK"
            print(f"\n[Turn {move_record['turn']}] {team_name}: {render_move(selected_move)}", end="")
            if damage > 0:
                print(f" ({damage} damage!)")
            else:
                print()
            print(render_board(simulator.state))
            time.sleep(delay)

    # Game over
    if verbose:
        print("\n" + "=" * 50)
        if simulator.state.winner == Team.PLAYER:
            print("GAME OVER - WHITE WINS!")
        elif simulator.state.winner == Team.ENEMY:
            print("GAME OVER - BLACK WINS!")
        else:
            print("GAME OVER - DRAW")
        print(f"Total turns: {simulator.state.turn_number}")
        print("=" * 50)

    return GameReplay(
        moves=moves_record,
        initial_state=initial_state,
        final_state=simulator.state.to_json(),
        winner=int(simulator.state.winner) if simulator.state.winner is not None else None,
        total_turns=simulator.state.turn_number,
    )


def replay_game(replay: GameReplay, delay: float = 0.5) -> None:
    """Replay a recorded game move by move."""
    state = GameState.from_json(replay.initial_state)

    print("\n" + "=" * 50)
    print("REPLAY START")
    print("=" * 50)
    print(render_game_state(state))

    input("\nPress Enter to start replay...")

    for move_data in replay.moves:
        team_name = "WHITE" if move_data["team"] == 0 else "BLACK"
        piece_name = PieceType(move_data["piece_type"]).name.capitalize()
        from_pos = pos_to_chess(move_data["from"][0], move_data["from"][1])
        to_pos = pos_to_chess(move_data["to"][0], move_data["to"][1])

        move_str = f"{piece_name} {from_pos} → {to_pos}"
        if move_data.get("damage", 0) > 0:
            move_str += f" ({move_data['damage']} dmg)"

        print(f"\n[Turn {move_data['turn']}] {team_name}: {move_str}")

        # We'd need to reconstruct state here for full replay
        # For now just show the move record
        time.sleep(delay)

    print("\n" + "=" * 50)
    winner = replay.winner
    if winner == 0:
        print("WHITE WINS!")
    elif winner == 1:
        print("BLACK WINS!")
    else:
        print("DRAW")
    print("=" * 50)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="EXCHANGE Game Viewer")
    parser.add_argument("--watch", action="store_true", help="Watch a live game")
    parser.add_argument("--model", type=str, help="Path to model checkpoint for AI player")
    parser.add_argument("--replay", type=str, help="Path to replay file to view")
    parser.add_argument("--record", type=int, help="Record N random games to data/replays/")
    parser.add_argument("--delay", type=float, default=0.3, help="Delay between moves (seconds)")
    parser.add_argument("--no-unicode", action="store_true", help="Use ASCII instead of Unicode pieces")

    args = parser.parse_args()

    if args.replay:
        replay = GameReplay.load(args.replay)
        replay_game(replay, delay=args.delay)

    elif args.watch:
        network = None
        if args.model:
            import torch
            from .value_network import ExchangeValueNetwork
            print(f"Loading model: {args.model}")
            checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict) and "network_state_dict" in checkpoint:
                network = ExchangeValueNetwork.from_config(checkpoint.get("network_config", {}))
                network.load_state_dict(checkpoint["network_state_dict"])
            else:
                network = checkpoint
            network.eval()
            print("Model loaded! AI will play as both sides.\n")

        replay = play_and_record(network=network, delay=args.delay, verbose=True)

        # Save replay
        os.makedirs("data/replays", exist_ok=True)
        replay_path = f"data/replays/game_{int(time.time())}.json"
        replay.save(replay_path)
        print(f"\nReplay saved: {replay_path}")

    elif args.record:
        os.makedirs("data/replays", exist_ok=True)
        print(f"Recording {args.record} random games...")

        for i in range(args.record):
            sim = GameSimulator()
            sim.set_seed(i)
            replay = play_and_record(simulator=sim, verbose=False)
            replay_path = f"data/replays/random_{i:04d}.json"
            replay.save(replay_path)
            print(f"  Saved: {replay_path} ({replay.total_turns} turns, winner: {replay.winner})")

        print("Done!")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
