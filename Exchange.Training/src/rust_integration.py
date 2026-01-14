"""
Rust Simulator Integration

Provides a Python-compatible wrapper around the Rust game simulator.
Falls back to Python implementation if Rust extension is not available.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, TYPE_CHECKING

# Try to import Rust extension
try:
    from exchange_simulator import RustSimulator, RustMove, RustPiece
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    RustSimulator = None
    RustMove = None
    RustPiece = None

from .game_state import (
    GameState, Piece, PieceType, Team, AbilityId,
    PIECE_STATS, create_initial_state
)
from .game_simulator import Move, MoveType


def is_rust_available() -> bool:
    """Check if Rust simulator is available."""
    return RUST_AVAILABLE


class RustGameSimulator:
    """
    Fast game simulator backed by Rust.

    API-compatible with Python GameSimulator for drop-in replacement.
    """

    def __init__(self, state: Optional[GameState] = None):
        if not RUST_AVAILABLE:
            raise RuntimeError(
                "Rust simulator not available. "
                "Run 'cd rust_simulator && ./setup.sh' to build it."
            )

        self._sim = RustSimulator()

        # If a specific state was provided, we'd need to set it
        # For now, we only support starting from initial position
        if state is not None:
            # TODO: Could add state serialization if needed
            pass

    def reset(self) -> GameState:
        """Reset to initial state and return it."""
        self._sim.reset()
        return self._to_game_state()

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        self._sim.set_seed(seed)

    def set_playing_as(self, team: int) -> None:
        """Set which team we're playing as (0=White, 1=Black)."""
        self._sim.set_playing_as(team)

    def clone(self) -> RustGameSimulator:
        """Clone the simulator (for MCTS tree exploration)."""
        new_sim = RustGameSimulator.__new__(RustGameSimulator)
        new_sim._sim = self._sim.clone_sim()
        return new_sim

    @property
    def state(self) -> GameState:
        """Get current game state as Python GameState."""
        return self._to_game_state()

    def generate_moves(self, team: Team) -> list[Move]:
        """Generate all legal moves for a team."""
        # Rust generates for current side, verify it matches
        assert team.value == self._sim.side_to_move, \
            f"Team mismatch: requested {team}, current {self._sim.side_to_move}"

        rust_moves = self._sim.generate_moves()
        pieces = self._sim.get_pieces()

        return [self._rust_move_to_python(rm, pieces) for rm in rust_moves]

    def make_move(self, move: Move) -> int:
        """Execute a move and return damage dealt."""
        # Find the matching Rust move
        rust_moves = self._sim.generate_moves()

        for rm in rust_moves:
            if self._moves_match(move, rm):
                return self._sim.make_move(rm)

        raise ValueError(f"Move not found in legal moves: {move}")

    def make_move_by_index(self, index: int) -> int:
        """Make a move by index (faster for MCTS)."""
        return self._sim.make_move_by_index(index)

    def get_piece_at(self, x: int, y: int) -> Optional[Piece]:
        """Get piece at position."""
        rp = self._sim.get_piece_at(x, y)
        if rp is None:
            return None
        return self._rust_piece_to_python(rp)

    def is_on_board(self, x: int, y: int) -> bool:
        """Check if position is on the 8x8 board."""
        return 0 <= x < 8 and 0 <= y < 8

    def is_occupied(self, x: int, y: int) -> bool:
        """Check if position has a piece."""
        return self._sim.get_piece_at(x, y) is not None

    def num_moves(self) -> int:
        """Get number of legal moves (fast)."""
        return self._sim.num_moves()

    def _to_game_state(self) -> GameState:
        """Convert internal Rust state to Python GameState."""
        pieces = [self._rust_piece_to_python(rp) for rp in self._sim.get_pieces()]

        winner = None
        if self._sim.winner is not None:
            winner = Team(self._sim.winner)

        return GameState(
            pieces=pieces,
            side_to_move=Team(self._sim.side_to_move),
            royal_decree_active=None,  # TODO: expose from Rust if needed
            turn_number=self._sim.turn_number,
            moves_without_damage=0,  # TODO: expose from Rust if needed
            position_history=[],
            playing_as=Team.PLAYER,
            winner=winner,
            is_terminal=self._sim.is_terminal,
            is_draw=self._sim.is_draw,
        )

    def _rust_piece_to_python(self, rp: "RustPiece") -> Piece:
        """Convert Rust piece to Python Piece."""
        return Piece(
            piece_type=PieceType(rp.piece_type),
            team=Team(rp.team),
            x=rp.x,
            y=rp.y,
            current_hp=rp.current_hp,
            max_hp=rp.max_hp,
            base_damage=rp.base_damage,
            ability_cooldown=rp.ability_cooldown,
            ability_used_this_match=rp.ability_used_this_match,
            interpose_active=rp.interpose_active,
            was_pawn=rp.was_pawn,
            advance_cooldown_turns=rp.advance_cooldown_turns,
        )

    def _rust_move_to_python(self, rm: "RustMove", pieces: list["RustPiece"]) -> Move:
        """Convert Rust move to Python Move."""
        rp = pieces[rm.piece_idx]
        piece = self._rust_piece_to_python(rp)

        attack_pos = None
        if rm.attack_x is not None and rm.attack_y is not None:
            attack_pos = (rm.attack_x, rm.attack_y)

        ability_target = None
        if rm.ability_target_x is not None and rm.ability_target_y is not None:
            ability_target = (rm.ability_target_x, rm.ability_target_y)

        ability_id = None
        if rm.ability_id is not None:
            ability_id = AbilityId(rm.ability_id)

        return Move(
            piece=piece,
            move_type=MoveType(rm.move_type),
            from_pos=(rm.from_x, rm.from_y),
            to_pos=(rm.to_x, rm.to_y),
            attack_pos=attack_pos,
            ability_target=ability_target,
            ability_id=ability_id,
        )

    def _moves_match(self, py_move: Move, rust_move: "RustMove") -> bool:
        """Check if Python move matches Rust move."""
        if py_move.move_type.value != rust_move.move_type:
            return False
        if py_move.from_pos != (rust_move.from_x, rust_move.from_y):
            return False
        if py_move.to_pos != (rust_move.to_x, rust_move.to_y):
            return False

        # Check attack position
        if py_move.attack_pos is not None:
            if rust_move.attack_x is None or rust_move.attack_y is None:
                return False
            if py_move.attack_pos != (rust_move.attack_x, rust_move.attack_y):
                return False
        elif rust_move.attack_x is not None:
            return False

        # Check ability
        if py_move.ability_id is not None:
            if rust_move.ability_id is None:
                return False
            if py_move.ability_id.value != rust_move.ability_id:
                return False
        elif rust_move.ability_id is not None:
            return False

        return True


def get_simulator(use_rust: bool = True, state: Optional[GameState] = None):
    """
    Get a game simulator, preferring Rust if available.

    Args:
        use_rust: Whether to use Rust implementation if available
        state: Optional initial state (only works with Python impl)

    Returns:
        GameSimulator instance (Rust or Python)
    """
    if use_rust and RUST_AVAILABLE and state is None:
        return RustGameSimulator()
    else:
        from .game_simulator import GameSimulator
        return GameSimulator(state)
