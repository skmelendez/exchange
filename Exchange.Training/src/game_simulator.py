"""
Game Simulator for EXCHANGE

Implements the complete game rules for self-play training.
This is a Python port of the core game logic from the C# codebase,
optimized for fast batch simulation without any visual/Godot overhead.

Key differences from chess:
- HP-based combat (damage = base_damage + dice roll 1-6)
- Win by reducing enemy King to 0 HP
- Pieces have unique attack patterns (different from movement!)
- Knight must move before attacking (move+attack combo)
- Various abilities with cooldowns
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np

from .game_state import GameState, Piece, PieceType, Team, PIECE_STATS, AbilityId, PIECE_ABILITIES


class MoveType(IntEnum):
    """Type of move/action."""
    MOVE = 0
    ATTACK = 1
    MOVE_AND_ATTACK = 2  # Knight special
    ABILITY = 3


@dataclass
class Move:
    """Represents a move in the game."""
    piece: Piece
    move_type: MoveType
    from_pos: tuple[int, int]
    to_pos: tuple[int, int]
    attack_pos: Optional[tuple[int, int]] = None  # For MoveAndAttack or Skirmish reposition
    ability_target: Optional[tuple[int, int]] = None  # Target for abilities like Consecration
    ability_id: Optional[AbilityId] = None  # Which ability is being used
    expected_damage: int = 0
    expected_heal: int = 0  # For Consecration

    def __repr__(self) -> str:
        if self.move_type == MoveType.MOVE:
            return f"{self.piece.piece_type.name} {_pos_to_chess(self.from_pos)}->{_pos_to_chess(self.to_pos)}"
        elif self.move_type == MoveType.ATTACK:
            return f"{self.piece.piece_type.name} x{_pos_to_chess(self.to_pos)}"
        elif self.move_type == MoveType.MOVE_AND_ATTACK:
            return f"Knight {_pos_to_chess(self.from_pos)}->{_pos_to_chess(self.to_pos)} x{_pos_to_chess(self.attack_pos)}"
        elif self.move_type == MoveType.ABILITY and self.ability_id is not None:
            ability_names = {
                AbilityId.ROYAL_DECREE: "Royal Decree",
                AbilityId.OVEREXTEND: "Overextend",
                AbilityId.INTERPOSE: "Interpose",
                AbilityId.CONSECRATION: "Consecration",
                AbilityId.SKIRMISH: "Skirmish",
                AbilityId.ADVANCE: "Advance",
            }
            return f"{self.piece.piece_type.name} {ability_names.get(self.ability_id, 'ABILITY')}"
        else:
            return f"{self.piece.piece_type.name} ABILITY"


def _pos_to_chess(pos: tuple[int, int]) -> str:
    """Convert (x, y) to chess notation."""
    return f"{chr(ord('a') + pos[0])}{pos[1] + 1}"


# Direction vectors
CARDINAL = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
ALL_DIRS = CARDINAL + DIAGONAL
KNIGHT_MOVES = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]


class GameSimulator:
    """
    Fast game simulator for self-play training.

    Handles move generation, move execution, and game outcome determination.
    Designed for maximum simulation throughput without any visualization.
    """

    def __init__(self, state: Optional[GameState] = None):
        """Initialize with optional starting state."""
        self.state = state or self._create_initial_state()
        self._rng = random.Random()

    def _create_initial_state(self) -> GameState:
        """Create standard starting position."""
        from .game_state import create_initial_state
        return create_initial_state()

    def reset(self) -> GameState:
        """Reset to initial state and return it."""
        self.state = self._create_initial_state()
        return self.state

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        self._rng.seed(seed)

    def is_on_board(self, x: int, y: int) -> bool:
        """Check if position is on the 8x8 board."""
        return 0 <= x < 8 and 0 <= y < 8

    def get_piece_at(self, x: int, y: int) -> Optional[Piece]:
        """Get piece at position."""
        return self.state.get_piece_at(x, y)

    def is_occupied(self, x: int, y: int) -> bool:
        """Check if position has a piece."""
        return self.get_piece_at(x, y) is not None

    def generate_moves(self, team: Team) -> list[Move]:
        """Generate all legal moves for a team."""
        moves: list[Move] = []

        for piece in self.state.get_pieces(team):
            # Generate attacks first (usually higher priority)
            if piece.piece_type != PieceType.KNIGHT:
                moves.extend(self._generate_attacks(piece))

            # Generate regular moves
            moves.extend(self._generate_moves(piece))

            # Knight move+attack combos
            if piece.piece_type == PieceType.KNIGHT:
                moves.extend(self._generate_knight_combos(piece))

            # Generate ability moves
            if piece.can_use_ability:
                moves.extend(self._generate_ability_moves(piece))

        return moves

    def _generate_moves(self, piece: Piece) -> list[Move]:
        """Generate movement options for a piece."""
        positions = self._get_move_positions(piece)
        return [
            Move(
                piece=piece,
                move_type=MoveType.MOVE,
                from_pos=(piece.x, piece.y),
                to_pos=pos,
            )
            for pos in positions
        ]

    def _generate_attacks(self, piece: Piece) -> list[Move]:
        """Generate attack options for a piece."""
        positions = self._get_attack_positions(piece)
        moves: list[Move] = []

        for x, y in positions:
            target = self.get_piece_at(x, y)
            if target and target.team != piece.team:
                expected_damage = piece.base_damage + 4  # Average dice roll
                moves.append(Move(
                    piece=piece,
                    move_type=MoveType.ATTACK,
                    from_pos=(piece.x, piece.y),
                    to_pos=(x, y),
                    expected_damage=expected_damage,
                ))

        return moves

    def _generate_knight_combos(self, piece: Piece) -> list[Move]:
        """Generate Knight move+attack combinations."""
        moves: list[Move] = []
        move_positions = self._get_knight_move_positions(piece)

        for move_x, move_y in move_positions:
            # From the landing position, check cardinal adjacent for attacks
            for dx, dy in CARDINAL:
                ax, ay = move_x + dx, move_y + dy
                if not self.is_on_board(ax, ay):
                    continue

                target = self.get_piece_at(ax, ay)
                if target and target.team != piece.team:
                    expected_damage = piece.base_damage + 4
                    moves.append(Move(
                        piece=piece,
                        move_type=MoveType.MOVE_AND_ATTACK,
                        from_pos=(piece.x, piece.y),
                        to_pos=(move_x, move_y),
                        attack_pos=(ax, ay),
                        expected_damage=expected_damage,
                    ))

        return moves

    def _get_move_positions(self, piece: Piece) -> list[tuple[int, int]]:
        """Get valid move positions for a piece."""
        if piece.piece_type == PieceType.KING:
            return self._get_king_moves(piece)
        elif piece.piece_type == PieceType.QUEEN:
            return self._get_sliding_moves(piece, ALL_DIRS)
        elif piece.piece_type == PieceType.ROOK:
            return self._get_sliding_moves(piece, CARDINAL)
        elif piece.piece_type == PieceType.BISHOP:
            return self._get_sliding_moves(piece, DIAGONAL)
        elif piece.piece_type == PieceType.KNIGHT:
            return self._get_knight_move_positions(piece)
        elif piece.piece_type == PieceType.PAWN:
            return self._get_pawn_moves(piece)
        return []

    def _get_attack_positions(self, piece: Piece) -> list[tuple[int, int]]:
        """Get valid attack positions for a piece."""
        if piece.piece_type == PieceType.KING:
            return self._get_adjacent_positions(piece)
        elif piece.piece_type == PieceType.QUEEN:
            return self._get_queen_attacks(piece)
        elif piece.piece_type == PieceType.ROOK:
            return self._get_rook_attacks(piece)
        elif piece.piece_type == PieceType.BISHOP:
            return self._get_bishop_attacks(piece)
        elif piece.piece_type == PieceType.KNIGHT:
            return []  # Knight must move first, handled by combos
        elif piece.piece_type == PieceType.PAWN:
            return self._get_pawn_attacks(piece)
        return []

    def _get_king_moves(self, piece: Piece) -> list[tuple[int, int]]:
        """King moves: adjacent squares."""
        positions = []
        for dx, dy in ALL_DIRS:
            nx, ny = piece.x + dx, piece.y + dy
            if self.is_on_board(nx, ny) and not self.is_occupied(nx, ny):
                positions.append((nx, ny))
        return positions

    def _get_sliding_moves(self, piece: Piece, directions: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Sliding piece moves (Queen, Rook, Bishop)."""
        positions = []
        for dx, dy in directions:
            nx, ny = piece.x + dx, piece.y + dy
            while self.is_on_board(nx, ny):
                if self.is_occupied(nx, ny):
                    break
                positions.append((nx, ny))
                nx += dx
                ny += dy
        return positions

    def _get_knight_move_positions(self, piece: Piece) -> list[tuple[int, int]]:
        """Knight L-shape moves."""
        positions = []
        for dx, dy in KNIGHT_MOVES:
            nx, ny = piece.x + dx, piece.y + dy
            if self.is_on_board(nx, ny) and not self.is_occupied(nx, ny):
                positions.append((nx, ny))
        return positions

    def _get_pawn_moves(self, piece: Piece) -> list[tuple[int, int]]:
        """Pawn forward moves."""
        positions = []
        direction = 1 if piece.team == Team.PLAYER else -1
        start_row = 1 if piece.team == Team.PLAYER else 6

        # Single step
        nx, ny = piece.x, piece.y + direction
        if self.is_on_board(nx, ny) and not self.is_occupied(nx, ny):
            positions.append((nx, ny))

            # Double step from starting position
            if piece.y == start_row:
                nx2, ny2 = piece.x, piece.y + 2 * direction
                if not self.is_occupied(nx2, ny2):
                    positions.append((nx2, ny2))

        return positions

    def _get_adjacent_positions(self, piece: Piece) -> list[tuple[int, int]]:
        """Get all adjacent positions with enemies."""
        positions = []
        for dx, dy in ALL_DIRS:
            nx, ny = piece.x + dx, piece.y + dy
            if self.is_on_board(nx, ny):
                target = self.get_piece_at(nx, ny)
                if target and target.team != piece.team:
                    positions.append((nx, ny))
        return positions

    def _get_queen_attacks(self, piece: Piece) -> list[tuple[int, int]]:
        """Queen attacks: all 8 directions, range 1-3, blocked."""
        QUEEN_RANGE = 3
        positions = []
        for dx, dy in ALL_DIRS:
            nx, ny = piece.x + dx, piece.y + dy
            for _ in range(QUEEN_RANGE):
                if not self.is_on_board(nx, ny):
                    break
                target = self.get_piece_at(nx, ny)
                if target:
                    if target.team != piece.team:
                        positions.append((nx, ny))
                    break
                nx += dx
                ny += dy
        return positions

    def _get_rook_attacks(self, piece: Piece) -> list[tuple[int, int]]:
        """Rook attacks: adjacent (8 dirs) + cardinal range 2."""
        ROOK_RANGE = 2
        positions = []

        # Adjacent (all 8 directions)
        for dx, dy in ALL_DIRS:
            nx, ny = piece.x + dx, piece.y + dy
            if self.is_on_board(nx, ny):
                target = self.get_piece_at(nx, ny)
                if target and target.team != piece.team:
                    positions.append((nx, ny))

        # Cardinal extended range
        for dx, dy in CARDINAL:
            nx, ny = piece.x + dx, piece.y + dy
            for _ in range(ROOK_RANGE):
                if not self.is_on_board(nx, ny):
                    break
                target = self.get_piece_at(nx, ny)
                if target:
                    if target.team != piece.team and (nx, ny) not in positions:
                        positions.append((nx, ny))
                    break
                nx += dx
                ny += dy

        return positions

    def _get_bishop_attacks(self, piece: Piece) -> list[tuple[int, int]]:
        """Bishop attacks: diagonal range 2-3 (NOT adjacent), blocked."""
        BISHOP_MIN = 2
        BISHOP_MAX = 3
        positions = []

        for dx, dy in DIAGONAL:
            nx, ny = piece.x, piece.y
            for dist in range(1, BISHOP_MAX + 1):
                nx += dx
                ny += dy
                if not self.is_on_board(nx, ny):
                    break

                target = self.get_piece_at(nx, ny)

                if dist < BISHOP_MIN:
                    # Can't attack adjacent, but blocked by pieces
                    if target:
                        break
                    continue

                # Range 2-3: can attack
                if target:
                    if target.team != piece.team:
                        positions.append((nx, ny))
                    break

        return positions

    def _get_pawn_attacks(self, piece: Piece) -> list[tuple[int, int]]:
        """Pawn attacks: forward diagonal only."""
        positions = []
        direction = 1 if piece.team == Team.PLAYER else -1

        for dx in [-1, 1]:
            nx, ny = piece.x + dx, piece.y + direction
            if self.is_on_board(nx, ny):
                target = self.get_piece_at(nx, ny)
                if target and target.team != piece.team:
                    positions.append((nx, ny))

        return positions

    def _generate_ability_moves(self, piece: Piece) -> list[Move]:
        """Generate ability moves for a piece that can use its ability."""
        moves: list[Move] = []
        ability_id = PIECE_ABILITIES.get(piece.piece_type)
        if ability_id is None:
            return moves

        if ability_id == AbilityId.ROYAL_DECREE:
            # King: Once per match, all allied rolls +1 until next turn
            # No target needed - just activate
            moves.append(Move(
                piece=piece,
                move_type=MoveType.ABILITY,
                from_pos=(piece.x, piece.y),
                to_pos=(piece.x, piece.y),  # Stays in place
                ability_id=AbilityId.ROYAL_DECREE,
            ))

        elif ability_id == AbilityId.OVEREXTEND:
            # Queen: Move then attack, take 2 self-damage
            # Generate all move positions, then from each, all attack positions
            move_positions = self._get_sliding_moves(piece, ALL_DIRS)
            for mx, my in move_positions:
                # From this move position, find attackable targets
                for dx, dy in ALL_DIRS:
                    nx, ny = mx + dx, my + dy
                    for dist in range(3):  # Queen attack range
                        if not self.is_on_board(nx, ny):
                            break
                        target = self.get_piece_at(nx, ny)
                        if target:
                            if target.team != piece.team:
                                expected_damage = piece.base_damage + 4
                                moves.append(Move(
                                    piece=piece,
                                    move_type=MoveType.ABILITY,
                                    from_pos=(piece.x, piece.y),
                                    to_pos=(mx, my),  # Move destination
                                    attack_pos=(nx, ny),  # Attack target
                                    ability_id=AbilityId.OVEREXTEND,
                                    expected_damage=expected_damage,
                                ))
                            break  # Blocked by piece
                        nx += dx
                        ny += dy

        elif ability_id == AbilityId.INTERPOSE:
            # Rook: Damage to adjacent allies split with Rook
            # Only useful if there are adjacent allies to protect
            has_adjacent_ally = False
            for dx, dy in ALL_DIRS:
                nx, ny = piece.x + dx, piece.y + dy
                if self.is_on_board(nx, ny):
                    ally = self.get_piece_at(nx, ny)
                    if ally and ally.team == piece.team and ally != piece:
                        has_adjacent_ally = True
                        break
            if has_adjacent_ally:
                moves.append(Move(
                    piece=piece,
                    move_type=MoveType.ABILITY,
                    from_pos=(piece.x, piece.y),
                    to_pos=(piece.x, piece.y),
                    ability_id=AbilityId.INTERPOSE,
                ))

        elif ability_id == AbilityId.CONSECRATION:
            # Bishop: Heal diagonal ally 1d6 HP
            # Find diagonal allies that are damaged
            for dx, dy in DIAGONAL:
                nx, ny = piece.x + dx, piece.y + dy
                while self.is_on_board(nx, ny):
                    target = self.get_piece_at(nx, ny)
                    if target:
                        if target.team == piece.team and target.current_hp < target.max_hp:
                            moves.append(Move(
                                piece=piece,
                                move_type=MoveType.ABILITY,
                                from_pos=(piece.x, piece.y),
                                to_pos=(piece.x, piece.y),
                                ability_target=(nx, ny),
                                ability_id=AbilityId.CONSECRATION,
                                expected_heal=4,  # Average of 1d6
                            ))
                        break  # Blocked by piece (ally or enemy)
                    nx += dx
                    ny += dy

        elif ability_id == AbilityId.SKIRMISH:
            # Knight: Attack then reposition 1 tile (can escape after attack)
            # From current position, attack adjacent enemy, then move to adjacent empty
            attack_positions = []
            for dx, dy in CARDINAL:
                ax, ay = piece.x + dx, piece.y + dy
                if self.is_on_board(ax, ay):
                    target = self.get_piece_at(ax, ay)
                    if target and target.team != piece.team:
                        attack_positions.append((ax, ay))

            for attack_pos in attack_positions:
                # After attacking, can reposition to any adjacent empty tile
                for dx, dy in ALL_DIRS:
                    rx, ry = piece.x + dx, piece.y + dy
                    if self.is_on_board(rx, ry) and not self.is_occupied(rx, ry):
                        expected_damage = piece.base_damage + 4
                        moves.append(Move(
                            piece=piece,
                            move_type=MoveType.ABILITY,
                            from_pos=(piece.x, piece.y),
                            to_pos=(rx, ry),  # Reposition destination
                            attack_pos=attack_pos,  # Attack target
                            ability_id=AbilityId.SKIRMISH,
                            expected_damage=expected_damage,
                        ))

        elif ability_id == AbilityId.ADVANCE:
            # Pawn: Move forward extra tile (2 tiles from any position)
            # CANNOT use on consecutive turns (C# rule: UsedAdvanceLastTurn)
            if piece.advance_cooldown_turns > 0:
                return moves  # Cannot use Advance this turn

            direction = 1 if piece.team == Team.PLAYER else -1
            # First tile must be empty
            nx1, ny1 = piece.x, piece.y + direction
            if self.is_on_board(nx1, ny1) and not self.is_occupied(nx1, ny1):
                # Second tile
                nx2, ny2 = piece.x, piece.y + 2 * direction
                if self.is_on_board(nx2, ny2) and not self.is_occupied(nx2, ny2):
                    moves.append(Move(
                        piece=piece,
                        move_type=MoveType.ABILITY,
                        from_pos=(piece.x, piece.y),
                        to_pos=(nx2, ny2),
                        ability_id=AbilityId.ADVANCE,
                    ))

        return moves

    def make_move(self, move: Move) -> int:
        """
        Execute a move and return damage dealt (if any).

        Returns the actual damage dealt for training signal.
        """
        piece = move.piece
        damage_dealt = 0

        if move.move_type == MoveType.MOVE:
            # Simple move
            piece.x, piece.y = move.to_pos

            # Check pawn promotion
            self._check_promotion(piece)

        elif move.move_type == MoveType.ATTACK:
            # Attack target
            target = self.get_piece_at(*move.to_pos)
            if target:
                damage_dealt = self._apply_damage(piece, target)

        elif move.move_type == MoveType.MOVE_AND_ATTACK:
            # Knight: move then attack
            piece.x, piece.y = move.to_pos
            target = self.get_piece_at(*move.attack_pos)
            if target:
                damage_dealt = self._apply_damage(piece, target)

        elif move.move_type == MoveType.ABILITY:
            damage_dealt = self._execute_ability(move)

        # Track moves without damage for draw condition
        if damage_dealt > 0:
            self.state.moves_without_damage = 0
        else:
            self.state.moves_without_damage += 1

        # Switch sides
        self.state.side_to_move = (
            Team.ENEMY if self.state.side_to_move == Team.PLAYER else Team.PLAYER
        )
        self.state.turn_number += 1

        # Tick cooldowns for the team that just moved
        for p in self.state.get_pieces(piece.team):
            if p.ability_cooldown > 0:
                p.ability_cooldown -= 1
            # Deactivate Interpose at start of Rook's team's turn
            if p.piece_type == PieceType.ROOK:
                p.interpose_active = False

        # Decrement Advance cooldown for pawns of the team about to move
        for p in self.state.get_pieces(self.state.side_to_move):
            if p.piece_type == PieceType.PAWN and p.advance_cooldown_turns > 0:
                p.advance_cooldown_turns -= 1

        # Record position for repetition detection, then check terminal
        self.state.record_position()
        self.state.check_terminal()

        return damage_dealt

    def _apply_damage(self, attacker: Piece, target: Piece) -> int:
        """Apply combat damage with dice roll."""
        dice_roll = self._rng.randint(1, 6)
        damage = attacker.base_damage + dice_roll

        # Royal Decree bonus
        if self.state.royal_decree_active == attacker.team:
            damage += 1

        # Apply damage (potentially split via Interpose)
        actual_damage = self._apply_damage_with_interpose(target, damage)

        return actual_damage

    def _apply_damage_with_interpose(self, target: Piece, damage: int) -> int:
        """Apply damage to target, potentially splitting with Interposing Rook."""
        # Check if there's an active Interposing Rook adjacent to target
        for rook in self.state.get_pieces(target.team):
            if (rook.piece_type == PieceType.ROOK and
                rook.interpose_active and
                rook != target):
                # Check if Rook is adjacent to target
                dx = abs(rook.x - target.x)
                dy = abs(rook.y - target.y)
                if dx <= 1 and dy <= 1 and (dx + dy) > 0:
                    # Split damage between target and Rook
                    half_damage = damage // 2
                    remainder = damage % 2
                    target.current_hp -= half_damage
                    rook.current_hp -= (half_damage + remainder)  # Rook takes odd remainder
                    return damage

        # No Interpose - apply full damage
        target.current_hp -= damage
        return damage

    def _execute_ability(self, move: Move) -> int:
        """Execute an ability move. Returns damage dealt."""
        piece = move.piece
        ability_id = move.ability_id
        damage_dealt = 0

        if ability_id == AbilityId.ROYAL_DECREE:
            # King: All allied rolls +1 until next turn
            self.state.royal_decree_active = piece.team
            piece.ability_used_this_match = True
            # No cooldown - once per match

        elif ability_id == AbilityId.OVEREXTEND:
            # Queen: Move then attack, take 2 self-damage
            piece.x, piece.y = move.to_pos
            if move.attack_pos:
                target = self.get_piece_at(*move.attack_pos)
                if target:
                    damage_dealt = self._apply_damage(piece, target)
            # Self-damage
            piece.current_hp -= 2
            piece.ability_cooldown = PIECE_STATS[PieceType.QUEEN]["cooldown_max"]

        elif ability_id == AbilityId.INTERPOSE:
            # Rook: Activate damage splitting for adjacent allies
            piece.interpose_active = True
            piece.ability_cooldown = PIECE_STATS[PieceType.ROOK]["cooldown_max"]

        elif ability_id == AbilityId.CONSECRATION:
            # Bishop: Heal diagonal ally 1d6 HP
            if move.ability_target:
                target = self.get_piece_at(*move.ability_target)
                if target and target.team == piece.team:
                    heal = self._rng.randint(1, 6)
                    # Royal Decree bonus applies to heals too
                    if self.state.royal_decree_active == piece.team:
                        heal += 1
                    target.current_hp = min(target.max_hp, target.current_hp + heal)
            piece.ability_cooldown = PIECE_STATS[PieceType.BISHOP]["cooldown_max"]

        elif ability_id == AbilityId.SKIRMISH:
            # Knight: Attack then reposition 1 tile
            if move.attack_pos:
                target = self.get_piece_at(*move.attack_pos)
                if target:
                    damage_dealt = self._apply_damage(piece, target)
            # Reposition
            piece.x, piece.y = move.to_pos
            piece.ability_cooldown = PIECE_STATS[PieceType.KNIGHT]["cooldown_max"]

        elif ability_id == AbilityId.ADVANCE:
            # Pawn: Move forward extra tile
            piece.x, piece.y = move.to_pos
            self._check_promotion(piece)
            piece.ability_cooldown = PIECE_STATS[PieceType.PAWN]["cooldown_max"]
            piece.advance_cooldown_turns = 2  # Cannot use Advance next turn (decrements each turn)

        return damage_dealt

    def _check_promotion(self, piece: Piece) -> None:
        """Check and apply pawn promotion."""
        if piece.piece_type != PieceType.PAWN:
            return

        promotion_row = 7 if piece.team == Team.PLAYER else 0
        if piece.y == promotion_row:
            # Promote to Queen (AI default)
            hp_ratio = piece.current_hp / piece.max_hp
            queen_stats = PIECE_STATS[PieceType.QUEEN]

            piece.piece_type = PieceType.QUEEN
            piece.max_hp = queen_stats["max_hp"]
            piece.current_hp = max(1, int(queen_stats["max_hp"] * hp_ratio))
            piece.base_damage = queen_stats["base_damage"]
            piece.was_pawn = True  # Track that this was a promoted pawn

    def play_random_game(self) -> tuple[GameState, list[tuple[np.ndarray, float]]]:
        """
        Play a random game for initial data generation.

        Game ends when:
        - A king dies (win/loss)
        - Only kings remain (draw - insufficient material)
        - 50 moves without damage (draw - stalemate)

        Returns:
            Tuple of (final_state, training_data)
            where training_data is list of (state_tensor, outcome_value)
        """
        history: list[tuple[np.ndarray, Team]] = []

        while not self.state.is_terminal:
            # Record state before move
            tensor = self.state.to_tensor()
            side = self.state.side_to_move
            history.append((tensor, side))

            # Generate and pick random move
            moves = self.generate_moves(self.state.side_to_move)
            if not moves:
                # No legal moves - shouldn't happen often
                break

            move = self._rng.choice(moves)
            self.make_move(move)

        # Determine outcome
        if self.state.winner == Team.PLAYER:
            player_value = 1.0
        elif self.state.winner == Team.ENEMY:
            player_value = -1.0
        else:
            player_value = 0.0  # Draw / timeout

        # Convert history to training data
        training_data: list[tuple[np.ndarray, float]] = []
        for tensor, side in history:
            if side == Team.PLAYER:
                value = player_value
            else:
                value = -player_value
            training_data.append((tensor, value))

        return self.state, training_data


def play_self_play_game(
    simulator: GameSimulator,
    network: "torch.nn.Module",
    temperature: float = 1.0,
) -> tuple[GameState, list[tuple[np.ndarray, float]]]:
    """
    Play a self-play game using the network for move selection.

    Uses softmax temperature for exploration during training.

    Game ends when:
    - A king dies (win/loss)
    - Only kings remain (draw - insufficient material)
    - 50 moves without damage (draw - stalemate)
    """
    import torch

    history: list[tuple[np.ndarray, Team]] = []
    simulator.reset()

    while not simulator.state.is_terminal:
        # Record state
        tensor = simulator.state.to_tensor()
        side = simulator.state.side_to_move
        history.append((tensor, side))

        # Generate moves
        moves = simulator.generate_moves(simulator.state.side_to_move)
        if not moves:
            break

        # Evaluate each move
        move_values: list[float] = []
        for move in moves:
            # Clone state, make move, evaluate
            sim_copy = GameSimulator(simulator.state.clone())
            sim_copy.make_move(Move(
                piece=sim_copy.get_piece_at(move.piece.x, move.piece.y),  # Get piece from copied state
                move_type=move.move_type,
                from_pos=move.from_pos,
                to_pos=move.to_pos,
                attack_pos=move.attack_pos,
            ))

            # Get value from network
            with torch.no_grad():
                state_tensor = torch.tensor(sim_copy.state.to_tensor()).unsqueeze(0)
                value = network(state_tensor).item()

            # Negate because we evaluated from opponent's perspective
            move_values.append(-value)

        # Select move with temperature
        if temperature > 0:
            values = np.array(move_values)
            exp_values = np.exp(values / temperature)
            probs = exp_values / exp_values.sum()
            move_idx = np.random.choice(len(moves), p=probs)
        else:
            move_idx = np.argmax(move_values)

        # Execute selected move
        simulator.make_move(moves[move_idx])

    # Determine outcome
    if simulator.state.winner == Team.PLAYER:
        player_value = 1.0
    elif simulator.state.winner == Team.ENEMY:
        player_value = -1.0
    else:
        player_value = 0.0

    # Convert to training data
    training_data: list[tuple[np.ndarray, float]] = []
    for tensor, side in history:
        value = player_value if side == Team.PLAYER else -player_value
        training_data.append((tensor, value))

    return simulator.state, training_data
