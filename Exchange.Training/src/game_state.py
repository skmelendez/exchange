"""
Game State Representation for EXCHANGE

This module provides a Python representation of the EXCHANGE board state
that can be efficiently converted to tensors for neural network training.

The state encoding is designed to capture all information relevant to
position evaluation:
- Piece positions (12 binary planes: 6 types x 2 teams)
- HP information (6 normalized planes per piece type)
- Ability cooldowns (6 normalized planes)
- Special state flags (Royal Decree active, Interpose active, etc.)
- Side to move
"""

from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class PieceType(IntEnum):
    """Piece types matching C# enum order."""
    KING = 0
    QUEEN = 1
    ROOK = 2
    BISHOP = 3
    KNIGHT = 4
    PAWN = 5


class Team(IntEnum):
    """Teams matching C# enum order."""
    PLAYER = 0
    ENEMY = 1


class AbilityId(IntEnum):
    """Ability IDs matching C# enum order."""
    ROYAL_DECREE = 0   # King: Once per match, all allied rolls +1 until next turn
    OVEREXTEND = 1     # Queen: Move then attack, take 2 self-damage (3 turn CD)
    INTERPOSE = 2      # Rook: Damage to adjacent allies split with Rook (3 turn CD)
    CONSECRATION = 3   # Bishop: Heal diagonal ally 1d6 HP (3 turn CD)
    SKIRMISH = 4       # Knight: Attack then reposition 1 tile (3 turn CD)
    ADVANCE = 5        # Pawn: Move forward extra tile (1 turn CD)


# Map piece types to their abilities
PIECE_ABILITIES = {
    PieceType.KING: AbilityId.ROYAL_DECREE,
    PieceType.QUEEN: AbilityId.OVEREXTEND,
    PieceType.ROOK: AbilityId.INTERPOSE,
    PieceType.BISHOP: AbilityId.CONSECRATION,
    PieceType.KNIGHT: AbilityId.SKIRMISH,
    PieceType.PAWN: AbilityId.ADVANCE,
}


# Piece stats from PieceData.cs
PIECE_STATS = {
    PieceType.KING:   {"max_hp": 25, "base_damage": 1, "cooldown_max": -1},  # -1 = once per match
    PieceType.QUEEN:  {"max_hp": 10, "base_damage": 3, "cooldown_max": 3},
    PieceType.ROOK:   {"max_hp": 13, "base_damage": 2, "cooldown_max": 3},
    PieceType.BISHOP: {"max_hp": 10, "base_damage": 2, "cooldown_max": 3},
    PieceType.KNIGHT: {"max_hp": 11, "base_damage": 2, "cooldown_max": 3},
    PieceType.PAWN:   {"max_hp":  7, "base_damage": 1, "cooldown_max": 1},
}


@dataclass
class Piece:
    """Represents a single piece on the board."""
    piece_type: PieceType
    team: Team
    x: int
    y: int
    current_hp: int
    max_hp: int
    base_damage: int
    ability_cooldown: int = 0
    ability_used_this_match: bool = False  # For once-per-match abilities
    interpose_active: bool = False  # Rook special
    was_pawn: bool = False  # Track promoted pieces
    advance_cooldown_turns: int = 0  # Pawn: cannot use Advance consecutively (counts down each turn)

    @classmethod
    def create(cls, piece_type: PieceType, team: Team, x: int, y: int) -> Piece:
        """Create a piece with default stats."""
        stats = PIECE_STATS[piece_type]
        return cls(
            piece_type=piece_type,
            team=team,
            x=x,
            y=y,
            current_hp=stats["max_hp"],
            max_hp=stats["max_hp"],
            base_damage=stats["base_damage"],
        )

    @property
    def is_alive(self) -> bool:
        return self.current_hp > 0

    @property
    def can_use_ability(self) -> bool:
        stats = PIECE_STATS[self.piece_type]
        if stats["cooldown_max"] == -1:  # Once per match
            return not self.ability_used_this_match
        return self.ability_cooldown == 0

    def clone(self) -> Piece:
        """Deep copy of this piece."""
        return Piece(
            piece_type=self.piece_type,
            team=self.team,
            x=self.x,
            y=self.y,
            current_hp=self.current_hp,
            max_hp=self.max_hp,
            base_damage=self.base_damage,
            ability_cooldown=self.ability_cooldown,
            ability_used_this_match=self.ability_used_this_match,
            interpose_active=self.interpose_active,
            was_pawn=self.was_pawn,
            advance_cooldown_turns=self.advance_cooldown_turns,
        )


# Draw condition constants
DRAW_MOVES_WITHOUT_DAMAGE = 30  # Draw if no damage dealt in 30 moves
DRAW_REPETITION_COUNT = 3  # Draw if same position occurs 3 times


@dataclass
class GameState:
    """
    Complete game state for EXCHANGE.

    Mirrors C# SimulatedBoardState but optimized for ML training.
    """
    pieces: list[Piece] = field(default_factory=list)
    side_to_move: Team = Team.PLAYER
    royal_decree_active: Team | None = None  # Which team has buff active
    turn_number: int = 0
    moves_without_damage: int = 0  # Counter for 30-move draw rule
    position_history: list[int] = field(default_factory=list)  # For threefold repetition

    # Color awareness - which side the network is "playing as" for the whole game
    # This lets the network learn color-specific strategies (e.g., White needs to be aggressive)
    playing_as: Team = Team.PLAYER

    # Result tracking for training
    winner: Team | None = None
    is_terminal: bool = False
    is_draw: bool = False  # True if game ended in draw

    def position_hash(self) -> int:
        """
        Compute a hash of the current position for repetition detection.
        Includes: piece positions, types, teams, HP, and side to move.
        """
        # Sort pieces for consistent ordering
        piece_data = []
        for p in self.pieces:
            if p.is_alive:
                piece_data.append((p.x, p.y, int(p.piece_type), int(p.team), p.current_hp))
        piece_data.sort()
        return hash((tuple(piece_data), int(self.side_to_move)))

    def clone(self, copy_history: bool = False) -> GameState:
        """Deep copy of this game state. Set copy_history=True only when needed."""
        return GameState(
            pieces=[p.clone() for p in self.pieces],
            side_to_move=self.side_to_move,
            royal_decree_active=self.royal_decree_active,
            turn_number=self.turn_number,
            moves_without_damage=self.moves_without_damage,
            position_history=self.position_history.copy() if copy_history else [],
            playing_as=self.playing_as,
            winner=self.winner,
            is_terminal=self.is_terminal,
            is_draw=self.is_draw,
        )

    def record_position(self) -> None:
        """Record current position hash for repetition detection."""
        self.position_history.append(self.position_hash())

    def is_threefold_repetition(self) -> bool:
        """Check if current position has occurred 3 times."""
        if len(self.position_history) < DRAW_REPETITION_COUNT:
            return False
        current_hash = self.position_hash()
        count = self.position_history.count(current_hash)
        return count >= DRAW_REPETITION_COUNT

    def get_piece_at(self, x: int, y: int) -> Optional[Piece]:
        """Get piece at board position."""
        for p in self.pieces:
            if p.x == x and p.y == y and p.is_alive:
                return p
        return None

    def get_pieces(self, team: Team) -> list[Piece]:
        """Get all alive pieces for a team."""
        return [p for p in self.pieces if p.team == team and p.is_alive]

    def get_king(self, team: Team) -> Optional[Piece]:
        """Get the king for a team."""
        for p in self.pieces:
            if p.team == team and p.piece_type == PieceType.KING and p.is_alive:
                return p
        return None

    def check_terminal(self) -> None:
        """Check if game has ended (king died or draw condition met)."""
        player_king = self.get_king(Team.PLAYER)
        enemy_king = self.get_king(Team.ENEMY)

        # Win condition: opponent's king is dead
        if player_king is None or not player_king.is_alive:
            self.is_terminal = True
            self.winner = Team.ENEMY
            return
        elif enemy_king is None or not enemy_king.is_alive:
            self.is_terminal = True
            self.winner = Team.PLAYER
            return

        # Draw condition 1: Only kings left (insufficient material)
        alive_pieces = [p for p in self.pieces if p.is_alive]
        non_king_pieces = [p for p in alive_pieces if p.piece_type != PieceType.KING]
        if len(non_king_pieces) == 0:
            self.is_terminal = True
            self.is_draw = True
            self.winner = None
            return

        # Draw condition 2: 30 moves without damage
        if self.moves_without_damage >= DRAW_MOVES_WITHOUT_DAMAGE:
            self.is_terminal = True
            self.is_draw = True
            self.winner = None
            return

        # Draw condition 3: Threefold repetition
        if self.is_threefold_repetition():
            self.is_terminal = True
            self.is_draw = True
            self.winner = None
            return

    def _get_attack_squares(self, piece: Piece) -> list[tuple[int, int]]:
        """Get all squares a piece can attack (used for attack maps)."""
        attacks = []
        x, y = piece.x, piece.y

        if piece.piece_type == PieceType.KING:
            # King attacks adjacent squares
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 8 and 0 <= ny < 8:
                        attacks.append((nx, ny))

        elif piece.piece_type == PieceType.QUEEN:
            # Queen = Rook + Bishop (sliding)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                for dist in range(1, 8):
                    nx, ny = x + dx * dist, y + dy * dist
                    if not (0 <= nx < 8 and 0 <= ny < 8):
                        break
                    attacks.append((nx, ny))
                    if self.get_piece_at(nx, ny):  # Blocked
                        break

        elif piece.piece_type == PieceType.ROOK:
            # Rook attacks horizontally and vertically
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                for dist in range(1, 8):
                    nx, ny = x + dx * dist, y + dy * dist
                    if not (0 <= nx < 8 and 0 <= ny < 8):
                        break
                    attacks.append((nx, ny))
                    if self.get_piece_at(nx, ny):
                        break

        elif piece.piece_type == PieceType.BISHOP:
            # Bishop attacks diagonally
            for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                for dist in range(1, 8):
                    nx, ny = x + dx * dist, y + dy * dist
                    if not (0 <= nx < 8 and 0 <= ny < 8):
                        break
                    attacks.append((nx, ny))
                    if self.get_piece_at(nx, ny):
                        break

        elif piece.piece_type == PieceType.KNIGHT:
            # Knight L-shape attacks
            for dx, dy in [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 8 and 0 <= ny < 8:
                    attacks.append((nx, ny))

        elif piece.piece_type == PieceType.PAWN:
            # Pawns attack diagonally forward
            direction = 1 if piece.team == Team.PLAYER else -1
            for dx in [-1, 1]:
                nx, ny = x + dx, y + direction
                if 0 <= nx < 8 and 0 <= ny < 8:
                    attacks.append((nx, ny))

        return attacks

    def to_tensor(self) -> np.ndarray:
        """
        Convert game state to tensor representation.

        Returns:
            numpy array of shape (C, 8, 8) where C is the number of channels.

        Channel Layout (58 total):
            0-5:   White piece positions (King, Queen, Rook, Bishop, Knight, Pawn)
            6-11:  Black piece positions
            12-17: HP planes (normalized by max HP)
            18-23: Cooldown planes (normalized)
            24:    Royal Decree active for side to move
            25:    Interpose positions
            26:    Side to move (1 = White, 0 = Black)
            27:    Playing as (1 = White, 0 = Black)

            === STRATEGIC AWARENESS CHANNELS ===
            28:    Turn number (normalized by 300)
            29:    Moves without damage (normalized by 30) - DRAW URGENCY
            30:    Position repetition count (normalized: 0, 0.33, 0.67, 1.0)
            31:    Total HP ratio (white_hp / total_hp)
            32:    Piece count ratio (white_pieces / total_pieces)
            33:    Material balance (normalized weighted value difference)
            34:    White attack map (squares White can attack)
            35:    Black attack map (squares Black can attack)
            36:    Contested squares (both sides can attack)
            37:    White king zone (king position + adjacent squares)
            38:    Black king zone (king position + adjacent squares)
            39:    White pawn advancement (normalized y position)
            40:    Black pawn advancement (normalized y position)
            41:    White abilities ready (count / piece count)
            42:    Black abilities ready (count / piece count)
            43:    Low HP targets (pieces with <40% HP)
            44:    Center control bonus (central squares weighted)
            45:    King tropism (distance between kings, normalized)

            === TACTICAL CHANNELS ===
            46:    Hanging white pieces (attacked, not defended)
            47:    Hanging black pieces (attacked, not defended)
            48:    Defended white pieces
            49:    Defended black pieces
            50:    White king attackers (count of black pieces attacking king zone)
            51:    Black king attackers (count of white pieces attacking king zone)
            52:    Safe squares for white king
            53:    Safe squares for black king
            54:    Passed pawns white (no enemy pawns blocking)
            55:    Passed pawns black
            56:    Open files (no pawns - rook highways)
            57:    Damage potential map (attacker damage sum per square)
        """
        tensor = np.zeros((INPUT_CHANNELS, 8, 8), dtype=np.float32)

        # Collect piece info
        white_pieces = self.get_pieces(Team.PLAYER)
        black_pieces = self.get_pieces(Team.ENEMY)
        white_king = self.get_king(Team.PLAYER)
        black_king = self.get_king(Team.ENEMY)

        # === ORIGINAL CHANNELS (0-27) ===
        for piece in self.pieces:
            if not piece.is_alive:
                continue

            x, y = piece.x, piece.y
            pt = int(piece.piece_type)
            team_offset = 0 if piece.team == Team.PLAYER else 6

            # Piece position plane
            tensor[team_offset + pt, y, x] = 1.0

            # HP plane (normalized by max HP)
            hp_ratio = piece.current_hp / piece.max_hp
            tensor[12 + pt, y, x] = hp_ratio

            # Cooldown plane (normalized by max cooldown)
            stats = PIECE_STATS[piece.piece_type]
            if stats["cooldown_max"] > 0:
                cd_ratio = piece.ability_cooldown / stats["cooldown_max"]
                tensor[18 + pt, y, x] = cd_ratio
            elif stats["cooldown_max"] == -1:  # Once per match
                tensor[18 + pt, y, x] = 1.0 if piece.ability_used_this_match else 0.0

            # Interpose plane
            if piece.interpose_active:
                tensor[25, y, x] = 1.0

        # Royal Decree plane
        if self.royal_decree_active == self.side_to_move:
            tensor[24, :, :] = 1.0

        # Side to move plane
        if self.side_to_move == Team.PLAYER:
            tensor[26, :, :] = 1.0

        # Playing as plane
        if self.playing_as == Team.PLAYER:
            tensor[27, :, :] = 1.0

        # === NEW STRATEGIC CHANNELS (28-45) ===

        # 28: Turn number (normalized by 300 - most games end before this)
        tensor[28, :, :] = min(self.turn_number / 300.0, 1.0)

        # 29: Moves without damage - DRAW URGENCY (critical for endgame!)
        tensor[29, :, :] = self.moves_without_damage / 30.0

        # 30: Position repetition count
        if self.position_history:
            current_hash = self.position_hash()
            rep_count = self.position_history.count(current_hash)
            tensor[30, :, :] = min(rep_count / 3.0, 1.0)

        # 31: Total HP ratio
        white_hp = sum(p.current_hp for p in white_pieces)
        black_hp = sum(p.current_hp for p in black_pieces)
        total_hp = white_hp + black_hp
        if total_hp > 0:
            tensor[31, :, :] = white_hp / total_hp
        else:
            tensor[31, :, :] = 0.5

        # 32: Piece count ratio
        total_pieces = len(white_pieces) + len(black_pieces)
        if total_pieces > 0:
            tensor[32, :, :] = len(white_pieces) / total_pieces
        else:
            tensor[32, :, :] = 0.5

        # 33: Material balance (weighted by piece values: Q=9, R=5, B=3, N=3, P=1, K=0)
        piece_values = {PieceType.KING: 0, PieceType.QUEEN: 9, PieceType.ROOK: 5,
                       PieceType.BISHOP: 3, PieceType.KNIGHT: 3, PieceType.PAWN: 1}
        white_material = sum(piece_values[p.piece_type] for p in white_pieces)
        black_material = sum(piece_values[p.piece_type] for p in black_pieces)
        max_material = 39  # Q + 2R + 2B + 2N + 8P = 9 + 10 + 6 + 6 + 8 = 39
        material_diff = (white_material - black_material) / max_material  # Range: -1 to 1
        tensor[33, :, :] = (material_diff + 1) / 2  # Normalize to 0-1

        # 34-35: Attack maps (cache per-piece attacks for reuse)
        white_attacks = set()
        black_attacks = set()
        white_piece_attacks = {}  # Cache: piece -> set of attack squares
        black_piece_attacks = {}
        for p in white_pieces:
            attacks = set(self._get_attack_squares(p))
            white_piece_attacks[id(p)] = attacks
            white_attacks.update(attacks)
        for p in black_pieces:
            attacks = set(self._get_attack_squares(p))
            black_piece_attacks[id(p)] = attacks
            black_attacks.update(attacks)

        for ax, ay in white_attacks:
            tensor[34, ay, ax] = 1.0
        for ax, ay in black_attacks:
            tensor[35, ay, ax] = 1.0

        # 36: Contested squares (both sides attack)
        contested = white_attacks & black_attacks
        for cx, cy in contested:
            tensor[36, cy, cx] = 1.0

        # 37-38: King zones (king + adjacent squares)
        if white_king:
            kx, ky = white_king.x, white_king.y
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = kx + dx, ky + dy
                    if 0 <= nx < 8 and 0 <= ny < 8:
                        tensor[37, ny, nx] = 1.0

        if black_king:
            kx, ky = black_king.x, black_king.y
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = kx + dx, ky + dy
                    if 0 <= nx < 8 and 0 <= ny < 8:
                        tensor[38, ny, nx] = 1.0

        # 39-40: Pawn advancement (how far pawns have pushed)
        for p in white_pieces:
            if p.piece_type == PieceType.PAWN:
                advancement = p.y / 7.0  # 0 at rank 0, 1 at rank 7
                tensor[39, p.y, p.x] = advancement
        for p in black_pieces:
            if p.piece_type == PieceType.PAWN:
                advancement = (7 - p.y) / 7.0  # 0 at rank 7, 1 at rank 0
                tensor[40, p.y, p.x] = advancement

        # 41-42: Abilities ready ratio
        white_abilities_ready = sum(1 for p in white_pieces if p.can_use_ability)
        black_abilities_ready = sum(1 for p in black_pieces if p.can_use_ability)
        if white_pieces:
            tensor[41, :, :] = white_abilities_ready / len(white_pieces)
        if black_pieces:
            tensor[42, :, :] = black_abilities_ready / len(black_pieces)

        # 43: Low HP targets (pieces with <40% HP - vulnerable!)
        for p in self.pieces:
            if p.is_alive and p.current_hp / p.max_hp < 0.4:
                tensor[43, p.y, p.x] = 1.0

        # 44: Center control bonus (central 4x4 weighted higher)
        center_weights = np.zeros((8, 8), dtype=np.float32)
        # Inner center (d4, d5, e4, e5)
        center_weights[3:5, 3:5] = 1.0
        # Extended center
        center_weights[2:6, 2:6] = np.maximum(center_weights[2:6, 2:6], 0.5)
        # Mark controlled central squares
        for cx, cy in white_attacks:
            tensor[44, cy, cx] += center_weights[cy, cx] * 0.5
        for cx, cy in black_attacks:
            tensor[44, cy, cx] -= center_weights[cy, cx] * 0.5
        tensor[44] = (tensor[44] + 1) / 2  # Normalize to 0-1

        # 45: King tropism (distance between kings - important for endgame)
        if white_king and black_king:
            king_dist = abs(white_king.x - black_king.x) + abs(white_king.y - black_king.y)
            # Normalize: max distance is 14 (corner to corner Manhattan), closer = higher value
            tensor[45, :, :] = 1.0 - (king_dist / 14.0)

        # === ADDITIONAL TACTICAL CHANNELS (46-57) ===
        # Reuse white_attacks/black_attacks as defense maps (same thing - squares pieces can reach)

        # 46: Hanging white pieces (attacked by black, NOT defended by white)
        for p in white_pieces:
            pos = (p.x, p.y)
            if pos in black_attacks and pos not in white_attacks:
                tensor[46, p.y, p.x] = 1.0

        # 47: Hanging black pieces (attacked by white, NOT defended by black)
        for p in black_pieces:
            pos = (p.x, p.y)
            if pos in white_attacks and pos not in black_attacks:
                tensor[47, p.y, p.x] = 1.0

        # 48: Defended white pieces (white pieces protected by other white pieces)
        for p in white_pieces:
            pos = (p.x, p.y)
            if pos in white_attacks:
                tensor[48, p.y, p.x] = 1.0

        # 49: Defended black pieces
        for p in black_pieces:
            pos = (p.x, p.y)
            if pos in black_attacks:
                tensor[49, p.y, p.x] = 1.0

        # 50: White king attackers (how many black pieces attack white king zone)
        # Reuse cached king zones from channels 37-38
        if white_king:
            white_king_zone = set()
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = white_king.x + dx, white_king.y + dy
                    if 0 <= nx < 8 and 0 <= ny < 8:
                        white_king_zone.add((nx, ny))
            # Use cached attacks instead of recomputing
            attackers = sum(1 for p in black_pieces if black_piece_attacks[id(p)] & white_king_zone)
            tensor[50, :, :] = min(attackers / 6.0, 1.0)

        # 51: Black king attackers (how many white pieces attack black king zone)
        if black_king:
            black_king_zone = set()
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = black_king.x + dx, black_king.y + dy
                    if 0 <= nx < 8 and 0 <= ny < 8:
                        black_king_zone.add((nx, ny))
            # Use cached attacks instead of recomputing
            attackers = sum(1 for p in white_pieces if white_piece_attacks[id(p)] & black_king_zone)
            tensor[51, :, :] = min(attackers / 6.0, 1.0)

        # 52: Safe squares for white king (king can move there, not attacked by black)
        if white_king:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = white_king.x + dx, white_king.y + dy
                    if 0 <= nx < 8 and 0 <= ny < 8:
                        # Safe if not attacked by black and not occupied by friendly
                        if (nx, ny) not in black_attacks:
                            occupant = self.get_piece_at(nx, ny)
                            if occupant is None or occupant.team != Team.PLAYER:
                                tensor[52, ny, nx] = 1.0

        # 53: Safe squares for black king
        if black_king:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = black_king.x + dx, black_king.y + dy
                    if 0 <= nx < 8 and 0 <= ny < 8:
                        if (nx, ny) not in white_attacks:
                            occupant = self.get_piece_at(nx, ny)
                            if occupant is None or occupant.team != Team.ENEMY:
                                tensor[53, ny, nx] = 1.0

        # 54: Passed pawns white (no enemy pawns blocking or adjacent to their file ahead)
        for p in white_pieces:
            if p.piece_type == PieceType.PAWN:
                is_passed = True
                for check_y in range(p.y + 1, 8):  # Check squares ahead
                    for check_x in [p.x - 1, p.x, p.x + 1]:  # File and adjacent files
                        if 0 <= check_x < 8:
                            blocker = self.get_piece_at(check_x, check_y)
                            if blocker and blocker.team == Team.ENEMY and blocker.piece_type == PieceType.PAWN:
                                is_passed = False
                                break
                    if not is_passed:
                        break
                if is_passed:
                    tensor[54, p.y, p.x] = 1.0

        # 55: Passed pawns black
        for p in black_pieces:
            if p.piece_type == PieceType.PAWN:
                is_passed = True
                for check_y in range(p.y - 1, -1, -1):  # Check squares ahead (downward for black)
                    for check_x in [p.x - 1, p.x, p.x + 1]:
                        if 0 <= check_x < 8:
                            blocker = self.get_piece_at(check_x, check_y)
                            if blocker and blocker.team == Team.PLAYER and blocker.piece_type == PieceType.PAWN:
                                is_passed = False
                                break
                    if not is_passed:
                        break
                if is_passed:
                    tensor[55, p.y, p.x] = 1.0

        # 56: Open files (files with no pawns - highways for rooks)
        for file_x in range(8):
            has_pawn = False
            for check_y in range(8):
                piece = self.get_piece_at(file_x, check_y)
                if piece and piece.piece_type == PieceType.PAWN:
                    has_pawn = True
                    break
            if not has_pawn:
                tensor[56, :, file_x] = 1.0

        # 57: Damage potential map (sum of attacker base damage per square)
        # Use cached attacks instead of recomputing
        for p in white_pieces:
            for ax, ay in white_piece_attacks[id(p)]:
                tensor[57, ay, ax] += p.base_damage / 10.0  # Normalize
        for p in black_pieces:
            for ax, ay in black_piece_attacks[id(p)]:
                tensor[57, ay, ax] -= p.base_damage / 10.0  # Negative for black
        tensor[57] = (tensor[57] + 1) / 2  # Normalize to 0-1

        return tensor

    @classmethod
    def from_json(cls, data: dict) -> GameState:
        """Load game state from JSON (output from C# simulator)."""
        pieces = []
        for p_data in data.get("pieces", []):
            piece = Piece(
                piece_type=PieceType(p_data["pieceType"]),
                team=Team(p_data["team"]),
                x=p_data["x"],
                y=p_data["y"],
                current_hp=p_data["currentHp"],
                max_hp=p_data["maxHp"],
                base_damage=p_data["baseDamage"],
                ability_cooldown=p_data.get("abilityCooldown", 0),
                ability_used_this_match=p_data.get("abilityUsedThisMatch", False),
                interpose_active=p_data.get("interposeActive", False),
                was_pawn=p_data.get("wasPawn", False),
            )
            pieces.append(piece)

        return cls(
            pieces=pieces,
            side_to_move=Team(data.get("sideToMove", 0)),
            royal_decree_active=Team(data["royalDecreeActive"]) if data.get("royalDecreeActive") is not None else None,
            turn_number=data.get("turnNumber", 0),
            moves_without_damage=data.get("movesWithoutDamage", 0),
            winner=Team(data["winner"]) if data.get("winner") is not None else None,
            is_terminal=data.get("isTerminal", False),
            is_draw=data.get("isDraw", False),
        )

    def to_json(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "pieces": [
                {
                    "pieceType": int(p.piece_type),
                    "team": int(p.team),
                    "x": p.x,
                    "y": p.y,
                    "currentHp": p.current_hp,
                    "maxHp": p.max_hp,
                    "baseDamage": p.base_damage,
                    "abilityCooldown": p.ability_cooldown,
                    "abilityUsedThisMatch": p.ability_used_this_match,
                    "interposeActive": p.interpose_active,
                    "wasPawn": p.was_pawn,
                }
                for p in self.pieces
            ],
            "sideToMove": int(self.side_to_move),
            "royalDecreeActive": int(self.royal_decree_active) if self.royal_decree_active is not None else None,
            "turnNumber": self.turn_number,
            "movesWithoutDamage": self.moves_without_damage,
            "winner": int(self.winner) if self.winner is not None else None,
            "isTerminal": self.is_terminal,
            "isDraw": self.is_draw,
        }


def create_initial_state() -> GameState:
    """Create the standard starting position for EXCHANGE."""
    state = GameState()

    # Standard chess-like setup
    # Player/White (bottom, rows 0-1)
    back_rank = [PieceType.ROOK, PieceType.KNIGHT, PieceType.BISHOP, PieceType.QUEEN,
                 PieceType.KING, PieceType.BISHOP, PieceType.KNIGHT, PieceType.ROOK]

    for x, pt in enumerate(back_rank):
        state.pieces.append(Piece.create(pt, Team.PLAYER, x, 0))
    for x in range(8):
        state.pieces.append(Piece.create(PieceType.PAWN, Team.PLAYER, x, 1))

    # Enemy/Black (top, rows 6-7)
    for x, pt in enumerate(back_rank):
        state.pieces.append(Piece.create(pt, Team.ENEMY, x, 7))
    for x in range(8):
        state.pieces.append(Piece.create(PieceType.PAWN, Team.ENEMY, x, 6))

    return state


# Tensor shape constants for network architecture
INPUT_CHANNELS = 58  # 28 original + 18 strategic + 12 tactical channels
BOARD_SIZE = 8


# Channel names for logging/visualization
CHANNEL_NAMES = [
    # Original channels (0-27)
    "white_king", "white_queen", "white_rook", "white_bishop", "white_knight", "white_pawn",
    "black_king", "black_queen", "black_rook", "black_bishop", "black_knight", "black_pawn",
    "hp_king", "hp_queen", "hp_rook", "hp_bishop", "hp_knight", "hp_pawn",
    "cd_king", "cd_queen", "cd_rook", "cd_bishop", "cd_knight", "cd_pawn",
    "royal_decree_active", "interpose_positions", "side_to_move", "playing_as",
    # Strategic channels (28-45)
    "turn_number", "moves_without_damage", "position_repetition", "hp_ratio",
    "piece_count_ratio", "material_balance", "white_attack_map", "black_attack_map",
    "contested_squares", "white_king_zone", "black_king_zone", "white_pawn_advance",
    "black_pawn_advance", "white_abilities_ready", "black_abilities_ready", "low_hp_targets",
    "center_control", "king_tropism",
    # Tactical channels (46-57)
    "hanging_white", "hanging_black", "defended_white", "defended_black",
    "white_king_attackers", "black_king_attackers", "safe_squares_white_king", "safe_squares_black_king",
    "passed_pawns_white", "passed_pawns_black", "open_files", "damage_potential",
]
