using Godot;
using Exchange.Core;
using Exchange.Board;

namespace Exchange.Pieces;

/// <summary>
/// Bishop: 10 HP, 2 Base Damage
/// Movement: Any distance diagonally (blocked by pieces)
/// Attack: Any distance diagonally, CANNOT attack adjacent
/// Ability: Consecration (3 turn CD) - Heal diagonal ally 1d6 HP (cannot self-target)
/// </summary>
public partial class BishopPiece : BasePiece
{
    private const int AttackRange = 3; // Limited snipe range for tactical play

    public BishopPiece(Team team) : base(PieceType.Bishop, team) { }

    public override List<Vector2I> GetValidMoves(GameBoard board)
    {
        var moves = new List<Vector2I>();

        // Bishop moves diagonally until blocked
        foreach (var dir in Vector2IExtensions.DiagonalDirections)
        {
            var current = BoardPosition + dir;
            while (current.IsOnBoard())
            {
                if (board.IsOccupied(current))
                    break;
                moves.Add(current);
                current += dir;
            }
        }

        return moves;
    }

    public override List<Vector2I> GetAttackablePositions(GameBoard board)
    {
        var positions = new List<Vector2I>();

        // Bishop attacks diagonally but CANNOT attack adjacent tiles, limited range
        foreach (var dir in Vector2IExtensions.DiagonalDirections)
        {
            var current = BoardPosition + dir;

            // Skip adjacent tile (range 1)
            if (!current.IsOnBoard()) continue;
            if (board.IsOccupied(current))
                continue; // Adjacent is blocked, can't shoot through

            current += dir; // Start from 2 tiles away

            // Continue checking up to AttackRange (starting from range 2)
            for (int range = 2; range <= AttackRange && current.IsOnBoard(); range++)
            {
                var piece = board.GetPieceAt(current);
                if (piece != null)
                {
                    if (piece.Team != Team)
                        positions.Add(current);
                    break;
                }
                current += dir;
            }
        }

        return positions;
    }

    public override List<Vector2I> GetThreatenedPositions(GameBoard board)
    {
        var positions = new List<Vector2I>();

        // Bishop threatens diagonal tiles (except adjacent), limited range
        foreach (var dir in Vector2IExtensions.DiagonalDirections)
        {
            var current = BoardPosition + dir;

            // Skip adjacent - Bishop cannot attack adjacent
            if (!current.IsOnBoard()) continue;
            bool adjacentBlocked = board.IsOccupied(current);

            current += dir; // Start threatening from 2 tiles away

            // Continue checking up to AttackRange (starting from range 2)
            for (int range = 2; range <= AttackRange && current.IsOnBoard(); range++)
            {
                if (adjacentBlocked)
                    break; // Line of sight blocked by adjacent piece

                positions.Add(current);
                if (board.IsOccupied(current))
                    break;
                current += dir;
            }
        }

        return positions;
    }

    /// <summary>
    /// Gets valid targets for Consecration (diagonal allies, not self)
    /// </summary>
    public List<Vector2I> GetHealTargets(GameBoard board)
    {
        var targets = new List<Vector2I>();

        foreach (var dir in Vector2IExtensions.DiagonalDirections)
        {
            var current = BoardPosition + dir;
            while (current.IsOnBoard())
            {
                var piece = board.GetPieceAt(current);
                if (piece != null)
                {
                    if (piece.Team == Team && piece != this)
                        targets.Add(current);
                    break;
                }
                current += dir;
            }
        }

        return targets;
    }
}
