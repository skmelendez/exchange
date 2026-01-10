using Godot;
using Exchange.Core;
using Exchange.Board;

namespace Exchange.Pieces;

/// <summary>
/// Queen: 10 HP, 3 Base Damage
/// Movement: Any distance straight or diagonal (blocked by pieces)
/// Attack: Any distance straight or diagonal (blocked by pieces)
/// Ability: Overextend (3 turn CD) - Move then attack, take 2 damage afterward
/// </summary>
public partial class QueenPiece : BasePiece
{
    private const int AttackRange = 3; // Limited snipe range for tactical play

    public QueenPiece(Team team) : base(PieceType.Queen, team) { }

    public override List<Vector2I> GetValidMoves(GameBoard board)
    {
        var moves = new List<Vector2I>();

        // Queen moves in all 8 directions until blocked
        foreach (var dir in Vector2IExtensions.AllDirections)
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

        // Queen attacks in all 8 directions, limited range
        foreach (var dir in Vector2IExtensions.AllDirections)
        {
            var current = BoardPosition + dir;
            for (int range = 1; range <= AttackRange && current.IsOnBoard(); range++)
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

        // Queen threatens tiles in line of sight, limited range
        foreach (var dir in Vector2IExtensions.AllDirections)
        {
            var current = BoardPosition + dir;
            for (int range = 1; range <= AttackRange && current.IsOnBoard(); range++)
            {
                positions.Add(current);
                if (board.IsOccupied(current))
                    break;
                current += dir;
            }
        }

        return positions;
    }
}
