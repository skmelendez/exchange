using Godot;
using Exchange.Core;
using Exchange.Board;

namespace Exchange.Pieces;

/// <summary>
/// Rook: 13 HP, 2 Base Damage
/// Movement: Any distance in cardinal directions (blocked by pieces)
/// Attack: Adjacent or short straight-line range (2-3 tiles based on doc interpretation)
/// Ability: Interpose (3 turn CD) - Damage to adjacent allies split with Rook
/// </summary>
public partial class RookPiece : BasePiece
{
    private const int AttackRange = 2; // "short straight-line range"

    /// <summary>
    /// Tracks if Interpose ability is currently active (damage to adjacent allies split with Rook)
    /// </summary>
    public bool InterposeActive { get; set; } = false;

    public RookPiece(Team team) : base(PieceType.Rook, team) { }

    public override void ResetTurnState()
    {
        base.ResetTurnState();
        // Interpose lasts until the Rook's next turn, then deactivates
        InterposeActive = false;
    }

    public override List<Vector2I> GetValidMoves(GameBoard board)
    {
        var moves = new List<Vector2I>();

        // Rook moves in cardinal directions until blocked
        foreach (var dir in Vector2IExtensions.CardinalDirections)
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

        // Adjacent attacks (all 8 directions)
        foreach (var dir in Vector2IExtensions.AllDirections)
        {
            var pos = BoardPosition + dir;
            if (!pos.IsOnBoard()) continue;

            var piece = board.GetPieceAt(pos);
            if (piece != null && piece.Team != Team)
                positions.Add(pos);
        }

        // Short-range straight line attacks (cardinal only, up to AttackRange)
        foreach (var dir in Vector2IExtensions.CardinalDirections)
        {
            var current = BoardPosition + dir;
            for (int i = 0; i < AttackRange && current.IsOnBoard(); i++)
            {
                var piece = board.GetPieceAt(current);
                if (piece != null)
                {
                    if (piece.Team != Team && !positions.Contains(current))
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

        // Threatens adjacent tiles
        foreach (var dir in Vector2IExtensions.AllDirections)
        {
            var pos = BoardPosition + dir;
            if (pos.IsOnBoard())
                positions.Add(pos);
        }

        // Threatens short-range cardinal directions
        foreach (var dir in Vector2IExtensions.CardinalDirections)
        {
            var current = BoardPosition + dir;
            for (int i = 0; i < AttackRange && current.IsOnBoard(); i++)
            {
                positions.Add(current);
                if (board.IsOccupied(current))
                    break;
                current += dir;
            }
        }

        return positions.Distinct().ToList();
    }
}
