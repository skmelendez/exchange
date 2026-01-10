using Godot;
using Exchange.Core;
using Exchange.Board;

namespace Exchange.Pieces;

/// <summary>
/// Knight: 11 HP, 2 Base Damage
/// Movement: L-shape (2+1 or 1+2), can jump over pieces
/// Attack: Adjacent only
/// SPECIAL RULE: Knight may move THEN base attack in same turn
///   - Must move first
///   - Attack must be adjacent to landing square
///   - Cannot attack without moving
///   - Cannot move again after attacking
///   - Using ability replaces this behavior
/// Ability: Skirmish (3 turn CD) - Attack then reposition 1 tile
/// </summary>
public partial class KnightPiece : BasePiece
{
    public bool HasMovedThisTurn { get; set; } = false;

    public KnightPiece(Team team) : base(PieceType.Knight, team) { }

    public override List<Vector2I> GetValidMoves(GameBoard board)
    {
        var moves = new List<Vector2I>();

        // Knight moves in L-shape, can jump
        foreach (var offset in Vector2IExtensions.KnightMoves)
        {
            var pos = BoardPosition + offset;
            if (!pos.IsOnBoard()) continue;
            if (board.IsOccupied(pos)) continue;

            moves.Add(pos);
        }

        return moves;
    }

    public override List<Vector2I> GetAttackablePositions(GameBoard board)
    {
        var positions = new List<Vector2I>();

        // Knight can only attack CARDINAL adjacent tiles (up/down/left/right, NOT diagonal)
        foreach (var dir in Vector2IExtensions.CardinalDirections)
        {
            var pos = BoardPosition + dir;
            if (!pos.IsOnBoard()) continue;

            var piece = board.GetPieceAt(pos);
            if (piece != null && piece.Team != Team)
                positions.Add(pos);
        }

        return positions;
    }

    public override List<Vector2I> GetThreatenedPositions(GameBoard board)
    {
        // Knight threatens CARDINAL adjacent tiles only (up/down/left/right, NOT diagonal)
        var positions = new List<Vector2I>();
        foreach (var dir in Vector2IExtensions.CardinalDirections)
        {
            var pos = BoardPosition + dir;
            if (pos.IsOnBoard())
                positions.Add(pos);
        }
        return positions;
    }

    /// <summary>
    /// Gets valid skirmish reposition tiles (1 tile in any direction after attack)
    /// </summary>
    public List<Vector2I> GetSkirmishRepositionTiles(GameBoard board)
    {
        var positions = new List<Vector2I>();
        foreach (var dir in Vector2IExtensions.AllDirections)
        {
            var pos = BoardPosition + dir;
            if (pos.IsOnBoard() && !board.IsOccupied(pos))
                positions.Add(pos);
        }
        return positions;
    }

    public override void ResetTurnState()
    {
        base.ResetTurnState();
        HasMovedThisTurn = false;
    }
}
