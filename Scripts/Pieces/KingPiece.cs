using Godot;
using Exchange.Core;
using Exchange.Board;

namespace Exchange.Pieces;

/// <summary>
/// King: 15 HP, 1 Base Damage
/// Movement: One tile in any direction
/// Attack: Adjacent (all directions)
/// Ability: Royal Decree (once per match) - All allied combat rolls +1 until next turn
/// Special: Cannot move into threatened tiles
/// </summary>
public partial class KingPiece : BasePiece
{
    public KingPiece(Team team) : base(PieceType.King, team) { }

    public override List<Vector2I> GetValidMoves(GameBoard board)
    {
        var moves = new List<Vector2I>();

        foreach (var dir in Vector2IExtensions.AllDirections)
        {
            var pos = BoardPosition + dir;
            if (!pos.IsOnBoard()) continue;

            // Cannot move to occupied tiles
            if (board.IsOccupied(pos)) continue;

            // King cannot move into threatened tiles
            var tile = board.GetTile(pos);
            if (tile.IsThreatened(Team)) continue;

            moves.Add(pos);
        }

        return moves;
    }

    public override List<Vector2I> GetAttackablePositions(GameBoard board)
    {
        var positions = new List<Vector2I>();

        // King attacks adjacent tiles (all 8 directions)
        foreach (var dir in Vector2IExtensions.AllDirections)
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
        // King threatens all adjacent tiles regardless of occupancy
        var positions = new List<Vector2I>();
        foreach (var dir in Vector2IExtensions.AllDirections)
        {
            var pos = BoardPosition + dir;
            if (pos.IsOnBoard())
                positions.Add(pos);
        }
        return positions;
    }
}
