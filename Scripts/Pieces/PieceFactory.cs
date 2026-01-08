using Exchange.Core;

namespace Exchange.Pieces;

/// <summary>
/// Factory for creating piece instances
/// </summary>
public static class PieceFactory
{
    public static BasePiece Create(PieceType type, Team team)
    {
        return type switch
        {
            PieceType.King => new KingPiece(team),
            PieceType.Queen => new QueenPiece(team),
            PieceType.Rook => new RookPiece(team),
            PieceType.Bishop => new BishopPiece(team),
            PieceType.Knight => new KnightPiece(team),
            PieceType.Pawn => new PawnPiece(team),
            _ => throw new ArgumentException($"Unknown piece type: {type}")
        };
    }
}
