using Godot;
using Exchange.Core;
using Exchange.Board;

namespace Exchange.Pieces;

/// <summary>
/// Pawn: 7 HP, 1 Base Damage
/// Movement: One tile forward (toward enemy side)
/// Attack: Diagonal-adjacent forward only
/// Ability: Advance (1 turn CD) - Move forward one extra tile. Cannot use consecutively.
/// </summary>
public partial class PawnPiece : BasePiece
{
    public bool UsedAdvanceLastTurn { get; set; } = false;

    public PawnPiece(Team team) : base(PieceType.Pawn, team) { }

    /// <summary>
    /// Forward direction depends on team
    /// Player pawns move toward row 8 (+Y), Enemy pawns toward row 1 (-Y)
    /// </summary>
    public int ForwardDirection => Team == Team.Player ? 1 : -1;

    /// <summary>
    /// Starting row for this pawn (to check if it can move 2 squares)
    /// </summary>
    private int StartingRow => Team == Team.Player ? 1 : 6;

    public bool IsOnStartingRow => BoardPosition.Y == StartingRow;

    /// <summary>
    /// The row where this pawn promotes (row 7 for Player, row 0 for Enemy)
    /// </summary>
    private int PromotionRow => Team == Team.Player ? 7 : 0;

    /// <summary>
    /// Returns true if the pawn is on the promotion row and should promote
    /// </summary>
    public bool ShouldPromote => BoardPosition.Y == PromotionRow;

    /// <summary>
    /// Returns true if moving to the given position would result in promotion
    /// </summary>
    public bool WouldPromoteAt(Vector2I position) => position.Y == PromotionRow;

    public override List<Vector2I> GetValidMoves(GameBoard board)
    {
        var moves = new List<Vector2I>();

        // One tile forward
        var forward = new Vector2I(BoardPosition.X, BoardPosition.Y + ForwardDirection);
        if (forward.IsOnBoard() && !board.IsOccupied(forward))
        {
            moves.Add(forward);

            // Two tiles forward on first move (traditional chess rule)
            if (IsOnStartingRow)
            {
                var twoForward = new Vector2I(BoardPosition.X, BoardPosition.Y + (ForwardDirection * 2));
                if (twoForward.IsOnBoard() && !board.IsOccupied(twoForward))
                    moves.Add(twoForward);
            }
        }

        return moves;
    }

    public override List<Vector2I> GetAttackablePositions(GameBoard board)
    {
        var positions = new List<Vector2I>();

        // Pawn attacks diagonally forward only
        var leftDiag = new Vector2I(BoardPosition.X - 1, BoardPosition.Y + ForwardDirection);
        var rightDiag = new Vector2I(BoardPosition.X + 1, BoardPosition.Y + ForwardDirection);

        if (leftDiag.IsOnBoard())
        {
            var piece = board.GetPieceAt(leftDiag);
            if (piece != null && piece.Team != Team)
                positions.Add(leftDiag);
        }

        if (rightDiag.IsOnBoard())
        {
            var piece = board.GetPieceAt(rightDiag);
            if (piece != null && piece.Team != Team)
                positions.Add(rightDiag);
        }

        return positions;
    }

    public override List<Vector2I> GetThreatenedPositions(GameBoard board)
    {
        // Pawn threatens forward diagonals regardless of occupancy
        var positions = new List<Vector2I>();

        var leftDiag = new Vector2I(BoardPosition.X - 1, BoardPosition.Y + ForwardDirection);
        var rightDiag = new Vector2I(BoardPosition.X + 1, BoardPosition.Y + ForwardDirection);

        if (leftDiag.IsOnBoard())
            positions.Add(leftDiag);
        if (rightDiag.IsOnBoard())
            positions.Add(rightDiag);

        return positions;
    }

    /// <summary>
    /// Gets the Advance ability target tile (2 tiles forward if valid)
    /// </summary>
    public Vector2I? GetAdvanceTarget(GameBoard board)
    {
        // Cannot use consecutively
        if (UsedAdvanceLastTurn) return null;

        var oneForward = new Vector2I(BoardPosition.X, BoardPosition.Y + ForwardDirection);
        var twoForward = new Vector2I(BoardPosition.X, BoardPosition.Y + (ForwardDirection * 2));

        // Must have clear path
        if (!oneForward.IsOnBoard() || board.IsOccupied(oneForward))
            return null;

        if (!twoForward.IsOnBoard() || board.IsOccupied(twoForward))
            return null;

        return twoForward;
    }

    public override void ResetTurnState()
    {
        base.ResetTurnState();
        // Note: UsedAdvanceLastTurn is tracked across turns, reset separately
    }
}
