using Exchange.Core;

namespace Exchange.AI;

/// <summary>
/// Orders moves for better alpha-beta pruning efficiency.
/// Good move ordering dramatically reduces search time.
///
/// Priority order:
/// 1. Hash move (best move from previous search / transposition table)
/// 2. King captures (game-winning)
/// 3. Winning captures (capturing higher value pieces)
/// 4. Equal captures
/// 5. Losing captures (capturing lower value pieces)
/// 6. Killer moves (moves that caused cutoffs at this depth)
/// 7. Regular moves (ordered by positional heuristics)
/// </summary>
public class MoveOrderer
{
    private static readonly Dictionary<PieceType, int> PieceValues = new()
    {
        { PieceType.King, 10000 },
        { PieceType.Queen, 900 },
        { PieceType.Rook, 500 },
        { PieceType.Bishop, 330 },
        { PieceType.Knight, 320 },
        { PieceType.Pawn, 100 }
    };

    // MVV-LVA (Most Valuable Victim - Least Valuable Attacker) table
    // Higher score = better capture to search first
    // Capturing a Queen with a Pawn is much better than capturing a Pawn with a Queen
    private static readonly int[,] MvvLvaTable = new int[6, 6];

    static MoveOrderer()
    {
        // Initialize MVV-LVA scores
        // Index: [victim][attacker]
        PieceType[] pieces = { PieceType.Pawn, PieceType.Knight, PieceType.Bishop, PieceType.Rook, PieceType.Queen, PieceType.King };

        for (int victim = 0; victim < 6; victim++)
        {
            for (int attacker = 0; attacker < 6; attacker++)
            {
                // Score = victim value * 10 - attacker value
                // This prioritizes capturing valuable pieces with cheap pieces
                int victimValue = PieceValues[pieces[victim]];
                int attackerValue = PieceValues[pieces[attacker]];
                MvvLvaTable[victim, attacker] = victimValue * 10 - attackerValue;
            }
        }
    }

    /// <summary>
    /// Order moves in-place for better pruning
    /// </summary>
    public void OrderMoves(List<SimulatedMove> moves, SimulatedBoardState state, TranspositionTable tt)
    {
        // Get hash move from transposition table
        var hashMove = tt.GetBestMove(state.ZobristHash);

        // Score each move
        var scores = new int[moves.Count];
        for (int i = 0; i < moves.Count; i++)
        {
            scores[i] = ScoreMove(moves[i], hashMove);
        }

        // Sort by score (descending) - simple insertion sort for small lists
        for (int i = 1; i < moves.Count; i++)
        {
            var move = moves[i];
            var score = scores[i];
            int j = i - 1;

            while (j >= 0 && scores[j] < score)
            {
                moves[j + 1] = moves[j];
                scores[j + 1] = scores[j];
                j--;
            }

            moves[j + 1] = move;
            scores[j + 1] = score;
        }
    }

    private int ScoreMove(SimulatedMove move, SimulatedMove? hashMove)
    {
        int score = 0;

        // Hash move gets highest priority
        if (hashMove.HasValue && MovesEqual(move, hashMove.Value))
        {
            return 1_000_000;
        }

        // Attacks/captures (including MoveAndAttack for Knight)
        if (move.MoveType == SimulatedMoveType.Attack || move.MoveType == SimulatedMoveType.MoveAndAttack)
        {
            // King capture = instant win - always search first!
            if (move.CapturedPiece?.PieceType == PieceType.King)
            {
                return 900_000;
            }

            // Will this attack kill the target?
            if (move.CapturedPiece != null)
            {
                // Killing captures - use MVV-LVA
                int victimIndex = GetPieceIndex(move.CapturedPiece.PieceType);
                int attackerIndex = GetPieceIndex(move.Piece.PieceType);
                int mvvlva = MvvLvaTable[victimIndex, attackerIndex];

                // Check if this is a winning trade (capturing more valuable piece)
                int victimValue = PieceValues[move.CapturedPiece.PieceType];
                int attackerValue = PieceValues[move.Piece.PieceType];

                if (victimValue >= attackerValue)
                {
                    // Good or equal trade - high priority
                    score += 100_000 + mvvlva;
                }
                else
                {
                    // Losing trade (capturing less valuable piece) - still search, but lower priority
                    // This will be validated by the evaluation function
                    score += 50_000 + mvvlva;
                }
            }
            else
            {
                // Attack that won't kill - much lower priority
                // These chip damage attacks should be searched after good captures
                // but before bad moves
                score += 5_000;
            }
        }
        else if (move.MoveType == SimulatedMoveType.Move)
        {
            // Regular moves - use positional heuristics
            score += ScorePositionalMove(move);
        }

        return score;
    }

    private int ScorePositionalMove(SimulatedMove move)
    {
        int score = 0;

        // Center control bonus
        int toCenterDist = GetCenterDistance(move.ToPos);
        int fromCenterDist = GetCenterDistance(move.FromPos);

        if (toCenterDist < fromCenterDist)
        {
            score += (fromCenterDist - toCenterDist) * 10;
        }

        // Advancement bonus (except for King)
        if (move.Piece.PieceType != PieceType.King)
        {
            int advDir = move.Piece.Team == Team.Player ? 1 : -1;
            int advancement = (move.ToPos.Y - move.FromPos.Y) * advDir;
            if (advancement > 0)
            {
                score += advancement * 5;
            }
        }

        // Development bonus for knights/bishops from back rank
        if ((move.Piece.PieceType == PieceType.Knight || move.Piece.PieceType == PieceType.Bishop))
        {
            int backRank = move.Piece.Team == Team.Player ? 0 : 7;
            if (move.FromPos.Y == backRank && move.ToPos.Y != backRank)
            {
                score += 20;
            }
        }

        return score;
    }

    private static int GetCenterDistance(Godot.Vector2I pos)
    {
        int dx = Math.Min(Math.Abs(pos.X - 3), Math.Abs(pos.X - 4));
        int dy = Math.Min(Math.Abs(pos.Y - 3), Math.Abs(pos.Y - 4));
        return Math.Max(dx, dy);
    }

    private static int GetPieceIndex(PieceType type) => type switch
    {
        PieceType.Pawn => 0,
        PieceType.Knight => 1,
        PieceType.Bishop => 2,
        PieceType.Rook => 3,
        PieceType.Queen => 4,
        PieceType.King => 5,
        _ => 0
    };

    private static bool MovesEqual(SimulatedMove a, SimulatedMove b)
    {
        if (a.FromPos != b.FromPos ||
            a.ToPos != b.ToPos ||
            a.MoveType != b.MoveType ||
            a.Piece.PieceType != b.Piece.PieceType ||
            a.Piece.Team != b.Piece.Team)
        {
            return false;
        }

        // For MoveAndAttack, also compare AttackPos
        if (a.MoveType == SimulatedMoveType.MoveAndAttack)
        {
            return a.AttackPos == b.AttackPos;
        }

        return true;
    }
}
