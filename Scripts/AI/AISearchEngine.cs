using Godot;
using Exchange.Core;
using Exchange.Board;
using Exchange.Pieces;

namespace Exchange.AI;

/// <summary>
/// High-performance minimax search engine with alpha-beta pruning and transposition tables.
/// Supports up to 5-ply lookahead with incremental cache reuse between moves.
/// </summary>
public class AISearchEngine
{
    private readonly TranspositionTable _transpositionTable = new();
    private readonly MoveOrderer _moveOrderer = new();

    private int _nodesSearched;
    private int _cacheHits;
    private int _cutoffs;

    /// <summary>
    /// Configuration for the search
    /// </summary>
    public int MaxDepth { get; set; } = 2;

    /// <summary>
    /// Find the best move using iterative deepening with alpha-beta search.
    /// </summary>
    public SearchResult FindBestMove(GameBoard board, Team sideToMove, int depth)
    {
        MaxDepth = depth;
        _nodesSearched = 0;
        _cacheHits = 0;
        _cutoffs = 0;

        // Create simulation state from real board
        var state = SimulatedBoardState.FromGameBoard(board);

        // Iterative deepening - helps with move ordering and allows time control
        SearchResult? bestResult = null;
        for (int d = 1; d <= depth; d++)
        {
            var result = SearchRoot(state, sideToMove, d);
            if (result.BestMove != null)
            {
                bestResult = result;
                GameLogger.Debug("AI", $"Depth {d}: {result.BestMove.Value.ToDebugString()} = {result.Score} ({_nodesSearched} nodes, {_cacheHits} cache hits, {_cutoffs} cutoffs)");
            }
        }

        GameLogger.Debug("AI", $"Search complete: {_nodesSearched} total nodes, {_cacheHits} cache hits, {_cutoffs} cutoffs");

        return bestResult ?? new SearchResult(null, 0, "No moves found");
    }

    /// <summary>
    /// Called after a move is made to age the transposition table.
    /// Entries from previous positions remain valid for transpositions.
    /// </summary>
    public void OnMoveMade()
    {
        _transpositionTable.AgeEntries();
    }

    /// <summary>
    /// Clear the transposition table (e.g., new game)
    /// </summary>
    public void ClearCache()
    {
        _transpositionTable.Clear();
    }

    private SearchResult SearchRoot(SimulatedBoardState state, Team sideToMove, int depth)
    {
        var moves = GenerateAllMoves(state, sideToMove);
        if (moves.Count == 0)
            return new SearchResult(null, sideToMove == Team.Enemy ? -99999 : 99999, "No moves");

        // Order moves for better pruning
        _moveOrderer.OrderMoves(moves, state, _transpositionTable);

        SimulatedMove? bestMove = null;
        int bestScore = int.MinValue;
        int alpha = int.MinValue;
        int beta = int.MaxValue;
        string bestReason = "";

        foreach (var move in moves)
        {
            state.MakeMove(move);

            // Negamax formulation: negate score and swap alpha/beta
            int score = -AlphaBeta(state, GetOpponent(sideToMove), depth - 1, -beta, -alpha);

            state.UndoMove(move);

            if (score > bestScore)
            {
                bestScore = score;
                bestMove = move;
                bestReason = move.Reason;
            }

            alpha = Math.Max(alpha, score);
        }

        return new SearchResult(bestMove, bestScore, bestReason);
    }

    private int AlphaBeta(SimulatedBoardState state, Team sideToMove, int depth, int alpha, int beta)
    {
        _nodesSearched++;

        // Check transposition table
        ulong hash = state.ZobristHash;
        if (_transpositionTable.TryGet(hash, depth, alpha, beta, out int ttScore))
        {
            _cacheHits++;
            return ttScore;
        }

        // Terminal conditions
        if (depth == 0)
        {
            int eval = Evaluate(state, sideToMove);
            _transpositionTable.Store(hash, depth, eval, TranspositionTable.NodeType.Exact, null);
            return eval;
        }

        // Check for king capture (game over)
        if (!state.HasKing(Team.Player))
            return sideToMove == Team.Enemy ? 50000 + depth : -50000 - depth;
        if (!state.HasKing(Team.Enemy))
            return sideToMove == Team.Player ? 50000 + depth : -50000 - depth;

        var moves = GenerateAllMoves(state, sideToMove);
        if (moves.Count == 0)
        {
            // No moves = bad (simplified, not checkmate detection)
            return -10000 + depth;
        }

        _moveOrderer.OrderMoves(moves, state, _transpositionTable);

        int originalAlpha = alpha;
        SimulatedMove? bestMove = null;

        foreach (var move in moves)
        {
            state.MakeMove(move);
            int score = -AlphaBeta(state, GetOpponent(sideToMove), depth - 1, -beta, -alpha);
            state.UndoMove(move);

            if (score >= beta)
            {
                _cutoffs++;
                _transpositionTable.Store(hash, depth, beta, TranspositionTable.NodeType.LowerBound, move);
                return beta; // Beta cutoff
            }

            if (score > alpha)
            {
                alpha = score;
                bestMove = move;
            }
        }

        // Store in transposition table
        var nodeType = alpha > originalAlpha
            ? TranspositionTable.NodeType.Exact
            : TranspositionTable.NodeType.UpperBound;
        _transpositionTable.Store(hash, depth, alpha, nodeType, bestMove);

        return alpha;
    }

    private List<SimulatedMove> GenerateAllMoves(SimulatedBoardState state, Team team)
    {
        var moves = new List<SimulatedMove>();
        var pieces = state.GetPieces(team);

        foreach (var piece in pieces)
        {
            // Generate attacks (usually higher priority) - except Knight which needs move first
            if (piece.PieceType != PieceType.Knight)
            {
                var attacks = GetAttackMoves(state, piece);
                moves.AddRange(attacks);
            }

            // Generate regular moves
            var regularMoves = GetRegularMoves(state, piece);
            moves.AddRange(regularMoves);

            // Generate Knight move+attack combos (Knight MUST move first, then attacks CARDINAL adjacent)
            if (piece.PieceType == PieceType.Knight)
            {
                var knightCombos = GetKnightMoveAttackCombos(state, piece);
                moves.AddRange(knightCombos);
            }

            // TODO: Generate ability moves if needed (Skirmish = attack first, then move)
        }

        return moves;
    }

    /// <summary>
    /// Generate Knight move+attack combos: Knight moves to position, then attacks CARDINAL adjacent.
    /// This is a compound move where the Knight must move first.
    /// </summary>
    private List<SimulatedMove> GetKnightMoveAttackCombos(SimulatedBoardState state, SimulatedPiece knight)
    {
        var combos = new List<SimulatedMove>();
        var movePositions = GetKnightMoves(state, knight);

        foreach (var movePos in movePositions)
        {
            // Simulate Knight at new position and check cardinal adjacent for attacks
            var attacksFromNewPos = GetKnightAttacksFromPosition(state, movePos, knight.Team);

            foreach (var attackPos in attacksFromNewPos)
            {
                var target = state.GetPieceAt(attackPos);
                if (target == null || target.Team == knight.Team) continue;

                // Calculate expected damage
                int expectedDamage = knight.BaseDamage + 4; // base + avg dice
                bool willKill = target.CurrentHp <= expectedDamage;

                string reason;
                if (willKill)
                {
                    if (target.PieceType == PieceType.King)
                        reason = $"Knight CHECKMATE: Move to {movePos.ToChessNotation()}, kill King!";
                    else
                        reason = $"Knight: Move to {movePos.ToChessNotation()}, kill {target.PieceType}";
                }
                else
                {
                    int hpAfter = target.CurrentHp - expectedDamage;
                    reason = $"Knight: Move to {movePos.ToChessNotation()}, attack {target.PieceType} ({expectedDamage} dmg, {hpAfter} HP left)";
                }

                combos.Add(new SimulatedMove
                {
                    Piece = knight,
                    FromPos = knight.Position,
                    ToPos = movePos,           // Knight moves here first
                    AttackPos = attackPos,     // Then attacks this square
                    MoveType = SimulatedMoveType.MoveAndAttack,
                    CapturedPiece = willKill ? target : null,
                    DamageDealt = expectedDamage,
                    Reason = reason
                });
            }
        }

        return combos;
    }

    private List<SimulatedMove> GetAttackMoves(SimulatedBoardState state, SimulatedPiece piece)
    {
        var moves = new List<SimulatedMove>();
        var attackPositions = GetAttackablePositions(state, piece);

        foreach (var targetPos in attackPositions)
        {
            var target = state.GetPieceAt(targetPos);
            if (target == null || target.Team == piece.Team) continue;

            // Calculate expected damage (base + average dice roll of 4)
            int expectedDamage = piece.BaseDamage + 4;
            bool willKill = target.CurrentHp <= expectedDamage;

            // Check if attacker is under counter-attack threat
            bool attackerThreatened = IsSquareAttacked(state, piece.Position, GetOpponent(piece.Team));

            string reason;
            if (willKill)
            {
                if (target.PieceType == PieceType.King)
                    reason = "CHECKMATE: Kill King!";
                else if (attackerThreatened)
                    reason = $"Trade: Kill {target.PieceType} (we're threatened)";
                else
                    reason = $"Kill {target.PieceType}";
            }
            else
            {
                int targetHpAfter = target.CurrentHp - expectedDamage;
                reason = $"Attack {target.PieceType} ({expectedDamage} dmg, {targetHpAfter} HP left)";
            }

            moves.Add(new SimulatedMove
            {
                Piece = piece,
                FromPos = piece.Position,
                ToPos = targetPos,
                MoveType = SimulatedMoveType.Attack,
                CapturedPiece = willKill ? target : null,
                DamageDealt = expectedDamage,
                Reason = reason
            });
        }

        return moves;
    }

    private List<SimulatedMove> GetRegularMoves(SimulatedBoardState state, SimulatedPiece piece)
    {
        var moves = new List<SimulatedMove>();
        var movePositions = GetValidMovePositions(state, piece);

        foreach (var toPos in movePositions)
        {
            moves.Add(new SimulatedMove
            {
                Piece = piece,
                FromPos = piece.Position,
                ToPos = toPos,
                MoveType = SimulatedMoveType.Move,
                Reason = $"Move to {toPos.ToChessNotation()}"
            });
        }

        return moves;
    }

    #region Move Generation (mirrors piece logic)

    private List<Vector2I> GetValidMovePositions(SimulatedBoardState state, SimulatedPiece piece)
    {
        return piece.PieceType switch
        {
            PieceType.King => GetKingMoves(state, piece),
            PieceType.Queen => GetSlidingMoves(state, piece, Vector2IExtensions.AllDirections),
            PieceType.Rook => GetSlidingMoves(state, piece, Vector2IExtensions.CardinalDirections),
            PieceType.Bishop => GetSlidingMoves(state, piece, Vector2IExtensions.DiagonalDirections),
            PieceType.Knight => GetKnightMoves(state, piece),
            PieceType.Pawn => GetPawnMoves(state, piece),
            _ => new List<Vector2I>()
        };
    }

    private List<Vector2I> GetAttackablePositions(SimulatedBoardState state, SimulatedPiece piece)
    {
        return piece.PieceType switch
        {
            PieceType.King => GetKingAttacks(state, piece),      // Adjacent (8 dirs)
            PieceType.Queen => GetQueenAttacks(state, piece),    // All 8 dirs, blocked
            PieceType.Rook => GetRookAttacks(state, piece),      // Adjacent + Cardinal range 3
            PieceType.Bishop => GetBishopAttacks(state, piece),  // Diagonal, NO adjacent, blocked
            PieceType.Knight => GetKnightAttacks(state, piece),  // Adjacent only!
            PieceType.Pawn => GetPawnAttacks(state, piece),      // Forward diagonal only
            _ => new List<Vector2I>()
        };
    }

    private List<Vector2I> GetKingMoves(SimulatedBoardState state, SimulatedPiece piece)
    {
        var moves = new List<Vector2I>();
        foreach (var dir in Vector2IExtensions.AllDirections)
        {
            var pos = piece.Position + dir;
            if (pos.IsOnBoard() && !state.IsOccupied(pos))
                moves.Add(pos);
        }
        return moves;
    }

    // === KING: Adjacent attacks (8 directions) ===
    private List<Vector2I> GetKingAttacks(SimulatedBoardState state, SimulatedPiece piece)
    {
        var attacks = new List<Vector2I>();
        foreach (var dir in Vector2IExtensions.AllDirections)
        {
            var pos = piece.Position + dir;
            if (pos.IsOnBoard() && state.IsOccupiedByTeam(pos, GetOpponent(piece.Team)))
                attacks.Add(pos);
        }
        return attacks;
    }

    // === QUEEN: All 8 directions, range 1-7, blocked ===
    private const int QueenAttackRange = 7;

    private List<Vector2I> GetQueenAttacks(SimulatedBoardState state, SimulatedPiece piece)
    {
        var attacks = new List<Vector2I>();
        foreach (var dir in Vector2IExtensions.AllDirections)
        {
            var pos = piece.Position + dir;
            for (int range = 1; range <= QueenAttackRange && pos.IsOnBoard(); range++)
            {
                var target = state.GetPieceAt(pos);
                if (target != null)
                {
                    if (target.Team != piece.Team)
                        attacks.Add(pos);
                    break; // Blocked
                }
                pos += dir;
            }
        }
        return attacks;
    }

    // === ROOK: All 8 directions range 1 + Cardinal range 1-3, blocked ===
    private const int RookCardinalRange = 3;

    private List<Vector2I> GetRookAttacks(SimulatedBoardState state, SimulatedPiece piece)
    {
        var attacks = new List<Vector2I>();

        // Range 1 all around (8 directions)
        foreach (var dir in Vector2IExtensions.AllDirections)
        {
            var pos = piece.Position + dir;
            if (pos.IsOnBoard() && state.IsOccupiedByTeam(pos, GetOpponent(piece.Team)))
                attacks.Add(pos);
        }

        // Cardinal directions range 2-3 (range 1 already covered above)
        foreach (var dir in Vector2IExtensions.CardinalDirections)
        {
            var pos = piece.Position + dir;
            for (int range = 1; range <= RookCardinalRange && pos.IsOnBoard(); range++)
            {
                var target = state.GetPieceAt(pos);
                if (target != null)
                {
                    if (target.Team != piece.Team && !attacks.Contains(pos))
                        attacks.Add(pos);
                    break; // Blocked
                }
                pos += dir;
            }
        }

        return attacks;
    }

    // === BISHOP: Diagonal only, range 2-4 (NOT adjacent), blocked ===
    private const int BishopMinRange = 2;
    private const int BishopMaxRange = 4;

    private List<Vector2I> GetBishopAttacks(SimulatedBoardState state, SimulatedPiece piece)
    {
        var attacks = new List<Vector2I>();

        foreach (var dir in Vector2IExtensions.DiagonalDirections)
        {
            var pos = piece.Position;

            // Check each tile in this diagonal direction
            for (int range = 1; range <= BishopMaxRange; range++)
            {
                pos += dir;
                if (!pos.IsOnBoard()) break;

                var target = state.GetPieceAt(pos);

                // Range 1 = adjacent, Bishop can't attack there
                if (range < BishopMinRange)
                {
                    // If something is blocking the adjacent square, can't see past it
                    if (target != null) break;
                    continue;
                }

                // Range 2-4: Can attack here
                if (target != null)
                {
                    if (target.Team != piece.Team)
                        attacks.Add(pos);
                    break; // Blocked by this piece
                }
            }
        }

        return attacks;
    }

    // === KNIGHT: Special - must MOVE first, then attacks CARDINAL adjacent only ===
    // Note: For AI simulation, Knight from current position can't attack directly.
    // We generate move+attack combos in GetKnightMoveAttacks instead.
    private List<Vector2I> GetKnightAttacks(SimulatedBoardState state, SimulatedPiece piece)
    {
        // Knight cannot attack from current position without moving first!
        // Return empty - the move+attack combos are generated separately
        return new List<Vector2I>();
    }

    /// <summary>
    /// Get attacks available from a position (for Knight move+attack simulation)
    /// Knight attacks CARDINAL adjacent only (4 squares: up, down, left, right)
    /// </summary>
    private List<Vector2I> GetKnightAttacksFromPosition(SimulatedBoardState state, Vector2I position, Team team)
    {
        var attacks = new List<Vector2I>();

        // Cardinal adjacent only (NOT diagonal)
        foreach (var dir in Vector2IExtensions.CardinalDirections)
        {
            var pos = position + dir;
            if (pos.IsOnBoard() && state.IsOccupiedByTeam(pos, GetOpponent(team)))
                attacks.Add(pos);
        }

        return attacks;
    }

    // === PAWN: Forward diagonal adjacent only (2 squares) ===
    private List<Vector2I> GetPawnAttacks(SimulatedBoardState state, SimulatedPiece piece)
    {
        var attacks = new List<Vector2I>();
        int direction = piece.Team == Team.Player ? 1 : -1;

        // Forward diagonal left
        var leftDiag = piece.Position + new Vector2I(-1, direction);
        if (leftDiag.IsOnBoard() && state.IsOccupiedByTeam(leftDiag, GetOpponent(piece.Team)))
            attacks.Add(leftDiag);

        // Forward diagonal right
        var rightDiag = piece.Position + new Vector2I(1, direction);
        if (rightDiag.IsOnBoard() && state.IsOccupiedByTeam(rightDiag, GetOpponent(piece.Team)))
            attacks.Add(rightDiag);

        return attacks;
    }

    #region Movement (separate from attacks)

    private List<Vector2I> GetSlidingMoves(SimulatedBoardState state, SimulatedPiece piece, Vector2I[] directions)
    {
        var moves = new List<Vector2I>();
        foreach (var dir in directions)
        {
            var pos = piece.Position + dir;
            while (pos.IsOnBoard())
            {
                if (state.IsOccupied(pos)) break;
                moves.Add(pos);
                pos += dir;
            }
        }
        return moves;
    }

    private List<Vector2I> GetKnightMoves(SimulatedBoardState state, SimulatedPiece piece)
    {
        var moves = new List<Vector2I>();
        int[] offsets = { -2, -1, 1, 2 };

        foreach (int dx in offsets)
        {
            foreach (int dy in offsets)
            {
                if (Math.Abs(dx) == Math.Abs(dy)) continue;
                var pos = piece.Position + new Vector2I(dx, dy);
                if (pos.IsOnBoard() && !state.IsOccupied(pos))
                    moves.Add(pos);
            }
        }
        return moves;
    }

    private List<Vector2I> GetPawnMoves(SimulatedBoardState state, SimulatedPiece piece)
    {
        var moves = new List<Vector2I>();
        int direction = piece.Team == Team.Player ? 1 : -1;
        int startRow = piece.Team == Team.Player ? 1 : 6;

        // Single step forward
        var oneStep = piece.Position + new Vector2I(0, direction);
        if (oneStep.IsOnBoard() && !state.IsOccupied(oneStep))
        {
            moves.Add(oneStep);

            // Double step from starting position
            if (piece.Position.Y == startRow)
            {
                var twoStep = piece.Position + new Vector2I(0, direction * 2);
                if (!state.IsOccupied(twoStep))
                    moves.Add(twoStep);
            }
        }

        return moves;
    }

    #endregion

    #endregion

    #region Evaluation

    private static readonly Dictionary<PieceType, int> PieceValues = new()
    {
        { PieceType.King, 10000 },
        { PieceType.Queen, 900 },
        { PieceType.Rook, 500 },
        { PieceType.Bishop, 330 },
        { PieceType.Knight, 320 },
        { PieceType.Pawn, 100 }
    };

    /// <summary>
    /// Evaluate position from perspective of sideToMove (positive = good for sideToMove)
    /// </summary>
    private int Evaluate(SimulatedBoardState state, Team sideToMove)
    {
        int score = 0;
        var opponent = GetOpponent(sideToMove);

        // Cache attacks for both sides (expensive to compute multiple times)
        var myAttackedSquares = GetAllAttackedSquares(state, sideToMove);
        var oppAttackedSquares = GetAllAttackedSquares(state, opponent);

        // === MATERIAL + HP EVALUATION (only ALIVE pieces!) ===
        foreach (var piece in state.GetPieces(sideToMove))
        {
            int pieceValue = PieceValues[piece.PieceType];
            int hpBonus = piece.CurrentHp * 3; // HP matters more in this game
            int total = pieceValue + hpBonus;

            // HANGING PIECE PENALTY: If this piece can be attacked and isn't defended
            if (oppAttackedSquares.Contains(piece.Position))
            {
                bool isDefended = myAttackedSquares.Contains(piece.Position);
                if (!isDefended)
                {
                    // Hanging! Huge penalty - we could lose this piece
                    total -= pieceValue / 2;
                }
                else
                {
                    // Under attack but defended - still risky, small penalty
                    total -= pieceValue / 6;
                }
            }

            score += total;
        }

        foreach (var piece in state.GetPieces(opponent))
        {
            int pieceValue = PieceValues[piece.PieceType];
            int hpBonus = piece.CurrentHp * 3;
            int total = pieceValue + hpBonus;

            // OPPORTUNITY: Enemy piece is attackable by us
            if (myAttackedSquares.Contains(piece.Position))
            {
                bool isDefended = oppAttackedSquares.Contains(piece.Position);
                if (!isDefended)
                {
                    // Hanging enemy piece! We can take it for free
                    total -= pieceValue / 2;
                }
                else
                {
                    // Attackable but defended - potential trade
                    total -= pieceValue / 8;
                }
            }

            score -= total;
        }

        // === KING SAFETY (Critical!) ===
        var myKing = state.GetKing(sideToMove);
        var oppKing = state.GetKing(opponent);

        if (myKing != null)
        {
            if (oppAttackedSquares.Contains(myKing.Position))
            {
                // Our king is in check - very bad!
                score -= 500;

                // Even worse if we have few escape squares
                var kingMoves = GetKingMoves(state, myKing);
                int safeSquares = kingMoves.Count(m => !oppAttackedSquares.Contains(m));
                if (safeSquares == 0)
                    score -= 2000; // Potential checkmate!
            }
        }

        if (oppKing != null)
        {
            if (myAttackedSquares.Contains(oppKing.Position))
            {
                // Enemy king in check - good!
                score += 400;

                var kingMoves = GetKingMoves(state, oppKing);
                int safeSquares = kingMoves.Count(m => !myAttackedSquares.Contains(m));
                if (safeSquares == 0)
                    score += 1500; // Potential checkmate!
            }
        }

        // === MOBILITY ===
        int myMobility = GenerateAllMoves(state, sideToMove).Count;
        int oppMobility = GenerateAllMoves(state, opponent).Count;
        score += (myMobility - oppMobility) * 8;

        // === CENTER CONTROL ===
        score += EvaluateCenterControl(state, sideToMove, myAttackedSquares, oppAttackedSquares);

        // === PIECE ACTIVITY ===
        score += EvaluatePieceActivity(state, sideToMove, myAttackedSquares);
        score -= EvaluatePieceActivity(state, opponent, oppAttackedSquares);

        return score;
    }

    /// <summary>
    /// Get all squares attacked by a team (cached for efficiency)
    /// </summary>
    private HashSet<Vector2I> GetAllAttackedSquares(SimulatedBoardState state, Team team)
    {
        var attacked = new HashSet<Vector2I>();
        foreach (var piece in state.GetPieces(team))
        {
            var attacks = GetAttackablePositions(state, piece);
            foreach (var pos in attacks)
                attacked.Add(pos);

            // Also include threatened squares (where piece COULD attack if enemy was there)
            var threats = GetThreatenedSquares(state, piece);
            foreach (var pos in threats)
                attacked.Add(pos);
        }
        return attacked;
    }

    /// <summary>
    /// Get squares a piece threatens (matches actual piece rules!)
    /// </summary>
    private List<Vector2I> GetThreatenedSquares(SimulatedBoardState state, SimulatedPiece piece)
    {
        return piece.PieceType switch
        {
            PieceType.King => GetKingThreats(piece),
            PieceType.Queen => GetQueenThreats(state, piece),
            PieceType.Rook => GetRookThreats(state, piece),
            PieceType.Bishop => GetBishopThreats(state, piece),
            PieceType.Knight => GetKnightThreats(piece),  // Adjacent only!
            PieceType.Pawn => GetPawnThreats(piece),
            _ => new List<Vector2I>()
        };
    }

    private List<Vector2I> GetKingThreats(SimulatedPiece piece)
    {
        var threats = new List<Vector2I>();
        foreach (var dir in Vector2IExtensions.AllDirections)
        {
            var pos = piece.Position + dir;
            if (pos.IsOnBoard()) threats.Add(pos);
        }
        return threats;
    }

    private List<Vector2I> GetQueenThreats(SimulatedBoardState state, SimulatedPiece piece)
    {
        var threats = new List<Vector2I>();
        foreach (var dir in Vector2IExtensions.AllDirections)
        {
            var pos = piece.Position;
            for (int range = 1; range <= QueenAttackRange && (pos += dir).IsOnBoard(); range++)
            {
                threats.Add(pos);
                if (state.IsOccupied(pos)) break;
            }
        }
        return threats;
    }

    private List<Vector2I> GetRookThreats(SimulatedBoardState state, SimulatedPiece piece)
    {
        var threats = new List<Vector2I>();

        // Adjacent threats (all 8 directions)
        foreach (var dir in Vector2IExtensions.AllDirections)
        {
            var pos = piece.Position + dir;
            if (pos.IsOnBoard()) threats.Add(pos);
        }

        // Cardinal threats (up to range 3)
        foreach (var dir in Vector2IExtensions.CardinalDirections)
        {
            var pos = piece.Position + dir;
            for (int i = 0; i < RookCardinalRange && pos.IsOnBoard(); i++)
            {
                if (!threats.Contains(pos)) threats.Add(pos);
                if (state.IsOccupied(pos)) break;
                pos += dir;
            }
        }

        return threats;
    }

    private List<Vector2I> GetBishopThreats(SimulatedBoardState state, SimulatedPiece piece)
    {
        var threats = new List<Vector2I>();

        foreach (var dir in Vector2IExtensions.DiagonalDirections)
        {
            var pos = piece.Position;

            // Check each tile in this diagonal direction (respecting max range of 4)
            for (int range = 1; range <= BishopMaxRange; range++)
            {
                pos += dir;
                if (!pos.IsOnBoard()) break;

                var target = state.GetPieceAt(pos);

                // Range 1 = adjacent, Bishop can't attack/threaten there
                if (range < BishopMinRange)
                {
                    if (target != null) break; // Blocked
                    continue;
                }

                // Range 2-4: Bishop threatens this square
                threats.Add(pos);
                if (target != null) break; // Blocked
            }
        }

        return threats;
    }

    private List<Vector2I> GetKnightThreats(SimulatedPiece piece)
    {
        // Knight threatens CARDINAL ADJACENT only (4 squares: up, down, left, right)
        // Note: This represents where Knight could attack AFTER moving
        var threats = new List<Vector2I>();
        foreach (var dir in Vector2IExtensions.CardinalDirections)
        {
            var pos = piece.Position + dir;
            if (pos.IsOnBoard()) threats.Add(pos);
        }
        return threats;
    }

    private List<Vector2I> GetPawnThreats(SimulatedPiece pawn)
    {
        var threats = new List<Vector2I>();
        int dir = pawn.Team == Team.Player ? 1 : -1;

        var leftDiag = pawn.Position + new Vector2I(-1, dir);
        var rightDiag = pawn.Position + new Vector2I(1, dir);

        if (leftDiag.IsOnBoard()) threats.Add(leftDiag);
        if (rightDiag.IsOnBoard()) threats.Add(rightDiag);

        return threats;
    }

    private int EvaluateCenterControl(SimulatedBoardState state, Team team,
        HashSet<Vector2I> myAttacks, HashSet<Vector2I> oppAttacks)
    {
        int score = 0;
        Vector2I[] centerSquares = { new(3, 3), new(3, 4), new(4, 3), new(4, 4) };
        Vector2I[] extendedCenter = { new(2, 2), new(2, 3), new(2, 4), new(2, 5),
                                      new(3, 2), new(3, 5), new(4, 2), new(4, 5),
                                      new(5, 2), new(5, 3), new(5, 4), new(5, 5) };

        // Core center control
        foreach (var pos in centerSquares)
        {
            var piece = state.GetPieceAt(pos);
            if (piece != null && piece.IsAlive)
                score += piece.Team == team ? 25 : -25;

            // Attacking center is also valuable
            if (myAttacks.Contains(pos)) score += 10;
            if (oppAttacks.Contains(pos)) score -= 10;
        }

        // Extended center
        foreach (var pos in extendedCenter)
        {
            if (myAttacks.Contains(pos)) score += 3;
            if (oppAttacks.Contains(pos)) score -= 3;
        }

        return score;
    }

    private int EvaluatePieceActivity(SimulatedBoardState state, Team team, HashSet<Vector2I> attacks)
    {
        int score = 0;

        foreach (var piece in state.GetPieces(team))
        {
            // Pieces controlling more squares are more active
            var pieceAttacks = GetAttackablePositions(state, piece);
            score += pieceAttacks.Count * 2;

            // Development bonus (not on back rank)
            int backRank = team == Team.Player ? 0 : 7;
            if (piece.PieceType != PieceType.King && piece.Position.Y != backRank)
            {
                score += 5;
            }

            // Rooks on open files
            if (piece.PieceType == PieceType.Rook)
            {
                bool openFile = true;
                for (int y = 0; y < 8; y++)
                {
                    var p = state.GetPieceAt(new Vector2I(piece.Position.X, y));
                    if (p != null && p.PieceType == PieceType.Pawn)
                    {
                        openFile = false;
                        break;
                    }
                }
                if (openFile) score += 20;
            }
        }

        return score;
    }

    private bool IsSquareAttacked(SimulatedBoardState state, Vector2I square, Team byTeam)
    {
        foreach (var piece in state.GetPieces(byTeam))
        {
            var attacks = GetAttackablePositions(state, piece);
            if (attacks.Contains(square))
                return true;
        }
        return false;
    }

    /// <summary>
    /// Get the lowest value attacker of a square (for SEE-like analysis)
    /// </summary>
    private SimulatedPiece? GetLowestValueAttacker(SimulatedBoardState state, Vector2I square, Team byTeam)
    {
        SimulatedPiece? lowest = null;
        int lowestValue = int.MaxValue;

        foreach (var piece in state.GetPieces(byTeam))
        {
            var attacks = GetAttackablePositions(state, piece);
            if (attacks.Contains(square))
            {
                int value = PieceValues[piece.PieceType];
                if (value < lowestValue)
                {
                    lowestValue = value;
                    lowest = piece;
                }
            }
        }

        return lowest;
    }

    #endregion

    private static Team GetOpponent(Team team) => team == Team.Player ? Team.Enemy : Team.Player;
}

/// <summary>
/// Result of a search operation
/// </summary>
public record SearchResult(SimulatedMove? BestMove, int Score, string Reason);
