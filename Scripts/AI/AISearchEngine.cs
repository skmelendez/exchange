using Godot;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using Exchange.Core;
using Exchange.Board;
using Exchange.Pieces;

namespace Exchange.AI;

/// <summary>
/// High-performance minimax search engine with alpha-beta pruning and transposition tables.
/// Supports up to 5-ply lookahead with incremental cache reuse between moves.
/// Now with parallel root move evaluation for multi-core performance! 🚀
/// </summary>
public class AISearchEngine
{
    private readonly TranspositionTable _transpositionTable = new();
    private readonly MoveOrderer _moveOrderer = new();

    private int _nodesSearched;
    private int _cacheHits;
    private int _cutoffs;

    // Pondering state
    private CancellationTokenSource? _ponderCts;
    private Task<SearchResult>? _ponderTask;
    private Vector2I? _ponderedPlayerMove;  // The player move we're pondering on

    /// <summary>
    /// Enable parallel evaluation of root moves across multiple CPU cores.
    /// </summary>
    public bool UseParallelSearch { get; set; } = true;

    /// <summary>
    /// Maximum degree of parallelism (0 = use all cores)
    /// </summary>
    public int MaxParallelism { get; set; } = 0;

    /// <summary>
    /// Enable pondering (thinking during opponent's turn)
    /// </summary>
    public bool UsePondering { get; set; } = true;

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
            int nodesBefore = _nodesSearched;
            var result = SearchRoot(state, sideToMove, d);
            int nodesThisDepth = _nodesSearched - nodesBefore;

            if (result.BestMove != null)
            {
                bestResult = result;
                GameLogger.Debug("AI", $"Depth {d}: {result.BestMove.Value.ToDebugString()} = {result.Score} ({nodesThisDepth} nodes this depth, {_nodesSearched} total)");
            }
            else
            {
                GameLogger.Warning("AI", $"Depth {d}: No best move found! ({nodesThisDepth} nodes searched)");
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

    #region Pondering

    /// <summary>
    /// Start pondering on the expected player response.
    /// Call this after AI makes its move to think during player's turn.
    /// </summary>
    public void StartPondering(GameBoard board, Team aiTeam, int depth)
    {
        if (!UsePondering) return;

        StopPondering(); // Cancel any existing ponder

        _ponderCts = new CancellationTokenSource();
        var token = _ponderCts.Token;

        // Create snapshot for pondering
        var state = SimulatedBoardState.FromGameBoard(board);
        var playerTeam = aiTeam == Team.Enemy ? Team.Player : Team.Enemy;

        // Find best player move to ponder on
        var playerMoves = GenerateAllMoves(state, playerTeam);
        if (playerMoves.Count == 0) return;

        _moveOrderer.OrderMoves(playerMoves, state, _transpositionTable);
        var expectedPlayerMove = playerMoves[0]; // Best move from player's perspective
        _ponderedPlayerMove = expectedPlayerMove.ToPos;

        GameLogger.Debug("AI-Ponder", $"Starting ponder on player move: {expectedPlayerMove.ToDebugString()}");

        // Start background search assuming player makes this move
        _ponderTask = Task.Run(() =>
        {
            try
            {
                // Make the expected player move
                state.MakeMove(expectedPlayerMove);

                // Now search for AI's response
                _nodesSearched = 0;
                _cacheHits = 0;
                _cutoffs = 0;

                var result = SearchRoot(state, aiTeam, depth);

                if (!token.IsCancellationRequested)
                {
                    GameLogger.Debug("AI-Ponder", $"Ponder complete: {_nodesSearched} nodes, {_cacheHits} hits, best={result.Score}");
                }

                return result;
            }
            catch (OperationCanceledException)
            {
                return new SearchResult(null, 0, "Cancelled");
            }
        }, token);
    }

    /// <summary>
    /// Stop any active pondering
    /// </summary>
    public void StopPondering()
    {
        if (_ponderCts != null)
        {
            _ponderCts.Cancel();
            _ponderCts.Dispose();
            _ponderCts = null;
        }
        _ponderTask = null;
        _ponderedPlayerMove = null;
    }

    /// <summary>
    /// Check if we have a pondered result ready for this player move
    /// </summary>
    public async Task<SearchResult?> GetPonderedResultAsync(Vector2I playerMoveTo)
    {
        if (_ponderTask == null || _ponderedPlayerMove == null)
            return null;

        // Check if player made the move we were pondering on
        if (_ponderedPlayerMove.Value == playerMoveTo)
        {
            GameLogger.Debug("AI-Ponder", "Player made predicted move! Using pondered result.");
            try
            {
                // Wait for ponder to complete (should be done or nearly done)
                var result = await _ponderTask;
                StopPondering();
                return result;
            }
            catch
            {
                StopPondering();
                return null;
            }
        }
        else
        {
            GameLogger.Debug("AI-Ponder", $"Player made different move. Pondered: {_ponderedPlayerMove.Value.ToChessNotation()}, Actual: {playerMoveTo.ToChessNotation()}");
            StopPondering();
            return null; // Player made a different move, TT still helps though!
        }
    }

    /// <summary>
    /// Get TT statistics for debugging
    /// </summary>
    public (int used, int total, int hitRate) GetCacheStats()
    {
        var (used, total) = _transpositionTable.GetStats();
        int hitRate = _nodesSearched > 0 ? (_cacheHits * 100) / _nodesSearched : 0;
        return (used, total, hitRate);
    }

    #endregion

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
        {
            GameLogger.Warning("AI", $"SearchRoot depth {depth}: No moves generated for {sideToMove}! Pieces: {state.GetPieces(sideToMove).Count()}");
            return new SearchResult(null, sideToMove == Team.Enemy ? -99999 : 99999, "No moves");
        }

        // Debug: Log move counts
        int attackMoves = moves.Count(m => m.MoveType == SimulatedMoveType.Attack);
        int regularMoves = moves.Count(m => m.MoveType == SimulatedMoveType.Move);
        int combos = moves.Count(m => m.MoveType == SimulatedMoveType.MoveAndAttack);
        GameLogger.Debug("AI-Moves", $"Generated {moves.Count} moves for {sideToMove}: {attackMoves} attacks, {regularMoves} moves, {combos} combos");

        // Order moves for better pruning
        _moveOrderer.OrderMoves(moves, state, _transpositionTable);

        // Use parallel search for depth >= 3 with enough moves to benefit
        if (UseParallelSearch && depth >= 3 && moves.Count >= 4)
        {
            return SearchRootParallel(state, sideToMove, depth, moves);
        }

        // Sequential fallback for shallow depths or few moves
        return SearchRootSequential(state, sideToMove, depth, moves);
    }

    /// <summary>
    /// Sequential root search - used for shallow depths or few moves
    /// </summary>
    private SearchResult SearchRootSequential(SimulatedBoardState state, Team sideToMove, int depth, List<SimulatedMove> moves)
    {
        SimulatedMove? bestMove = null;
        int bestScore = -99999999;
        int alpha = -99999999;
        int beta = 99999999;
        string bestReason = "";

        foreach (var move in moves)
        {
            state.MakeMove(move);
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

    /// <summary>
    /// Parallel root search - evaluates top-level moves across multiple CPU cores 🔥
    /// Uses "Young Brothers Wait" concept: first move is searched sequentially for a good alpha,
    /// then remaining moves are searched in parallel with that alpha bound.
    /// </summary>
    private SearchResult SearchRootParallel(SimulatedBoardState state, Team sideToMove, int depth, List<SimulatedMove> moves)
    {
        // First move searched sequentially to establish a good alpha bound
        var firstMove = moves[0];
        state.MakeMove(firstMove);
        int firstScore = -AlphaBeta(state, GetOpponent(sideToMove), depth - 1, -99999999, 99999999);
        state.UndoMove(firstMove);

        var bestResult = new ConcurrentBag<(SimulatedMove Move, int Score)>
        {
            (firstMove, firstScore)
        };

        int sharedAlpha = firstScore;

        // Remaining moves searched in parallel
        var parallelMoves = moves.Skip(1).ToList();

        int parallelism = MaxParallelism > 0 ? MaxParallelism : System.Environment.ProcessorCount;
        var options = new ParallelOptions { MaxDegreeOfParallelism = parallelism };

        // Each thread gets its own copy of the board state
        Parallel.ForEach(parallelMoves, options, move =>
        {
            // Clone the state for this thread
            var localState = state.Clone();

            // Read current alpha (may be stale but that's OK for parallel search)
            int localAlpha = Volatile.Read(ref sharedAlpha);

            localState.MakeMove(move);

            // Use aspiration window around current best
            int score = -AlphaBetaThreadSafe(localState, GetOpponent(sideToMove), depth - 1, -99999999, -localAlpha);

            localState.UndoMove(move);

            // Track this result
            bestResult.Add((move, score));

            // Update shared alpha if we found better (atomic)
            int current;
            do
            {
                current = Volatile.Read(ref sharedAlpha);
                if (score <= current) break;
            } while (Interlocked.CompareExchange(ref sharedAlpha, score, current) != current);

            Interlocked.Increment(ref _nodesSearched);
        });

        // Find best move from all results
        var best = bestResult.OrderByDescending(r => r.Score).First();

        GameLogger.Debug("AI", $"Parallel search: {parallelism} threads, {moves.Count} moves, best={best.Score}");

        return new SearchResult(best.Move, best.Score, best.Move.Reason);
    }

    /// <summary>
    /// Thread-safe version of AlphaBeta that uses shared transposition table.
    /// TT access is inherently thread-safe for reads (may get stale data, acceptable).
    /// Writes may race but worst case is slightly suboptimal replacement.
    /// </summary>
    private int AlphaBetaThreadSafe(SimulatedBoardState state, Team sideToMove, int depth, int alpha, int beta)
    {
        Interlocked.Increment(ref _nodesSearched);

        // Check transposition table (thread-safe read)
        ulong hash = state.ZobristHash;
        if (_transpositionTable.TryGet(hash, depth, alpha, beta, out int ttScore))
        {
            Interlocked.Increment(ref _cacheHits);
            return ttScore;
        }

        // Terminal conditions
        if (depth == 0)
        {
            int eval = Evaluate(state, sideToMove);
            _transpositionTable.Store(hash, depth, eval, TranspositionTable.NodeType.Exact, null);
            return eval;
        }

        // Check for king death
        if (!state.HasKing(Team.Player))
            return sideToMove == Team.Enemy ? 50000 + depth : -50000 - depth;
        if (!state.HasKing(Team.Enemy))
            return sideToMove == Team.Player ? 50000 + depth : -50000 - depth;

        var moves = GenerateAllMoves(state, sideToMove);
        if (moves.Count == 0)
        {
            return -10000 + depth;
        }

        // Use TT for move ordering (best move from previous search first)
        _moveOrderer.OrderMoves(moves, state, _transpositionTable);

        int originalAlpha = alpha;
        SimulatedMove? bestMove = null;
        int moveIndex = 0;

        foreach (var move in moves)
        {
            state.MakeMove(move);

            int score;
            bool isCapture = move.MoveType == SimulatedMoveType.Attack || move.MoveType == SimulatedMoveType.MoveAndAttack;

            // Late Move Reductions for quiet moves
            if (moveIndex >= 4 && depth >= 3 && !isCapture)
            {
                score = -AlphaBetaThreadSafe(state, GetOpponent(sideToMove), depth - 2, -beta, -alpha);
                if (score > alpha)
                {
                    score = -AlphaBetaThreadSafe(state, GetOpponent(sideToMove), depth - 1, -beta, -alpha);
                }
            }
            else
            {
                score = -AlphaBetaThreadSafe(state, GetOpponent(sideToMove), depth - 1, -beta, -alpha);
            }

            state.UndoMove(move);
            moveIndex++;

            if (score >= beta)
            {
                Interlocked.Increment(ref _cutoffs);
                _transpositionTable.Store(hash, depth, beta, TranspositionTable.NodeType.LowerBound, move);
                return beta;
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

        // Check for king death (HP-based: only game over if king is actually DEAD)
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
        int moveIndex = 0;

        foreach (var move in moves)
        {
            state.MakeMove(move);

            int score;
            bool isCapture = move.MoveType == SimulatedMoveType.Attack || move.MoveType == SimulatedMoveType.MoveAndAttack;

            // Late Move Reductions (LMR): Search later quiet moves at reduced depth
            // This dramatically speeds up search by quickly pruning unpromising moves
            if (moveIndex >= 4 && depth >= 3 && !isCapture)
            {
                // Search at reduced depth first
                score = -AlphaBeta(state, GetOpponent(sideToMove), depth - 2, -beta, -alpha);

                // If it looks promising, re-search at full depth
                if (score > alpha)
                {
                    score = -AlphaBeta(state, GetOpponent(sideToMove), depth - 1, -beta, -alpha);
                }
            }
            else
            {
                // Full depth search for important moves
                score = -AlphaBeta(state, GetOpponent(sideToMove), depth - 1, -beta, -alpha);
            }

            state.UndoMove(move);
            moveIndex++;

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

                // Simple reason to avoid string allocations
                string reason = willKill ? "KnightKill" : "KnightAttack";

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

            // Use simple reason strings to avoid allocations in hot path
            // Full reason is only needed for the final chosen move
            string reason = willKill ? "Kill" : "Attack";

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
            PieceType.Rook => GetRookAttacks(state, piece),      // Adjacent + Cardinal range 2
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

    // === QUEEN: All 8 directions, range 1-4, blocked ===
    private const int QueenAttackRange = 3;

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

    // === ROOK: All 8 directions range 1 + Cardinal range 1-2, blocked ===
    private const int RookCardinalRange = 2;

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
    private const int BishopMaxRange = 3;

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

        // IMPORTANT: Two different "attack" concepts:
        // 1. CaptureSquares = where we can ACTUALLY capture a piece RIGHT NOW
        // 2. ThreatZones = all squares we control/threaten (for safety evaluation)
        var myCaptureSquares = GetActualCaptureSquares(state, sideToMove);
        var oppCaptureSquares = GetActualCaptureSquares(state, opponent);
        var myThreatZones = GetAllThreatZones(state, sideToMove);
        var oppThreatZones = GetAllThreatZones(state, opponent);

        // === AGGRESSION BONUS ===
        // In HP-based combat, having attack opportunities is inherently valuable!
        // Reward positions where we can deal damage.
        int attackOpportunities = myCaptureSquares.Count;
        int oppAttackOpportunities = oppCaptureSquares.Count;
        score += (attackOpportunities - oppAttackOpportunities) * 25;  // Each attack option is worth ~25

        // Bonus for attacking LOW HP targets (potential kills!)
        foreach (var piece in state.GetPieces(opponent))
        {
            if (myCaptureSquares.Contains(piece.Position))
            {
                // Estimate damage we can deal (base damage + avg dice = ~6-8)
                int potentialDamage = 7;
                if (piece.CurrentHp <= potentialDamage)
                {
                    // Can likely kill this piece! Big bonus based on piece value
                    score += PieceValues[piece.PieceType] / 2;
                }
                else if (piece.CurrentHp <= potentialDamage * 2)
                {
                    // Piece is wounded - good target
                    score += 30;
                }
            }
        }

        // === MATERIAL + HP EVALUATION (only ALIVE pieces!) ===
        const int HpWeight = 20;

        foreach (var piece in state.GetPieces(sideToMove))
        {
            // Skip Kings here - they have their own evaluation section
            if (piece.PieceType == PieceType.King) continue;

            int pieceValue = PieceValues[piece.PieceType];
            int hpBonus = piece.CurrentHp * HpWeight;
            int total = pieceValue + hpBonus;

            // HANGING PIECE PENALTY: Only if opponent can ACTUALLY capture this piece
            if (oppCaptureSquares.Contains(piece.Position))
            {
                bool isDefended = myThreatZones.Contains(piece.Position);
                if (!isDefended)
                {
                    // Truly hanging! Apply significant penalty
                    // But not 2x piece value - that was too extreme
                    total -= pieceValue + (piece.CurrentHp * HpWeight / 2);
                }
                else
                {
                    // Under attack but defended - evaluate trade
                    // Find who's attacking us and compare values
                    var attackers = GetAttackersOfSquare(state, piece.Position, opponent);
                    if (attackers.Count > 0)
                    {
                        var cheapestAttacker = attackers.OrderBy(a => PieceValues[a.PieceType]).First();
                        int attackerValue = PieceValues[cheapestAttacker.PieceType];

                        // Bad trade if we lose more value
                        if (pieceValue > attackerValue)
                            total -= (pieceValue - attackerValue) / 2;
                        // Penalty reduced for equal/good trades
                        else
                            total -= pieceValue / 6;
                    }
                }
            }
            // Piece in THREAT ZONE but not immediately capturable - small caution penalty
            else if (oppThreatZones.Contains(piece.Position))
            {
                total -= 15; // Minor penalty for being in enemy's sphere of influence
            }

            score += total;
        }

        foreach (var piece in state.GetPieces(opponent))
        {
            // Skip Kings here - they have their own evaluation section
            if (piece.PieceType == PieceType.King) continue;

            int pieceValue = PieceValues[piece.PieceType];
            int hpBonus = piece.CurrentHp * HpWeight;
            int total = pieceValue + hpBonus;

            // OPPORTUNITY: Enemy piece is ACTUALLY capturable by us
            if (myCaptureSquares.Contains(piece.Position))
            {
                bool isDefended = oppThreatZones.Contains(piece.Position);
                if (!isDefended)
                {
                    // Hanging enemy piece! Big opportunity
                    total -= pieceValue + (piece.CurrentHp * HpWeight / 2);
                }
                else
                {
                    // Attackable but defended - evaluate trade potential
                    var ourAttackers = GetAttackersOfSquare(state, piece.Position, sideToMove);
                    if (ourAttackers.Count > 0)
                    {
                        var cheapestAttacker = ourAttackers.OrderBy(a => PieceValues[a.PieceType]).First();
                        int attackerValue = PieceValues[cheapestAttacker.PieceType];

                        // Good trade if we win material
                        if (pieceValue > attackerValue)
                            total -= (pieceValue - attackerValue) / 3;
                    }
                }
            }

            score -= total;
        }

        // === HP-BASED KING EVALUATION (NOT checkmate-style!) ===
        // This game is about DAMAGE over time, not instant checkmate
        // Key considerations:
        // 1. How much damage can be dealt to king this turn?
        // 2. Can attackers be killed BEFORE they deal damage?
        // 3. Is a suicide attack worth the damage dealt?
        // 4. King HP remaining matters proportionally

        var myKing = state.GetKing(sideToMove);
        var oppKing = state.GetKing(opponent);

        if (myKing != null)
        {
            // === OUR KING'S SAFETY ===
            int kingHpPercent = (myKing.CurrentHp * 100) / 25; // 25 is max HP

            // Low HP is dangerous but not game over
            if (kingHpPercent < 40)
                score -= (40 - kingHpPercent) * 10; // Penalty scales with low HP

            // Check if king is under attack
            var kingAttackers = GetAttackersOfSquare(state, myKing.Position, opponent);
            if (kingAttackers.Count > 0)
            {
                // Calculate total potential damage from attackers
                int totalPotentialDamage = 0;
                int attackersThatCanBeKilled = 0;

                foreach (var attacker in kingAttackers)
                {
                    int expectedDmg = attacker.BaseDamage + 4; // base + avg dice
                    totalPotentialDamage += expectedDmg;

                    // Can we kill this attacker before they attack?
                    if (myCaptureSquares.Contains(attacker.Position))
                    {
                        // We can counter-attack!
                        var defender = GetLowestValueAttacker(state, attacker.Position, sideToMove);
                        if (defender != null)
                        {
                            int defenderDmg = defender.BaseDamage + 4;
                            if (attacker.CurrentHp <= defenderDmg)
                            {
                                attackersThatCanBeKilled++;
                                totalPotentialDamage -= expectedDmg; // Won't deal damage if killed
                            }
                        }
                    }
                }

                // Penalty based on net damage we can't prevent
                if (totalPotentialDamage > 0)
                {
                    // Would this kill our king?
                    if (myKing.CurrentHp <= totalPotentialDamage)
                        score -= 8000; // Very dangerous! But not instant game over
                    else
                        score -= totalPotentialDamage * 40; // Proportional to damage
                }

                // Small penalty for being under attack at all
                score -= (kingAttackers.Count - attackersThatCanBeKilled) * 50;
            }

            // Bonus for escape routes (can move to safety)
            var kingMoves = GetKingMoves(state, myKing);
            int safeSquares = kingMoves.Count(m => !oppThreatZones.Contains(m));
            score += safeSquares * 15;
        }

        if (oppKing != null)
        {
            // === ENEMY KING EVALUATION ===
            int oppKingHpPercent = (oppKing.CurrentHp * 100) / 25;

            // Low HP enemy king is an opportunity
            if (oppKingHpPercent < 40)
                score += (40 - oppKingHpPercent) * 10;

            // Check if we can attack enemy king
            var ourKingAttackers = GetAttackersOfSquare(state, oppKing.Position, sideToMove);
            if (ourKingAttackers.Count > 0)
            {
                int totalDamageWeCanDeal = 0;
                int attackersThatWillDie = 0;

                foreach (var attacker in ourKingAttackers)
                {
                    int expectedDmg = attacker.BaseDamage + 4;
                    totalDamageWeCanDeal += expectedDmg;

                    // Will our attacker be killed after?
                    if (oppThreatZones.Contains(attacker.Position))
                    {
                        var defender = GetLowestValueAttacker(state, attacker.Position, opponent);
                        if (defender != null)
                        {
                            int defenderDmg = defender.BaseDamage + 4;
                            if (attacker.CurrentHp <= defenderDmg)
                                attackersThatWillDie++;
                        }
                    }
                }

                // === SACRIFICE EVALUATION ===
                // Is the damage worth the piece(s) we'll lose?
                int valueOfAttackersThatWillDie = 0;
                if (attackersThatWillDie > 0)
                {
                    foreach (var attacker in ourKingAttackers)
                    {
                        if (oppThreatZones.Contains(attacker.Position))
                            valueOfAttackersThatWillDie += PieceValues[attacker.PieceType];
                    }
                }

                // Would this kill enemy king?
                if (oppKing.CurrentHp <= totalDamageWeCanDeal)
                {
                    // Can kill the king! Huge bonus, minus piece sacrifice cost
                    score += 10000 - valueOfAttackersThatWillDie;
                }
                else
                {
                    // Dealing damage - is it worth the sacrifice?
                    int damageValue = totalDamageWeCanDeal * 50; // Each HP damage to king is valuable
                    int netGain = damageValue - (valueOfAttackersThatWillDie / 2); // Sacrifices are costly

                    // Only attack king if it's net positive or we're getting good damage
                    if (netGain > 0)
                        score += netGain;
                    else if (totalDamageWeCanDeal >= 5)
                        score += totalDamageWeCanDeal * 20; // Still valuable if significant damage
                }
            }
        }

        // === MOBILITY ===
        int myMobility = GenerateAllMoves(state, sideToMove).Count;
        int oppMobility = GenerateAllMoves(state, opponent).Count;
        score += (myMobility - oppMobility) * 8;

        // === CENTER CONTROL ===
        score += EvaluateCenterControl(state, sideToMove, myThreatZones, oppThreatZones);

        // === PIECE ACTIVITY ===
        score += EvaluatePieceActivity(state, sideToMove, myThreatZones);
        score -= EvaluatePieceActivity(state, opponent, oppThreatZones);

        // === PAWN STRUCTURE ===
        // Pawns are valuable as shields in this HP-based game
        score += EvaluatePawnStructure(state, sideToMove, myKing);
        score -= EvaluatePawnStructure(state, opponent, oppKing);

        return score;
    }

    /// <summary>
    /// Evaluate pawn structure - walls protect pieces, connected pawns support each other
    /// </summary>
    private int EvaluatePawnStructure(SimulatedBoardState state, Team team, SimulatedPiece? king)
    {
        int score = 0;
        var pawns = state.GetPieces(team).Where(p => p.PieceType == PieceType.Pawn).ToList();

        foreach (var pawn in pawns)
        {
            // Connected pawns (adjacent pawns support each other)
            int connectedBonus = 0;
            foreach (var dir in Vector2IExtensions.AllDirections)
            {
                var adjacentPos = pawn.Position + dir;
                var adjacent = state.GetPieceAt(adjacentPos);
                if (adjacent != null && adjacent.Team == team && adjacent.PieceType == PieceType.Pawn)
                {
                    connectedBonus += 15; // Each adjacent friendly pawn adds support
                }
            }
            score += connectedBonus;

            // Pawn shields in front of valuable pieces (especially King)
            if (king != null)
            {
                int forwardDir = team == Team.Player ? 1 : -1;
                var kingFront = king.Position + new Vector2I(0, forwardDir);
                var kingFrontLeft = king.Position + new Vector2I(-1, forwardDir);
                var kingFrontRight = king.Position + new Vector2I(1, forwardDir);

                if (pawn.Position == kingFront || pawn.Position == kingFrontLeft || pawn.Position == kingFrontRight)
                {
                    score += 40; // Pawn shielding king is very valuable
                }
            }

            // Advanced pawns are more threatening
            int advanceRank = team == Team.Player ? pawn.Position.Y : (7 - pawn.Position.Y);
            if (advanceRank >= 4) // Past the midpoint
            {
                score += (advanceRank - 3) * 20; // Bonus for advanced pawns
            }

            // Isolated pawn penalty (no friendly pawns on adjacent files)
            bool hasFileNeighbor = pawns.Any(p =>
                p != pawn &&
                Math.Abs(p.Position.X - pawn.Position.X) == 1);
            if (!hasFileNeighbor)
            {
                score -= 10; // Isolated pawn is weaker
            }
        }

        return score;
    }

    /// <summary>
    /// Get squares where pieces can ACTUALLY be captured right now.
    /// Only includes positions where an enemy piece exists and can be attacked.
    /// </summary>
    private HashSet<Vector2I> GetActualCaptureSquares(SimulatedBoardState state, Team team)
    {
        var captures = new HashSet<Vector2I>();
        foreach (var piece in state.GetPieces(team))
        {
            var attacks = GetAttackablePositions(state, piece);
            foreach (var pos in attacks)
                captures.Add(pos);
        }
        return captures;
    }

    /// <summary>
    /// Get all squares a team threatens (controls), including empty squares.
    /// Used for evaluating piece safety and board control.
    /// </summary>
    private HashSet<Vector2I> GetAllThreatZones(SimulatedBoardState state, Team team)
    {
        var zones = new HashSet<Vector2I>();
        foreach (var piece in state.GetPieces(team))
        {
            var threats = GetThreatenedSquares(state, piece);
            foreach (var pos in threats)
                zones.Add(pos);
        }
        return zones;
    }

    /// <summary>
    /// Legacy function - combines both (for backwards compatibility in king safety, etc.)
    /// </summary>
    private HashSet<Vector2I> GetAllAttackedSquares(SimulatedBoardState state, Team team)
    {
        var attacked = new HashSet<Vector2I>();
        foreach (var piece in state.GetPieces(team))
        {
            var attacks = GetAttackablePositions(state, piece);
            foreach (var pos in attacks)
                attacked.Add(pos);

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
    /// Get all pieces that can attack a specific square.
    /// Used for HP-based damage calculation and sacrifice evaluation.
    /// </summary>
    private List<SimulatedPiece> GetAttackersOfSquare(SimulatedBoardState state, Vector2I square, Team byTeam)
    {
        var attackers = new List<SimulatedPiece>();
        foreach (var piece in state.GetPieces(byTeam))
        {
            var attacks = GetAttackablePositions(state, piece);
            if (attacks.Contains(square))
                attackers.Add(piece);

            // Also check Knight move+attack combos (Knight attacks AFTER moving)
            if (piece.PieceType == PieceType.Knight)
            {
                var combos = GetKnightMoveAttackCombos(state, piece);
                if (combos.Any(c => c.AttackPos == square))
                    attackers.Add(piece);
            }
        }
        return attackers;
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
