using Godot;
using Exchange.Core;
using Exchange.Board;
using Exchange.Pieces;
using Exchange.AI;

namespace Exchange.Controllers;

/// <summary>
/// Advanced AI controller for enemy decision-making.
/// Features:
/// - True N-ply minimax with alpha-beta pruning (up to 5 depth)
/// - Transposition table with incremental cache reuse
/// - Zobrist hashing for fast position comparison
/// - MVV-LVA move ordering for optimal pruning
/// - Fallback heuristic evaluation for depth 1
/// - Configurable depth per act for difficulty scaling
/// </summary>
public partial class AIController : Node
{
    private GameBoard _board = null!;
    private GameState _gameState = null!;
    private TurnController _turnController = null!;

    /// <summary>
    /// The minimax search engine with transposition tables.
    /// Persists between turns for cache reuse.
    /// </summary>
    private readonly AISearchEngine _searchEngine = new();

    /// <summary>
    /// Minimax lookahead depth (1-5). Higher = smarter but slower.
    /// Recommended: Act 1 = 2, Act 2 = 3, Act 3 = 4-5
    /// </summary>
    [Export] public int LookaheadDepth { get; set; } = 2;

    /// <summary>
    /// Use heuristic-only mode (faster, for depth 1 or testing)
    /// </summary>
    [Export] public bool UseHeuristicFallback { get; set; } = false;

    private static readonly RandomNumberGenerator _rng = new();

    // Piece values (centipawn-like scale)
    private static readonly Dictionary<PieceType, int> PieceValues = new()
    {
        { PieceType.King, 10000 },
        { PieceType.Queen, 900 },
        { PieceType.Rook, 500 },
        { PieceType.Bishop, 330 },
        { PieceType.Knight, 320 },
        { PieceType.Pawn, 100 }
    };

    public void Initialize(GameBoard board, GameState gameState, TurnController turnController)
    {
        _board = board;
        _gameState = gameState;
        _turnController = turnController;
    }

    public async void ExecuteTurn()
    {
        GameLogger.Debug("AI", $"=== ExecuteTurn START === (Depth: {LookaheadDepth}, Heuristic: {UseHeuristicFallback})");
        await ToSignal(GetTree().CreateTimer(0.2f), SceneTreeTimer.SignalName.Timeout);

        AIDecision? decision = null;

        // Use true minimax search for depth >= 2 (unless heuristic fallback is forced)
        if (LookaheadDepth >= 2 && !UseHeuristicFallback)
        {
            decision = FindBestActionWithMinimax();
        }

        // Fallback to heuristic evaluation
        if (decision == null)
        {
            GameLogger.Debug("AI", "Using heuristic fallback...");
            decision = FindBestActionWithHeuristic();
        }

        if (decision == null)
        {
            GameLogger.Error("AI", "No valid action found!");
            return;
        }

        GameLogger.Debug("AI", $"Best: {decision.Piece.PieceType} {decision.Action} " +
                 $"{(decision.Target.HasValue ? $"-> {decision.Target.Value.ToChessNotation()}" : "")} " +
                 $"(score: {decision.Score}, {decision.Reason})");

        ExecuteDecision(decision);

        // Age transposition table entries for cache reuse
        _searchEngine.OnMoveMade();

        GameLogger.Debug("AI", "=== ExecuteTurn END ===");
    }

    /// <summary>
    /// Clears the AI search cache (call on new game/match)
    /// </summary>
    public void ClearSearchCache()
    {
        _searchEngine.ClearCache();
        GameLogger.Debug("AI", "Search cache cleared");
    }

    #region Minimax Search

    private AIDecision? FindBestActionWithMinimax()
    {
        GameLogger.Debug("AI", $"Starting {LookaheadDepth}-ply minimax search...");

        var result = _searchEngine.FindBestMove(_board, Team.Enemy, LookaheadDepth);

        if (result.BestMove == null)
        {
            GameLogger.Warning("AI", "Minimax search found no moves");
            return null;
        }

        var move = result.BestMove.Value;

        // Convert SimulatedMove back to AIDecision with actual piece reference
        var piece = FindActualPiece(move.Piece);
        if (piece == null)
        {
            GameLogger.Error("AI", "Could not find piece for simulated move");
            return null;
        }

        var action = move.MoveType switch
        {
            SimulatedMoveType.Attack => ActionType.Attack,
            SimulatedMoveType.Move => ActionType.Move,
            SimulatedMoveType.Ability => ActionType.Ability,
            _ => ActionType.Move
        };

        return new AIDecision(piece, action, move.ToPos, result.Score, result.Reason);
    }

    private BasePiece? FindActualPiece(SimulatedPiece simPiece)
    {
        // Find the actual piece on the board that matches the simulated piece
        foreach (var piece in _board.EnemyPieces)
        {
            if (piece.BoardPosition == simPiece.Position &&
                piece.PieceType == simPiece.PieceType &&
                piece.Team == simPiece.Team)
            {
                return piece;
            }
        }
        return null;
    }

    #endregion

    private record AIDecision(BasePiece Piece, ActionType Action, Vector2I? Target, int Score, string Reason);

    #region Heuristic Fallback (Original Implementation)

    private AIDecision? FindBestActionWithHeuristic()
    {
        var candidates = new List<AIDecision>();
        int currentEval = EvaluatePosition(Team.Enemy);

        GameLogger.Debug("AI", $"Position eval: {currentEval}");

        // First, check if we need to respond to threats
        var defensiveMove = FindDefensiveMove();
        if (defensiveMove != null && defensiveMove.Score > 500)
        {
            GameLogger.Debug("AI", $"DEFENSIVE PRIORITY: {defensiveMove.Reason}");
            candidates.Add(defensiveMove);
        }

        foreach (var piece in _board.EnemyPieces.ToList())
        {
            EvaluateAttacksWithLookahead(piece, candidates);
            EvaluateMovesWithLookahead(piece, candidates);
            EvaluateAbilities(piece, candidates);
        }

        if (candidates.Count == 0)
            return null;

        // Sort and pick best (add small random variance to prevent predictable ties)
        candidates = candidates.OrderByDescending(c => c.Score + _rng.RandiRange(-3, 3)).ToList();

        // Log top candidates
        GameLogger.Debug("AI", "Top 3:");
        foreach (var c in candidates.Take(3))
            GameLogger.Debug("AI", $"  {c.Piece.PieceType} {c.Action} {c.Target?.ToChessNotation() ?? ""}: {c.Score} ({c.Reason})");

        return candidates[0];
    }

    private void EvaluateAttacksWithLookahead(BasePiece piece, List<AIDecision> candidates)
    {
        var attacks = piece.GetAttackablePositions(_board);

        foreach (var targetPos in attacks)
        {
            var target = _board.GetPieceAt(targetPos);
            if (target == null) continue;

            int score = 0;
            var reasons = new List<string>();

            // Base capture value
            int captureValue = PieceValues[target.PieceType];
            score += captureValue;
            reasons.Add($"capture {target.PieceType}");

            // Kill bonus
            int expectedDamage = piece.BaseDamage + 4;
            if (target.CurrentHp <= expectedDamage)
            {
                score += 200;
                reasons.Add("likely kill");
            }

            // KING ATTACK - highest priority
            if (target.PieceType == PieceType.King)
            {
                score += 5000;
                reasons.Add("KING!");
            }

            // Lookahead: will we lose our attacker?
            if (IsSquareAttackedBy(piece.BoardPosition, Team.Player))
            {
                int ourValue = PieceValues[piece.PieceType];
                // If we're trading down, it's bad
                if (ourValue > captureValue)
                {
                    score -= (ourValue - captureValue);
                    reasons.Add($"bad trade -{ourValue - captureValue}");
                }
                else if (ourValue < captureValue)
                {
                    score += 100; // Good trade bonus
                    reasons.Add("good trade");
                }
            }

            // Undefended target bonus
            if (!IsSquareDefendedBy(targetPos, Team.Player))
            {
                score += 50;
                reasons.Add("undefended");
            }

            candidates.Add(new AIDecision(piece, ActionType.Attack, targetPos, score, string.Join(", ", reasons)));
        }
    }

    private void EvaluateMovesWithLookahead(BasePiece piece, List<AIDecision> candidates)
    {
        var moves = piece.GetValidMoves(_board);

        foreach (var movePos in moves)
        {
            int score = 0;
            var reasons = new List<string>();

            // === SAFETY (Critical with lookahead) ===
            bool toSafe = !IsSquareAttackedBy(movePos, Team.Player);
            bool fromSafe = !IsSquareAttackedBy(piece.BoardPosition, Team.Player);

            if (!toSafe)
            {
                // Moving into attacked square
                if (fromSafe)
                {
                    // Safe -> Danger = BAD
                    score -= PieceValues[piece.PieceType];
                    reasons.Add("into danger!");
                }
                else
                {
                    // Danger -> Different Danger
                    score -= 30;
                    reasons.Add("still unsafe");
                }
            }
            else if (!fromSafe)
            {
                // Escaping attack!
                score += PieceValues[piece.PieceType] / 2;
                reasons.Add("ESCAPE!");
            }

            // King safety - never move into check
            if (piece.PieceType == PieceType.King && !toSafe)
            {
                continue; // Skip entirely
            }

            // === POSITIONAL ===
            score += EvaluatePositionalMove(piece, movePos, reasons);

            // === PIECE COORDINATION ===
            score += EvaluatePieceCoordination(piece, movePos, reasons);

            // === PAWN STRUCTURE ===
            if (piece.PieceType == PieceType.Pawn)
            {
                score += EvaluatePawnMove(piece, movePos, reasons);
            }

            // === LOOKAHEAD: What attacks does this create? ===
            int attacksCreated = SimulateAttacksFromPosition(piece, movePos);
            if (attacksCreated > 0)
            {
                score += attacksCreated * 30;
                reasons.Add($"+{attacksCreated} attacks");
            }

            // Skip terrible moves
            if (score < -300)
                continue;

            candidates.Add(new AIDecision(piece, ActionType.Move, movePos, score, string.Join(", ", reasons)));
        }
    }

    #endregion

    #region Defensive Awareness

    private AIDecision? FindDefensiveMove()
    {
        // Check if any of our valuable pieces are under attack
        foreach (var piece in _board.EnemyPieces.OrderByDescending(p => PieceValues[p.PieceType]))
        {
            if (piece.PieceType == PieceType.Pawn) continue; // Don't overreact to pawn threats

            if (IsSquareAttackedBy(piece.BoardPosition, Team.Player))
            {
                int pieceValue = PieceValues[piece.PieceType];

                // Option 1: Move the threatened piece to safety
                var safeMoves = piece.GetValidMoves(_board)
                    .Where(m => !IsSquareAttackedBy(m, Team.Player))
                    .ToList();

                if (safeMoves.Count > 0)
                {
                    var bestSafe = safeMoves
                        .OrderByDescending(m => EvaluatePositionalMove(piece, m, new List<string>()))
                        .First();

                    return new AIDecision(piece, ActionType.Move, bestSafe,
                        pieceValue / 2 + 100,
                        $"save {piece.PieceType} from attack");
                }

                // Option 2: Can we capture the attacker?
                foreach (var defender in _board.EnemyPieces)
                {
                    var attacks = defender.GetAttackablePositions(_board);
                    foreach (var attackPos in attacks)
                    {
                        var target = _board.GetPieceAt(attackPos);
                        if (target != null && target.Team == Team.Player)
                        {
                            // Is this the piece attacking us?
                            if (target.GetAttackablePositions(_board).Contains(piece.BoardPosition))
                            {
                                return new AIDecision(defender, ActionType.Attack, attackPos,
                                    PieceValues[target.PieceType] + 200,
                                    $"counter-attack {target.PieceType} threatening {piece.PieceType}");
                            }
                        }
                    }
                }

                // Option 3: Block or interpose (complex, skip for now)
            }
        }

        // Check if our King is in danger
        if (_board.EnemyKing != null && IsSquareAttackedBy(_board.EnemyKing.BoardPosition, Team.Player))
        {
            // King under attack! Must respond
            var kingMoves = _board.EnemyKing.GetValidMoves(_board)
                .Where(m => !IsSquareAttackedBy(m, Team.Player))
                .ToList();

            if (kingMoves.Count > 0)
            {
                return new AIDecision(_board.EnemyKing, ActionType.Move, kingMoves[0],
                    5000, "KING ESCAPE!");
            }
        }

        return null;
    }

    #endregion

    #region Pawn Structure Evaluation

    private int EvaluatePawnMove(BasePiece pawn, Vector2I toPos, List<string> reasons)
    {
        int score = 0;

        // Advancement is good
        int advancement = pawn.BoardPosition.Y - toPos.Y;
        if (advancement > 0)
        {
            score += advancement * 15;

            // Bonus for passed pawn potential (no enemy pawns ahead)
            bool isPassed = true;
            for (int y = toPos.Y - 1; y >= 0; y--)
            {
                for (int dx = -1; dx <= 1; dx++)
                {
                    var checkPos = new Vector2I(toPos.X + dx, y);
                    var p = _board.GetPieceAt(checkPos);
                    if (p != null && p.Team == Team.Player && p.PieceType == PieceType.Pawn)
                    {
                        isPassed = false;
                        break;
                    }
                }
            }
            if (isPassed)
            {
                score += 40;
                reasons.Add("passed pawn");
            }
        }

        // Penalty for doubled pawns
        int pawnsOnFile = _board.EnemyPieces
            .Count(p => p.PieceType == PieceType.Pawn && p.BoardPosition.X == toPos.X && p != pawn);
        if (pawnsOnFile > 0)
        {
            score -= 30;
            reasons.Add("doubled");
        }

        // Bonus for connected pawns (pawn next to another pawn)
        bool hasNeighbor = false;
        foreach (var dx in new[] { -1, 1 })
        {
            var neighbor = _board.GetPieceAt(new Vector2I(toPos.X + dx, toPos.Y));
            if (neighbor != null && neighbor.Team == Team.Enemy && neighbor.PieceType == PieceType.Pawn)
            {
                hasNeighbor = true;
                break;
            }
        }
        if (hasNeighbor)
        {
            score += 20;
            reasons.Add("connected");
        }

        // Penalty for isolated pawn (no friendly pawns on adjacent files)
        bool hasAdjFilePawn = false;
        foreach (var dx in new[] { -1, 1 })
        {
            int adjFile = toPos.X + dx;
            if (adjFile >= 0 && adjFile < 8)
            {
                if (_board.EnemyPieces.Any(p => p.PieceType == PieceType.Pawn && p.BoardPosition.X == adjFile))
                {
                    hasAdjFilePawn = true;
                    break;
                }
            }
        }
        if (!hasAdjFilePawn)
        {
            score -= 20;
            reasons.Add("isolated");
        }

        return score;
    }

    #endregion

    #region Piece Coordination

    private int EvaluatePieceCoordination(BasePiece piece, Vector2I toPos, List<string> reasons)
    {
        int score = 0;

        // Bonus for defending other pieces
        int piecesDefended = 0;
        var attacksFromNew = SimulateThreatsFromPosition(piece, toPos);
        foreach (var threatPos in attacksFromNew)
        {
            var ally = _board.GetPieceAt(threatPos);
            if (ally != null && ally.Team == Team.Enemy && ally != piece)
            {
                piecesDefended++;
            }
        }
        if (piecesDefended > 0)
        {
            score += piecesDefended * 15;
            reasons.Add($"defends {piecesDefended}");
        }

        // Rook on open file bonus
        if (piece.PieceType == PieceType.Rook)
        {
            bool openFile = true;
            for (int y = 0; y < 8; y++)
            {
                var p = _board.GetPieceAt(new Vector2I(toPos.X, y));
                if (p != null && p.PieceType == PieceType.Pawn)
                {
                    openFile = false;
                    break;
                }
            }
            if (openFile)
            {
                score += 30;
                reasons.Add("open file");
            }

            // Connected rooks (same rank or file)
            foreach (var other in _board.EnemyPieces.Where(p => p.PieceType == PieceType.Rook && p != piece))
            {
                if (other.BoardPosition.X == toPos.X || other.BoardPosition.Y == toPos.Y)
                {
                    score += 25;
                    reasons.Add("connected rooks");
                    break;
                }
            }
        }

        // Bishop pair bonus (if we have both bishops)
        if (piece.PieceType == PieceType.Bishop)
        {
            int bishopCount = _board.EnemyPieces.Count(p => p.PieceType == PieceType.Bishop);
            if (bishopCount >= 2)
            {
                score += 30;
            }
        }

        // Knight outpost (knight on advanced square defended by pawn)
        if (piece.PieceType == PieceType.Knight && toPos.Y <= 4)
        {
            bool defendedByPawn = false;
            foreach (var dx in new[] { -1, 1 })
            {
                var pawnPos = new Vector2I(toPos.X + dx, toPos.Y + 1);
                var p = _board.GetPieceAt(pawnPos);
                if (p != null && p.Team == Team.Enemy && p.PieceType == PieceType.Pawn)
                {
                    defendedByPawn = true;
                    break;
                }
            }
            if (defendedByPawn)
            {
                score += 35;
                reasons.Add("outpost");
            }
        }

        return score;
    }

    #endregion

    #region Positional Evaluation

    private int EvaluatePositionalMove(BasePiece piece, Vector2I toPos, List<string> reasons)
    {
        int score = 0;

        // Center control
        int centerDist = GetCenterDistance(toPos);
        int oldCenterDist = GetCenterDistance(piece.BoardPosition);
        if (centerDist < oldCenterDist)
        {
            int bonus = (3 - centerDist) * 10;
            score += bonus;
            if (bonus > 0) reasons.Add($"center +{bonus}");
        }

        // Development (early game)
        if (_gameState.TurnNumber <= 12)
        {
            if ((piece.PieceType == PieceType.Knight || piece.PieceType == PieceType.Bishop) &&
                piece.BoardPosition.Y >= 6)
            {
                score += 40;
                reasons.Add("develop");
            }
        }

        // King safety - keep King back in middlegame
        if (piece.PieceType == PieceType.King && _gameState.TurnNumber < 30)
        {
            if (toPos.Y < 6)
            {
                score -= 50;
                reasons.Add("King too forward");
            }
        }

        // Advancement (for non-King pieces)
        if (piece.PieceType != PieceType.King)
        {
            int adv = piece.BoardPosition.Y - toPos.Y;
            if (adv > 0)
            {
                score += adv * 8;
            }
        }

        return score;
    }

    private int EvaluatePosition(Team forTeam)
    {
        int score = 0;

        // Material
        foreach (var p in _board.EnemyPieces)
            score += PieceValues[p.PieceType] + (p.CurrentHp * 2);
        foreach (var p in _board.PlayerPieces)
            score -= PieceValues[p.PieceType] + (p.CurrentHp * 2);

        // King safety
        if (_board.EnemyKing != null && IsSquareAttackedBy(_board.EnemyKing.BoardPosition, Team.Player))
            score -= 300;
        if (_board.PlayerKing != null && IsSquareAttackedBy(_board.PlayerKing.BoardPosition, Team.Enemy))
            score += 200;

        // Mobility (number of legal moves)
        int enemyMobility = _board.EnemyPieces.Sum(p => p.GetValidMoves(_board).Count);
        int playerMobility = _board.PlayerPieces.Sum(p => p.GetValidMoves(_board).Count);
        score += (enemyMobility - playerMobility) * 5;

        return forTeam == Team.Enemy ? score : -score;
    }

    #endregion

    #region Helper Methods

    private bool IsSquareAttackedBy(Vector2I square, Team team)
    {
        var pieces = team == Team.Player ? _board.PlayerPieces : _board.EnemyPieces;
        return pieces.Any(p => p.GetThreatenedPositions(_board).Contains(square));
    }

    private bool IsSquareDefendedBy(Vector2I square, Team team)
    {
        var pieces = team == Team.Player ? _board.PlayerPieces : _board.EnemyPieces;
        return pieces.Any(p => p.GetThreatenedPositions(_board).Contains(square));
    }

    private int SimulateAttacksFromPosition(BasePiece piece, Vector2I fromPos)
    {
        int count = 0;
        foreach (var dir in GetDirections(piece))
        {
            var checkPos = fromPos + dir;
            while (checkPos.IsOnBoard())
            {
                var target = _board.GetPieceAt(checkPos);
                if (target != null)
                {
                    if (target.Team == Team.Player)
                        count++;
                    break;
                }
                if (!IsRangePiece(piece)) break;
                checkPos += dir;
            }
        }
        return count;
    }

    private List<Vector2I> SimulateThreatsFromPosition(BasePiece piece, Vector2I fromPos)
    {
        var threats = new List<Vector2I>();
        foreach (var dir in GetDirections(piece))
        {
            var checkPos = fromPos + dir;
            while (checkPos.IsOnBoard())
            {
                threats.Add(checkPos);
                if (_board.IsOccupied(checkPos)) break;
                if (!IsRangePiece(piece)) break;
                checkPos += dir;
            }
        }
        return threats;
    }

    private Vector2I[] GetDirections(BasePiece piece) => piece.PieceType switch
    {
        PieceType.Queen => Vector2IExtensions.AllDirections,
        PieceType.Rook => Vector2IExtensions.CardinalDirections,
        PieceType.Bishop => Vector2IExtensions.DiagonalDirections,
        _ => Vector2IExtensions.AllDirections
    };

    private bool IsRangePiece(BasePiece piece) =>
        piece.PieceType is PieceType.Queen or PieceType.Rook or PieceType.Bishop;

    private int GetCenterDistance(Vector2I pos)
    {
        int dx = Math.Min(Math.Abs(pos.X - 3), Math.Abs(pos.X - 4));
        int dy = Math.Min(Math.Abs(pos.Y - 3), Math.Abs(pos.Y - 4));
        return Math.Max(dx, dy);
    }

    private void EvaluateAbilities(BasePiece piece, List<AIDecision> candidates)
    {
        if (!piece.CanUseAbility) return;

        int score = 0;
        string reason = "";

        switch (piece.AbilityId)
        {
            case AbilityId.RoyalDecree:
                int attackers = _board.EnemyPieces.Count(p => p.GetAttackablePositions(_board).Count > 0);
                if (attackers >= 2)
                {
                    score = 60 + attackers * 25;
                    reason = $"Royal Decree: {attackers} attackers";
                }
                break;

            case AbilityId.Consecration when piece is BishopPiece bishop:
                foreach (var pos in bishop.GetHealTargets(_board))
                {
                    var target = _board.GetPieceAt(pos);
                    if (target != null && target.CurrentHp < target.MaxHp * 0.6f)
                    {
                        int healScore = (int)((1 - (float)target.CurrentHp / target.MaxHp) * 150);
                        if (healScore > score)
                        {
                            score = healScore;
                            reason = $"Heal {target.PieceType}";
                        }
                    }
                }
                break;

            case AbilityId.Advance when piece is PawnPiece pawn:
                var advTarget = pawn.GetAdvanceTarget(_board);
                if (advTarget.HasValue && !IsSquareAttackedBy(advTarget.Value, Team.Player))
                {
                    score = 35;
                    reason = "Safe advance";
                }
                break;
        }

        if (score > 0)
            candidates.Add(new AIDecision(piece, ActionType.Ability, null, score, reason));
    }

    private void ExecuteDecision(AIDecision decision)
    {
        switch (decision.Action)
        {
            case ActionType.Attack when decision.Target.HasValue:
                var target = _board.GetPieceAt(decision.Target.Value);
                if (target != null)
                    _turnController.ExecuteAttack(decision.Piece, target);
                break;

            case ActionType.Move when decision.Target.HasValue:
                _turnController.ExecuteMove(decision.Piece, decision.Target.Value);
                break;

            case ActionType.Ability:
                _turnController.ExecuteAbility(decision.Piece, decision.Target);
                break;
        }
    }

    #endregion
}
