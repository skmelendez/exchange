using Godot;
using System.Linq;
using Exchange.Core;
using Exchange.Board;
using Exchange.Pieces;
using Exchange.Combat;

namespace Exchange.Controllers;

/// <summary>
/// Manages the turn-based game flow and enforces turn economy rules.
/// Per design doc: Each turn, exactly ONE piece may act. Action is ONE of: Move, Attack, or Ability.
/// </summary>
public partial class TurnController : Node
{
    // C# events for complex types (Godot signals have type restrictions)
    public event Action<Team, int>? TurnStarted;
    public event Action<Team>? TurnEnded;
    public event Action<BasePiece, ActionType>? ActionCompleted;
    public event Action<Team>? MatchEnded;
    public event Action<string>? MatchDrawn;  // Called when match ends in draw (reason provided)
    public event Action<BasePiece?>? PieceSelected;
    public event Action<List<Vector2I>>? ValidMovesCalculated;
    public event Action<List<Vector2I>>? ValidAttacksCalculated;
    public event Action<BasePiece, Vector2I, Vector2I>? PieceMoved;  // piece, from, to
    public event Action<PawnPiece>? PawnPromotionRequired;  // For player pawn promotion UI

    private GameState _gameState = null!;
    private GameBoard _board = null!;
    private CombatResolver _combatResolver = null!;
    private AIController? _aiController;

    // Current turn state
    private BasePiece? _selectedPiece;
    private List<Vector2I> _validMoves = new();
    private List<Vector2I> _validAttacks = new();
    private bool _awaitingAction = false;

    // Multi-step ability state
    private AbilityPhase _abilityPhase = AbilityPhase.None;
    private BasePiece? _abilityPiece;
    private List<Vector2I> _abilityValidPositions = new();

    // Pawn promotion state
    private PawnPiece? _pendingPromotionPawn;
    public bool IsAwaitingPromotion => _pendingPromotionPawn != null;

    // Damage tracking for 50-move draw rule
    private bool _damageDealtThisTurn = false;

    public BasePiece? SelectedPiece => _selectedPiece;
    public bool IsPlayerTurn => _gameState.CurrentPhase == GamePhase.PlayerTurn;
    public bool IsAwaitingPlayerInput => IsPlayerTurn && _awaitingAction;
    public AbilityPhase CurrentAbilityPhase => _abilityPhase;

    public void Initialize(GameState gameState, GameBoard board, CombatResolver combatResolver, AIController? aiController = null)
    {
        _gameState = gameState;
        _board = board;
        _combatResolver = combatResolver;
        _aiController = aiController;
    }

    public void StartMatch()
    {
        _gameState.ResetForNewMatch();
        _board.RecalculateThreatZones();

        // Reset all pieces
        foreach (var piece in _board.PlayerPieces)
            piece.ResetForNewMatch();
        foreach (var piece in _board.EnemyPieces)
            piece.ResetForNewMatch();

        StartTurn();
    }

    private void StartTurn()
    {
        GameLogger.Debug("Turn", $"=== StartTurn called === Phase={_gameState.CurrentPhase}, Team={_gameState.CurrentTeam}");

        // Tick cooldowns for current team
        var currentTeamPieces = _gameState.CurrentTeam == Team.Player
            ? _board.PlayerPieces
            : _board.EnemyPieces;

        foreach (var piece in currentTeamPieces)
        {
            piece.TickCooldown();
            piece.ResetTurnState();
        }

        // Reset pawn Advance tracking
        foreach (var piece in currentTeamPieces.OfType<PawnPiece>())
        {
            piece.UsedAdvanceLastTurn = false;
        }

        // Clear Royal Decree at start of the team's turn (it lasts until their next turn)
        if (_gameState.CurrentTeam == Team.Player)
            _gameState.PlayerRoyalDecreeActive = false;
        else
            _gameState.EnemyRoyalDecreeActive = false;

        // Apply Boss Rules
        ApplyBossRulesForTurn();

        _awaitingAction = true;
        GameLogger.Debug("Turn", "_awaitingAction set to TRUE, emitting TurnStarted");
        TurnStarted?.Invoke(_gameState.CurrentTeam, _gameState.TurnNumber);

        GameLogger.Debug("Turn", $"{_gameState.CurrentTeam}'s turn #{_gameState.TurnNumber}");

        // If enemy turn, trigger AI
        if (_gameState.CurrentPhase == GamePhase.EnemyTurn && _aiController != null)
        {
            GameLogger.Debug("Turn", "Enemy turn - triggering AI...");
            _aiController.ExecuteTurn();
        }
        else
        {
            GameLogger.Debug("Turn", "Player turn - awaiting input");
        }
    }

    public void SelectPiece(BasePiece? piece)
    {
        if (!IsAwaitingPlayerInput) return;

        // Don't allow piece selection changes during multi-step ability execution
        if (_abilityPhase != AbilityPhase.None)
        {
            GameLogger.Debug("Select", $"Blocked - in ability phase {_abilityPhase}");
            return;
        }

        if (piece != null && piece.Team != Team.Player) return;

        _selectedPiece = piece;
        _validMoves.Clear();
        _validAttacks.Clear();

        if (piece != null)
        {
            _validMoves = piece.GetValidMoves(_board);
            _validAttacks = piece.GetAttackablePositions(_board);

            // Knight special rule: MUST move before attacking
            if (piece is KnightPiece knight)
            {
                if (knight.HasMovedThisTurn)
                {
                    // Already moved - can attack but not move again
                    _validMoves.Clear();
                }
                else
                {
                    // Hasn't moved yet - can move but NOT attack standing still
                    _validAttacks.Clear();
                }
            }
        }

        PieceSelected?.Invoke(piece);
        ValidMovesCalculated?.Invoke(_validMoves);
        ValidAttacksCalculated?.Invoke(_validAttacks);
    }

    public bool TryMove(Vector2I targetPosition)
    {
        if (!IsAwaitingPlayerInput) return false;

        // Handle Overextend move phase
        if (_abilityPhase == AbilityPhase.OverextendSelectMove && _abilityPiece != null)
        {
            if (!_abilityValidPositions.Contains(targetPosition)) return false;
            ExecuteOverextendMove(_abilityPiece, targetPosition);
            return true;
        }

        if (_selectedPiece == null) return false;
        if (!_validMoves.Contains(targetPosition)) return false;

        ExecuteMove(_selectedPiece, targetPosition);
        return true;
    }

    public bool TryAttack(Vector2I targetPosition)
    {
        if (!IsAwaitingPlayerInput) return false;

        // Handle Overextend attack phase
        if (_abilityPhase == AbilityPhase.OverextendSelectAttack && _abilityPiece != null)
        {
            if (!_abilityValidPositions.Contains(targetPosition)) return false;
            var target = _board.GetPieceAt(targetPosition);
            if (target == null) return false;
            ExecuteOverextendAttack(_abilityPiece, target);
            return true;
        }

        // Handle Skirmish attack phase
        if (_abilityPhase == AbilityPhase.SkirmishSelectAttack && _abilityPiece != null)
        {
            if (!_abilityValidPositions.Contains(targetPosition)) return false;
            var target = _board.GetPieceAt(targetPosition);
            if (target == null) return false;
            ExecuteSkirmishAttack(_abilityPiece as KnightPiece, target);
            return true;
        }

        if (_selectedPiece == null) return false;
        if (!_validAttacks.Contains(targetPosition)) return false;

        var attackTarget = _board.GetPieceAt(targetPosition);
        if (attackTarget == null) return false;

        ExecuteAttack(_selectedPiece, attackTarget);
        return true;
    }

    public bool TryUseAbility(Vector2I? targetPosition = null)
    {
        if (!IsAwaitingPlayerInput || _selectedPiece == null) return false;
        if (!_selectedPiece.CanUseAbility) return false;

        ExecuteAbility(_selectedPiece, targetPosition);
        return true;
    }

    /// <summary>
    /// Handle Skirmish reposition phase
    /// </summary>
    public bool TryReposition(Vector2I targetPosition)
    {
        if (!IsAwaitingPlayerInput) return false;

        if (_abilityPhase == AbilityPhase.SkirmishSelectReposition && _abilityPiece is KnightPiece knight)
        {
            if (!_abilityValidPositions.Contains(targetPosition)) return false;
            ExecuteSkirmishReposition(knight, targetPosition);
            return true;
        }

        return false;
    }

    /// <summary>
    /// Cancel multi-step ability execution
    /// Note: Some phases cannot be cancelled (e.g., Skirmish reposition - Knight MUST move after attacking)
    /// </summary>
    public void CancelAbility()
    {
        if (_abilityPhase == AbilityPhase.None)
            return;

        // Skirmish reposition CANNOT be cancelled - Knight must move after attacking
        if (_abilityPhase == AbilityPhase.SkirmishSelectReposition)
        {
            GameLogger.Debug("Ability", "Cannot cancel Skirmish reposition - Knight must move!");
            return;
        }

        GameLogger.Debug("Ability", $"Cancelled {_abilityPhase}");
        _abilityPhase = AbilityPhase.None;
        _abilityPiece = null;
        _abilityValidPositions.Clear();

        // Re-emit selection to restore normal UI
        PieceSelected?.Invoke(_selectedPiece);
        if (_selectedPiece != null)
        {
            ValidMovesCalculated?.Invoke(_selectedPiece.GetValidMoves(_board));
            ValidAttacksCalculated?.Invoke(_selectedPiece.GetAttackablePositions(_board));
        }
    }

    /// <summary>
    /// Execute a move action
    /// </summary>
    public void ExecuteMove(BasePiece piece, Vector2I targetPosition)
    {
        // VALIDATION: Ensure this is actually a valid move for this piece
        var validMoves = piece.GetValidMoves(_board);
        if (!validMoves.Contains(targetPosition))
        {
            GameLogger.Error("Move", $"INVALID MOVE BLOCKED! {piece.Team} {piece.PieceType} at {piece.BoardPosition.ToChessNotation()} tried to move to {targetPosition.ToChessNotation()}");
            GameLogger.Error("Move", $"Valid moves were: [{string.Join(", ", validMoves.Select(m => m.ToChessNotation()))}]");

            // Log what's blocking the path
            var blockers = new List<string>();
            foreach (var dir in Vector2IExtensions.AllDirections)
            {
                var checkPos = piece.BoardPosition + dir;
                while (checkPos.IsOnBoard())
                {
                    var blocker = _board.GetPieceAt(checkPos);
                    if (blocker != null)
                    {
                        blockers.Add($"{blocker.Team} {blocker.PieceType} at {checkPos.ToChessNotation()}");
                        break;
                    }
                    checkPos += dir;
                }
            }
            GameLogger.Error("Move", $"Pieces on board in LOS: [{string.Join(", ", blockers)}]");

            // For AI, end turn anyway to prevent getting stuck
            if (piece.Team == Team.Enemy)
            {
                GameLogger.Warning("Move", "AI made invalid move - ending turn to prevent stuck state");
                EndAction(piece, ActionType.Move);
            }
            return;
        }

        // Check if entering threat zone
        var targetTile = _board.GetTile(targetPosition);
        if (targetTile.IsThreatened(piece.Team))
            piece.EnteredThreatZoneThisTurn = true;

        var fromPos = piece.BoardPosition;
        _board.MovePiece(piece, targetPosition);
        _board.RecalculateThreatZones();

        string team = piece.Team == Team.Player ? "W" : "B";
        GameLogger.Debug("Action", $"[{team}] {piece.PieceType} moves {fromPos.ToChessNotation()} -> {targetPosition.ToChessNotation()}");

        // Notify UI of move
        PieceMoved?.Invoke(piece, fromPos, targetPosition);

        // Check for pawn promotion
        if (piece is PawnPiece pawn && pawn.ShouldPromote)
        {
            HandlePawnPromotion(pawn, ActionType.Move);
            return; // Don't end turn yet - waiting for promotion selection
        }

        // Knight special rule: can still attack after moving
        if (piece is KnightPiece knight)
        {
            knight.HasMovedThisTurn = true;
            // Recalculate attacks from new position
            _validMoves.Clear();
            _validAttacks = piece.GetAttackablePositions(_board);

            if (_validAttacks.Count > 0)
            {
                // If it's the AI's turn, let AI handle the attack decision
                if (piece.Team == Team.Enemy)
                {
                    GameLogger.Debug("Knight", $"AI Knight has {_validAttacks.Count} adjacent enemy(s) - AI will decide");
                    // Don't wait for input, just end the move action
                    // AI will need to handle Knight attack separately if we want that behavior
                    // For now, AI Knight just ends turn after moving (simplification)
                }
                else
                {
                    GameLogger.Debug("Knight", $"May now attack {_validAttacks.Count} adjacent enemy(s) or press SPACE to end turn");
                    // Re-emit piece selected to clear old highlights, then show new attacks
                    PieceSelected?.Invoke(piece);
                    ValidMovesCalculated?.Invoke(new List<Vector2I>());
                    ValidAttacksCalculated?.Invoke(_validAttacks);
                    return; // Don't end turn yet - waiting for player attack or Space
                }
            }
            else
            {
                GameLogger.Debug("Knight", "No adjacent enemies to attack, ending turn");
            }
        }

        EndAction(piece, ActionType.Move);
    }

    /// <summary>
    /// Execute an attack action
    /// </summary>
    public void ExecuteAttack(BasePiece attacker, BasePiece defender)
    {
        // VALIDATION: Ensure this is actually a valid attack for this piece
        var validAttacks = attacker.GetAttackablePositions(_board);
        if (!validAttacks.Contains(defender.BoardPosition))
        {
            GameLogger.Error("Attack", $"INVALID ATTACK BLOCKED! {attacker.Team} {attacker.PieceType} at {attacker.BoardPosition.ToChessNotation()} tried to attack {defender.PieceType} at {defender.BoardPosition.ToChessNotation()}");
            GameLogger.Error("Attack", $"Valid attack targets were: [{string.Join(", ", validAttacks.Select(a => a.ToChessNotation()))}]");

            // For AI, end turn anyway to prevent getting stuck
            if (attacker.Team == Team.Enemy)
            {
                GameLogger.Warning("Attack", "AI made invalid attack - ending turn to prevent stuck state");
                EndAction(attacker, ActionType.Attack);
            }
            return;
        }

        var result = _combatResolver.ResolveAttack(attacker, defender, _board);

        // Track damage for 50-move draw rule
        if (result.DamageDealt > 0)
            _damageDealtThisTurn = true;

        if (result.DefenderDestroyed)
        {
            _board.RemovePiece(defender);
            _board.RecalculateThreatZones();

            // Check win condition
            if (defender.PieceType == PieceType.King)
            {
                EndMatch(attacker.Team);
                return;
            }
        }

        EndAction(attacker, ActionType.Attack);
    }

    /// <summary>
    /// Execute an ability
    /// </summary>
    public void ExecuteAbility(BasePiece piece, Vector2I? targetPosition)
    {
        switch (piece.AbilityId)
        {
            case AbilityId.RoyalDecree:
                piece.StartAbilityCooldown();
                ExecuteRoyalDecree(piece);
                EndAction(piece, ActionType.Ability);
                break;

            case AbilityId.Overextend:
                StartOverextend(piece);
                // Don't end action yet - multi-step
                break;

            case AbilityId.Interpose:
                piece.StartAbilityCooldown();
                ExecuteInterpose(piece as RookPiece);
                EndAction(piece, ActionType.Ability);
                break;

            case AbilityId.Consecration:
                piece.StartAbilityCooldown();
                if (targetPosition.HasValue)
                    ExecuteConsecration(piece as BishopPiece, targetPosition.Value);
                EndAction(piece, ActionType.Ability);
                break;

            case AbilityId.Skirmish:
                StartSkirmish(piece as KnightPiece);
                // Don't end action yet - multi-step
                break;

            case AbilityId.Advance:
                piece.StartAbilityCooldown();
                ExecuteAdvance(piece as PawnPiece);
                // Only end action if no promotion occurred
                // (HandlePawnPromotion will call EndAction after promotion completes)
                if (!IsAwaitingPromotion)
                    EndAction(piece, ActionType.Ability);
                break;
        }
    }

    private void ExecuteRoyalDecree(BasePiece king)
    {
        if (king.Team == Team.Player)
            _gameState.PlayerRoyalDecreeActive = true;
        else
            _gameState.EnemyRoyalDecreeActive = true;

        GameLogger.Debug("Ability", $"Royal Decree activated! All {king.Team} combat rolls +1 until next turn.");
    }

    private void ExecuteInterpose(RookPiece? rook)
    {
        if (rook == null) return;

        rook.InterposeActive = true;
        GameLogger.Debug("Ability", "Interpose activated! Damage to adjacent allies will be split with Rook.");
    }

    #region Overextend (Queen)

    private void StartOverextend(BasePiece piece)
    {
        var moves = piece.GetValidMoves(_board);
        if (moves.Count == 0)
        {
            GameLogger.Debug("Ability", "Overextend: No valid moves available!");
            return;
        }

        _abilityPhase = AbilityPhase.OverextendSelectMove;
        _abilityPiece = piece;
        _abilityValidPositions = moves;

        GameLogger.Debug("Ability", $"Overextend: Select move target ({moves.Count} options)");

        // Show move options
        PieceSelected?.Invoke(piece);
        ValidMovesCalculated?.Invoke(moves);
        ValidAttacksCalculated?.Invoke(new List<Vector2I>());
    }

    private void ExecuteOverextendMove(BasePiece piece, Vector2I targetPosition)
    {
        // Check threat zone
        var fromPos = piece.BoardPosition;
        var targetTile = _board.GetTile(targetPosition);
        if (targetTile.IsThreatened(piece.Team))
            piece.EnteredThreatZoneThisTurn = true;

        _board.MovePiece(piece, targetPosition);
        _board.RecalculateThreatZones();

        string team = piece.Team == Team.Player ? "W" : "B";
        GameLogger.Debug("Ability", $"[{team}] Queen Overextend {fromPos.ToChessNotation()} -> {targetPosition.ToChessNotation()}");

        // Notify UI of move
        PieceMoved?.Invoke(piece, fromPos, targetPosition);

        // Transition to attack phase
        var attacks = piece.GetAttackablePositions(_board);
        if (attacks.Count == 0)
        {
            GameLogger.Debug("Ability", "Overextend: No attack targets! Queen takes 2 damage anyway.");
            piece.StartAbilityCooldown();
            piece.TakeDamage(2);
            GameLogger.Debug("Ability", $"Overextend: Queen HP now {piece.CurrentHp}/{piece.MaxHp}");
            CompleteOverextend(piece);
            return;
        }

        _abilityPhase = AbilityPhase.OverextendSelectAttack;
        _abilityValidPositions = attacks;

        GameLogger.Debug("Ability", $"Overextend: Select attack target ({attacks.Count} options)");

        // Show attack options
        ValidMovesCalculated?.Invoke(new List<Vector2I>());
        ValidAttacksCalculated?.Invoke(attacks);
    }

    private void ExecuteOverextendAttack(BasePiece piece, BasePiece target)
    {
        piece.StartAbilityCooldown();

        var result = _combatResolver.ResolveAttack(piece, target, _board);

        if (result.DefenderDestroyed)
        {
            _board.RemovePiece(target);
            _board.RecalculateThreatZones();

            // Check win condition
            if (target.PieceType == PieceType.King)
            {
                EndMatch(piece.Team);
                return;
            }
        }

        // Queen takes 2 self-damage
        piece.TakeDamage(2);
        GameLogger.Debug("Ability", $"Overextend: Queen takes 2 self-damage! HP now {piece.CurrentHp}/{piece.MaxHp}");

        // Check if Queen died from self-damage
        if (!piece.IsAlive)
        {
            GameLogger.Debug("Ability", "Overextend: Queen died from self-damage!");
            _board.RemovePiece(piece);
            _board.RecalculateThreatZones();
        }

        CompleteOverextend(piece);
    }

    private void CompleteOverextend(BasePiece piece)
    {
        _abilityPhase = AbilityPhase.None;
        _abilityPiece = null;
        _abilityValidPositions.Clear();

        EndAction(piece, ActionType.Ability);
    }

    #endregion

    #region Skirmish (Knight)

    private void StartSkirmish(KnightPiece? knight)
    {
        if (knight == null) return;

        var attacks = knight.GetAttackablePositions(_board);
        if (attacks.Count == 0)
        {
            GameLogger.Debug("Ability", "Skirmish: No adjacent enemies to attack!");
            return;
        }

        _abilityPhase = AbilityPhase.SkirmishSelectAttack;
        _abilityPiece = knight;
        _abilityValidPositions = attacks;

        GameLogger.Debug("Ability", $"Skirmish: Select attack target ({attacks.Count} options)");

        // Show attack options
        PieceSelected?.Invoke(knight);
        ValidMovesCalculated?.Invoke(new List<Vector2I>());
        ValidAttacksCalculated?.Invoke(attacks);
    }

    private void ExecuteSkirmishAttack(KnightPiece? knight, BasePiece target)
    {
        if (knight == null) return;

        knight.StartAbilityCooldown();

        var result = _combatResolver.ResolveAttack(knight, target, _board);

        if (result.DefenderDestroyed)
        {
            _board.RemovePiece(target);
            _board.RecalculateThreatZones();

            // Check win condition
            if (target.PieceType == PieceType.King)
            {
                EndMatch(knight.Team);
                return;
            }
        }

        // Transition to reposition phase
        var repositionTiles = knight.GetSkirmishRepositionTiles(_board);
        if (repositionTiles.Count == 0)
        {
            GameLogger.Debug("Ability", "Skirmish: No reposition tiles available!");
            CompleteSkirmish(knight);
            return;
        }

        _abilityPhase = AbilityPhase.SkirmishSelectReposition;
        _abilityValidPositions = repositionTiles;

        GameLogger.Debug("Ability", $"Skirmish: Select reposition tile ({repositionTiles.Count} options)");

        // Show reposition options (as move highlights)
        ValidMovesCalculated?.Invoke(repositionTiles);
        ValidAttacksCalculated?.Invoke(new List<Vector2I>());
    }

    private void ExecuteSkirmishReposition(KnightPiece knight, Vector2I targetPosition)
    {
        // Check threat zone
        var fromPos = knight.BoardPosition;
        var targetTile = _board.GetTile(targetPosition);
        if (targetTile.IsThreatened(knight.Team))
            knight.EnteredThreatZoneThisTurn = true;

        _board.MovePiece(knight, targetPosition);
        _board.RecalculateThreatZones();

        string team = knight.Team == Team.Player ? "W" : "B";
        GameLogger.Debug("Ability", $"[{team}] Knight Skirmish reposition {fromPos.ToChessNotation()} -> {targetPosition.ToChessNotation()}");

        // Notify UI of move
        PieceMoved?.Invoke(knight, fromPos, targetPosition);

        CompleteSkirmish(knight);
    }

    private void CompleteSkirmish(KnightPiece knight)
    {
        _abilityPhase = AbilityPhase.None;
        _abilityPiece = null;
        _abilityValidPositions.Clear();

        EndAction(knight, ActionType.Ability);
    }

    #endregion

    private void ExecuteConsecration(BishopPiece? bishop, Vector2I targetPos)
    {
        if (bishop == null) return;

        var target = _board.GetPieceAt(targetPos);
        if (target == null || target.Team != bishop.Team) return;

        int healed = _combatResolver.ResolveHealing(bishop, target);
        GameLogger.Debug("Ability", $"Consecration heals {target.PieceType} for {healed}!");
    }

    private void ExecuteAdvance(PawnPiece? pawn)
    {
        if (pawn == null) return;

        var target = pawn.GetAdvanceTarget(_board);
        if (!target.HasValue) return;

        // Check threat zone
        var targetTile = _board.GetTile(target.Value);
        if (targetTile.IsThreatened(pawn.Team))
            pawn.EnteredThreatZoneThisTurn = true;

        _board.MovePiece(pawn, target.Value);
        pawn.UsedAdvanceLastTurn = true;
        _board.RecalculateThreatZones();

        GameLogger.Debug("Ability", $"Advance! Pawn moves to {target.Value.ToChessNotation()}");

        // Check for promotion after Advance
        if (pawn.ShouldPromote)
        {
            // Ability already started cooldown in ExecuteAbility, just handle promotion
            HandlePawnPromotion(pawn, ActionType.Ability);
            // Don't call EndAction - HandlePawnPromotion will do that via CompletePromotion
        }
    }

    #region Pawn Promotion

    /// <summary>
    /// Handles pawn promotion when a pawn reaches the opposite end
    /// </summary>
    private void HandlePawnPromotion(PawnPiece pawn, ActionType originalAction)
    {
        GameLogger.Info("Promotion", $"{pawn.Team} Pawn at {pawn.BoardPosition.ToChessNotation()} is ready for promotion!");

        _pendingPromotionPawn = pawn;

        if (pawn.Team == Team.Player)
        {
            // Player pawn: emit event so UI can show selection dialog
            PawnPromotionRequired?.Invoke(pawn);
            // Turn will continue when CompletePromotion is called
        }
        else
        {
            // AI pawn: auto-promote to Queen (strongest piece)
            CompletePromotion(PieceType.Queen, originalAction);
        }
    }

    /// <summary>
    /// Complete the promotion after piece type selection.
    /// Called by UI for player pawns, or automatically for AI pawns.
    /// </summary>
    public void CompletePromotion(PieceType promotionType, ActionType? actionOverride = null)
    {
        if (_pendingPromotionPawn == null)
        {
            GameLogger.Error("Promotion", "CompletePromotion called but no pending promotion!");
            return;
        }

        var pawn = _pendingPromotionPawn;
        var originalAction = actionOverride ?? ActionType.Move;
        _pendingPromotionPawn = null;

        // Perform the actual promotion
        var newPiece = _board.PromotePawn(pawn, promotionType, PieceFactory.Create);

        GameLogger.Info("Promotion", $"Pawn promoted to {promotionType} at {newPiece.BoardPosition.ToChessNotation()}");

        // Recalculate threat zones with new piece
        _board.RecalculateThreatZones();

        // End the turn
        EndAction(newPiece, originalAction);
    }

    #endregion

    private void EndAction(BasePiece piece, ActionType action)
    {
        piece.HasActedThisTurn = true;
        _awaitingAction = false;

        ActionCompleted?.Invoke(piece, action);
        GameLogger.Debug("Action", $"{piece.PieceType} completed {action}");

        EndTurn();
    }

    private void EndTurn()
    {
        GameLogger.Debug("Turn", $"=== EndTurn called === Current phase: {_gameState.CurrentPhase}");

        // Update moves-without-damage counter for 50-move draw rule
        if (_damageDealtThisTurn)
        {
            _gameState.MovesWithoutDamage = 0;
            _damageDealtThisTurn = false;
        }
        else
        {
            _gameState.MovesWithoutDamage++;
        }

        // Check for draw conditions
        if (CheckDrawConditions())
            return;

        // Check king threat status for next turn bonus
        _gameState.PlayerKingThreatened = _board.IsKingThreatened(Team.Player);
        _gameState.EnemyKingThreatened = _board.IsKingThreatened(Team.Enemy);

        // Boss Rule 5: Player King is always considered threatened during Room 5 Boss
        if (_gameState.IsBossMatch && _gameState.CurrentRoom == 5)
        {
            _gameState.PlayerKingThreatened = true;
            GameLogger.Debug("Boss", "Rule 5: Player King is ALWAYS threatened!");
        }

        if (_gameState.PlayerKingThreatened)
            GameLogger.Debug("Status", "Player King is threatened! Enemy gains +1 dice next attack.");
        if (_gameState.EnemyKingThreatened)
            GameLogger.Debug("Status", "Enemy King is threatened! Player gains +1 dice next attack.");

        TurnEnded?.Invoke(_gameState.CurrentTeam);

        // Switch turns
        var oldPhase = _gameState.CurrentPhase;
        _gameState.CurrentPhase = _gameState.CurrentPhase == GamePhase.PlayerTurn
            ? GamePhase.EnemyTurn
            : GamePhase.PlayerTurn;

        GameLogger.Debug("Turn", $"Phase switched: {oldPhase} -> {_gameState.CurrentPhase}");

        if (_gameState.CurrentPhase == GamePhase.PlayerTurn)
            _gameState.TurnNumber++;

        _selectedPiece = null;
        _validMoves.Clear();
        _validAttacks.Clear();

        GameLogger.Debug("Turn", "Calling StartTurn for next turn...");
        StartTurn();
    }

    private void EndMatch(Team winner)
    {
        _gameState.CurrentPhase = GamePhase.MatchEnd;
        _awaitingAction = false;

        GameLogger.Info("Match", $"{winner} wins!");
        MatchEnded?.Invoke(winner);
    }

    /// <summary>
    /// Check for draw conditions and end match if draw occurred
    /// </summary>
    private bool CheckDrawConditions()
    {
        // Draw condition 1: Only kings left (insufficient material)
        var alivePieces = _board.PlayerPieces.Concat(_board.EnemyPieces).Where(p => p.IsAlive).ToList();
        var nonKingPieces = alivePieces.Where(p => p.PieceType != PieceType.King).ToList();

        if (nonKingPieces.Count == 0)
        {
            DrawMatch("Only kings remain - insufficient material");
            return true;
        }

        // Draw condition 2: 50 moves without damage (stalemate)
        if (_gameState.MovesWithoutDamage >= GameState.DrawMovesWithoutDamage)
        {
            DrawMatch($"{GameState.DrawMovesWithoutDamage} moves without damage - stalemate");
            return true;
        }

        return false;
    }

    private void DrawMatch(string reason)
    {
        _gameState.CurrentPhase = GamePhase.MatchEnd;
        _awaitingAction = false;

        GameLogger.Info("Match", $"DRAW: {reason}");
        MatchDrawn?.Invoke(reason);
    }

    /// <summary>
    /// Apply boss rules at the start of each turn
    /// </summary>
    private void ApplyBossRulesForTurn()
    {
        if (!_gameState.IsBossMatch) return;

        int bossRoom = _gameState.CurrentRoom;

        switch (bossRoom)
        {
            case 1:
                // Boss Rule 1: Threat penalties are -2 (handled in CombatResolver)
                GameLogger.Debug("Boss", "Rule 1: Threat penalties are -2!");
                break;

            case 3:
                // Boss Rule 3: Enemy King can move into threatened tiles
                if (_board.EnemyKing is KingPiece enemyKing)
                {
                    enemyKing.CanIgnoreThreatZones = true;
                    GameLogger.Debug("Boss", "Rule 3: Enemy King can move into threatened tiles!");
                }
                break;

            case 4:
                // Boss Rule 4: Enemy abilities have no cooldowns
                // Reset all enemy cooldowns to 0 at start of enemy turn
                if (_gameState.CurrentTeam == Team.Enemy)
                {
                    foreach (var piece in _board.EnemyPieces)
                    {
                        piece.AbilityCooldownCurrent = 0;
                    }
                    GameLogger.Debug("Boss", "Rule 4: Enemy abilities have no cooldowns!");
                }
                break;

            case 5:
                // Boss Rule 5: Player King always considered threatened
                // This is handled in EndTurn where we set PlayerKingThreatened
                GameLogger.Debug("Boss", "Rule 5: Player King is always considered threatened!");
                break;
        }
    }

    /// <summary>
    /// DEBUG: Force end the match with a specific winner
    /// </summary>
    public void ForceMatchEnd(Team winner)
    {
        GameLogger.Debug("Debug", $"ForceMatchEnd called - {winner} wins!");
        EndMatch(winner);
    }

    /// <summary>
    /// Knight: End turn without attacking after moving
    /// </summary>
    public void KnightEndTurnWithoutAttack()
    {
        if (_selectedPiece is KnightPiece knight && knight.HasMovedThisTurn)
        {
            EndAction(knight, ActionType.Move);
        }
    }

    /// <summary>
    /// Execute Knight's special move+attack combo (AI use).
    /// Knight moves to moveTarget, then attacks the piece at attackTarget.
    /// </summary>
    public void ExecuteKnightMoveAndAttack(BasePiece knight, Vector2I moveTarget, Vector2I attackTarget)
    {
        if (knight.PieceType != PieceType.Knight)
        {
            GameLogger.Error("Knight", "ExecuteKnightMoveAndAttack called on non-Knight piece!");
            return;
        }

        // Validate the move
        var validMoves = knight.GetValidMoves(_board);
        if (!validMoves.Contains(moveTarget))
        {
            GameLogger.Error("Knight", $"Invalid move target {moveTarget.ToChessNotation()} for Knight at {knight.BoardPosition.ToChessNotation()}");
            EndAction(knight, ActionType.Move);
            return;
        }

        // Execute the move
        var fromPos = knight.BoardPosition;
        var targetTile = _board.GetTile(moveTarget);
        if (targetTile.IsThreatened(knight.Team))
            knight.EnteredThreatZoneThisTurn = true;

        _board.MovePiece(knight, moveTarget);
        _board.RecalculateThreatZones();

        string team = knight.Team == Team.Player ? "W" : "B";
        GameLogger.Debug("Knight", $"[{team}] Knight {fromPos.ToChessNotation()} -> {moveTarget.ToChessNotation()}");

        // Notify UI of move
        PieceMoved?.Invoke(knight, fromPos, moveTarget);

        // Now validate and execute the attack
        var validAttacks = knight.GetAttackablePositions(_board);
        if (!validAttacks.Contains(attackTarget))
        {
            GameLogger.Error("Knight", $"Invalid attack target {attackTarget.ToChessNotation()} from {moveTarget.ToChessNotation()}");
            GameLogger.Error("Knight", $"Valid attacks: [{string.Join(", ", validAttacks.Select(a => a.ToChessNotation()))}]");
            EndAction(knight, ActionType.Move);
            return;
        }

        var defender = _board.GetPieceAt(attackTarget);
        if (defender == null)
        {
            GameLogger.Error("Knight", $"No piece at attack target {attackTarget.ToChessNotation()}");
            EndAction(knight, ActionType.Move);
            return;
        }

        // Execute the attack
        var result = _combatResolver.ResolveAttack(knight, defender, _board);

        if (result.DefenderDestroyed)
        {
            _board.RemovePiece(defender);
            _board.RecalculateThreatZones();

            if (defender.PieceType == PieceType.King)
            {
                EndMatch(knight.Team);
                return;
            }
        }

        GameLogger.Debug("Knight", $"Knight attacked {defender.PieceType} at {attackTarget.ToChessNotation()}");
        EndAction(knight, ActionType.MoveAndAttack);
    }
}
