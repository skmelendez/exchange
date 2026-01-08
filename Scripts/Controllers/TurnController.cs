using Godot;
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
    public event Action<BasePiece?>? PieceSelected;
    public event Action<List<Vector2I>>? ValidMovesCalculated;
    public event Action<List<Vector2I>>? ValidAttacksCalculated;

    private GameState _gameState = null!;
    private GameBoard _board = null!;
    private CombatResolver _combatResolver = null!;
    private AIController? _aiController;

    // Current turn state
    private BasePiece? _selectedPiece;
    private List<Vector2I> _validMoves = new();
    private List<Vector2I> _validAttacks = new();
    private bool _awaitingAction = false;

    public BasePiece? SelectedPiece => _selectedPiece;
    public bool IsPlayerTurn => _gameState.CurrentPhase == GamePhase.PlayerTurn;
    public bool IsAwaitingPlayerInput => IsPlayerTurn && _awaitingAction;

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
        GD.Print($"[TURN] === StartTurn called === Phase={_gameState.CurrentPhase}, Team={_gameState.CurrentTeam}");

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

        _awaitingAction = true;
        GD.Print($"[TURN] _awaitingAction set to TRUE, emitting TurnStarted");
        TurnStarted?.Invoke(_gameState.CurrentTeam, _gameState.TurnNumber);

        GD.Print($"[TURN] {_gameState.CurrentTeam}'s turn #{_gameState.TurnNumber}");

        // If enemy turn, trigger AI
        if (_gameState.CurrentPhase == GamePhase.EnemyTurn && _aiController != null)
        {
            GD.Print("[TURN] Enemy turn - triggering AI...");
            _aiController.ExecuteTurn();
        }
        else
        {
            GD.Print("[TURN] Player turn - awaiting input");
        }
    }

    public void SelectPiece(BasePiece? piece)
    {
        if (!IsAwaitingPlayerInput) return;
        if (piece != null && piece.Team != Team.Player) return;

        _selectedPiece = piece;
        _validMoves.Clear();
        _validAttacks.Clear();

        if (piece != null)
        {
            _validMoves = piece.GetValidMoves(_board);
            _validAttacks = piece.GetAttackablePositions(_board);

            // Knight special: if already moved, can still attack
            if (piece is KnightPiece knight && knight.HasMovedThisTurn)
            {
                _validMoves.Clear(); // Can't move again
            }
        }

        PieceSelected?.Invoke(piece);
        ValidMovesCalculated?.Invoke(_validMoves);
        ValidAttacksCalculated?.Invoke(_validAttacks);
    }

    public bool TryMove(Vector2I targetPosition)
    {
        if (!IsAwaitingPlayerInput || _selectedPiece == null) return false;
        if (!_validMoves.Contains(targetPosition)) return false;

        ExecuteMove(_selectedPiece, targetPosition);
        return true;
    }

    public bool TryAttack(Vector2I targetPosition)
    {
        if (!IsAwaitingPlayerInput || _selectedPiece == null) return false;
        if (!_validAttacks.Contains(targetPosition)) return false;

        var target = _board.GetPieceAt(targetPosition);
        if (target == null) return false;

        ExecuteAttack(_selectedPiece, target);
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
    /// Execute a move action
    /// </summary>
    public void ExecuteMove(BasePiece piece, Vector2I targetPosition)
    {
        // Check if entering threat zone
        var targetTile = _board.GetTile(targetPosition);
        if (targetTile.IsThreatened(piece.Team))
            piece.EnteredThreatZoneThisTurn = true;

        _board.MovePiece(piece, targetPosition);
        _board.RecalculateThreatZones();

        GD.Print($"[Action] {piece.PieceType} moves to {targetPosition.ToChessNotation()}");

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
                    GD.Print($"[Knight] AI Knight has {_validAttacks.Count} adjacent enemy(s) - AI will decide");
                    // Don't wait for input, just end the move action
                    // AI will need to handle Knight attack separately if we want that behavior
                    // For now, AI Knight just ends turn after moving (simplification)
                }
                else
                {
                    GD.Print($"[Knight] May now attack {_validAttacks.Count} adjacent enemy(s) or press SPACE to end turn");
                    // Re-emit piece selected to clear old highlights, then show new attacks
                    PieceSelected?.Invoke(piece);
                    ValidMovesCalculated?.Invoke(new List<Vector2I>());
                    ValidAttacksCalculated?.Invoke(_validAttacks);
                    return; // Don't end turn yet - waiting for player attack or Space
                }
            }
            else
            {
                GD.Print("[Knight] No adjacent enemies to attack, ending turn");
            }
        }

        EndAction(piece, ActionType.Move);
    }

    /// <summary>
    /// Execute an attack action
    /// </summary>
    public void ExecuteAttack(BasePiece attacker, BasePiece defender)
    {
        var result = _combatResolver.ResolveAttack(attacker, defender, _board);

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
        piece.StartAbilityCooldown();

        switch (piece.AbilityId)
        {
            case AbilityId.RoyalDecree:
                ExecuteRoyalDecree(piece);
                break;

            case AbilityId.Overextend:
                // This is a special case - needs move then attack selection
                // For now, we'll handle it in a simplified way
                GD.Print("[Ability] Overextend requires move+attack selection (TODO: full implementation)");
                break;

            case AbilityId.Interpose:
                ExecuteInterpose(piece);
                break;

            case AbilityId.Consecration:
                if (targetPosition.HasValue)
                    ExecuteConsecration(piece as BishopPiece, targetPosition.Value);
                break;

            case AbilityId.Skirmish:
                // Attack then reposition - needs target selection
                GD.Print("[Ability] Skirmish requires attack+reposition selection (TODO: full implementation)");
                break;

            case AbilityId.Advance:
                ExecuteAdvance(piece as PawnPiece);
                break;
        }

        EndAction(piece, ActionType.Ability);
    }

    private void ExecuteRoyalDecree(BasePiece king)
    {
        if (king.Team == Team.Player)
            _gameState.PlayerRoyalDecreeActive = true;
        else
            _gameState.EnemyRoyalDecreeActive = true;

        GD.Print($"[Ability] Royal Decree activated! All {king.Team} combat rolls +1 until next turn.");
    }

    private void ExecuteInterpose(BasePiece rook)
    {
        // Interpose is a persistent effect tracked separately
        // For now, log it - full implementation needs effect system
        GD.Print($"[Ability] Interpose activated! Damage to adjacent allies will be split.");
    }

    private void ExecuteConsecration(BishopPiece? bishop, Vector2I targetPos)
    {
        if (bishop == null) return;

        var target = _board.GetPieceAt(targetPos);
        if (target == null || target.Team != bishop.Team) return;

        int healed = _combatResolver.ResolveHealing(bishop, target);
        GD.Print($"[Ability] Consecration heals {target.PieceType} for {healed}!");
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

        GD.Print($"[Ability] Advance! Pawn moves to {target.Value.ToChessNotation()}");
    }

    private void EndAction(BasePiece piece, ActionType action)
    {
        piece.HasActedThisTurn = true;
        _awaitingAction = false;

        ActionCompleted?.Invoke(piece, action);
        GD.Print($"[Action] {piece.PieceType} completed {action}");

        EndTurn();
    }

    private void EndTurn()
    {
        GD.Print($"[TURN] === EndTurn called === Current phase: {_gameState.CurrentPhase}");

        // Check king threat status for next turn bonus
        _gameState.PlayerKingThreatened = _board.IsKingThreatened(Team.Player);
        _gameState.EnemyKingThreatened = _board.IsKingThreatened(Team.Enemy);

        if (_gameState.PlayerKingThreatened)
            GD.Print("[Status] Player King is threatened! Enemy gains +1 dice next attack.");
        if (_gameState.EnemyKingThreatened)
            GD.Print("[Status] Enemy King is threatened! Player gains +1 dice next attack.");

        TurnEnded?.Invoke(_gameState.CurrentTeam);

        // Switch turns
        var oldPhase = _gameState.CurrentPhase;
        _gameState.CurrentPhase = _gameState.CurrentPhase == GamePhase.PlayerTurn
            ? GamePhase.EnemyTurn
            : GamePhase.PlayerTurn;

        GD.Print($"[TURN] Phase switched: {oldPhase} -> {_gameState.CurrentPhase}");

        if (_gameState.CurrentPhase == GamePhase.PlayerTurn)
            _gameState.TurnNumber++;

        _selectedPiece = null;
        _validMoves.Clear();
        _validAttacks.Clear();

        GD.Print("[TURN] Calling StartTurn for next turn...");
        StartTurn();
    }

    private void EndMatch(Team winner)
    {
        _gameState.CurrentPhase = GamePhase.MatchEnd;
        _awaitingAction = false;

        GD.Print($"[Match] {winner} wins!");
        MatchEnded?.Invoke(winner);
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
}
