using Godot;
using Exchange.Core;
using Exchange.Board;
using Exchange.Pieces;
using Exchange.Combat;
using Exchange.UI;

namespace Exchange.Controllers;

// Note: ESC key for pause menu is handled by GameManager/MainGameController.
// This controller only handles in-combat cancellation (selection, abilities).

/// <summary>
/// Main game controller that wires all systems together.
/// Handles input and coordinates between board, turn controller, and UI.
/// </summary>
public partial class GameController : Node2D
{
    /// <summary>Emitted when combat match ends. Parameter is winner team (0=Player, 1=Enemy).</summary>
    [Signal] public delegate void CombatMatchEndedEventHandler(int winnerTeam);

    private GameState _gameState = null!;
    private GameBoard _board = null!;
    private TurnController _turnController = null!;
    private CombatResolver _combatResolver = null!;
    private AIController _aiController = null!;
    private GameUI _ui = null!;

    // Input state
    private bool _isSelectingTarget = false;
    private ActionType _pendingAction = ActionType.None;

    // AI configuration (set before adding to tree)
    private int _aiDepth = 2;

    /// <summary>
    /// Set AI search depth before adding this node to the tree.
    /// </summary>
    public void SetAiDepth(int depth)
    {
        _aiDepth = depth;
        GameLogger.Info("GameController", $"AI depth configured: {depth}-ply");
    }

    public override void _Ready()
    {
        InitializeGame();
        SetupInput();
    }

    private void InitializeGame()
    {
        // Create game state
        _gameState = new GameState();
        AddChild(_gameState);

        // Create board
        _board = new GameBoard
        {
            Position = new Vector2(100, 50) // Offset for UI space
        };
        AddChild(_board);

        // Create combat resolver
        _combatResolver = new CombatResolver();
        _combatResolver.Initialize(_gameState);
        AddChild(_combatResolver);

        // Create AI controller with configured depth
        _aiController = new AIController();
        _aiController.LookaheadDepth = _aiDepth;
        GameLogger.Info("GameController", $"AI initialized with {_aiDepth}-ply search");
        AddChild(_aiController);

        // Create turn controller
        _turnController = new TurnController();
        _turnController.Initialize(_gameState, _board, _combatResolver, _aiController);
        AddChild(_turnController);

        // Initialize AI with references
        _aiController.Initialize(_board, _gameState, _turnController);

        // Create UI
        _ui = new GameUI();
        _ui.Initialize(_gameState, _turnController);
        AddChild(_ui);

        // Connect signals
        ConnectSignals();

        // Setup board with standard chess position
        _board.SetupStandardPosition(PieceFactory.Create);

        // Start the match
        _turnController.StartMatch();
    }

    private void ConnectSignals()
    {
        _turnController.TurnStarted += OnTurnStarted;
        _turnController.MatchEnded += OnMatchEnded;
        _turnController.PieceSelected += OnPieceSelected;
        _turnController.ValidMovesCalculated += OnValidMovesCalculated;
        _turnController.ValidAttacksCalculated += OnValidAttacksCalculated;
        _turnController.PieceMoved += OnPieceMoved;

        _combatResolver.CombatResolved += OnCombatResolved;
    }

    private void SetupInput()
    {
        // Input is handled in _Input
    }

    public override void _Input(InputEvent @event)
    {
        if (@event is InputEventMouseButton mouseButton && mouseButton.Pressed)
        {
            if (mouseButton.ButtonIndex == MouseButton.Left)
            {
                GameLogger.Debug("Input", $"Left click at {mouseButton.Position}");
                HandleLeftClick(mouseButton.Position);
            }
            else if (mouseButton.ButtonIndex == MouseButton.Right)
            {
                GameLogger.Debug("Input", "Right click - canceling");
                HandleRightClick();
            }
        }

        // Keyboard shortcuts (ESC handled by parent - see HandleCancelInput)
        if (@event is InputEventKey key && key.Pressed)
        {
            GameLogger.Debug("GameController", $"Key pressed: {key.Keycode}");
            switch (key.Keycode)
            {
                case Key.Space:
                    GameLogger.Debug("GameController", "Space - Knight end turn");
                    _turnController.KnightEndTurnWithoutAttack();
                    break;
                case Key.A:
                    if (_turnController.SelectedPiece != null)
                        _pendingAction = ActionType.Attack;
                    break;
                case Key.E:
                    GameLogger.Debug("GameController", "E - Use ability");
                    if (_turnController.SelectedPiece != null && _turnController.SelectedPiece.CanUseAbility)
                        _turnController.TryUseAbility();
                    break;
            }
        }
    }

    /// <summary>
    /// Handle cancel input (ESC key) - called by parent controller.
    /// Cancels abilities or piece selection in combat.
    /// </summary>
    public void HandleCancelInput()
    {
        GameLogger.Debug("GameController", "Cancel input received");

        if (_turnController.CurrentAbilityPhase != AbilityPhase.None)
        {
            GameLogger.Debug("GameController", "Canceling ability");
            _turnController.CancelAbility();
        }
        else
        {
            CancelSelection();
        }
    }

    /// <summary>
    /// Called externally (from debug menu) to force win the current match.
    /// </summary>
    public void ForceAutoWin()
    {
        if (_gameState.CurrentPhase == GamePhase.MatchEnd)
        {
            GameLogger.Warning("GameController", "Force auto-win called but match already ended");
            return;
        }

        // Kill the enemy king to trigger win
        if (_board.EnemyKing != null)
        {
            GameLogger.Info("GameController", $"[DEBUG] Force destroying enemy King at {_board.EnemyKing.BoardPosition.ToChessNotation()}");
            _board.EnemyKing.TakeDamage(9999);
            _board.RemovePiece(_board.EnemyKing);
            _turnController.ForceMatchEnd(Team.Player);
        }
        else
        {
            GameLogger.Error("GameController", "Force auto-win failed - no enemy King found!");
        }
    }

    private void HandleLeftClick(Vector2 screenPos)
    {
        GameLogger.Debug("Click", $"IsAwaitingPlayerInput={_turnController.IsAwaitingPlayerInput}, IsPlayerTurn={_turnController.IsPlayerTurn}");

        if (!_turnController.IsAwaitingPlayerInput)
        {
            GameLogger.Debug("Click", "Not awaiting player input, ignoring click");
            return;
        }

        // Convert screen position to board position
        var localPos = screenPos - _board.Position;
        var boardPos = ScreenToBoardPosition(localPos);

        GameLogger.Debug("Click", $"Board position: {boardPos.ToChessNotation()} ({boardPos})");

        if (!boardPos.IsOnBoard())
        {
            GameLogger.Debug("Click", "Position off board, ignoring");
            return;
        }

        var clickedPiece = _board.GetPieceAt(boardPos);
        GameLogger.Debug("Click", $"Piece at position: {(clickedPiece != null ? $"{clickedPiece.Team} {clickedPiece.PieceType}" : "none")}");
        GameLogger.Debug("Click", $"Currently selected: {(_turnController.SelectedPiece != null ? $"{_turnController.SelectedPiece.Team} {_turnController.SelectedPiece.PieceType}" : "none")}");
        GameLogger.Debug("Click", $"Ability phase: {_turnController.CurrentAbilityPhase}");

        // Handle multi-step ability phases
        var abilityPhase = _turnController.CurrentAbilityPhase;

        // Skirmish reposition: click empty tile to reposition
        if (abilityPhase == AbilityPhase.SkirmishSelectReposition && clickedPiece == null)
        {
            GameLogger.Debug("Click", "Attempting Skirmish reposition...");
            if (_turnController.TryReposition(boardPos))
            {
                GameLogger.Debug("Click", "Skirmish reposition successful!");
                return;
            }
            GameLogger.Debug("Click", "Skirmish reposition failed");
            return;
        }

        // Overextend move phase: click empty tile to move
        if (abilityPhase == AbilityPhase.OverextendSelectMove && clickedPiece == null)
        {
            GameLogger.Debug("Click", "Attempting Overextend move...");
            if (_turnController.TryMove(boardPos))
            {
                GameLogger.Debug("Click", "Overextend move successful!");
                return;
            }
            GameLogger.Debug("Click", "Overextend move failed");
            return;
        }

        // Overextend/Skirmish attack phase: click enemy to attack
        if ((abilityPhase == AbilityPhase.OverextendSelectAttack || abilityPhase == AbilityPhase.SkirmishSelectAttack)
            && clickedPiece != null && clickedPiece.Team == Team.Enemy)
        {
            GameLogger.Debug("Click", "Attempting ability attack...");
            if (_turnController.TryAttack(boardPos))
            {
                GameLogger.Debug("Click", "Ability attack successful!");
                return;
            }
            GameLogger.Debug("Click", "Ability attack failed");
            return;
        }

        // If we have a selected piece, try to act
        if (_turnController.SelectedPiece != null)
        {
            // Try attack first if clicking enemy
            if (clickedPiece != null && clickedPiece.Team == Team.Enemy)
            {
                GameLogger.Debug("Click", "Attempting attack on enemy...");
                if (_turnController.TryAttack(boardPos))
                {
                    GameLogger.Debug("Click", "Attack successful!");
                    return;
                }
                GameLogger.Debug("Click", "Attack failed (not in valid attacks)");
            }

            // Try move if clicking empty tile
            if (clickedPiece == null)
            {
                GameLogger.Debug("Click", "Attempting move to empty tile...");
                if (_turnController.TryMove(boardPos))
                {
                    GameLogger.Debug("Click", "Move successful!");
                    return;
                }
                GameLogger.Debug("Click", "Move failed (not in valid moves)");
            }
        }

        // Select/deselect piece
        if (clickedPiece != null && clickedPiece.Team == Team.Player)
        {
            GameLogger.Debug("Click", $"Selecting player piece: {clickedPiece.PieceType}");
            _turnController.SelectPiece(clickedPiece);
        }
        else
        {
            GameLogger.Debug("Click", "Deselecting (clicked empty or enemy without selection)");
            _turnController.SelectPiece(null);
        }
    }

    private void HandleRightClick()
    {
        CancelSelection();
    }

    private void CancelSelection()
    {
        _turnController.SelectPiece(null);
        _pendingAction = ActionType.None;
        _isSelectingTarget = false;
        ClearHighlights();
    }

    private Vector2I ScreenToBoardPosition(Vector2 localPos)
    {
        int x = (int)(localPos.X / Tile.TileSize);
        int y = 7 - (int)(localPos.Y / Tile.TileSize); // Flip Y
        return new Vector2I(x, y);
    }

    private void OnTurnStarted(Team team, int turnNumber)
    {
        _ui.UpdateTurnDisplay(team, turnNumber);
        ClearHighlights();
    }

    private void OnMatchEnded(Team winner)
    {
        _ui.ShowMatchResult(winner);
        GameLogger.Info("GameController", $"Match ended - Winner: {winner}");

        // Emit signal for MainGameController to handle
        EmitSignal(SignalName.CombatMatchEnded, (int)winner);
    }

    private void OnPieceSelected(BasePiece? piece)
    {
        ClearHighlights();
        _ui.UpdateSelectedPiece(piece);

        // Highlight the selected piece's tile
        if (piece != null)
        {
            var tile = _board.GetTile(piece.BoardPosition);
            tile.ShowHighlight(true, new Color(0.9f, 0.9f, 0.2f, 0.5f)); // Yellow for selected
        }
    }

    private void OnValidMovesCalculated(List<Vector2I> moves)
    {
        foreach (var pos in moves)
        {
            var tile = _board.GetTile(pos);
            tile.ShowHighlight(true, new Color(0.2f, 0.8f, 0.2f, 0.4f)); // Green for moves
        }
    }

    private void OnValidAttacksCalculated(List<Vector2I> attacks)
    {
        foreach (var pos in attacks)
        {
            var tile = _board.GetTile(pos);
            tile.ShowHighlight(true, new Color(0.9f, 0.3f, 0.3f, 0.5f)); // Red for attacks
        }

        // Log for debugging
        if (attacks.Count > 0)
            GameLogger.Debug("UI", $"Showing {attacks.Count} attack target(s)");
    }

    private void OnPieceMoved(BasePiece piece, Vector2I fromPos, Vector2I toPos)
    {
        _ui.ShowMove(piece, fromPos, toPos);
    }

    private void OnCombatResolved(CombatResolver.CombatResult result)
    {
        // Show attack line effect from attacker to defender
        _board.ShowAttackEffect(
            result.Attacker.BoardPosition,
            result.Defender.BoardPosition,
            result.Attacker.Team
        );

        _ui.ShowCombatResult(result);
    }

    private void ClearHighlights()
    {
        for (int x = 0; x < 8; x++)
        {
            for (int y = 0; y < 8; y++)
            {
                var tile = _board.GetTile(new Vector2I(x, y));
                tile.ShowHighlight(false);
            }
        }
        _board.RecalculateThreatZones();
    }
}
