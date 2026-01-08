using Godot;
using Exchange.Core;
using Exchange.Board;
using Exchange.Pieces;
using Exchange.Combat;
using Exchange.UI;

namespace Exchange.Controllers;

/// <summary>
/// Main game controller that wires all systems together.
/// Handles input and coordinates between board, turn controller, and UI.
/// </summary>
public partial class GameController : Node2D
{
    private GameState _gameState = null!;
    private GameBoard _board = null!;
    private TurnController _turnController = null!;
    private CombatResolver _combatResolver = null!;
    private AIController _aiController = null!;
    private GameUI _ui = null!;

    // Input state
    private bool _isSelectingTarget = false;
    private ActionType _pendingAction = ActionType.None;

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

        // Create AI controller
        _aiController = new AIController();
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
                GD.Print($"[INPUT] Left click at {mouseButton.Position}");
                HandleLeftClick(mouseButton.Position);
            }
            else if (mouseButton.ButtonIndex == MouseButton.Right)
            {
                GD.Print("[INPUT] Right click - canceling");
                HandleRightClick();
            }
        }

        // Keyboard shortcuts
        if (@event is InputEventKey key && key.Pressed)
        {
            GD.Print($"[INPUT] Key pressed: {key.Keycode}");
            switch (key.Keycode)
            {
                case Key.Escape:
                    CancelSelection();
                    break;
                case Key.Space:
                    GD.Print("[INPUT] Space - Knight end turn");
                    _turnController.KnightEndTurnWithoutAttack();
                    break;
                case Key.A:
                    if (_turnController.SelectedPiece != null)
                        _pendingAction = ActionType.Attack;
                    break;
                case Key.E:
                    GD.Print("[INPUT] E - Use ability");
                    if (_turnController.SelectedPiece != null && _turnController.SelectedPiece.CanUseAbility)
                        _turnController.TryUseAbility();
                    break;
            }
        }
    }

    private void HandleLeftClick(Vector2 screenPos)
    {
        GD.Print($"[CLICK] IsAwaitingPlayerInput={_turnController.IsAwaitingPlayerInput}, IsPlayerTurn={_turnController.IsPlayerTurn}");

        if (!_turnController.IsAwaitingPlayerInput)
        {
            GD.Print("[CLICK] Not awaiting player input, ignoring click");
            return;
        }

        // Convert screen position to board position
        var localPos = screenPos - _board.Position;
        var boardPos = ScreenToBoardPosition(localPos);

        GD.Print($"[CLICK] Board position: {boardPos.ToChessNotation()} ({boardPos})");

        if (!boardPos.IsOnBoard())
        {
            GD.Print("[CLICK] Position off board, ignoring");
            return;
        }

        var clickedPiece = _board.GetPieceAt(boardPos);
        GD.Print($"[CLICK] Piece at position: {(clickedPiece != null ? $"{clickedPiece.Team} {clickedPiece.PieceType}" : "none")}");
        GD.Print($"[CLICK] Currently selected: {(_turnController.SelectedPiece != null ? $"{_turnController.SelectedPiece.Team} {_turnController.SelectedPiece.PieceType}" : "none")}");

        // If we have a selected piece, try to act
        if (_turnController.SelectedPiece != null)
        {
            // Try attack first if clicking enemy
            if (clickedPiece != null && clickedPiece.Team == Team.Enemy)
            {
                GD.Print("[CLICK] Attempting attack on enemy...");
                if (_turnController.TryAttack(boardPos))
                {
                    GD.Print("[CLICK] Attack successful!");
                    return;
                }
                GD.Print("[CLICK] Attack failed (not in valid attacks)");
            }

            // Try move if clicking empty tile
            if (clickedPiece == null)
            {
                GD.Print("[CLICK] Attempting move to empty tile...");
                if (_turnController.TryMove(boardPos))
                {
                    GD.Print("[CLICK] Move successful!");
                    return;
                }
                GD.Print("[CLICK] Move failed (not in valid moves)");
            }
        }

        // Select/deselect piece
        if (clickedPiece != null && clickedPiece.Team == Team.Player)
        {
            GD.Print($"[CLICK] Selecting player piece: {clickedPiece.PieceType}");
            _turnController.SelectPiece(clickedPiece);
        }
        else
        {
            GD.Print("[CLICK] Deselecting (clicked empty or enemy without selection)");
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
            GD.Print($"[UI] Showing {attacks.Count} attack target(s)");
    }

    private void OnCombatResolved(CombatResolver.CombatResult result)
    {
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
