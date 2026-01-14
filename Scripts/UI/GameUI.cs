using Godot;
using Exchange.Core;
using Exchange.Pieces;
using Exchange.Combat;
using Exchange.Controllers;

namespace Exchange.UI;

/// <summary>
/// Callback type for pawn promotion selection
/// </summary>
public delegate void PromotionSelectedHandler(PieceType selectedType);

/// <summary>
/// Main game UI displaying turn info, selected piece, combat log, etc.
/// </summary>
public partial class GameUI : CanvasLayer
{
    private GameState? _gameState;
    private TurnController? _turnController;

    // UI Elements
    private Label _turnLabel = null!;
    private Label _selectedPieceLabel = null!;
    private Label _combatLogLabel = null!;
    private VBoxContainer _pieceStatsPanel = null!;
    private Label _instructionsLabel = null!;

    private List<string> _combatLog = new();
    private const int MaxLogEntries = 8;

    // Promotion dialog
    private Control? _promotionDialog;
    private PromotionSelectedHandler? _promotionCallback;
    public bool IsPromotionDialogOpen => _promotionDialog != null && _promotionDialog.Visible;

    public void Initialize(GameState gameState, TurnController turnController)
    {
        _gameState = gameState;
        _turnController = turnController;
    }

    public override void _Ready()
    {
        BuildUI();
    }

    private void BuildUI()
    {
        // Main container on right side
        var rightPanel = new VBoxContainer
        {
            Position = new Vector2(650, 50),
            CustomMinimumSize = new Vector2(600, 600)
        };
        AddChild(rightPanel);

        // Turn indicator
        _turnLabel = new Label { Text = "Turn: Player #1" };
        _turnLabel.AddThemeColorOverride("font_color", Colors.White);
        _turnLabel.AddThemeFontSizeOverride("font_size", 24);
        rightPanel.AddChild(_turnLabel);

        rightPanel.AddChild(new HSeparator());

        // Selected piece info
        _selectedPieceLabel = new Label { Text = "No piece selected" };
        _selectedPieceLabel.AddThemeColorOverride("font_color", Colors.LightGray);
        _selectedPieceLabel.AddThemeFontSizeOverride("font_size", 16);
        rightPanel.AddChild(_selectedPieceLabel);

        // Piece stats panel
        _pieceStatsPanel = new VBoxContainer();
        rightPanel.AddChild(_pieceStatsPanel);

        rightPanel.AddChild(new HSeparator());

        // Instructions
        _instructionsLabel = new Label
        {
            Text = "Controls:\n" +
                   "Left Click: Select piece / Move / Attack\n" +
                   "Right Click: Cancel selection\n" +
                   "Space: Knight - end turn after move\n" +
                   "E: Use ability\n" +
                   "Esc: Cancel"
        };
        _instructionsLabel.AddThemeColorOverride("font_color", Colors.LightGray);
        _instructionsLabel.AddThemeFontSizeOverride("font_size", 12);
        rightPanel.AddChild(_instructionsLabel);

        rightPanel.AddChild(new HSeparator());

        // Combat log header
        var logHeader = new Label { Text = "Combat Log:" };
        logHeader.AddThemeColorOverride("font_color", Colors.Yellow);
        logHeader.AddThemeFontSizeOverride("font_size", 14);
        rightPanel.AddChild(logHeader);

        // Combat log
        _combatLogLabel = new Label
        {
            Text = "",
            AutowrapMode = TextServer.AutowrapMode.Word,
            CustomMinimumSize = new Vector2(300, 200)
        };
        _combatLogLabel.AddThemeColorOverride("font_color", Colors.White);
        _combatLogLabel.AddThemeFontSizeOverride("font_size", 11);
        rightPanel.AddChild(_combatLogLabel);

        // Room/Match indicator at top
        var topPanel = new HBoxContainer { Position = new Vector2(100, 10) };
        AddChild(topPanel);

        var roomLabel = new Label { Text = "Room 1 | Match 1 (Entry)" };
        roomLabel.AddThemeColorOverride("font_color", Colors.Gold);
        roomLabel.AddThemeFontSizeOverride("font_size", 16);
        topPanel.AddChild(roomLabel);
    }

    public void UpdateTurnDisplay(Team team, int turnNumber)
    {
        string teamText = team == Team.Player ? "YOUR TURN" : "ENEMY TURN";
        _turnLabel.Text = $"{teamText} - Turn #{turnNumber}";
        _turnLabel.AddThemeColorOverride("font_color",
            team == Team.Player ? Colors.LightGreen : Colors.OrangeRed);
    }

    public void UpdateSelectedPiece(BasePiece? piece)
    {
        // Clear stats panel
        foreach (var child in _pieceStatsPanel.GetChildren())
            child.QueueFree();

        if (piece == null)
        {
            _selectedPieceLabel.Text = "No piece selected";
            return;
        }

        _selectedPieceLabel.Text = $"Selected: {piece.Team} {piece.PieceType}";

        // Add stats
        AddStatLabel($"HP: {piece.CurrentHp}/{piece.MaxHp}");
        AddStatLabel($"Base Damage: {piece.BaseDamage}");
        AddStatLabel($"Ability: {piece.AbilityId}");

        // Build ability status string
        string cooldownText;
        if (piece.AbilityCooldownMax == -1)
        {
            cooldownText = piece.AbilityUsedThisMatch ? "Used" : "Ready (once)";
        }
        else if (piece.AbilityCooldownCurrent > 0)
        {
            cooldownText = $"Cooldown: {piece.AbilityCooldownCurrent}";
        }
        else
        {
            cooldownText = "Ready";
        }

        // Show remaining charges for limited-use abilities
        if (piece.AbilityUsesRemaining == -1)
        {
            // Unlimited ability
            AddStatLabel($"Ability Status: {cooldownText}");
        }
        else if (piece.AbilityUsesRemaining == 0)
        {
            // No charges left
            AddStatLabel($"Ability Status: No charges left", Colors.Gray);
        }
        else
        {
            // Has charges remaining
            var chargeColor = piece.AbilityUsesRemaining <= 1 ? Colors.OrangeRed : Colors.Cyan;
            AddStatLabel($"Ability Status: {cooldownText}");
            AddStatLabel($"Charges: {piece.AbilityUsesRemaining} remaining", chargeColor);
        }

        if (piece.EnteredThreatZoneThisTurn)
            AddStatLabel("!! Entered Threat Zone (-1 dice)", Colors.OrangeRed);
    }

    private void AddStatLabel(string text, Color? color = null)
    {
        var label = new Label { Text = text };
        label.AddThemeColorOverride("font_color", color ?? Colors.White);
        label.AddThemeFontSizeOverride("font_size", 12);
        _pieceStatsPanel.AddChild(label);
    }

    public void ShowCombatResult(CombatResolver.CombatResult result)
    {
        string atkTeam = result.Attacker.Team == Team.Player ? "W" : "B";
        string defTeam = result.Defender.Team == Team.Player ? "W" : "B";
        string atkPos = result.Attacker.BoardPosition.ToChessNotation();
        string defPos = result.Defender.BoardPosition.ToChessNotation();

        string entry = $"[{atkTeam}] {result.Attacker.PieceType} ({atkPos}) x [{defTeam}] {result.Defender.PieceType} ({defPos}): " +
                       $"{result.DamageDealt} dmg";
        if (result.DefenderDestroyed)
            entry += " [DESTROYED]";

        AddToLog(entry);
    }

    public void ShowMove(BasePiece piece, Vector2I fromPos, Vector2I toPos)
    {
        string team = piece.Team == Team.Player ? "W" : "B";
        string entry = $"[{team}] {piece.PieceType} {fromPos.ToChessNotation()} -> {toPos.ToChessNotation()}";
        AddToLog(entry);
    }

    public void ShowAbilityUse(BasePiece piece, string abilityName)
    {
        string team = piece.Team == Team.Player ? "W" : "B";
        string pos = piece.BoardPosition.ToChessNotation();
        string entry = $"[{team}] {piece.PieceType} ({pos}) uses {abilityName}";
        AddToLog(entry);
    }

    public void ShowMatchResult(Team winner)
    {
        string msg = winner == Team.Player ? "VICTORY!" : "DEFEAT!";
        AddToLog($"=== {msg} ===");

        // Show popup
        var popup = new Label
        {
            Text = msg,
            Position = new Vector2(300, 300)
        };
        popup.AddThemeColorOverride("font_color", winner == Team.Player ? Colors.Gold : Colors.Red);
        popup.AddThemeFontSizeOverride("font_size", 48);
        AddChild(popup);
    }

    public void ShowDrawResult(string reason)
    {
        AddToLog($"=== DRAW: {reason} ===");

        // Show popup
        var popup = new Label
        {
            Text = "DRAW!",
            Position = new Vector2(300, 300)
        };
        popup.AddThemeColorOverride("font_color", Colors.Gray);
        popup.AddThemeFontSizeOverride("font_size", 48);
        AddChild(popup);
    }

    private void AddToLog(string entry)
    {
        _combatLog.Insert(0, entry);
        if (_combatLog.Count > MaxLogEntries)
            _combatLog.RemoveAt(_combatLog.Count - 1);

        _combatLogLabel.Text = string.Join("\n", _combatLog);
    }

    #region Pawn Promotion Dialog

    /// <summary>
    /// Shows the pawn promotion dialog and calls the callback when a piece is selected.
    /// </summary>
    public void ShowPromotionDialog(Vector2I pawnPosition, PromotionSelectedHandler onSelected)
    {
        _promotionCallback = onSelected;

        // Create semi-transparent background overlay
        var overlay = new ColorRect
        {
            Color = new Color(0, 0, 0, 0.6f),
            Size = GetViewport().GetVisibleRect().Size,
            Position = Vector2.Zero,
            ZIndex = 90
        };

        // Create dialog container
        _promotionDialog = new Control { ZIndex = 100 };
        _promotionDialog.AddChild(overlay);

        // Dialog panel
        var panel = new PanelContainer
        {
            Position = new Vector2(250, 200)
        };
        _promotionDialog.AddChild(panel);

        var vbox = new VBoxContainer();
        panel.AddChild(vbox);

        // Title
        var title = new Label
        {
            Text = "Pawn Promotion!",
            HorizontalAlignment = HorizontalAlignment.Center
        };
        title.AddThemeColorOverride("font_color", Colors.Gold);
        title.AddThemeFontSizeOverride("font_size", 24);
        vbox.AddChild(title);

        var subtitle = new Label
        {
            Text = $"Choose a piece for your pawn at {pawnPosition.ToChessNotation()}:",
            HorizontalAlignment = HorizontalAlignment.Center
        };
        subtitle.AddThemeColorOverride("font_color", Colors.White);
        subtitle.AddThemeFontSizeOverride("font_size", 14);
        vbox.AddChild(subtitle);

        vbox.AddChild(new HSeparator());

        // Piece selection buttons
        var buttonContainer = new HBoxContainer();
        buttonContainer.AddThemeConstantOverride("separation", 10);
        vbox.AddChild(buttonContainer);

        CreatePromotionButton(buttonContainer, PieceType.Queen, "Q", "Queen (15 HP, 3 DMG)");
        CreatePromotionButton(buttonContainer, PieceType.Rook, "R", "Rook (12 HP, 2 DMG)");
        CreatePromotionButton(buttonContainer, PieceType.Bishop, "B", "Bishop (8 HP, 2 DMG)");
        CreatePromotionButton(buttonContainer, PieceType.Knight, "N", "Knight (10 HP, 2 DMG)");

        AddChild(_promotionDialog);

        AddToLog($"Pawn at {pawnPosition.ToChessNotation()} ready for promotion!");
    }

    private void CreatePromotionButton(HBoxContainer container, PieceType pieceType, string letter, string tooltip)
    {
        var button = new Button
        {
            CustomMinimumSize = new Vector2(80, 80),
            TooltipText = tooltip
        };

        var vbox = new VBoxContainer
        {
            AnchorsPreset = (int)Control.LayoutPreset.FullRect
        };
        button.AddChild(vbox);

        var letterLabel = new Label
        {
            Text = letter,
            HorizontalAlignment = HorizontalAlignment.Center
        };
        letterLabel.AddThemeFontSizeOverride("font_size", 32);
        letterLabel.AddThemeColorOverride("font_color", Colors.White);
        vbox.AddChild(letterLabel);

        var nameLabel = new Label
        {
            Text = pieceType.ToString(),
            HorizontalAlignment = HorizontalAlignment.Center
        };
        nameLabel.AddThemeFontSizeOverride("font_size", 12);
        nameLabel.AddThemeColorOverride("font_color", Colors.LightGray);
        vbox.AddChild(nameLabel);

        button.Pressed += () => OnPromotionSelected(pieceType);
        container.AddChild(button);
    }

    private void OnPromotionSelected(PieceType selectedType)
    {
        GameLogger.Info("UI", $"Promotion selected: {selectedType}");
        AddToLog($"Pawn promoted to {selectedType}!");

        // Hide and cleanup dialog
        if (_promotionDialog != null)
        {
            _promotionDialog.QueueFree();
            _promotionDialog = null;
        }

        // Invoke callback
        _promotionCallback?.Invoke(selectedType);
        _promotionCallback = null;
    }

    /// <summary>
    /// Force close the promotion dialog (e.g., on match end)
    /// </summary>
    public void ClosePromotionDialog()
    {
        if (_promotionDialog != null)
        {
            _promotionDialog.QueueFree();
            _promotionDialog = null;
        }
        _promotionCallback = null;
    }

    #endregion
}
