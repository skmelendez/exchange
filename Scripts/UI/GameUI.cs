using Godot;
using Exchange.Core;
using Exchange.Pieces;
using Exchange.Combat;
using Exchange.Controllers;

namespace Exchange.UI;

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

        string cooldownText = piece.AbilityCooldownMax == -1
            ? (piece.AbilityUsedThisMatch ? "Used" : "Ready (once)")
            : (piece.AbilityCooldownCurrent > 0 ? $"Cooldown: {piece.AbilityCooldownCurrent}" : "Ready");
        AddStatLabel($"Ability Status: {cooldownText}");

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

    private void AddToLog(string entry)
    {
        _combatLog.Insert(0, entry);
        if (_combatLog.Count > MaxLogEntries)
            _combatLog.RemoveAt(_combatLog.Count - 1);

        _combatLogLabel.Text = string.Join("\n", _combatLog);
    }
}
