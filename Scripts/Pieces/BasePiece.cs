using Godot;
using Exchange.Core;
using Exchange.Board;

namespace Exchange.Pieces;

/// <summary>
/// Base class for all chess pieces. Handles HP, movement patterns, and ability state.
/// Pieces are Node2D instances that live as children of GameBoard.
/// </summary>
public abstract partial class BasePiece : Node2D
{
    /// <summary>Emitted when piece HP changes.</summary>
    [Signal] public delegate void HealthChangedEventHandler(int newHealth, int maxHealth);

    /// <summary>Emitted when piece HP reaches zero.</summary>
    [Signal] public delegate void PieceDiedEventHandler(BasePiece piece);

    /// <summary>Emitted when piece uses its ability.</summary>
    [Signal] public delegate void AbilityUsedEventHandler(AbilityId ability);

    /// <summary>The type of chess piece (King, Queen, etc.).</summary>
    public PieceType PieceType { get; protected set; }

    /// <summary>Which team this piece belongs to (Player or Enemy).</summary>
    public Team Team { get; set; }

    /// <summary>Current position on the 8x8 board grid.</summary>
    public Vector2I BoardPosition { get; set; }

    /// <summary>Maximum hit points for this piece type.</summary>
    public int MaxHp { get; protected set; }

    /// <summary>Current hit points. Piece is destroyed at 0.</summary>
    public int CurrentHp { get; protected set; }

    /// <summary>Base damage dealt before dice roll modifier.</summary>
    public int BaseDamage { get; protected set; }

    /// <summary>The unique ability this piece type has.</summary>
    public AbilityId AbilityId { get; protected set; }

    /// <summary>Cooldown duration in turns (-1 = once per match).</summary>
    public int AbilityCooldownMax { get; protected set; }

    /// <summary>Remaining cooldown turns before ability is available.</summary>
    public int AbilityCooldownCurrent { get; set; }

    /// <summary>For once-per-match abilities: whether it has been used.</summary>
    public bool AbilityUsedThisMatch { get; set; }

    /// <summary>Whether this piece moved into a threat zone this turn (-1 dice penalty).</summary>
    public bool EnteredThreatZoneThisTurn { get; set; }

    /// <summary>Whether this piece has taken its action this turn.</summary>
    public bool HasActedThisTurn { get; set; }

    // Visual components
    protected ColorRect? _visual;
    protected Label? _label;
    protected Label? _hpLabel;

    protected static readonly Color PlayerColor = new(0.2f, 0.4f, 0.8f);
    protected static readonly Color EnemyColor = new(0.8f, 0.2f, 0.2f);

    public bool IsAlive => CurrentHp > 0;
    public bool CanUseAbility => AbilityCooldownCurrent == 0 &&
        (AbilityCooldownMax != -1 || !AbilityUsedThisMatch);

    protected BasePiece(PieceType type, Team team)
    {
        PieceType = type;
        Team = team;

        var stats = PieceData.Stats[type];
        MaxHp = stats.MaxHp;
        CurrentHp = stats.MaxHp;
        BaseDamage = stats.BaseDamage;
        AbilityId = stats.Ability;
        AbilityCooldownMax = stats.AbilityCooldown;
    }

    public override void _Ready()
    {
        SetupVisuals();
    }

    protected virtual void SetupVisuals()
    {
        // Placeholder visual - colored square
        _visual = new ColorRect
        {
            Size = new Vector2(Tile.TileSize - 8, Tile.TileSize - 8),
            Position = new Vector2(4, 4),
            Color = Team == Team.Player ? PlayerColor : EnemyColor
        };
        AddChild(_visual);

        // Piece letter
        _label = new Label
        {
            Text = PieceData.GetNotation(PieceType, Team).ToString(),
            Position = new Vector2(Tile.TileSize / 2 - 8, Tile.TileSize / 2 - 12)
        };
        _label.AddThemeColorOverride("font_color", Colors.White);
        AddChild(_label);

        // HP display
        _hpLabel = new Label
        {
            Text = CurrentHp.ToString(),
            Position = new Vector2(4, Tile.TileSize - 20)
        };
        _hpLabel.AddThemeColorOverride("font_color", Colors.Yellow);
        _hpLabel.AddThemeFontSizeOverride("font_size", 10);
        AddChild(_hpLabel);
    }

    public void TakeDamage(int amount)
    {
        CurrentHp = Math.Max(0, CurrentHp - amount);
        UpdateHpDisplay();
        EmitSignal(SignalName.HealthChanged, CurrentHp, MaxHp);

        if (CurrentHp <= 0)
            EmitSignal(SignalName.PieceDied, this);
    }

    public void Heal(int amount)
    {
        CurrentHp = Math.Min(MaxHp, CurrentHp + amount);
        UpdateHpDisplay();
        EmitSignal(SignalName.HealthChanged, CurrentHp, MaxHp);
    }

    protected void UpdateHpDisplay()
    {
        if (_hpLabel != null)
            _hpLabel.Text = CurrentHp.ToString();
    }

    public void TickCooldown()
    {
        if (AbilityCooldownCurrent > 0)
            AbilityCooldownCurrent--;
    }

    public void StartAbilityCooldown()
    {
        if (AbilityCooldownMax == -1)
            AbilityUsedThisMatch = true;
        else
            AbilityCooldownCurrent = AbilityCooldownMax;
    }

    public void ResetForNewMatch()
    {
        CurrentHp = MaxHp;
        AbilityCooldownCurrent = 0;
        AbilityUsedThisMatch = false;
        EnteredThreatZoneThisTurn = false;
        HasActedThisTurn = false;
        UpdateHpDisplay();
    }

    public virtual void ResetTurnState()
    {
        EnteredThreatZoneThisTurn = false;
        HasActedThisTurn = false;
    }

    /// <summary>
    /// Gets all valid movement positions for this piece
    /// </summary>
    public abstract List<Vector2I> GetValidMoves(GameBoard board);

    /// <summary>
    /// Gets all positions this piece can attack from current position
    /// </summary>
    public abstract List<Vector2I> GetAttackablePositions(GameBoard board);

    /// <summary>
    /// Gets all positions this piece threatens (for threat zone calculation)
    /// May differ from attackable positions for some pieces
    /// </summary>
    public virtual List<Vector2I> GetThreatenedPositions(GameBoard board) =>
        GetAttackablePositions(board);

    /// <summary>
    /// Helper to get positions in a direction until blocked
    /// </summary>
    protected List<Vector2I> GetPositionsInDirection(GameBoard board, Vector2I direction, int maxDistance = 8)
    {
        var positions = new List<Vector2I>();
        var current = BoardPosition + direction;

        for (int i = 0; i < maxDistance && current.IsOnBoard(); i++)
        {
            var piece = board.GetPieceAt(current);
            if (piece != null)
            {
                // Can attack enemy pieces, but blocked after
                if (piece.Team != Team)
                    positions.Add(current);
                break;
            }
            positions.Add(current);
            current += direction;
        }

        return positions;
    }

    /// <summary>
    /// Helper to get adjacent positions (optionally filtered)
    /// </summary>
    protected List<Vector2I> GetAdjacentPositions(GameBoard board, bool includeOccupiedByEnemy = true)
    {
        var positions = new List<Vector2I>();
        foreach (var dir in Vector2IExtensions.AllDirections)
        {
            var pos = BoardPosition + dir;
            if (!pos.IsOnBoard()) continue;

            var piece = board.GetPieceAt(pos);
            if (piece == null || (includeOccupiedByEnemy && piece.Team != Team))
                positions.Add(pos);
        }
        return positions;
    }
}
