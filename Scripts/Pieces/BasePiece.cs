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

    /// <summary>Maximum uses for this ability (0 = unlimited).</summary>
    public int AbilityMaxUses { get; protected set; }

    /// <summary>Remaining uses for this ability (-1 = unlimited).</summary>
    public int AbilityUsesRemaining { get; set; }

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
    protected static readonly Color DamageFlashColor = new(1f, 0.3f, 0.3f);

    private const float HitAnimationDuration = 0.15f;
    private const float ShakeAmount = 4f;

    public bool IsAlive => CurrentHp > 0;

    /// <summary>
    /// Whether this piece can currently use its ability.
    /// Checks cooldown and remaining uses.
    /// </summary>
    public bool CanUseAbility =>
        AbilityCooldownCurrent <= 0 &&
        (AbilityUsesRemaining == -1 || AbilityUsesRemaining > 0);

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
        AbilityMaxUses = stats.AbilityMaxUses;

        // Initialize ability uses remaining
        // 0 means unlimited (-1 internally), >0 means limited uses
        if (AbilityMaxUses == 0)
        {
            AbilityUsesRemaining = -1; // Unlimited
        }
        else
        {
            AbilityUsesRemaining = AbilityMaxUses;

            // Asymmetric balancing: Black Rook gets +1 Interpose to compensate for first-mover disadvantage
            if (type == PieceType.Rook && team == Team.Enemy)
            {
                AbilityUsesRemaining += 1;
            }
        }
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

        // Play hit animation
        PlayHitAnimation(amount);

        if (CurrentHp <= 0)
            EmitSignal(SignalName.PieceDied, this);
    }

    /// <summary>
    /// Sets HP directly without animation (used for promotion, initialization)
    /// </summary>
    public void SetHpDirect(int hp)
    {
        CurrentHp = Math.Clamp(hp, 0, MaxHp);
        UpdateHpDisplay();
    }

    /// <summary>
    /// Plays visual feedback when piece takes damage: flash + shake + floating damage number
    /// </summary>
    private void PlayHitAnimation(int damageAmount)
    {
        if (_visual == null) return;

        var originalColor = Team == Team.Player ? PlayerColor : EnemyColor;
        var originalPos = _visual.Position;

        // Create damage number
        var damageLabel = new Label
        {
            Text = $"-{damageAmount}",
            Position = new Vector2(Tile.TileSize / 2 - 10, -5),
            ZIndex = 100
        };
        damageLabel.AddThemeColorOverride("font_color", Colors.Red);
        damageLabel.AddThemeFontSizeOverride("font_size", 16);
        AddChild(damageLabel);

        // Animate damage number floating up and fading
        var damageTween = CreateTween();
        damageTween.SetParallel(true);
        damageTween.TweenProperty(damageLabel, "position", damageLabel.Position + new Vector2(0, -30), 0.6f)
            .SetTrans(Tween.TransitionType.Quad)
            .SetEase(Tween.EaseType.Out);
        damageTween.TweenProperty(damageLabel, "modulate:a", 0f, 0.6f)
            .SetDelay(0.3f);
        damageTween.SetParallel(false);
        damageTween.TweenCallback(Callable.From(() => damageLabel.QueueFree()));

        // Flash and shake the piece
        var hitTween = CreateTween();

        // Flash to damage color
        hitTween.TweenProperty(_visual, "color", DamageFlashColor, HitAnimationDuration / 2)
            .SetTrans(Tween.TransitionType.Quad);

        // Shake effect (quick back and forth)
        hitTween.TweenProperty(_visual, "position", originalPos + new Vector2(ShakeAmount, 0), HitAnimationDuration / 4)
            .SetTrans(Tween.TransitionType.Sine);
        hitTween.TweenProperty(_visual, "position", originalPos + new Vector2(-ShakeAmount, 0), HitAnimationDuration / 4)
            .SetTrans(Tween.TransitionType.Sine);
        hitTween.TweenProperty(_visual, "position", originalPos, HitAnimationDuration / 4)
            .SetTrans(Tween.TransitionType.Sine);

        // Return to original color
        hitTween.TweenProperty(_visual, "color", originalColor, HitAnimationDuration / 2)
            .SetTrans(Tween.TransitionType.Quad);
    }

    public void Heal(int amount)
    {
        int actualHeal = Math.Min(amount, MaxHp - CurrentHp);
        CurrentHp = Math.Min(MaxHp, CurrentHp + amount);
        UpdateHpDisplay();
        EmitSignal(SignalName.HealthChanged, CurrentHp, MaxHp);

        // Play heal animation
        if (actualHeal > 0)
            PlayHealAnimation(actualHeal);
    }

    /// <summary>
    /// Plays visual feedback when piece is healed: green flash + floating heal number
    /// </summary>
    private void PlayHealAnimation(int healAmount)
    {
        if (_visual == null) return;

        var originalColor = Team == Team.Player ? PlayerColor : EnemyColor;

        // Create heal number
        var healLabel = new Label
        {
            Text = $"+{healAmount}",
            Position = new Vector2(Tile.TileSize / 2 - 10, -5),
            ZIndex = 100
        };
        healLabel.AddThemeColorOverride("font_color", Colors.LimeGreen);
        healLabel.AddThemeFontSizeOverride("font_size", 16);
        AddChild(healLabel);

        // Animate heal number floating up and fading
        var healTween = CreateTween();
        healTween.SetParallel(true);
        healTween.TweenProperty(healLabel, "position", healLabel.Position + new Vector2(0, -30), 0.6f)
            .SetTrans(Tween.TransitionType.Quad)
            .SetEase(Tween.EaseType.Out);
        healTween.TweenProperty(healLabel, "modulate:a", 0f, 0.6f)
            .SetDelay(0.3f);
        healTween.SetParallel(false);
        healTween.TweenCallback(Callable.From(() => healLabel.QueueFree()));

        // Flash green
        var flashTween = CreateTween();
        flashTween.TweenProperty(_visual, "color", Colors.LimeGreen, HitAnimationDuration / 2)
            .SetTrans(Tween.TransitionType.Quad);
        flashTween.TweenProperty(_visual, "color", originalColor, HitAnimationDuration / 2)
            .SetTrans(Tween.TransitionType.Quad);
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

        // Decrement uses remaining for limited-use abilities
        if (AbilityUsesRemaining > 0)
            AbilityUsesRemaining--;
    }

    public void ResetForNewMatch()
    {
        CurrentHp = MaxHp;
        AbilityCooldownCurrent = 0;
        AbilityUsedThisMatch = false;
        EnteredThreatZoneThisTurn = false;
        HasActedThisTurn = false;

        // Reset ability uses
        if (AbilityMaxUses == 0)
        {
            AbilityUsesRemaining = -1; // Unlimited
        }
        else
        {
            AbilityUsesRemaining = AbilityMaxUses;
            // Asymmetric balancing: Black Rook gets +1 Interpose
            if (PieceType == PieceType.Rook && Team == Team.Enemy)
            {
                AbilityUsesRemaining += 1;
            }
        }

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
