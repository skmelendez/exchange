using Godot;
using Exchange.Core;
using Exchange.Pieces;

namespace Exchange.Board;

/// <summary>
/// Represents a single tile on the 8x8 board.
/// Tracks occupancy and threat state.
/// </summary>
public partial class Tile : Node2D
{
    [Export] public Vector2I BoardPosition { get; set; }

    public BasePiece? OccupyingPiece { get; set; }
    public bool IsOccupied => OccupyingPiece != null;

    // Threat tracking - which teams threaten this tile
    public bool ThreatenedByPlayer { get; set; }
    public bool ThreatenedByEnemy { get; set; }

    public bool IsThreatened(Team forTeam) => forTeam == Team.Player ? ThreatenedByEnemy : ThreatenedByPlayer;

    // Visual components
    private ColorRect? _background;
    private ColorRect? _threatOverlay;

    private static readonly Color LightTile = new(0.93f, 0.93f, 0.82f);
    private static readonly Color DarkTile = new(0.46f, 0.59f, 0.34f);
    private static readonly Color ThreatColor = new(0.8f, 0.2f, 0.2f, 0.3f);
    private static readonly Color HighlightColor = new(0.2f, 0.6f, 0.9f, 0.4f);

    public const int TileSize = 64;

    public override void _Ready()
    {
        SetupVisuals();
    }

    private void SetupVisuals()
    {
        // Background tile (checkerboard pattern)
        _background = new ColorRect
        {
            Size = new Vector2(TileSize, TileSize),
            Color = (BoardPosition.X + BoardPosition.Y) % 2 == 0 ? LightTile : DarkTile
        };
        AddChild(_background);

        // Threat overlay (hidden by default)
        _threatOverlay = new ColorRect
        {
            Size = new Vector2(TileSize, TileSize),
            Color = ThreatColor,
            Visible = false
        };
        AddChild(_threatOverlay);
    }

    public void ShowThreat(bool show)
    {
        if (_threatOverlay != null)
            _threatOverlay.Visible = show;
    }

    public void ShowHighlight(bool show, Color? customColor = null)
    {
        if (_threatOverlay != null)
        {
            _threatOverlay.Color = customColor ?? HighlightColor;
            _threatOverlay.Visible = show;
        }
    }

    public void ResetThreatState()
    {
        ThreatenedByPlayer = false;
        ThreatenedByEnemy = false;
        ShowThreat(false);
    }

    public string GetChessNotation() => BoardPosition.ToChessNotation();
}
