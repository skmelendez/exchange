using Godot;
using Exchange.Core;
using Exchange.Pieces;

namespace Exchange.Board;

/// <summary>
/// Manages the 8x8 game board, piece placement, and threat zone calculation.
/// </summary>
public partial class GameBoard : Node2D
{
    [Signal] public delegate void PieceSelectedEventHandler(BasePiece piece);
    [Signal] public delegate void TileSelectedEventHandler(Vector2I position);
    [Signal] public delegate void PieceDestroyedEventHandler(BasePiece piece);
    [Signal] public delegate void PieceMoveAnimationCompleteEventHandler();
    [Signal] public delegate void AttackAnimationCompleteEventHandler();

    private const float MoveAnimationDuration = 0.2f; // seconds
    private const float AttackAnimationDuration = 0.35f; // Total time for attack effect

    private Tile[,] _tiles = new Tile[8, 8];
    private List<BasePiece> _playerPieces = new();
    private List<BasePiece> _enemyPieces = new();
    private bool _isAnimating = false;
    private bool _isAttackAnimating = false;

    public IReadOnlyList<BasePiece> PlayerPieces => _playerPieces;
    public IReadOnlyList<BasePiece> EnemyPieces => _enemyPieces;

    public BasePiece? PlayerKing { get; private set; }
    public BasePiece? EnemyKing { get; private set; }

    public override void _Ready()
    {
        CreateBoard();
    }

    private void CreateBoard()
    {
        for (int x = 0; x < 8; x++)
        {
            for (int y = 0; y < 8; y++)
            {
                var tile = new Tile
                {
                    BoardPosition = new Vector2I(x, y),
                    Position = new Vector2(x * Tile.TileSize, (7 - y) * Tile.TileSize) // Flip Y for display
                };
                _tiles[x, y] = tile;
                AddChild(tile);
            }
        }
    }

    public Tile GetTile(Vector2I pos) => _tiles[pos.X, pos.Y];
    public Tile? GetTileSafe(Vector2I pos) => pos.IsOnBoard() ? _tiles[pos.X, pos.Y] : null;

    public BasePiece? GetPieceAt(Vector2I pos)
    {
        if (!pos.IsOnBoard()) return null;
        return _tiles[pos.X, pos.Y].OccupyingPiece;
    }

    public bool IsOccupied(Vector2I pos) => GetPieceAt(pos) != null;
    public bool IsOccupiedByTeam(Vector2I pos, Team team) => GetPieceAt(pos)?.Team == team;

    /// <summary>
    /// Places a piece on the board at the specified position
    /// </summary>
    public void PlacePiece(BasePiece piece, Vector2I position)
    {
        if (!position.IsOnBoard())
        {
            GameLogger.Error("Board", $"Invalid board position: {position}");
            return;
        }

        var tile = GetTile(position);
        if (tile.IsOccupied)
        {
            GameLogger.Error("Board", $"Tile {position.ToChessNotation()} already occupied!");
            return;
        }

        piece.BoardPosition = position;
        piece.Position = new Vector2(position.X * Tile.TileSize, (7 - position.Y) * Tile.TileSize);
        tile.OccupyingPiece = piece;

        if (piece.Team == Team.Player)
        {
            _playerPieces.Add(piece);
            if (piece.PieceType == PieceType.King)
                PlayerKing = piece;
        }
        else
        {
            _enemyPieces.Add(piece);
            if (piece.PieceType == PieceType.King)
                EnemyKing = piece;
        }

        AddChild(piece);
    }

    /// <summary>
    /// Moves a piece to a new position with animation
    /// </summary>
    public bool MovePiece(BasePiece piece, Vector2I newPosition, bool animate = true)
    {
        if (!newPosition.IsOnBoard()) return false;

        var targetTile = GetTile(newPosition);
        if (targetTile.IsOccupied) return false;

        // Clear old tile
        var oldTile = GetTile(piece.BoardPosition);
        oldTile.OccupyingPiece = null;

        // Update logical position immediately
        piece.BoardPosition = newPosition;

        // Calculate target visual position
        var targetVisualPos = new Vector2(newPosition.X * Tile.TileSize, (7 - newPosition.Y) * Tile.TileSize);

        if (animate && MoveAnimationDuration > 0)
        {
            // Animate the piece movement
            _isAnimating = true;
            var tween = CreateTween();
            tween.TweenProperty(piece, "position", targetVisualPos, MoveAnimationDuration)
                 .SetTrans(Tween.TransitionType.Quad)
                 .SetEase(Tween.EaseType.Out);
            tween.TweenCallback(Callable.From(() =>
            {
                _isAnimating = false;
                EmitSignal(SignalName.PieceMoveAnimationComplete);
            }));
        }
        else
        {
            // Instant move (no animation)
            piece.Position = targetVisualPos;
        }

        // Set new tile
        targetTile.OccupyingPiece = piece;

        return true;
    }

    /// <summary>
    /// Whether a piece movement animation is currently playing
    /// </summary>
    public bool IsAnimating => _isAnimating;

    /// <summary>
    /// Whether an attack animation is currently playing
    /// </summary>
    public bool IsAttackAnimating => _isAttackAnimating;

    /// <summary>
    /// Shows a visual attack effect from attacker to defender (strike line)
    /// </summary>
    public void ShowAttackEffect(Vector2I attackerPos, Vector2I defenderPos, Team attackerTeam)
    {
        _isAttackAnimating = true;

        // Calculate pixel positions (center of tiles)
        float halfTile = Tile.TileSize / 2f;
        var startPos = new Vector2(attackerPos.X * Tile.TileSize + halfTile, (7 - attackerPos.Y) * Tile.TileSize + halfTile);
        var endPos = new Vector2(defenderPos.X * Tile.TileSize + halfTile, (7 - defenderPos.Y) * Tile.TileSize + halfTile);

        // Create a Line2D for the attack effect
        var attackLine = new Line2D
        {
            Width = 4f,
            DefaultColor = attackerTeam == Team.Player ? new Color(0.4f, 0.6f, 1f, 0.9f) : new Color(1f, 0.4f, 0.3f, 0.9f),
            ZIndex = 50,
            BeginCapMode = Line2D.LineCapMode.Round,
            EndCapMode = Line2D.LineCapMode.Round
        };
        attackLine.AddPoint(startPos);
        attackLine.AddPoint(startPos); // Start with zero length, animate to endPos
        AddChild(attackLine);

        // Animate the line extending to target, then fading
        var lineTween = CreateTween();

        // Extend line to target (quick strike)
        lineTween.TweenMethod(
            Callable.From<float>((t) => {
                if (IsInstanceValid(attackLine) && attackLine.GetPointCount() >= 2)
                    attackLine.SetPointPosition(1, startPos.Lerp(endPos, t));
            }),
            0f, 1f, 0.1f
        ).SetTrans(Tween.TransitionType.Quad).SetEase(Tween.EaseType.Out);

        // Brief hold
        lineTween.TweenInterval(0.05f);

        // Fade out
        lineTween.TweenProperty(attackLine, "modulate:a", 0f, 0.15f);

        // Cleanup and signal completion
        lineTween.TweenCallback(Callable.From(() =>
        {
            attackLine.QueueFree();
            _isAttackAnimating = false;
            EmitSignal(SignalName.AttackAnimationComplete);
        }));
    }

    /// <summary>
    /// Promotes a pawn to a new piece type at the same position.
    /// Returns the newly created piece.
    /// </summary>
    public BasePiece PromotePawn(BasePiece pawn, PieceType promotionType, Func<PieceType, Team, BasePiece> pieceFactory)
    {
        if (pawn.PieceType != PieceType.Pawn)
        {
            GameLogger.Error("Board", $"Cannot promote non-pawn piece: {pawn.PieceType}");
            return pawn;
        }

        if (promotionType == PieceType.King || promotionType == PieceType.Pawn)
        {
            GameLogger.Error("Board", $"Cannot promote pawn to {promotionType}");
            return pawn;
        }

        var position = pawn.BoardPosition;
        var team = pawn.Team;

        // Calculate HP ratio to preserve relative health
        float hpRatio = (float)pawn.CurrentHp / pawn.MaxHp;

        // Remove the pawn from the board (don't use RemovePiece - it queues free)
        var tile = GetTile(position);
        tile.OccupyingPiece = null;

        if (team == Team.Player)
            _playerPieces.Remove(pawn);
        else
            _enemyPieces.Remove(pawn);

        pawn.QueueFree();

        // Create the promoted piece
        var newPiece = pieceFactory(promotionType, team);

        // Set HP proportionally (promoted piece keeps relative health)
        int newHp = Math.Max(1, (int)(newPiece.MaxHp * hpRatio));

        // Place the new piece
        newPiece.BoardPosition = position;
        newPiece.Position = new Vector2(position.X * Tile.TileSize, (7 - position.Y) * Tile.TileSize);
        tile.OccupyingPiece = newPiece;

        if (team == Team.Player)
            _playerPieces.Add(newPiece);
        else
            _enemyPieces.Add(newPiece);

        AddChild(newPiece);

        // Set HP proportionally (deferred to ensure _Ready has run)
        CallDeferred(nameof(SetPromotedPieceHp), newPiece, newHp);

        GameLogger.Info("Board", $"Pawn promoted to {promotionType} at {position.ToChessNotation()} with {newHp}/{newPiece.MaxHp} HP");

        return newPiece;
    }

    /// <summary>
    /// Helper for deferred HP setting after promotion
    /// </summary>
    private void SetPromotedPieceHp(BasePiece piece, int hp)
    {
        piece.SetHpDirect(hp);
    }

    /// <summary>
    /// Removes a piece from the board (destroyed)
    /// </summary>
    public void RemovePiece(BasePiece piece)
    {
        var tile = GetTile(piece.BoardPosition);
        tile.OccupyingPiece = null;

        if (piece.Team == Team.Player)
            _playerPieces.Remove(piece);
        else
            _enemyPieces.Remove(piece);

        EmitSignal(SignalName.PieceDestroyed, piece);
        piece.QueueFree();
    }

    /// <summary>
    /// Recalculates all threat zones on the board
    /// </summary>
    public void RecalculateThreatZones()
    {
        // Clear all threats
        for (int x = 0; x < 8; x++)
            for (int y = 0; y < 8; y++)
                _tiles[x, y].ResetThreatState();

        // Calculate threats for all pieces
        foreach (var piece in _playerPieces)
            MarkThreatenedTiles(piece);

        foreach (var piece in _enemyPieces)
            MarkThreatenedTiles(piece);

        // Update visual threat indicators
        UpdateThreatVisuals();
    }

    private void MarkThreatenedTiles(BasePiece piece)
    {
        var attackableTiles = piece.GetAttackablePositions(this);
        foreach (var pos in attackableTiles)
        {
            var tile = GetTileSafe(pos);
            if (tile == null) continue;

            if (piece.Team == Team.Player)
                tile.ThreatenedByPlayer = true;
            else
                tile.ThreatenedByEnemy = true;
        }
    }

    private void UpdateThreatVisuals()
    {
        for (int x = 0; x < 8; x++)
        {
            for (int y = 0; y < 8; y++)
            {
                var tile = _tiles[x, y];
                // Show threat overlay if threatened by either side
                tile.ShowThreat(tile.ThreatenedByPlayer || tile.ThreatenedByEnemy);
            }
        }
    }

    /// <summary>
    /// Checks if a king is in a threatened position
    /// </summary>
    public bool IsKingThreatened(Team kingTeam)
    {
        var king = kingTeam == Team.Player ? PlayerKing : EnemyKing;
        if (king == null) return false;

        var tile = GetTile(king.BoardPosition);
        return tile.IsThreatened(kingTeam);
    }

    /// <summary>
    /// Sets up standard chess starting positions
    /// </summary>
    public void SetupStandardPosition(Func<PieceType, Team, BasePiece> pieceFactory)
    {
        // Player pieces (bottom, rows 1-2)
        PlacePiece(pieceFactory(PieceType.Rook, Team.Player), new Vector2I(0, 0));
        PlacePiece(pieceFactory(PieceType.Knight, Team.Player), new Vector2I(1, 0));
        PlacePiece(pieceFactory(PieceType.Bishop, Team.Player), new Vector2I(2, 0));
        PlacePiece(pieceFactory(PieceType.Queen, Team.Player), new Vector2I(3, 0));
        PlacePiece(pieceFactory(PieceType.King, Team.Player), new Vector2I(4, 0));
        PlacePiece(pieceFactory(PieceType.Bishop, Team.Player), new Vector2I(5, 0));
        PlacePiece(pieceFactory(PieceType.Knight, Team.Player), new Vector2I(6, 0));
        PlacePiece(pieceFactory(PieceType.Rook, Team.Player), new Vector2I(7, 0));

        for (int x = 0; x < 8; x++)
            PlacePiece(pieceFactory(PieceType.Pawn, Team.Player), new Vector2I(x, 1));

        // Enemy pieces (top, rows 7-8)
        PlacePiece(pieceFactory(PieceType.Rook, Team.Enemy), new Vector2I(0, 7));
        PlacePiece(pieceFactory(PieceType.Knight, Team.Enemy), new Vector2I(1, 7));
        PlacePiece(pieceFactory(PieceType.Bishop, Team.Enemy), new Vector2I(2, 7));
        PlacePiece(pieceFactory(PieceType.Queen, Team.Enemy), new Vector2I(3, 7));
        PlacePiece(pieceFactory(PieceType.King, Team.Enemy), new Vector2I(4, 7));
        PlacePiece(pieceFactory(PieceType.Bishop, Team.Enemy), new Vector2I(5, 7));
        PlacePiece(pieceFactory(PieceType.Knight, Team.Enemy), new Vector2I(6, 7));
        PlacePiece(pieceFactory(PieceType.Rook, Team.Enemy), new Vector2I(7, 7));

        for (int x = 0; x < 8; x++)
            PlacePiece(pieceFactory(PieceType.Pawn, Team.Enemy), new Vector2I(x, 6));

        RecalculateThreatZones();
    }

    /// <summary>
    /// Gets all pieces for a team
    /// </summary>
    public IEnumerable<BasePiece> GetPiecesForTeam(Team team) =>
        team == Team.Player ? _playerPieces : _enemyPieces;

    /// <summary>
    /// Clears the entire board
    /// </summary>
    public void ClearBoard()
    {
        foreach (var piece in _playerPieces.ToList())
        {
            GetTile(piece.BoardPosition).OccupyingPiece = null;
            piece.QueueFree();
        }
        foreach (var piece in _enemyPieces.ToList())
        {
            GetTile(piece.BoardPosition).OccupyingPiece = null;
            piece.QueueFree();
        }

        _playerPieces.Clear();
        _enemyPieces.Clear();
        PlayerKing = null;
        EnemyKing = null;

        for (int x = 0; x < 8; x++)
            for (int y = 0; y < 8; y++)
                _tiles[x, y].ResetThreatState();
    }
}
