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

    private Tile[,] _tiles = new Tile[8, 8];
    private List<BasePiece> _playerPieces = new();
    private List<BasePiece> _enemyPieces = new();

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
            GD.PrintErr($"Invalid board position: {position}");
            return;
        }

        var tile = GetTile(position);
        if (tile.IsOccupied)
        {
            GD.PrintErr($"Tile {position.ToChessNotation()} already occupied!");
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
    /// Moves a piece to a new position
    /// </summary>
    public bool MovePiece(BasePiece piece, Vector2I newPosition)
    {
        if (!newPosition.IsOnBoard()) return false;

        var targetTile = GetTile(newPosition);
        if (targetTile.IsOccupied) return false;

        // Clear old tile
        var oldTile = GetTile(piece.BoardPosition);
        oldTile.OccupyingPiece = null;

        // Update piece position
        piece.BoardPosition = newPosition;
        piece.Position = new Vector2(newPosition.X * Tile.TileSize, (7 - newPosition.Y) * Tile.TileSize);

        // Set new tile
        targetTile.OccupyingPiece = piece;

        return true;
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
