using Godot;

namespace Exchange.Core;

/// <summary>
/// Extension methods for Vector2I board coordinates.
/// Board uses chess coordinates: columns a-h (0-7), rows 1-8 (0-7 internally).
/// </summary>
public static class Vector2IExtensions
{
    public static readonly Vector2I[] CardinalDirections =
    {
        new(0, 1),   // Up
        new(0, -1),  // Down
        new(1, 0),   // Right
        new(-1, 0)   // Left
    };

    public static readonly Vector2I[] DiagonalDirections =
    {
        new(1, 1),   // Up-Right
        new(1, -1),  // Down-Right
        new(-1, 1),  // Up-Left
        new(-1, -1)  // Down-Left
    };

    public static readonly Vector2I[] AllDirections =
    {
        new(0, 1), new(0, -1), new(1, 0), new(-1, 0),
        new(1, 1), new(1, -1), new(-1, 1), new(-1, -1)
    };

    public static readonly Vector2I[] KnightMoves =
    {
        new(1, 2), new(2, 1), new(2, -1), new(1, -2),
        new(-1, -2), new(-2, -1), new(-2, 1), new(-1, 2)
    };

    public static bool IsOnBoard(this Vector2I pos) =>
        pos.X >= 0 && pos.X < 8 && pos.Y >= 0 && pos.Y < 8;

    public static bool IsAdjacent(this Vector2I a, Vector2I b) =>
        Math.Abs(a.X - b.X) <= 1 && Math.Abs(a.Y - b.Y) <= 1 && a != b;

    public static bool IsDiagonallyAdjacent(this Vector2I a, Vector2I b) =>
        Math.Abs(a.X - b.X) == 1 && Math.Abs(a.Y - b.Y) == 1;

    public static bool IsOrthogonallyAdjacent(this Vector2I a, Vector2I b) =>
        (Math.Abs(a.X - b.X) == 1 && a.Y == b.Y) || (Math.Abs(a.Y - b.Y) == 1 && a.X == b.X);

    /// <summary>
    /// Converts board position to chess notation (e.g., 0,0 -> "a1")
    /// </summary>
    public static string ToChessNotation(this Vector2I pos) =>
        $"{(char)('a' + pos.X)}{pos.Y + 1}";

    /// <summary>
    /// Parses chess notation to board position (e.g., "a1" -> 0,0)
    /// </summary>
    public static Vector2I FromChessNotation(string notation)
    {
        if (notation.Length != 2) throw new ArgumentException("Invalid notation");
        int x = notation[0] - 'a';
        int y = notation[1] - '1';
        return new Vector2I(x, y);
    }

    /// <summary>
    /// Gets Manhattan distance between two positions
    /// </summary>
    public static int ManhattanDistance(this Vector2I a, Vector2I b) =>
        Math.Abs(a.X - b.X) + Math.Abs(a.Y - b.Y);

    /// <summary>
    /// Gets Chebyshev distance (king distance) between two positions
    /// </summary>
    public static int ChebyshevDistance(this Vector2I a, Vector2I b) =>
        Math.Max(Math.Abs(a.X - b.X), Math.Abs(a.Y - b.Y));
}
