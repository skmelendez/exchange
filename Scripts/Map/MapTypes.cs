namespace Exchange.Map;

/// <summary>
/// Types of nodes that can appear on the act map
/// </summary>
public enum MapNodeType
{
    Combat,     // Standard enemy encounter
    Elite,      // Harder combat, better rewards (mid-act)
    Boss,       // End-of-act boss fight
    Treasure,   // Free relic/reward
    Shop,       // Spend coins
    Event,      // Random encounter with choices
    Start       // Starting node (visual only, immediately leads to first choice)
}

/// <summary>
/// Represents a position on the map grid
/// </summary>
public readonly struct MapPosition
{
    public int Column { get; }  // X: 0 = start, increasing = progress toward boss
    public int Row { get; }     // Y: 0 = top, increasing = down

    public MapPosition(int column, int row)
    {
        Column = column;
        Row = row;
    }

    public override string ToString() => $"({Column}, {Row})";

    public override bool Equals(object? obj) =>
        obj is MapPosition other && Column == other.Column && Row == other.Row;

    public override int GetHashCode() => HashCode.Combine(Column, Row);

    public static bool operator ==(MapPosition left, MapPosition right) => left.Equals(right);
    public static bool operator !=(MapPosition left, MapPosition right) => !left.Equals(right);
}

/// <summary>
/// Configuration for map generation per act.
/// Values are set via MapConstants.GetActConfig().
/// </summary>
public class ActConfig
{
    public int ActNumber { get; init; }
    public int Columns { get; init; } = 8;          // Total columns (including boss)
    public int MaxRows { get; init; } = 3;          // Max vertical divergence
    public int MinStartingPaths { get; init; } = 1;
    public int MaxStartingPaths { get; init; } = 3;
    public int EliteColumn { get; init; } = 4;      // Column where elite appears
    public int AiDepth { get; init; } = 2;          // Minimax lookahead depth

    // Node type distribution weights (for non-fixed columns)
    public float CombatWeight { get; init; } = 0.5f;
    public float EventWeight { get; init; } = 0.2f;
    public float TreasureWeight { get; init; } = 0.15f;
    public float ShopWeight { get; init; } = 0.15f;
}
