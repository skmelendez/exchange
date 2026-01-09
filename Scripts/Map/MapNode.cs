using Godot;

namespace Exchange.Map;

/// <summary>
/// Represents a single node on the act map.
/// Nodes are connected in a multi-path tree structure following STS-style pathing.
/// </summary>
public class MapNode
{
    private static readonly object _idLock = new();
    private static int _nextId;

    /// <summary>Unique identifier for this node within the current run.</summary>
    public int Id { get; }

    /// <summary>Grid position (Column, Row) on the act map.</summary>
    public MapPosition Position { get; }

    /// <summary>Type of encounter at this node (Combat, Elite, Boss, Shop, Event, Treasure).</summary>
    public MapNodeType NodeType { get; set; }

    /// <summary>Node IDs that have connections leading TO this node.</summary>
    public List<int> IncomingConnections { get; } = [];

    /// <summary>Node IDs that this node connects TO (player can travel to these).</summary>
    public List<int> OutgoingConnections { get; } = [];

    /// <summary>Whether the player has visited this node.</summary>
    public bool IsVisited { get; set; }

    /// <summary>Whether the node was successfully completed (combat won, event done, etc.).</summary>
    public bool IsCompleted { get; set; }

    /// <summary>Whether the player can currently move to this node.</summary>
    public bool IsAccessible { get; set; }

    /// <summary>Whether this is the player's current position on the map.</summary>
    public bool IsCurrentNode { get; set; }

    /// <summary>Difficulty scaling factor for combat nodes (optional).</summary>
    public int? EnemyDifficulty { get; set; }

    /// <summary>Coin reward for completing this node (optional).</summary>
    public int? RewardCoins { get; set; }

    /// <summary>Event identifier for Event nodes (optional).</summary>
    public string? EventId { get; set; }

    /// <summary>
    /// Creates a new map node at the specified position.
    /// </summary>
    /// <param name="position">Grid position on the map</param>
    /// <param name="nodeType">Type of encounter</param>
    public MapNode(MapPosition position, MapNodeType nodeType)
    {
        lock (_idLock)
        {
            Id = _nextId++;
        }
        Position = position;
        NodeType = nodeType;
    }

    /// <summary>
    /// Resets the static ID counter. Call when starting a new run to ensure
    /// consistent ID generation across map regenerations.
    /// </summary>
    public static void ResetIdCounter()
    {
        lock (_idLock)
        {
            _nextId = 0;
        }
    }

    /// <summary>Whether this node is a combat encounter (Combat, Elite, or Boss).</summary>
    public bool IsCombatNode => NodeType is MapNodeType.Combat or MapNodeType.Elite or MapNodeType.Boss;

    /// <inheritdoc/>
    public override string ToString() =>
        $"Node[{Id}] {NodeType} at {Position} " +
        $"(in:{IncomingConnections.Count}, out:{OutgoingConnections.Count})" +
        (IsVisited ? " [VISITED]" : "") +
        (IsAccessible ? " [ACCESSIBLE]" : "");
}
