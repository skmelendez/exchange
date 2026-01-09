using Godot;

namespace Exchange.Map;

/// <summary>
/// Generates act maps using STS-style multi-path tree algorithm.
/// Uses Godot's RandomNumberGenerator with seed for deterministic generation.
///
/// Algorithm:
/// 1. Create 1-3 starting nodes on column 0
/// 2. For each column, extend paths from previous column
/// 3. Each node connects to 1-3 nearest nodes on next column (up, straight, down)
/// 4. Paths cannot cross (connections cannot intersect)
/// 5. Paths may merge (multiple inputs to one node)
/// 6. All paths must converge at boss node on final column
/// 7. Node types assigned based on column (combat/event/treasure/shop/elite/boss)
/// </summary>
public static class MapGenerator
{
    /// <summary>
    /// Generate a complete act map with deterministic seed-based generation.
    /// </summary>
    /// <param name="config">Act configuration parameters</param>
    /// <param name="seed">Random seed for reproducible generation</param>
    /// <returns>A fully generated ActMap</returns>
    public static ActMap Generate(ActConfig config, int seed)
    {
        var rng = new RandomNumberGenerator();
        rng.Seed = (ulong)seed;
        var map = new ActMap(config, seed);
        MapNode.ResetIdCounter();

        // Step 1: Create starting nodes (column 0)
        int numStartingPaths = rng.RandiRange(config.MinStartingPaths, config.MaxStartingPaths);
        var startingRows = PickStartingRows(numStartingPaths, config.MaxRows, rng);

        GD.Print($"[MapGen] Generating Act {config.ActNumber} with {numStartingPaths} starting paths, seed {seed}");

        foreach (int row in startingRows)
        {
            var node = new MapNode(new MapPosition(0, row), MapNodeType.Combat);
            map.AddNode(node);
        }

        // Step 2: Generate paths column by column
        for (int col = 1; col < config.Columns; col++)
        {
            GenerateColumn(map, col, config, rng);
        }

        // Step 3: Ensure all paths reach the boss
        EnsureBossConnectivity(map, config);

        // Step 4: Assign node types based on column
        AssignNodeTypes(map, config, rng);

        // Step 5: Validate and fix any issues
        ValidateMap(map);

        map.DebugPrint();
        return map;
    }

    private static List<int> PickStartingRows(int count, int maxRows, RandomNumberGenerator rng)
    {
        var available = Enumerable.Range(0, maxRows).ToList();
        var selected = new List<int>();

        for (int i = 0; i < count && available.Count > 0; i++)
        {
            int idx = rng.RandiRange(0, available.Count - 1);
            selected.Add(available[idx]);
            available.RemoveAt(idx);
        }

        selected.Sort();
        return selected;
    }

    private static void GenerateColumn(ActMap map, int col, ActConfig config, RandomNumberGenerator rng)
    {
        // Get all nodes from previous column
        var prevNodes = map.Nodes
            .Where(n => n.Position.Column == col - 1)
            .OrderBy(n => n.Position.Row)
            .ToList();

        if (prevNodes.Count == 0)
        {
            GD.PrintErr($"[MapGen] No nodes in column {col - 1}!");
            return;
        }

        bool isBossColumn = col == config.Columns - 1;

        if (isBossColumn)
        {
            // Boss column: single node, all previous nodes connect to it
            // Place boss in middle row
            int bossRow = config.MaxRows / 2;
            var bossNode = new MapNode(new MapPosition(col, bossRow), MapNodeType.Boss);
            map.AddNode(bossNode);

            foreach (var prev in prevNodes)
            {
                map.Connect(prev, bossNode);
            }
            return;
        }

        // For each node in previous column, create connections to this column
        // Track which rows we've created nodes for
        var createdNodes = new Dictionary<int, MapNode>();

        foreach (var prevNode in prevNodes)
        {
            // Determine valid target rows (no crossing)
            var validRows = GetValidTargetRows(prevNode, prevNodes, config.MaxRows, rng);

            if (validRows.Count == 0)
            {
                // Force at least one connection (straight ahead or closest)
                validRows.Add(Math.Clamp(prevNode.Position.Row, 0, config.MaxRows - 1));
            }

            // Pick 1-2 target rows (more connections = more merging potential)
            int numConnections = rng.RandiRange(1, Math.Min(2, validRows.Count));
            var targetRows = validRows.OrderBy(_ => rng.Randi()).Take(numConnections).ToList();

            foreach (int row in targetRows)
            {
                // Get or create node at this position
                if (!createdNodes.TryGetValue(row, out var targetNode))
                {
                    targetNode = new MapNode(new MapPosition(col, row), MapNodeType.Combat);
                    map.AddNode(targetNode);
                    createdNodes[row] = targetNode;
                }

                map.Connect(prevNode, targetNode);
            }
        }

        // Ensure we have at least one node in this column
        if (createdNodes.Count == 0)
        {
            int row = rng.RandiRange(0, config.MaxRows - 1);
            var node = new MapNode(new MapPosition(col, row), MapNodeType.Combat);
            map.AddNode(node);

            // Connect from all previous nodes
            foreach (var prev in prevNodes)
            {
                map.Connect(prev, node);
            }
        }
    }

    private static List<int> GetValidTargetRows(MapNode source, List<MapNode> allPrevNodes, int maxRows, RandomNumberGenerator rng)
    {
        var validRows = new List<int>();
        int sourceRow = source.Position.Row;

        // Can connect to: same row, one above, one below (within bounds)
        for (int delta = -1; delta <= 1; delta++)
        {
            int targetRow = sourceRow + delta;
            if (targetRow < 0 || targetRow >= maxRows)
                continue;

            // Check for crossing: would this connection cross any existing connection?
            bool wouldCross = false;

            foreach (var otherNode in allPrevNodes)
            {
                if (otherNode == source) continue;

                // A crossing occurs if:
                // - source is above other, but we're connecting below other's row
                // - source is below other, but we're connecting above other's row
                // (simplified: connections from different rows going to opposite directions)

                int otherRow = otherNode.Position.Row;

                // If source is above other node
                if (sourceRow < otherRow)
                {
                    // We can't connect to a row below the other node's row
                    // (that would cross their potential straight connection)
                    // Actually more nuanced - we need to check their actual connections
                    // For now, use simple heuristic: don't connect far below if we're above
                    if (targetRow > otherRow)
                        wouldCross = true;
                }
                // If source is below other node
                else if (sourceRow > otherRow)
                {
                    if (targetRow < otherRow)
                        wouldCross = true;
                }
            }

            if (!wouldCross)
                validRows.Add(targetRow);
        }

        return validRows;
    }

    private static void EnsureBossConnectivity(ActMap map, ActConfig config)
    {
        if (map.BossNode == null)
        {
            GD.PrintErr("[MapGen] No boss node found!");
            return;
        }

        // Check that all nodes have a path to boss
        var visited = new HashSet<int>();
        var toVisit = new Queue<int>();
        toVisit.Enqueue(map.BossNode.Id);

        // BFS backwards from boss
        while (toVisit.Count > 0)
        {
            var nodeId = toVisit.Dequeue();
            if (visited.Contains(nodeId)) continue;
            visited.Add(nodeId);

            var node = map.GetNodeById(nodeId);
            if (node == null) continue;

            foreach (var incomingId in node.IncomingConnections)
            {
                if (!visited.Contains(incomingId))
                    toVisit.Enqueue(incomingId);
            }
        }

        // Find disconnected nodes and connect them
        var disconnected = map.Nodes.Where(n => !visited.Contains(n.Id) && n.Position.Column < config.Columns - 1).ToList();

        foreach (var node in disconnected)
        {
            // Find nearest connected node in next column
            var nextColNodes = map.Nodes
                .Where(n => n.Position.Column == node.Position.Column + 1 && visited.Contains(n.Id))
                .OrderBy(n => Math.Abs(n.Position.Row - node.Position.Row))
                .ToList();

            if (nextColNodes.Count > 0)
            {
                map.Connect(node, nextColNodes[0]);
                GD.Print($"[MapGen] Connected disconnected node {node.Position} to {nextColNodes[0].Position}");
            }
        }
    }

    private static void AssignNodeTypes(ActMap map, ActConfig config, RandomNumberGenerator rng)
    {
        foreach (var node in map.Nodes)
        {
            int col = node.Position.Column;

            // Column 0: Always combat (starting nodes)
            if (col == 0)
            {
                node.NodeType = MapNodeType.Combat;
                continue;
            }

            // Boss column: Already set
            if (node.NodeType == MapNodeType.Boss)
                continue;

            // Elite column: One node becomes elite
            if (col == config.EliteColumn)
            {
                // Pick one random node in this column to be elite
                var colNodes = map.Nodes.Where(n => n.Position.Column == col).ToList();
                var eliteNode = colNodes[rng.RandiRange(0, colNodes.Count - 1)];
                eliteNode.NodeType = MapNodeType.Elite;

                // Others in this column are combat
                foreach (var n in colNodes.Where(n => n != eliteNode))
                {
                    n.NodeType = MapNodeType.Combat;
                }
                continue;
            }

            // Middle columns: weighted random
            float roll = rng.Randf();
            float cumulative = 0;

            cumulative += config.CombatWeight;
            if (roll < cumulative)
            {
                node.NodeType = MapNodeType.Combat;
                continue;
            }

            cumulative += config.EventWeight;
            if (roll < cumulative)
            {
                node.NodeType = MapNodeType.Event;
                continue;
            }

            cumulative += config.TreasureWeight;
            if (roll < cumulative)
            {
                node.NodeType = MapNodeType.Treasure;
                continue;
            }

            node.NodeType = MapNodeType.Shop;
        }

        // Ensure at least one shop in the act (if not already)
        if (!map.Nodes.Any(n => n.NodeType == MapNodeType.Shop))
        {
            // Convert a random non-combat/elite/boss node to shop
            var candidates = map.Nodes
                .Where(n => n.NodeType == MapNodeType.Event || n.NodeType == MapNodeType.Treasure)
                .ToList();

            if (candidates.Count > 0)
            {
                candidates[rng.RandiRange(0, candidates.Count - 1)].NodeType = MapNodeType.Shop;
            }
        }
    }

    private static void ValidateMap(ActMap map)
    {
        // Ensure all non-boss nodes have at least one outgoing connection
        foreach (var node in map.Nodes.Where(n => n.NodeType != MapNodeType.Boss))
        {
            if (node.OutgoingConnections.Count == 0)
            {
                GD.PrintErr($"[MapGen] Node {node.Position} has no outgoing connections!");
            }
        }

        // Ensure boss has incoming connections
        if (map.BossNode != null && map.BossNode.IncomingConnections.Count == 0)
        {
            GD.PrintErr("[MapGen] Boss node has no incoming connections!");
        }

        // Ensure starting nodes have no incoming connections
        foreach (var start in map.StartingNodes)
        {
            if (start.IncomingConnections.Count > 0)
            {
                GD.PrintErr($"[MapGen] Starting node {start.Position} has incoming connections!");
            }
        }
    }
}
