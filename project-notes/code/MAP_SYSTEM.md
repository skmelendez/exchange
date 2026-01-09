# Map System (STS-Style Node Tree)

## Overview

The map system implements a Slay the Spire-style branching path structure where players choose their route through an act.

---

## Data Structures

### ActMap
```csharp
ActMap {
    ActNumber: int          // 1-4 (4 = Final Boss)
    Config: ActConfig       // Generation parameters
    Nodes: List<MapNode>    // All nodes in this act
    Seed: int               // Deterministic generation seed

    StartingNodes: List<MapNode>  // Column 0 nodes
    BossNode: MapNode?            // Final column node
    CurrentNode: MapNode?         // Player's current position
}
```

### MapNode
```csharp
MapNode {
    Id: int                     // Unique identifier
    Position: MapPosition       // (Column, Row)
    NodeType: MapNodeType       // Combat/Elite/Boss/Shop/Event/Treasure

    IncomingConnections: List<int>   // Node IDs that lead here
    OutgoingConnections: List<int>   // Node IDs this leads to

    IsVisited: bool       // Player has been here
    IsAccessible: bool    // Player can move here now
    IsCurrentNode: bool   // Player is here now
}
```

### MapPosition
```csharp
MapPosition {
    Column: int    // X: 0 = start, increasing toward boss
    Row: int       // Y: 0 = top, increasing downward
}
```

---

## Generation Algorithm

```
1. CREATE STARTING NODES (Column 0)
   - Pick 1-3 random rows
   - Create Combat nodes at each

2. GENERATE COLUMNS (1 to N-2)
   For each node in previous column:
   - Get valid target rows (no crossing paths)
   - Connect to 1-2 nodes in next column
   - Create nodes if they don't exist
   - Nodes may merge (multiple incoming)

3. CREATE BOSS NODE (Column N-1)
   - Single node in middle row
   - All previous column nodes connect to it

4. ENSURE CONNECTIVITY
   - BFS backwards from boss
   - Connect any orphaned nodes

5. ASSIGN NODE TYPES
   - Column 0: Combat (starting nodes)
   - Elite column: One random node becomes Elite
   - Other columns: Weighted random (Combat/Event/Treasure/Shop)
   - Final column: Boss
   - Ensure at least one Shop exists
```

---

## Path Crossing Prevention

```
For source node connecting to target row:
  For each other node in same column:
    If source is above other AND target row is below other:
      CROSSING - reject this connection
    If source is below other AND target row is above other:
      CROSSING - reject this connection
```

---

## Act Configurations

| Act | Columns | Max Rows | Elite Col | AI Depth |
|-----|---------|----------|-----------|----------|
| 1   | 8       | 3        | 4         | 2-ply    |
| 2   | 9       | 3        | 5         | 3-ply    |
| 3   | 10      | 3        | 5         | 4-ply    |
| Final | 2     | 1        | -         | 4-ply    |

---

## Node Type Distribution

Default weights per column (non-fixed):
- Combat: 50-65%
- Event: 15-20%
- Treasure: 10-15%
- Shop: 10-15%

Fixed assignments:
- Column 0: Always Combat
- Elite column: One node becomes Elite
- Final column: Always Boss

---

## Navigation Flow

```
1. Player on Map Screen
   └── CurrentNode = null (or visited node)
       └── StartingNodes or OutgoingConnections are Accessible

2. Player clicks accessible node
   └── RunManager.SelectNode()
       └── SetCurrentNode() updates accessibility
       └── NodeSelected event fires

3. MainGameController.EnterNode()
   └── Combat: StartCombat() → GameController instance
   └── Shop: EnterShop() → (TODO: Shop UI)
   └── Event: EnterEvent() → (TODO: Event UI)
   └── Treasure: OpenTreasure() → (TODO: Treasure UI)

4. Node completed
   └── RunManager.CompleteCurrentNode()
       └── Award coins
       └── If Boss: CompleteAct() → Start next act
       └── Update map accessibility
       └── Return to map screen
```

---

## Visual Representation (MapUI)

```
Layout:
- Nodes: 40px circles
- Horizontal spacing: 120px between columns
- Vertical spacing: 80px between rows
- Left margin: 80px
- Top margin: 100px

Colors by type:
- Combat: Red (0.7, 0.3, 0.3)
- Elite: Orange/Gold (0.8, 0.6, 0.2)
- Boss: Bright Red (0.9, 0.2, 0.2)
- Treasure: Yellow (0.9, 0.8, 0.2)
- Shop: Green (0.3, 0.7, 0.3)
- Event: Blue (0.5, 0.5, 0.8)

Symbols:
- Combat: !
- Elite: E
- Boss: B
- Treasure: T
- Shop: $
- Event: ?

States:
- Current: White border, glow
- Accessible: Green border, pulsing
- Visited: Darkened, gray border
- Unexplored: Slightly dark, thin border

Connections:
- Active path: Bright green, thick
- Partially traveled: Gray, medium
- Unexplored: Dark gray, thin
- Drawn as bezier curves
```
