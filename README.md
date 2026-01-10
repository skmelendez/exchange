# Exchange

**by Dead Letter Games**

---

A turn-based roguelike strategy game that reimagines chess as a tactical RPG. Pieces have hit points, abilities, and persistence across battles. Navigate a branching map, collect relics, and dethrone the enemy King.

## The Core Concept

Exchange takes the familiar framework of chess and transforms it into something entirely new:

- **Pieces have HP** - Attacks deal damage based on dice rolls; pieces don't die in one hit
- **One action per turn** - Each turn, ONE piece performs ONE action: Move, Attack, or Ability
- **Roguelike progression** - Navigate through acts, collect relics, and face escalating challenges
- **Persistent army** - Your pieces carry their HP and cooldowns between battles

---

## Combat Rules

### Turn Economy

Each turn, exactly **one piece** may act. An action is one of:
- **Move** - Standard chess-inspired movement patterns
- **Base Attack** - Deal damage to an enemy (replaces movement)
- **Use Ability** - Unique power for each piece type

No chaining allowed, except for the Knight's special rule.

### Damage Calculation

```
Total Damage = Base Damage + Dice Roll (1d6, clamped 1-6)
```

Modifiers affect your roll:
- **Entered threat zone this turn:** -1 to roll
- **Enemy King is threatened:** +1 to your roll
- **Royal Decree active:** +1 to all allied rolls

### Victory Condition

- **Win:** Enemy King reaches 0 HP
- **Lose:** Your King reaches 0 HP

---

## Piece Stats

| Piece  | HP  | Base Damage | Attack Range |
|--------|-----|-------------|--------------|
| King   | 15  | 1           | Adjacent (all 8 directions) |
| Queen  | 10  | 3           | Up to 3 tiles in any direction |
| Rook   | 13  | 2           | Up to 2 tiles orthogonally |
| Knight | 11  | 2           | Adjacent only (but can move+attack) |
| Bishop | 10  | 2           | Up to 3 tiles diagonally (not adjacent) |
| Pawn   | 7   | 1           | Diagonal-adjacent forward only |

---

## Abilities

Each piece type has a unique ability that replaces movement and attacks:

| Piece  | Ability | Cooldown | Effect |
|--------|---------|----------|--------|
| **King** | Royal Decree | Once/match | All allied combat rolls gain +1 until your next turn |
| **Queen** | Overextend | 3 turns | Move, then attack. Queen takes 2 self-damage afterward |
| **Rook** | Interpose | 3 turns | Damage to adjacent allies is split between ally and Rook |
| **Bishop** | Consecration | 3 turns | Heal a diagonal ally for 1d6 HP (cannot target self) |
| **Knight** | Skirmish | 3 turns | Attack, then reposition 1 tile |
| **Pawn** | Advance | 1 turn | Move forward one additional tile (cannot use consecutively) |

### Knight Special Rule

The Knight is unique: it may **move then attack** in the same turn.
- Must move first
- Attack must be adjacent to landing square
- Cannot attack without moving first
- Using an ability replaces this behavior

---

## Threat Zones

- A **Threat Zone** is any tile a piece could attack from its current position
- Threat zones are always visible on the board
- **Penalty:** Moving into a threat zone applies -1 to your next combat roll
- **King Safety:** The King cannot move into threatened tiles
- **King Threatened:** If your King ends a turn in a threat zone, the enemy gains +1 to their next attack

---

## Map System

Exchange features a **Slay the Spire-style branching map** that offers player agency:

### Structure
- **3 Acts + Final Boss**
- Node tree flows **left to right**
- Player chooses their path through branching nodes
- All paths converge at the Boss node

### Node Types

| Node | Description |
|------|-------------|
| **Combat** | Standard battle against enemy chess army |
| **Elite** | Harder combat with better rewards |
| **Treasure** | Free relic or resource pickup |
| **Shop** | Spend coins on relics, healing, upgrades |
| **Event** | Random encounter with risk/reward choices |
| **Boss** | End-of-act mandatory fight |

### AI Difficulty Scaling

The enemy AI grows smarter as you progress:

| Act | Search Depth | Behavior |
|-----|--------------|----------|
| Act 1 | 2-ply | Basic tactical awareness |
| Act 2 | 3-ply | Sees your response to their moves |
| Act 3 | 4-ply | Deep strategic planning |
| Final Boss | 5-ply | Maximum difficulty |

---

## Relics

Relics are passive items that bend the rules in your favor. Categories include:

### Dice Relics
- **Ivory Die** - First roll each match cannot be below 2
- **Weighted Die** - Dice rolls of 6 deal +1 damage
- **Fate Weights** - First roll each match is treated as 6

### Formation Relics
- **Battle Standard** - Adjacent allies take -1 damage
- **Command Flag** - Adjacent allies gain +1 dice on attacks
- **Shield Emblem** - Adjacent allies ignore threat penalties

### Rule-Benders (Rare)
- **Double-Headed Knight** - First non-Knight attack may move then attack
- **Split Square** - First piece each match ignores move/attack exclusivity
- **False Crown** - First time the King would die, it remains at 1 HP

---

## Boss Modifiers

Each act boss introduces a unique rule that changes combat:

1. **Act 1 Boss:** Threat penalties are -2 instead of -1
2. **Act 2 Boss:** Enemies adjacent to allies take -1 damage
3. **Act 3 Boss:** Enemy King may move into threatened tiles
4. **Final Boss:** Player King is always considered threatened (+1 to all enemy rolls)

---

## Technical Architecture

### Tech Stack
- **Engine:** Godot 4.5+ with .NET 8.0
- **Language:** C# 12 with file-scoped namespaces
- **Pattern:** Composition over inheritance, event-driven communication

### AI System
- True N-ply minimax search with alpha-beta pruning
- Transposition tables with Zobrist hashing for position caching
- Parallel search using "Young Brothers Wait" strategy
- Pondering during player's turn for faster response
- MVV-LVA move ordering for efficient pruning

### Namespace Structure
```
Exchange/
├── Core/           # Enums, data structures, game state
├── Board/          # Tile and GameBoard management
├── Pieces/         # All chess piece implementations
├── Combat/         # Dice rolling and combat resolution
├── Controllers/    # Game flow, turns, AI
├── AI/             # Minimax engine, transposition tables
├── Map/            # Roguelike map generation
└── UI/             # User interface components
```

---

## Systems In Development

### Implemented
- [x] Full combat loop with HP-based damage
- [x] All piece abilities with cooldowns
- [x] Threat zone visualization and penalties
- [x] Dice-based combat with modifiers
- [x] N-ply minimax AI with alpha-beta pruning
- [x] Transposition tables with incremental reuse
- [x] Parallel AI search (multi-core)
- [x] AI pondering during player turn
- [x] STS-style branching map structure
- [x] Act progression with difficulty scaling
- [x] Animated piece movement and attacks

### Upcoming
- [ ] Relic system implementation
- [ ] Shop UI and purchasing
- [ ] Event encounters with choices
- [ ] Treasure rewards
- [ ] Coin economy and rewards
- [ ] Meta-progression / unlocks
- [ ] Sound effects and music
- [ ] Save/load system
- [ ] Opening book for AI
- [ ] Quiescence search extension

---

## Building

```bash
dotnet build
```

Run from Godot Editor or build with:
```bash
godot --export-release "platform" output_path
```

---

## License

All rights reserved. Copyright Dead Letter Games.
