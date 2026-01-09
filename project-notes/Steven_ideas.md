# Steven's Ideas - EXCHANGE Evolution

This document captures design ideas that evolve beyond the original design doc. These represent the direction we want to take the game.

---

## Run Structure Overhaul: STS-Style Node Tree

### Overview
Replace the linear 5-room × 3-match structure with a **Slay the Spire-style node tree** that offers player agency and variety.

### Structure
- **3 Acts + 1 Final Boss**
- Node tree flows **left to right** on screen
- Player chooses their path through branching nodes

### Node Types
| Node | Description |
|------|-------------|
| **Combat** | Standard battle against enemy chess army |
| **Elite** | Harder combat with better rewards |
| **Treasure** | Free relic or resource pickup |
| **Shop** | Spend coins on relics, healing, etc. |
| **Event** | Random encounter with choices (risk/reward) |
| **Boss** | End-of-act mandatory fight |

### Visual Layout (Left to Right)
```
Column:  0      1      2      3      4      5      6      7
       START                 MID                         BOSS
Row 0:   O------O------O------E------O------O------O------B
          \      \    /              \    /      /
Row 1:     O------O--O                O--O------O
            \    /                        \    /
Row 2:       O--O                          O--O
```

### Path Generation Algorithm (STS-Style)

**Core Concept:** Multi-path, non-branching tree with controlled merging

**Parameters for Exchange:**
- **Grid:** ~8-10 columns × 3 rows (max divergence)
- **Paths:** 1-3 starting paths (seeded random)
- **Combats per act:** 7 (including mid-act elite)
- **Mid-act elite:** Around column 3-4 (forced on one random path)
- **Final merge:** All paths converge at boss node (last column)

**Generation Rules:**
1. Start with 1-3 nodes on column 0 (starting options)
2. Each node connects to 1-3 nearest nodes on next column (up, straight, down)
3. **No paths cross** - connections cannot intersect
4. Paths may **merge** at later columns (multiple inputs → one node)
5. Boss node is single node at final column - all paths must reach it
6. Seed-based deterministic generation for run consistency

**Node Distribution per Act:**
| Column | Node Type | Notes |
|--------|-----------|-------|
| 0 | Combat | 1-3 starting options |
| 1-2 | Combat/Event/Treasure | Mix |
| 3-4 | Elite (mid-act) | At least one path hits elite |
| 5-6 | Combat/Shop/Event | Mix |
| 7 | Boss | Single node, all paths merge |

---

## AI Difficulty Scaling

### Per-Act Lookahead Depth
The enemy AI gets smarter as the run progresses:

| Act | Minimax Depth | Description |
|-----|---------------|-------------|
| Act 1 | 2-ply | Basic tactical awareness |
| Act 2 | 3-ply | Sees your response to their response |
| Act 3 | 4-ply | Deep strategic planning |
| Final | 4-ply+ | Plus unique boss mechanics |

### Additional Scaling Ideas (TBD)
- [ ] Act-specific enemy compositions
- [ ] Elite enemies with unique abilities
- [ ] Boss-specific rule modifiers (keep from original doc?)
- [ ] Enemy buff pools that increase per act

---

## Debug Features Needed

### Auto-Win Match
- Hotkey or debug menu option to instantly win current match
- Allows testing game flow, shop, events, node progression
- Should award standard rewards as if player won normally

### Other Debug Ideas
- [ ] Skip to specific act
- [ ] Spawn specific relics
- [ ] Set coin amount
- [ ] Toggle god mode (invincible King)
- [ ] Force specific dice rolls

---

## Open Questions

1. **How many nodes per act?** STS has ~15 floors per act. What feels right for chess battles?
2. **Path width?** How many parallel paths should exist?
3. **Elite frequency?** Every X nodes? Player choice?
4. **Event pool size?** Need to design events
5. **Relic balance?** Original doc has ~30 relics - enough for 3-act run?
6. **Healing between battles?** Auto-heal? Heal nodes? Shop healing?

---

## Migration Notes

### Original Design (for reference)
- 5 Rooms × 3 Matches = 15 total battles
- Linear progression
- Boss rules per room
- Shop after every win

### New Design
- 3 Acts × ~10-15 nodes = 30-45 potential encounters
- Branching paths with player choice
- Boss at end of each act
- Shops/treasures/events as dedicated nodes

---

## Priority Order

1. **Debug auto-win** - Unblocks testing
2. **Node tree data structure** - Define Act, Node types
3. **Node tree UI** - Left-to-right visual map
4. **Navigation flow** - Click node to enter, return to map after
5. **AI depth scaling** - Parameterize minimax depth per act
6. **Event system** - Random encounters with choices
7. **Balance pass** - Tune difficulty curve

---

*Last updated: Session with Claude - Combat loop completion*
