# Exchange - Architecture Overview

## Tech Stack
- **Engine:** Godot 4.5+ with .NET 8.0
- **Language:** C# 12 with file-scoped namespaces
- **Pattern:** Composition over inheritance, event-driven communication

---

## Namespace Structure

```
Exchange/
├── Core/           # Enums, data structures, game state
├── Board/          # Tile and GameBoard management
├── Pieces/         # All chess piece implementations
├── Combat/         # Dice rolling and combat resolution
├── Controllers/    # Game flow, turns, AI
├── Map/            # Roguelike map generation (STS-style)
└── UI/             # All user interface components
```

---

## Core Flow

```
MainGameController (Entry Point)
    │
    ├── RunManager (Roguelike meta-progression)
    │       │
    │       └── MapGenerator → ActMap → MapNode[]
    │
    ├── MapUI (Node selection screen)
    │
    └── GameController (Combat instance)
            │
            ├── GameState (Turn/match state)
            ├── GameBoard (8x8 tiles + pieces)
            ├── TurnController (Turn economy)
            ├── CombatResolver (Damage calculation)
            ├── AIController (Enemy decisions)
            └── GameUI (Combat HUD)
```

---

## Scene Hierarchy (Runtime)

```
MainGameController (Node2D)
├── RunManager (Node)
├── MapContainer (Control)
│   └── MapUI (Control)
├── CombatContainer (Node2D)
│   └── GameController* (Node2D, created per combat)
│       ├── GameState (Node)
│       ├── GameBoard (Node2D)
│       │   ├── Tile[64] (Node2D)
│       │   └── BasePiece[] (Node2D)
│       ├── CombatResolver (Node)
│       ├── AIController (Node)
│       └── GameUI (CanvasLayer)
├── OverlayUI (CanvasLayer)
├── PauseMenu (CanvasLayer)
└── DebugMenu (CanvasLayer)
```

---

## Key Patterns

### Signal/Event Communication
- Godot `[Signal]` delegates for simple types
- C# `event Action<T>` for complex types (records, custom classes)

### Turn Economy
Each turn: **ONE piece** performs **ONE action** (Move OR Attack OR Ability)

### Piece Inheritance
```
BasePiece (abstract)
├── KingPiece
├── QueenPiece
├── RookPiece
├── BishopPiece
├── KnightPiece
└── PawnPiece
```

---

## Data vs Node Classes

| Class | Type | Notes |
|-------|------|-------|
| `GameState` | Node | Needs to be in scene tree |
| `GameBoard` | Node2D | Contains Tile children |
| `BasePiece` | Node2D | Visual representation |
| `ActMap` | Plain C# | Data-only, no scene tree |
| `MapNode` | Plain C# | Data-only, no scene tree |
| `ActConfig` | Plain C# | Configuration record |
| `PieceData` | Static | Compile-time constants |
| `MapConstants` | Static | Compile-time constants |

---

## Build Command
```bash
dotnet build
```

Godot automatically builds when running from editor.
