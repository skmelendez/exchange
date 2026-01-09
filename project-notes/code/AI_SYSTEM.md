# AI System

## Overview

The AI uses a **true N-ply minimax search with alpha-beta pruning** and **transposition tables** for optimal play. Supports depths from 1-5 with incremental cache reuse between moves.

---

## Architecture

```
Scripts/AI/
├── AISearchEngine.cs      # Main minimax engine with alpha-beta
├── SimulatedBoardState.cs # Lightweight board for simulation + Zobrist hashing
├── TranspositionTable.cs  # Position cache with depth-aware storage
└── MoveOrderer.cs         # MVV-LVA move ordering for pruning efficiency

Scripts/Controllers/
└── AIController.cs        # Integration with game, fallback heuristics
```

---

## Minimax Search

### Algorithm
```
SearchRoot(state, depth):
    Generate all moves
    Order moves (hash move first, then captures by MVV-LVA)
    For each move:
        state.MakeMove(move)
        score = -AlphaBeta(state, depth-1, -beta, -alpha)  // Negamax
        state.UndoMove(move)
        Track best move
    Return best move + score

AlphaBeta(state, depth, alpha, beta):
    Check transposition table → return cached if valid
    If depth == 0: return Evaluate(state)
    If king captured: return ±50000

    Generate and order moves
    For each move:
        state.MakeMove(move)
        score = -AlphaBeta(state, depth-1, -beta, -alpha)
        state.UndoMove(move)

        If score >= beta: return beta  // Beta cutoff
        If score > alpha: alpha = score

    Store in transposition table
    Return alpha
```

### Iterative Deepening
Searches depth 1, then 2, then 3... up to target depth. Benefits:
- Better move ordering (uses previous iteration's best move)
- Can implement time control (stop when time runs out)
- Shows progressive improvement in logs

---

## Transposition Table

### Zobrist Hashing
Each board position has a unique 64-bit hash computed via XOR:
```csharp
hash = 0
For each piece on board:
    hash ^= PieceKeys[pieceType, team, x, y]
    hash ^= HpKeys[pieceType, team, currentHp]  // HP matters in this game!
```

### Cache Entry
```csharp
struct Entry {
    ulong Hash;           // Position identifier
    int Depth;            // Search depth
    int Score;            // Evaluation
    NodeType Type;        // Exact, LowerBound, UpperBound
    SimulatedMove BestMove;
    byte Age;             // For replacement policy
}
```

### Incremental Reuse
After each move:
1. Call `OnMoveMade()` to increment age
2. Previous positions remain cached for transpositions
3. Only need to search 1 additional ply forward (not full recalc!)

---

## Move Ordering (MVV-LVA)

**Most Valuable Victim - Least Valuable Attacker**

Priority order:
1. Hash move (from transposition table)
2. King captures (instant win)
3. Winning captures (QxP > PxQ)
4. Equal captures
5. Losing captures
6. Quiet moves (sorted by positional value)

```
Score = VictimValue × 10 - AttackerValue
Example: Pawn takes Queen = 900×10 - 100 = 8900 (search first!)
```

---

## Simulated Board State

Lightweight simulation layer (no Godot Node overhead):

```csharp
SimulatedBoardState:
    - 8x8 piece array
    - Piece list for fast iteration
    - Make/Undo move with Zobrist updates
    - Clone for parallel search (if needed)

SimulatedPiece:
    - PieceType, Team, Position
    - CurrentHp, MaxHp, BaseDamage
    - No visual components
```

---

## Position Evaluation

```csharp
Evaluate(state, sideToMove):
    score = 0

    // Material + HP
    For each piece:
        value = PieceValue + CurrentHp × 2
        score += (piece.Team == sideToMove) ? value : -value

    // Mobility
    myMoves = CountMoves(sideToMove)
    oppMoves = CountMoves(opponent)
    score += (myMoves - oppMoves) × 5

    // King Safety
    If my king attacked: score -= 200
    If opp king attacked: score += 150

    // Center Control
    score += CenterBonus(sideToMove)

    // Pawn Structure
    score += PawnStructure(sideToMove)

    return score
```

---

## Piece Values (Centipawn Scale)

| Piece | Value |
|-------|-------|
| King | 10000 |
| Queen | 900 |
| Rook | 500 |
| Bishop | 330 |
| Knight | 320 |
| Pawn | 100 |

---

## AI Depth Scaling

| Act | Depth | Behavior | Approx Think Time |
|-----|-------|----------|-------------------|
| 1 | 2-ply | Basic tactics | <0.5s |
| 2 | 3-ply | Sees your response | ~1s |
| 3 | 4-ply | Tactical combinations | ~2-3s |
| Boss | 5-ply | Deep strategic planning | ~3-5s |

Configure via `ActConfig.AiDepth` in MapTypes.cs.

---

## Heuristic Fallback

For depth 1 or as backup, the original heuristic system remains:

```
1. DEFENSIVE RESPONSE (if valuable piece under attack)
   - Move threatened piece to safety
   - Counter-attack the attacker
   - Escape King if in danger

2. ATTACK EVALUATION
   - Material value of target
   - Kill probability bonus
   - KING attack = highest priority
   - Trade analysis (don't trade down)
   - Undefended target bonus

3. MOVE EVALUATION
   - Safety (don't move into danger)
   - Escape from attack
   - Positional value (center control)
   - Piece coordination (defend allies)
   - Pawn structure (passed/connected/isolated)
   - Attack creation (new threats from position)

4. ABILITY EVALUATION
   - Royal Decree: Based on number of attackers
   - Consecration: Based on ally HP deficit
   - Advance: Only if safe destination
```

---

## Debug Logging

```
[AI] === ExecuteTurn START === (Depth: 4, Heuristic: False)
[AI] Starting 4-ply minimax search...
[AI] Depth 1: Queen b4->d6 = 230 (1247 nodes, 0 cache hits, 89 cutoffs)
[AI] Depth 2: Rook a7->a3 = 415 (3892 nodes, 423 cache hits, 1203 cutoffs)
[AI] Depth 3: Rook a7->a3 = 380 (8234 nodes, 2891 cache hits, 4521 cutoffs)
[AI] Depth 4: Bishop c8->f5 = 445 (12893 nodes, 6234 cache hits, 8923 cutoffs)
[AI] Search complete: 12893 total nodes, 6234 cache hits, 8923 cutoffs
[AI] Best: Bishop Attack -> f5 (score: 445, Attack Knight)
```

---

## Future Improvements

- [x] True N-ply minimax with alpha-beta pruning
- [x] Transposition tables for repeated positions
- [x] MVV-LVA move ordering
- [x] Incremental cache reuse
- [ ] Opening book for standard openings
- [ ] Endgame tablebase for King+piece endings
- [ ] Monte Carlo tree search for complex positions
- [ ] Quiescence search (extend search for captures)
- [ ] Null move pruning
- [ ] Late move reductions
