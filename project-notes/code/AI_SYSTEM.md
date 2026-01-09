# AI System

## Overview

The AI uses a heuristic-based decision system with minimax-style lookahead. Currently implements 2-ply evaluation with plans for variable depth per act.

---

## Decision Priority

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

## Position Evaluation

```
EvaluatePosition(forTeam) {
    score = 0

    // Material
    For each enemy piece: score += PieceValue + (CurrentHP * 2)
    For each player piece: score -= PieceValue + (CurrentHP * 2)

    // King Safety
    If enemy King attacked: score -= 300
    If player King attacked: score += 200

    // Mobility
    enemyMoves = sum of valid moves for all enemy pieces
    playerMoves = sum of valid moves for all player pieces
    score += (enemyMoves - playerMoves) * 5

    return forTeam == Enemy ? score : -score
}
```

---

## Lookahead Analysis

### Attack Lookahead
```
For each attack:
    score = targetValue

    If attacker is also under attack:
        If our piece worth more than target:
            score -= (ourValue - targetValue)  // Bad trade
        Else:
            score += 100  // Good trade
```

### Move Lookahead
```
For each move:
    toSafe = !IsSquareAttackedBy(dest, Player)
    fromSafe = !IsSquareAttackedBy(current, Player)

    If moving safe → danger:
        score -= pieceValue
    If moving danger → danger:
        score -= 30
    If escaping (danger → safe):
        score += pieceValue / 2
```

### Attack Creation
```
SimulateAttacksFromPosition(piece, newPos):
    Count how many enemy pieces would be attackable
    Each new attack = +30 points
```

---

## Positional Heuristics

### Center Control
- Distance to center (d3/d4/e3/e4) matters
- Closer = better, bonus = (3 - distance) * 10

### Development (Early Game)
- Knights/Bishops still on back rank = +40 to develop

### King Safety
- King advancing before turn 30 = -50

### Pawn Structure
- Advancement = +15 per row
- Passed pawn = +40
- Doubled pawn = -30
- Connected pawns = +20
- Isolated pawn = -20

### Piece Coordination
- Defending allies = +15 per piece
- Rook on open file = +30
- Connected rooks = +25
- Bishop pair = +30
- Knight outpost (defended by pawn) = +35

---

## AI Depth Scaling

| Act | Depth | Behavior |
|-----|-------|----------|
| 1 | 2-ply | Basic tactics, some blunders |
| 2 | 3-ply | Sees your response |
| 3 | 4-ply | Deep strategic planning |
| Final | 4-ply+ | Plus boss mechanics |

*Note: Full minimax not yet implemented - currently uses heuristic 2-ply.*

---

## Decision Record

```csharp
AIDecision {
    Piece: BasePiece      // Which piece acts
    Action: ActionType    // Move/Attack/Ability
    Target: Vector2I?     // Destination/target position
    Score: int            // Evaluation score
    Reason: string        // Human-readable explanation
}
```

Output example:
```
[AI] Best: Queen Attack -> e4 (score: 420, capture Rook, likely kill, good trade)
```

---

## Future Improvements

- [ ] True N-ply minimax with alpha-beta pruning
- [ ] Transposition tables for repeated positions
- [ ] Opening book for standard openings
- [ ] Endgame tablebase for King+piece endings
- [ ] Monte Carlo tree search for complex positions
