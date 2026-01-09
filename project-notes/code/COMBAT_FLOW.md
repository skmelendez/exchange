# Combat Flow

## Turn Sequence

```
┌─────────────────────────────────────────────────────────┐
│                    MATCH START                           │
│  GameState.ResetForNewMatch()                           │
│  All pieces: ResetForNewMatch()                         │
│  RecalculateThreatZones()                               │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    TURN START                            │
│  1. Tick cooldowns for current team                     │
│  2. Reset turn state (EnteredThreatZone, HasActed)      │
│  3. Reset Pawn UsedAdvanceLastTurn                      │
│  4. Clear Royal Decree from previous turn               │
│  5. Apply Boss Rules if applicable                      │
│  6. Emit TurnStarted event                              │
│  7. If Enemy turn → AIController.ExecuteTurn()          │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   PLAYER INPUT                           │
│  - Click to select piece                                │
│  - Click to move/attack                                 │
│  - Press E for ability                                  │
│  - Press Space (Knight only: end turn after move)       │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   ACTION EXECUTION                       │
│  ExecuteMove() OR ExecuteAttack() OR ExecuteAbility()   │
│  - Check/set EnteredThreatZoneThisTurn                  │
│  - Update board state                                   │
│  - RecalculateThreatZones()                             │
│  - Check win condition (King death)                     │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                     TURN END                             │
│  1. Mark piece HasActedThisTurn                         │
│  2. Check King threat status (for next turn bonus)      │
│  3. Apply Boss Rule 5 if applicable                     │
│  4. Switch CurrentPhase (Player ↔ Enemy)                │
│  5. Increment TurnNumber if back to Player              │
│  6. Clear selection state                               │
│  7. StartTurn() for next team                           │
└─────────────────────────────────────────────────────────┘
```

---

## Combat Resolution

```
CombatResolver.ResolveAttack(attacker, defender, board)
    │
    ├── Determine Modifiers
    │   ├── Royal Decree active? (+1)
    │   ├── Enemy King threatened? (+1)
    │   └── Entered threat zone? (-1 or -2 for Boss Room 1)
    │
    ├── DiceRoller.RollCombat()
    │   ├── Roll 1d6
    │   ├── Apply modifiers
    │   └── Clamp to 1-6
    │
    ├── Calculate Damage
    │   └── Total = BaseDamage + DiceResult
    │
    ├── Apply Boss Rules
    │   └── Room 2: Adjacent enemy allies take -1 damage
    │
    ├── Check Interpose (Rook ability)
    │   └── Split damage if adjacent Rook has InterposeActive
    │
    └── Apply Damage
        ├── defender.TakeDamage()
        └── Check if destroyed → emit PieceDestroyed
```

---

## Dice Roll Breakdown

```
DiceResult {
    RawRoll: 1-6 (actual die face)
    Modifiers: sum of all +/- adjustments
    FinalValue: clamped(RawRoll + Modifiers, 1, 6)
    Breakdown: "Roll: 4 | Threat Zone: -1 | Royal Decree: +1 | Final: 4"
}
```

---

## Multi-Step Abilities

Some abilities require multiple phases:

### Overextend (Queen)
```
1. OverextendSelectMove → Player picks move destination
2. ExecuteOverextendMove() → Queen moves
3. OverextendSelectAttack → Player picks attack target
4. ExecuteOverextendAttack() → Resolve attack
5. Queen takes 2 self-damage
6. CompleteOverextend() → End turn
```

### Skirmish (Knight)
```
1. SkirmishSelectAttack → Player picks adjacent enemy
2. ExecuteSkirmishAttack() → Resolve attack
3. SkirmishSelectReposition → Player picks 1-tile reposition
4. ExecuteSkirmishReposition() → Knight moves
5. CompleteSkirmish() → End turn
```

---

## Knight Special Rule

Knights can **move then attack** in the same turn (unique among pieces):

```
1. Select Knight
2. Move to new position (HasMovedThisTurn = true)
3. RecalculateAttacks from new position
4. If adjacent enemies exist:
   - Player can attack OR press Space to end turn
5. After attack (or Space): EndAction()
```

---

## Threat Zone Rules

- **Definition:** Any tile a piece could attack from current position
- **Visibility:** Shown with overlay on board
- **Penalty:** Moving into enemy threat zone = -1 to next combat roll
- **King Restriction:** King cannot move into threatened tiles (except Boss Rule 3)
- **King Threatened Bonus:** If your King is threatened at turn end, enemy gets +1 next attack
