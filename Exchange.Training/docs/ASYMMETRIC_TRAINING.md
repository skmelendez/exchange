# EXCHANGE AI: Asymmetric Training Architecture

## Overview

Asymmetric training uses a **Teacher-Student** approach where MCTS (Monte Carlo Tree Search) plays one side and a 1-ply neural network plays the other, alternating roles between games.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ASYMMETRIC TRAINING FLOW                             │
└─────────────────────────────────────────────────────────────────────────────┘

     GAME 0                          GAME 1                          GAME 2
  ┌──────────┐                    ┌──────────┐                    ┌──────────┐
  │  WHITE   │                    │  WHITE   │                    │  WHITE   │
  │  (MCTS)  │ ◄── Teacher        │(Network) │ ◄── Student        │  (MCTS)  │
  │ 50 sims  │                    │  1-ply   │                    │ 50 sims  │
  └────┬─────┘                    └────┬─────┘                    └────┬─────┘
       │                               │                               │
       ▼                               ▼                               ▼
  ┌──────────┐                    ┌──────────┐                    ┌──────────┐
  │  BLACK   │                    │  BLACK   │                    │  BLACK   │
  │(Network) │ ◄── Student        │  (MCTS)  │ ◄── Teacher        │(Network) │
  │  1-ply   │                    │ 50 sims  │                    │  1-ply   │
  └──────────┘                    └──────────┘                    └──────────┘
       │                               │                               │
       ▼                               ▼                               ▼
  Both sides                      Both sides                      Both sides
  contribute                      contribute                      contribute
  to training                     to training                     to training
```

---

## Why Asymmetric Training?

### The Self-Play Collapse Problem

Traditional self-play can degrade:
```
Iteration 0-10:   Network learns basic tactics
Iteration 10-20:  Network learns to counter itself
Iteration 20-30:  Both sides become overly defensive
Iteration 30+:    COLLAPSE - endless draws, repetitions
```

### Asymmetric Solution

```
┌─────────────────────────────────────────────────────────────────┐
│                    MCTS TEACHER (Stable)                        │
│  • Always searches 50 simulations ahead                         │
│  • Finds principled moves through tree search                   │
│  • Doesn't degrade with training                                │
│  • Provides consistent challenge                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼ Learns from
┌─────────────────────────────────────────────────────────────────┐
│                 NETWORK STUDENT (Improving)                     │
│  • Uses 1-ply lookahead only                                    │
│  • Learns to mimic MCTS quality without the search cost         │
│  • Improves each iteration                                      │
│  • Eventually can match MCTS strength with instant eval         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Complete Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING LOOP                                     │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────┐
  │   BOOTSTRAP PHASE   │ ◄─── Random games to initialize
  │   1000 games        │
  │   (random moves)    │
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                        ITERATION LOOP (x1000)                           │
  │                                                                         │
  │   ┌─────────────────────────────────────────────────────────────────┐   │
  │   │                   1. GENERATE GAMES                             │   │
  │   │                                                                 │   │
  │   │   100 games per iteration                                       │   │
  │   │   Each game has BOTH players: MCTS vs Network                   │   │
  │   │                                                                 │   │
  │   │   ┌─────────────────────────────────────────────────────────┐   │   │
  │   │   │  Game 0: MCTS=White vs Network=Black                    │   │   │
  │   │   │  Game 1: Network=White vs MCTS=Black                    │   │   │
  │   │   │  Game 2: MCTS=White vs Network=Black                    │   │   │
  │   │   │  ...alternating for all 100 games                       │   │   │
  │   │   │                                                         │   │   │
  │   │   │  Both sides contribute training data from EVERY game    │   │   │
  │   │   └─────────────────────────────────────────────────────────┘   │   │
  │   │                      │                                          │   │
  │   │                      ▼                                          │   │
  │   │            ┌─────────────────┐                                  │   │
  │   │            │ Training Data   │                                  │   │
  │   │            │ (state, value,  │                                  │   │
  │   │            │  policy_target) │                                  │   │
  │   │            └────────┬────────┘                                  │   │
  │   └─────────────────────┼───────────────────────────────────────────┘   │
  │                         ▼                                               │
  │   ┌─────────────────────────────────────────────────────────────────┐   │
  │   │                   2. TRAIN NETWORK                              │   │
  │   │                                                                 │   │
  │   │   Loss = MSE(value_pred, value_target)                          │   │
  │   │        + CrossEntropy(policy_pred, policy_target)               │   │
  │   │                                                                 │   │
  │   │   10 epochs per iteration                                       │   │
  │   │   Batch size: 256                                               │   │
  │   │   Learning rate: 1e-3 with cosine decay                         │   │
  │   └─────────────────────────────────────────────────────────────────┘   │
  │                         │                                               │
  │                         ▼                                               │
  │   ┌─────────────────────────────────────────────────────────────────┐   │
  │   │                   3. EVALUATE vs CHAMPION                       │   │
  │   │                                                                 │   │
  │   │   50 games: Current Network vs Champion Network                 │   │
  │   │   If win rate > 55%: Promote to new champion                    │   │
  │   └─────────────────────────────────────────────────────────────────┘   │
  │                         │                                               │
  │                         ▼                                               │
  │   ┌─────────────────────────────────────────────────────────────────┐   │
  │   │                   4. CHECKPOINT                                 │   │
  │   │                                                                 │   │
  │   │   Save: latest.pt, best.pt, iter_N.pt                           │   │
  │   │   Save: Replay files for analysis                               │   │
  │   └─────────────────────────────────────────────────────────────────┘   │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘
```

---

## Game State Tensor (75 Channels)

The neural network sees the board as a `[75, 8, 8]` tensor - 75 "heatmaps" stacked on top of each other.

### Channel Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TENSOR CHANNELS [75, 8, 8]                           │
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
                          PIECE POSITIONS (0-11)
═══════════════════════════════════════════════════════════════════════════════

  Channel 0: White King        ░░░░░░░░      1.0 where White King is
  Channel 1: White Queen       ░░░░░░░░      1.0 where White Queen is
  Channel 2: White Rook        ░░░░░░░░      1.0 where White Rooks are
  Channel 3: White Bishop      ░░░░░░░░      1.0 where White Bishops are
  Channel 4: White Knight      ░░░░░░░░      1.0 where White Knights are
  Channel 5: White Pawn        ░░░░░░░░      1.0 where White Pawns are

  Channel 6: Black King        ░░░░░░░░      Same for Black pieces
  Channel 7: Black Queen       ░░░░░░░░
  Channel 8: Black Rook        ░░░░░░░░
  Channel 9: Black Bishop      ░░░░░░░░
  Channel 10: Black Knight     ░░░░░░░░
  Channel 11: Black Pawn       ░░░░░░░░

  Example (Channel 0 - White King at e1):
  ┌─────────────────┐
  │ 0 0 0 0 0 0 0 0 │  8
  │ 0 0 0 0 0 0 0 0 │  7
  │ 0 0 0 0 0 0 0 0 │  6
  │ 0 0 0 0 0 0 0 0 │  5
  │ 0 0 0 0 0 0 0 0 │  4
  │ 0 0 0 0 0 0 0 0 │  3
  │ 0 0 0 0 0 0 0 0 │  2
  │ 0 0 0 0 1 0 0 0 │  1
  └─────────────────┘
    a b c d e f g h

═══════════════════════════════════════════════════════════════════════════════
                          HP INFORMATION (12-17)
═══════════════════════════════════════════════════════════════════════════════

  Channel 12-17: HP ratio per piece type (0.0 to 1.0)
                 current_hp / max_hp at each piece location

  Example (Channel 12 - King HP, if King has 15/25 HP):
  ┌─────────────────┐
  │ 0 0 0 0 0 0 0 0 │
  │ ... ... ... ... │
  │ 0 0 0 0 .6 0 0 0│  ◄── 0.6 = 15/25 HP
  └─────────────────┘

═══════════════════════════════════════════════════════════════════════════════
                        ABILITY COOLDOWNS (18-23)
═══════════════════════════════════════════════════════════════════════════════

  Channel 18-23: Cooldown ratio per piece type (0.0 to 1.0)
                 current_cooldown / max_cooldown
                 OR 1.0 if one-time ability already used

  Example (Channel 18 - King ability cooldown):
  • King's Royal Decree has no cooldown (3 uses per match)
  • Shows 0.0 if uses remaining, 1.0 if depleted

═══════════════════════════════════════════════════════════════════════════════
                        SPECIAL FLAGS (24-27)
═══════════════════════════════════════════════════════════════════════════════

  Channel 24: Royal Decree Active
              1.0 on ALL squares if current side has RD buff active
              0.0 otherwise
              (Uniform fill - tells network "we have +2 damage bonus!")

  Channel 25: Interpose Positions
              1.0 where Rooks have Interpose shield active
              (Can redirect attacks to this Rook)

  Channel 26: Side to Move
              1.0 on ALL squares if White to move
              0.0 on ALL squares if Black to move

  Channel 27: Playing As
              1.0 on ALL squares if we are White
              0.0 on ALL squares if we are Black

═══════════════════════════════════════════════════════════════════════════════
                       STRATEGIC CHANNELS (28-45)
═══════════════════════════════════════════════════════════════════════════════

  Channel 28: Turn Number (normalized)
              turn_number / 300, capped at 1.0
              Helps network understand game phase

  Channel 29: Moves Without Damage
              moves_without_damage / 30
              High = stalemate risk, need to be aggressive

  Channel 30: Position Repetition
              repetition_count / 3
              1.0 = threefold draw imminent!

  Channel 31: HP Ratio
              white_total_hp / (white_hp + black_hp)
              0.5 = even, >0.5 = white ahead

  Channel 32: Piece Count Ratio
              white_pieces / total_pieces

  Channel 33: Material Difference
              Normalized material balance (-1 to +1 scaled to 0-1)

  Channel 34: White Attack Map
              1.0 on squares White can attack
              ┌─────────────────┐
              │ 0 0 1 1 1 0 0 0 │
              │ 0 1 0 1 0 1 0 0 │
              │ 1 0 0 1 0 0 1 0 │  ◄── Shows all squares
              │ 0 0 0 ♕ 0 0 0 1 │      under White attack
              └─────────────────┘

  Channel 35: Black Attack Map
              Same for Black

  Channel 36: Contested Squares
              1.0 where BOTH sides attack (tension!)

  Channel 37: White King Zone (3x3 around King)
  Channel 38: Black King Zone

  Channel 39: White Pawn Advancement (y/7 for each pawn)
  Channel 40: Black Pawn Advancement

  Channel 41: Mobility White (# of legal moves per piece, normalized)
  Channel 42: Mobility Black

  Channel 43: Threats to White (enemy attacks on our pieces)
  Channel 44: Threats to Black

  Channel 45: Center Control
              Higher values for central squares under control

═══════════════════════════════════════════════════════════════════════════════
                       TACTICAL CHANNELS (46-57)
═══════════════════════════════════════════════════════════════════════════════

  Channel 46: Hanging White Pieces
              1.0 on undefended White pieces under attack (DANGER!)

  Channel 47: Hanging Black Pieces

  Channel 48: Defended White Pieces
              1.0 on White pieces protected by allies

  Channel 49: Defended Black Pieces

  Channel 50: White King Attackers
              Count of Black pieces attacking White king zone

  Channel 51: Black King Attackers

  Channel 52: Safe Squares White
              Squares White can move to without being captured

  Channel 53: Safe Squares Black

  Channel 54: Passed Pawns White
              Pawns with no enemy pawns blocking promotion

  Channel 55: Passed Pawns Black

  Channel 56: Open Files
              Files with no pawns (Rook highways!)

  Channel 57: Damage Potential
              Net damage balance per square
              white_damage_here - black_damage_here (normalized)

═══════════════════════════════════════════════════════════════════════════════
                      ABILITY TRACKING (58-61)
═══════════════════════════════════════════════════════════════════════════════

  Channel 58: Ability Charges Remaining
              uses_remaining / max_uses at each piece location
              Helps track King's 3 Royal Decree charges, etc.

  Channel 59: Royal Decree Turns Remaining
              0.0 = inactive, 0.5 = 1 turn left, 1.0 = 2 turns left
              (Uniform fill across board)

  Channel 60: RD Combo Potential
              1.0 on pieces that can attack AND have RD buff active
              "These pieces should attack NOW for bonus damage!"

  Channel 61: Promoted Pieces
              1.0 on Queens that were promoted from Pawns
              (May have different behavior/value)

═══════════════════════════════════════════════════════════════════════════════
                     NEW STRATEGIC CHANNELS (62-74)
═══════════════════════════════════════════════════════════════════════════════

  Channel 62: Threat Map White
              1.0 on White pieces that can be attacked next turn
              Highlights vulnerable pieces needing protection

  Channel 63: Threat Map Black
              Same for Black pieces

  Channel 64: Interpose Coverage White
              1.0 on White pieces within Rook's reactive Interpose range
              (Orthogonal LOS, max 3 squares from ally Rook)
              "These pieces have Rook backup if attacked!"

  Channel 65: Interpose Coverage Black
              Same for Black pieces

  Channel 66: Consecration Targets
              1.0 on ally pieces within Bishop heal range AND damaged
              Efficiency = HP that would be healed / HP currently missing

  Channel 67: Forcing Moves
              1.0 on squares where we have forcing/must-respond threats
              Checks, winning captures, promotion threats

  Channel 68: RD Combo Enhanced
              1.0 on pieces that can attack AND have RD buff active
              Enhanced version of channel 60 with better targeting

  Channel 69: Pawn Promotion Distance
              (8 - distance_to_promotion) / 7 per pawn location
              Higher = closer to promotion (0.86 for 1 square away)

  Channel 70: Enemy RD Turns
              enemy_rd_turns_remaining / 2.0 (uniform fill)
              Lets network anticipate enemy's buffed attacks

  Channel 71: Enemy Abilities Ready
              1.0 on enemy pieces with ability off cooldown
              "Watch out for these pieces - they can use abilities!"

  Channel 72: King Proximity Map
              (8 - distance_to_enemy_king) / 7 per piece
              Higher = closer to enemy king = better attacking position

  Channel 73: Safe Attack Squares
              1.0 on squares we can attack without counter-attack risk
              Calculated as: attack_map AND NOT(enemy_attack_map)

  Channel 74: Enemy Ability Charges
              charges_remaining / max_charges at each enemy piece
              Tracks enemy King's RD uses, etc.
```

---

## Network Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     POLICY-VALUE NETWORK (21M params)                       │
└─────────────────────────────────────────────────────────────────────────────┘

                              INPUT
                          [batch, 75, 8, 8]
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Input Convolution   │
                    │   75 → 256 channels   │
                    │   3x3 kernel          │
                    │   + BatchNorm + ReLU  │
                    └───────────┬───────────┘
                                │
                                ▼
              ┌─────────────────────────────────────┐
              │         RESIDUAL TOWER              │
              │                                     │
              │   ┌─────────────────────────────┐   │
              │   │     Residual Block x12      │   │
              │   │                             │   │
              │   │  ┌─────────────────────┐    │   │
              │   │  │ Conv 3x3 + BN + ReLU│    │   │
              │   │  └──────────┬──────────┘    │   │
              │   │             │               │   │
              │   │  ┌──────────▼──────────┐    │   │
              │   │  │ Conv 3x3 + BN       │    │   │
              │   │  └──────────┬──────────┘    │   │
              │   │             │               │   │
              │   │       ┌─────┴─────┐         │   │
              │   │       │  + Skip   │         │   │
              │   │       │ Connection│         │   │
              │   │       └─────┬─────┘         │   │
              │   │             │               │   │
              │   │        ReLU ▼               │   │
              │   └─────────────────────────────┘   │
              │                                     │
              └──────────────────┬──────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
        ┌───────────────────┐     ┌───────────────────┐
        │    VALUE HEAD     │     │   POLICY HEAD     │
        │                   │     │                   │
        │ Conv 1x1 → 1 ch   │     │ Conv 1x1 → 32 ch  │
        │ + BatchNorm       │     │ + BatchNorm       │
        │ + ReLU            │     │ + ReLU            │
        │       │           │     │       │           │
        │ Flatten (64)      │     │ Conv 1x1 → 73 ch  │
        │       │           │     │       │           │
        │ FC 64 → 256       │     │ Flatten (73×64)   │
        │ + ReLU            │     │       │           │
        │       │           │     │ FC → 4608         │
        │ FC 256 → 1        │     │       │           │
        │       │           │     │ Log-Softmax       │
        │    tanh           │     │                   │
        └───────┬───────────┘     └───────┬───────────┘
                │                         │
                ▼                         ▼
            VALUE                     POLICY
           [-1, +1]              [4608 move priors]

        "How good is         "Probability of each
         this position?"      move being best"
```

---

## Move Selection

### MCTS Side (Teacher)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MCTS SEARCH (50 sims)                             │
└─────────────────────────────────────────────────────────────────────────────┘

                         Current Position
                               │
                               ▼
                    ┌──────────────────────┐
                    │     ROOT NODE        │
                    │   N=50 (visits)      │
                    │   Q=0.3 (avg value)  │
                    └──────────┬───────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
    ┌───────────┐        ┌───────────┐        ┌───────────┐
    │  Move A   │        │  Move B   │        │  Move C   │
    │  N=20     │        │  N=25     │        │  N=5      │
    │  Q=0.4    │        │  Q=0.35   │        │  Q=-0.2   │
    └───────────┘        └───────────┘        └───────────┘
          │
          ▼
    ┌───────────────────────────────────────────────────────┐
    │                    SELECTION                          │
    │                                                       │
    │   PUCT = Q + c_puct * P * sqrt(N_parent) / (1 + N)    │
    │                                                       │
    │   Balance: exploitation (Q) vs exploration (P, N)     │
    └───────────────────────────────────────────────────────┘
          │
          ▼
    ┌───────────────────────────────────────────────────────┐
    │   After 50 simulations, select move with most visits  │
    │                                                       │
    │   Visit distribution → Policy Target for training     │
    │   [0.4, 0.5, 0.1, ...] (normalized visits)            │
    └───────────────────────────────────────────────────────┘
```

### 1-Ply Side (Student)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         1-PLY EVALUATION                                    │
└─────────────────────────────────────────────────────────────────────────────┘

                         Current Position
                               │
                               ▼
            ┌──────────────────────────────────────┐
            │       Generate All Legal Moves       │
            │       [Move A, Move B, Move C, ...]  │
            └──────────────────┬───────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
    ┌───────────┐        ┌───────────┐        ┌───────────┐
    │ Position  │        │ Position  │        │ Position  │
    │ after A   │        │ after B   │        │ after C   │
    └─────┬─────┘        └─────┬─────┘        └─────┬─────┘
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │
                               ▼
            ┌──────────────────────────────────────┐
            │    BATCH GPU EVALUATION              │
            │                                      │
            │    Network(positions) → values       │
            │    [-0.3, 0.5, 0.1, ...]             │
            │                                      │
            │    (Negated - opponent's view)       │
            └──────────────────┬───────────────────┘
                               │
                               ▼
            ┌──────────────────────────────────────┐
            │    SELECT BEST MOVE (greedy)         │
            │                                      │
            │    argmax([0.3, -0.5, -0.1, ...])    │
            │           ▲                          │
            │           └── Move A wins            │
            │                                      │
            │    Softmax(values) → Policy Target   │
            └──────────────────────────────────────┘
```

---

## Training Data Collection

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PER-MOVE DATA COLLECTION                                 │
└─────────────────────────────────────────────────────────────────────────────┘

     Each position in the game generates a training tuple:

     ┌─────────────────────────────────────────────────────────────────────┐
     │                                                                     │
     │   ( state_tensor,   value_target,   policy_target )                 │
     │                                                                     │
     │      [75,8,8]         float           [4608]                        │
     │                                                                     │
     │   Board state      Game outcome     Move distribution               │
     │   at this turn     from this        from MCTS visits                │
     │                    position         or 1-ply softmax                │
     │                                                                     │
     └─────────────────────────────────────────────────────────────────────┘


     VALUE TARGET CALCULATION (New Reward System):
     ┌─────────────────────────────────────────────────────────────────────┐
     │                                                                     │
     │   Base outcomes:                                                    │
     │   • Win:  +1.0 to +1.5 (early win bonus before turn 50)             │
     │   • Loss: -1.0                                                      │
     │   • Draw: -0.5 (only for insufficient material draws)               │
     │                                                                     │
     │   Per-Move Rewards (tracked via AbilityTracker):                    │
     │   • Damage dealt: +0.01 * damage * (1 + turn/100) [scales to 2x]    │
     │   • Attack action: +0.02 per attack                                 │
     │   • RD combo: +0.15 per attack during Royal Decree (max +0.60)      │
     │   • RD waste: -0.10 if RD expires with zero attacks                 │
     │   • King proximity: +0.02 per square closer to enemy King           │
     │   • King attack opportunity: +0.05 per piece threatening King       │
     │   • Interpose blocked: +0.02 per damage absorbed by Rook            │
     │   • Consecration: +0.10 * heal_efficiency                           │
     │   • Skirmish: +0.03 per use                                         │
     │   • Pawn advance: +0.02 per advance, +0.20 for promotion            │
     │                                                                     │
     │   Shuffle Penalty (exponential):                                    │
     │   • penalty = 0.01 * (2 ^ consecutive_no_damage_moves)              │
     │   • Capped at 1.0 (equivalent to loss)                              │
     │   • 5 moves = 0.32, 10 moves = 1.0 (capped)                         │
     │                                                                     │
     │   Final value flipped based on which side the position was for      │
     │                                                                     │
     └─────────────────────────────────────────────────────────────────────┘
```

---

## Batching Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      EFFICIENT BATCHED EVALUATION                           │
└─────────────────────────────────────────────────────────────────────────────┘

     100 games running in parallel, each at different positions
     Each game has MCTS on one side, Network on the other

     Step 1: CATEGORIZE BY WHOSE TURN IT IS
     ┌────────────────────────────────────────────────────────────────────┐
     │                                                                    │
     │   For each active game, check: Is it MCTS's turn or Network's?    │
     │                                                                    │
     │   Game 0 (MCTS=White): If White to move → MCTS turn               │
     │   Game 0 (MCTS=White): If Black to move → Network turn            │
     │   Game 1 (MCTS=Black): If White to move → Network turn            │
     │   Game 1 (MCTS=Black): If Black to move → MCTS turn               │
     │                                                                    │
     │   Result: Some games need MCTS search, others need 1-ply eval     │
     │                                                                    │
     └────────────────────────────────────────────────────────────────────┘

     Step 2: BATCH MCTS (for all games where it's MCTS's turn)
     ┌────────────────────────────────────────────────────────────────────┐
     │                                                                    │
     │   Collect all simulators where current side = MCTS side            │
     │   mcts.search_batch([sim_game0, sim_game3, sim_game7...], sims=50) │
     │                                                                    │
     │   All MCTS-turn positions searched together in Rust                │
     │   Network calls batched across all tree expansions                 │
     │                                                                    │
     └────────────────────────────────────────────────────────────────────┘

     Step 3: BATCH 1-PLY (for all games where it's Network's turn)
     ┌────────────────────────────────────────────────────────────────────┐
     │                                                                    │
     │   For each game where current side ≠ MCTS side:                    │
     │     Generate all legal moves → child positions                     │
     │                                                                    │
     │   Collect ALL child positions from ALL Network-turn games:         │
     │   [game2_move1, game2_move2, ..., game5_move1, game5_move2, ...]   │
     │                                                                    │
     │   SINGLE GPU BATCH:                                                │
     │   network(all_positions) → all_values                              │
     │                                                                    │
     │   Distribute values back to respective games, pick best move       │
     │                                                                    │
     └────────────────────────────────────────────────────────────────────┘

     Step 4: EXECUTE MOVES & REPEAT
     ┌────────────────────────────────────────────────────────────────────┐
     │                                                                    │
     │   Each game executes its selected move (from MCTS or 1-ply)        │
     │   Record training data: (state, policy_target)                     │
     │   Track combo attacks if Royal Decree was active                   │
     │                                                                    │
     │   Loop back to Step 1 until all games finish                       │
     │   Games finish when: checkmate, draw, or max turns                 │
     │                                                                    │
     │   Note: In a single iteration, a game alternates ~60-100 times     │
     │   between MCTS turns and Network turns                             │
     │                                                                    │
     └────────────────────────────────────────────────────────────────────┘
```

---

## Royal Decree Combo Tracking

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        COMBO BONUS SYSTEM (TRIPLED!)                        │
└─────────────────────────────────────────────────────────────────────────────┘

     Royal Decree: King ability that gives +2 damage to ALL allied attacks
                   for 2 turns (3 uses per match)

     PROBLEM: AI activates Royal Decree but doesn't attack during it!

     SOLUTION: Track and reward "combo attacks" (now with 3x bonus!)

     ┌────────────────────────────────────────────────────────────────────┐
     │   Before each move:                                                │
     │     rd_active = sim.is_royal_decree_active(current_team)           │
     │                                                                    │
     │   After move:                                                      │
     │     if rd_active AND damage_dealt > 0:                             │
     │       combo_attacks[team] += 1                                     │
     │                                                                    │
     │   At game end:                                                     │
     │     white_bonus = min(white_combos * 0.15, 0.6)   # TRIPLED!       │
     │     black_bonus = min(black_combos * 0.15, 0.6)                    │
     │     value_adjustment = white_bonus - black_bonus                   │
     │                                                                    │
     │   RD WASTE PENALTY (NEW):                                          │
     │     If RD expires with zero attacks: -0.10 penalty                 │
     │     Tracked per activation via AbilityTracker                      │
     └────────────────────────────────────────────────────────────────────┘

     Example:
       White uses Royal Decree, then attacks 4 times during buff
       Black uses Royal Decree, attacks only once

       white_bonus = min(4 * 0.15, 0.6) = 0.60
       black_bonus = min(1 * 0.15, 0.6) = 0.15
       net_bonus = 0.60 - 0.15 = +0.45 for White

       All White positions in this game get +0.45 value boost
       All Black positions get -0.45 value boost

       Network learns: "Attack while Royal Decree is active!"
```

---

## Reactive Interpose Mechanic

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      INTERPOSE REWORK (REACTIVE)                            │
└─────────────────────────────────────────────────────────────────────────────┘

     OLD MECHANIC (Proactive):
       • Rook chooses target to protect at start of turn
       • Must commit before knowing what enemy attacks
       • Rarely used, hard to optimize

     NEW MECHANIC (Reactive Interrupt):
       • Automatically triggers when ally within range is attacked
       • 50/50 damage split between target and Rook
       • No action cost - triggers reactively

     ┌────────────────────────────────────────────────────────────────────┐
     │   TRIGGER CONDITIONS:                                              │
     │   1. Ally within 3 squares takes damage                            │
     │   2. Rook has orthogonal line of sight to ally (same row/column)   │
     │   3. No blocking pieces in path                                    │
     │   4. Rook has Interpose charges remaining                          │
     │   5. Interpose not on cooldown (3 turn cooldown after trigger)     │
     │                                                                    │
     │   EFFECT:                                                          │
     │   • 50% damage goes to original target                             │
     │   • 50% damage goes to Interposing Rook                            │
     │   • Rook enters 3-turn cooldown                                    │
     │                                                                    │
     │   TRAINING REWARD:                                                 │
     │   • +0.02 per damage point absorbed by Rook                        │
     │   • Tracked via AbilityTracker.interpose_damage_blocked            │
     └────────────────────────────────────────────────────────────────────┘

     Channel 64/65 (Interpose Coverage):
       Shows which pieces are within Rook's protection range
       Helps network understand defensive positioning value
```

---

## Draw Rules (Simplified)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DRAW CONDITIONS (UPDATED)                           │
└─────────────────────────────────────────────────────────────────────────────┘

     REMOVED (was causing 41% draw rate):
       ✗ 30-move no-damage draw rule

     REMAINING DRAW CONDITIONS:
       ✓ Insufficient material (only Kings remain)
       ✓ Threefold position repetition
       ✓ Max turns reached (300)

     SHUFFLE PENALTY replaces passive draw:
       • Exponential penalty for consecutive no-damage moves
       • penalty = 0.01 * (2 ^ consecutive_no_damage_moves)
       • Capped at 1.0 (equivalent to loss)
       • Incentivizes aggressive play without hard draw cutoff
```

---

## AbilityTracker Dataclass

```python
@dataclass
class AbilityTracker:
    """Tracks ability usage and rewards per side."""

    # Royal Decree
    rd_activations: int = 0           # Times RD was activated
    rd_combo_attacks: int = 0         # Attacks during active RD
    rd_wasted: int = 0                # RD expired with 0 attacks

    # Interpose (Rook)
    interpose_triggers: int = 0       # Times Interpose activated
    interpose_damage_blocked: int = 0 # Total damage absorbed

    # Consecration (Bishop)
    consecration_uses: int = 0        # Times heal was used
    consecration_hp_healed: int = 0   # Total HP restored
    consecration_efficiency: float = 0.0  # healed / missing ratio

    # Skirmish (Knight)
    skirmish_uses: int = 0            # Times Skirmish used

    # Overextend (Queen)
    overextend_uses: int = 0          # Times Overextend used
    overextend_net_damage: int = 0    # damage dealt - 2 (self cost)

    # Pawn
    pawn_advances: int = 0            # Forward pawn moves
    pawn_promotions: int = 0          # Pawns promoted to Queen

    # King Hunt
    king_proximity_delta: int = 0     # Net squares moved closer to enemy King
    king_attack_opportunities: int = 0 # Pieces that can attack enemy King

    def calculate_total_reward(self, config: RewardConfig) -> float:
        """Sum all ability-specific rewards."""
        total = 0.0

        # RD rewards
        total += self.rd_activations * config.rd_activation_reward
        total += min(self.rd_combo_attacks * config.rd_combo_bonus_per_attack,
                     config.rd_combo_bonus_max)
        total -= self.rd_wasted * config.rd_waste_penalty

        # Ability rewards
        total += self.interpose_damage_blocked * config.interpose_damage_blocked_reward
        total += self.consecration_efficiency * config.consecration_efficiency_reward
        total += self.skirmish_uses * config.skirmish_reward
        total += self.overextend_net_damage * config.overextend_net_damage_reward

        # Pawn rewards
        total += self.pawn_advances * config.pawn_advance_reward
        total += self.pawn_promotions * config.pawn_promotion_reward

        # King hunt rewards
        total += self.king_proximity_delta * config.king_proximity_reward
        total += self.king_attack_opportunities * config.king_attack_opportunity_reward

        return total
```

---

## Command Reference

```bash
# Start asymmetric training
python3 scripts/train.py \
  --preset medium \
  --iterations 1000 \
  --games 100 \
  --asymmetric \
  --asymmetric-sims 50 \
  --output-dir runs/experiment_asymmetric

# Resume from checkpoint
python3 scripts/train.py \
  --asymmetric \
  --resume runs/experiment_asymmetric/checkpoints/latest.pt

# Disable combo bonus (not recommended)
python3 scripts/train.py --asymmetric --no-combo-bonus

# Adjust combo bonus
python3 scripts/train.py --asymmetric \
  --combo-bonus-per-attack 0.10 \
  --combo-bonus-max 0.5
```

---

## Expected Outcomes

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PROGRESSION                                │
└─────────────────────────────────────────────────────────────────────────────┘

  Iteration   MCTS Win%   Network Win%   Draw%   Notes
  ─────────   ─────────   ────────────   ─────   ─────────────────────────────
  0-50        ~70%        ~20%           ~10%    MCTS dominates (expected)
  50-100      ~60%        ~30%           ~10%    Network improving
  100-200     ~55%        ~40%           ~5%     Network catching up
  200-500     ~50%        ~48%           ~2%     Near parity
  500+        ~48%        ~50%           ~2%     Network matches MCTS quality

  Key metrics to watch (with new system):
  • MCTS should win more early (it's the teacher)
  • Network win rate should increase over time
  • Draw rate should stay VERY LOW (<10%, was 41% before!)
  • Game length: 80-100 turns (was 188 turns before!)
  • RD combo utilization: >50%
  • Interpose triggers: >30%
  • Consecration usage: >20%
  • Pawn promotions: >15%

  Ability Usage Targets:
  ┌────────────────────────────────────────────────────────────────────────┐
  │  Ability       │ Old Usage  │ Target Usage │ Reward Driver            │
  ├────────────────┼────────────┼──────────────┼──────────────────────────┤
  │  Royal Decree  │  ~10%      │  >50%        │ +0.15/attack, -0.10 waste│
  │  Interpose     │  ~5%       │  >30%        │ +0.02/dmg blocked        │
  │  Consecration  │  ~5%       │  >20%        │ +0.10 * efficiency       │
  │  Skirmish      │  ~40%      │  >60%        │ +0.03/use                │
  │  Overextend    │  ~30%      │  >50%        │ +0.05 * net damage       │
  │  Pawn Advance  │  ~8%       │  >15%        │ +0.02/advance, +0.20 promo│
  └────────────────────────────────────────────────────────────────────────┘
```
