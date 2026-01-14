# EXCHANGE: Signals, Channels, and Rewards Review

> Generated for comprehensive review before retraining. Use this to audit existing signals, identify gaps, and design improvements.

---

## Table of Contents
1. [Neural Network Input Channels (75 total)](#neural-network-input-channels)
2. [Piece Configuration](#piece-configuration)
3. [Reward & Penalty System](#reward--penalty-system)
4. [MCTS Configuration](#mcts-configuration)
5. [Training Configuration](#training-configuration)
6. [Analysis & Gaps](#analysis--gaps)

---

## Neural Network Input Channels

The network receives a `(75, 8, 8)` tensor. Each channel is an 8x8 plane encoding specific game information.

> **Updated**: Expanded from 62 to 75 channels to address gaps identified in review.

### Piece Position Channels (0-11)
| Channel | Name | Description | Format |
|---------|------|-------------|--------|
| 0 | White King | Binary position | 1.0 at king square |
| 1 | White Queen | Binary position | 1.0 at queen square(s) |
| 2 | White Rook | Binary positions | 1.0 at rook squares |
| 3 | White Bishop | Binary positions | 1.0 at bishop squares |
| 4 | White Knight | Binary positions | 1.0 at knight squares |
| 5 | White Pawn | Binary positions | 1.0 at pawn squares |
| 6 | Black King | Binary position | 1.0 at king square |
| 7 | Black Queen | Binary position | 1.0 at queen square(s) |
| 8 | Black Rook | Binary positions | 1.0 at rook squares |
| 9 | Black Bishop | Binary positions | 1.0 at bishop squares |
| 10 | Black Knight | Binary positions | 1.0 at knight squares |
| 11 | Black Pawn | Binary positions | 1.0 at pawn squares |

**Note**: Channels are one-hot by piece type. A square can only have one piece.

### HP Ratio Channels (12-17)
| Channel | Name | Description | Format |
|---------|------|-------------|--------|
| 12 | King HP | Health ratio | current_hp / max_hp (0.0-1.0) |
| 13 | Queen HP | Health ratio | current_hp / max_hp |
| 14 | Rook HP | Health ratio | current_hp / max_hp |
| 15 | Bishop HP | Health ratio | current_hp / max_hp |
| 16 | Knight HP | Health ratio | current_hp / max_hp |
| 17 | Pawn HP | Health ratio | current_hp / max_hp |

**Note**: HP is placed at the piece's position. Empty squares = 0.

### Ability Cooldown Channels (18-23)
| Channel | Name | Description | Format |
|---------|------|-------------|--------|
| 18 | King Cooldown | Ability cooldown | cooldown / max_cooldown |
| 19 | Queen Cooldown | Ability cooldown | cooldown / max_cooldown |
| 20 | Rook Cooldown | Ability cooldown | cooldown / max_cooldown |
| 21 | Bishop Cooldown | Ability cooldown | cooldown / max_cooldown |
| 22 | Knight Cooldown | Ability cooldown | cooldown / max_cooldown |
| 23 | Pawn Cooldown | Ability cooldown | cooldown / max_cooldown |

**Note**: For once-per-match abilities (cooldown_max = -1), value is 1.0 if used, 0.0 if available.

### Game State Channels (24-27)
| Channel | Name | Description | Format |
|---------|------|-------------|--------|
| 24 | Royal Decree Active | King's buff is active | Filled plane: 1.0 if active |
| 25 | Interpose Active | Rook shield positions | 1.0 at rook squares with interpose |
| 26 | Side to Move | Whose turn | Filled plane: 1.0 = White |
| 27 | Playing As | AI perspective | Filled plane: 1.0 = White |

### Strategic Feature Channels (28-45)
| Channel | Name | Description | Format |
|---------|------|-------------|--------|
| 28 | Turn Progress | Game length | turn_number / 300, clamped to 1.0 |
| 29 | Draw Progress | Damage drought | moves_without_damage / 30 |
| 30 | Repetition Count | Position repeats | count / 3, clamped to 1.0 |
| 31 | HP Balance | Health advantage | white_hp / total_hp |
| 32 | Piece Count Balance | Army size | white_pieces / total_pieces |
| 33 | Material Balance | Weighted value | (white_mat - black_mat) normalized |
| 34 | White Attacks | Attack map | 1.0 if White can attack square |
| 35 | Black Attacks | Attack map | 1.0 if Black can attack square |
| 36 | Contested Squares | Both attack | 1.0 if both sides attack |
| 37 | White King Zone | Safety zone | 1.0 in 3x3 around White king |
| 38 | Black King Zone | Safety zone | 1.0 in 3x3 around Black king |
| 39 | White Pawn Advancement | Pawn progress | y / 7 (0=home, 1=promotion) |
| 40 | Black Pawn Advancement | Pawn progress | (7 - y) / 7 |
| 41 | White Abilities Ready | Ready ratio | ready_count / piece_count |
| 42 | Black Abilities Ready | Ready ratio | ready_count / piece_count |
| 43 | Low HP Targets | Kill targets | 1.0 if piece HP < 40% |
| 44 | Center Control | Board control | Weighted center squares |
| 45 | King Tropism | King distance | 1.0 - manhattan_dist / 14 |

### Tactical Feature Channels (46-57)
| Channel | Name | Description | Format |
|---------|------|-------------|--------|
| 46 | White Hanging | Undefended & attacked | 1.0 at white piece attacked by black, not defended |
| 47 | Black Hanging | Undefended & attacked | 1.0 at black piece attacked by white, not defended |
| 48 | White Defended | Friendly protection | 1.0 at white pieces defended by white |
| 49 | Black Defended | Friendly protection | 1.0 at black pieces defended by black |
| 50 | White King Attackers | King pressure | attackers_count / 6 (filled plane) |
| 51 | Black King Attackers | King pressure | attackers_count / 6 (filled plane) |
| 52 | White King Safe Squares | Escape routes | 1.0 at safe adjacent squares |
| 53 | Black King Safe Squares | Escape routes | 1.0 at safe adjacent squares |
| 54 | White Passed Pawns | Promotion path clear | 1.0 at pawns with no blockers |
| 55 | Black Passed Pawns | Promotion path clear | 1.0 at pawns with no blockers |
| 56 | Open Files | Rook highways | 1.0 on files with no pawns |
| 57 | Damage Potential | Attack power | Sum of attacker damage per square |

### Ability Tracking Channels (58-61)
| Channel | Name | Description | Format |
|---------|------|-------------|--------|
| 58 | Ability Charges | Uses remaining | uses / max_uses (per piece) |
| 59 | Royal Decree Turns | Buff duration | turns_remaining / 2 |
| 60 | RD Combo Potential | Attack opportunity | 1.0 at pieces that can attack during RD |
| 61 | Promoted Pieces | Former pawns | 1.0 at Queens that were pawns |

### NEW Strategic Channels (62-74)
| Channel | Name | Description | Format |
|---------|------|-------------|--------|
| 62 | Threat Map White | White pieces under threat | 1.0 at attackable white pieces |
| 63 | Threat Map Black | Black pieces under threat | 1.0 at attackable black pieces |
| 64 | Interpose Coverage White | Protected by ally Rook | 1.0 if in Rook's reactive LOS (3 sq) |
| 65 | Interpose Coverage Black | Protected by ally Rook | 1.0 if in Rook's reactive LOS (3 sq) |
| 66 | Consecration Targets | Valid heal targets | 1.0 if in Bishop range AND damaged |
| 67 | Forcing Moves | Must-respond threats | 1.0 for checks, winning captures |
| 68 | RD Combo Enhanced | Enhanced attack targets | 1.0 at pieces with RD + attack available |
| 69 | Pawn Promotion Distance | Distance to promote | (8 - distance) / 7 per pawn |
| 70 | Enemy RD Turns | Enemy buff duration | enemy_turns / 2 (uniform fill) |
| 71 | Enemy Abilities Ready | Enemy threats | 1.0 at enemy pieces with ability ready |
| 72 | King Proximity Map | Distance to enemy King | (8 - manhattan_dist) / 7 per piece |
| 73 | Safe Attack Squares | Attack w/o counter | 1.0 at squares we attack, enemy doesn't |
| 74 | Enemy Ability Charges | Enemy resources | charges / max_charges per enemy piece |

---

## Piece Configuration

### Piece Stats
| Piece | Max HP | Base Damage | Cooldown Max | Ability Uses | Material Value |
|-------|--------|-------------|--------------|--------------|----------------|
| King | 25 | 1 | 0 | 3 | 0 (infinite) |
| Queen | 10 | 3 | 3 | Unlimited | 9 |
| Rook | 13 | 2 | 3 | 5 | 5 |
| Bishop | 10 | 2 | 3 | 3 | 3 |
| Knight | 11 | 2 | 3 | 5 | 3 |
| Pawn | 7 | 1 | 1 | Unlimited | 1 |

### Abilities
| Piece | Ability | Effect | Constraints |
|-------|---------|--------|-------------|
| King | Royal Decree | +2 damage for all allies for 2 turns | 3 uses per match |
| Queen | Overextend | Move again after attack (take 2 damage) | Unlimited, self-limiting |
| Rook | Interpose | Block next attack on adjacent ally | 5 uses, 3 turn cooldown |
| Bishop | Consecration | Heal all adjacent allies 2 HP | 3 uses, 3 turn cooldown |
| Knight | Skirmish | Move to different square after attack | 5 uses, 3 turn cooldown |
| Pawn | Advance | Move 2 squares forward | Unlimited, 1 turn cooldown |

---

## Reward & Penalty System

### Outcome Values

#### Win Value (with early-win bonus)
```
Turn 1-50:    +1.5 (50% bonus for quick wins)
Turn 50-100:  Linear decay from +1.5 to +1.0
Turn 100+:    +1.0 (base value)
```

**Rationale**: Incentivize aggressive, decisive play. Prolonged games suggest suboptimal play.

#### Loss Value
```
Standard:     -1.0 (loss_penalty config)
```

#### Draw Value (with early-draw penalty)
```
Turn 1-50:    -100.0 (CATASTROPHIC - never do this!)
Turn 50-150:  Linear decay from -100.0 to -0.8
Turn 150+:    -0.8 (base draw_penalty)
```

**Rationale**: Early draws indicate the AI found a "safe" strategy that avoids engagement. This is heavily punished.

#### Dynamic Contempt (Material-Based Draw Adjustment)
When drawing, penalty is adjusted based on material advantage:
```
Material Advantage > +30 HP:  -0.95 (threw away a win!)
Material Advantage > +15 HP:  -0.80 (should have closed)
Material Advantage > +5 HP:   -0.70 (base penalty)
Even position (±5 HP):        -0.60 (fight harder)
Material Disadvantage < -5:   -0.50 (decent save)
Material Disadvantage < -15:  -0.40 (good save)
Material Disadvantage < -30:  -0.30 (great save!)
```

### Move-Level Penalties

#### Repetition Penalty
```
Config:           repetition_penalty = 0.50
Per repeat:       -0.50 per occurrence in history
Effect:           2 repeats = -1.0 (effectively a loss!)
```

**Applied during**: Move selection (1-ply eval, not MCTS tree values)

#### Stalemate Penalty
```
Config:           stalemate_moves = 9999 (DISABLED)
Penalty:          -1.0 (same as loss)
```

**Note**: Currently disabled. Asymmetric training handles this better.

### Bonus Rewards

#### Combo Bonus (Royal Decree Attacks) - TRIPLED!
```
Config:           combo_bonus_enabled = True
Per attack:       +0.15 per attack while RD active (was 0.05)
Maximum:          +0.60 cap (was 0.30)
RD Waste Penalty: -0.10 if RD expires with 0 attacks (NEW)
```

**Rationale**: Strongly incentivize using the King's buff effectively. Triple reward makes RD combos highly valuable, waste penalty discourages defensive RD use.

#### King Hunt Rewards (NEW)
```
Proximity Delta:      +0.02 per square moved closer to enemy King
Attack Opportunity:   +0.05 per piece that can attack enemy King
```

**Rationale**: Encourage aggressive positioning toward the enemy King.

#### Ability-Specific Rewards (NEW via AbilityTracker)
```
Interpose Blocked:    +0.02 per damage absorbed by Rook
Consecration:         +0.10 * efficiency (healed / missing HP)
Skirmish:             +0.03 per use
Overextend:           +0.05 * (damage dealt - 2 self cost)
Pawn Advance:         +0.02 per advance
Pawn Promotion:       +0.20 per promotion
```

#### Shuffle Penalty (NEW)
```
Formula:         -0.01 * (2 ^ consecutive_no_damage_moves)
Maximum:         -1.0 (capped - equivalent to loss)
```

**Rationale**: Replaces 30-move draw rule. Exponential penalty makes shuffling increasingly painful.

---

## MCTS Configuration

### Search Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| num_simulations | 100 | Sims per move (training) |
| eval_simulations | 200 | Sims per move (evaluation) |
| c_puct | 3.0 | Exploration constant (high = more exploration) |

### Root Exploration Noise (Dirichlet)
| Parameter | Value | Description |
|-----------|-------|-------------|
| dirichlet_alpha | 0.15 | Concentration (lower = spikier) |
| dirichlet_epsilon | 0.50 | Noise weight at root (50%!) |
| noise_decay | True | Decay epsilon over iterations |
| noise_min_epsilon | 0.15 | Minimum after decay |
| noise_decay_iterations | 1000 | Iterations to fully decay |

### Move Selection
| Parameter | Value | Description |
|-----------|-------|-------------|
| temperature_start | 2.0 | Initial temp (very random) |
| temperature_end | 0.5 | Final temp (still exploring) |
| temperature_decay_iterations | 100 | Iterations to decay |

---

## Training Configuration

### Network & Optimization
| Parameter | Value | Description |
|-----------|-------|-------------|
| network_preset | "medium" | tiny/small/medium/large |
| batch_size | 256 | Training batch size |
| learning_rate | 1e-3 | Adam LR |
| weight_decay | 1e-4 | L2 regularization |
| epochs_per_iteration | 10 | Epochs per training cycle |
| gradient_clip | 1.0 | Gradient clipping |

### Self-Play Settings
| Parameter | Value | Description |
|-----------|-------|-------------|
| games_per_iteration | 200 | Games to generate |
| bootstrap_games | 1000 | Initial random games |
| bootstrap_epochs | 20 | Initial training epochs |

### Champion Evaluation
| Parameter | Value | Description |
|-----------|-------|-------------|
| champion_eval_games | 50 | Games vs champion |
| champion_win_threshold | 0.55 | Win rate to replace champion |
| champion_min_decisive | 5 | Minimum non-draws required |

---

## Analysis & Gaps

### Potential Issues with Current Signals

#### 1. Channel Redundancy
- **HP Ratio (12-17)** overlaps with **Low HP Targets (43)**
- **Material Balance (33)** overlaps with **Piece Count Balance (32)** and **HP Balance (31)**
- Consider consolidating or making more distinct

#### 2. Missing Signals
- ~~**Threat detection**: What pieces are under attack next turn?~~ ✅ **Channels 62-63**
- ~~**Forcing moves**: Is there a check/forced capture?~~ ✅ **Channel 67**
- **Tempo**: Who has initiative? *(still missing)*
- **Piece activity**: How many squares each piece controls *(still missing)*
- **Pawn structure**: Doubled/isolated/connected pawns *(still missing)*
- **Outposts**: Safe squares for knights *(still missing)*

#### 3. Ability-Specific Gaps
- ~~No signal for "Interpose can save this piece next turn"~~ ✅ **Channels 64-65**
- ~~No signal for "Consecration can save this piece"~~ ✅ **Channel 66**
- No signal for "Knight can Skirmish to safety after attack" *(still missing)*
- ~~No signal for "Pawn Advance creates promotion threat"~~ ✅ **Channel 69**

#### 4. Reward Design Questions
- ~~**Early draw penalty (-100)**: Is this too extreme?~~ ✅ **Replaced with shuffle penalty**
- ~~**Combo bonus**: Only rewards attacks, not other RD uses~~ ✅ **Tripled + waste penalty**
- **Repetition penalty**: Applied at move selection, not game outcome - creates inconsistency

#### 5. Temporal Information
- No history of recent moves/damage
- No "was this piece attacked last turn?"
- No momentum tracking

### Suggested Improvements

#### High Priority
1. Add **threat channel**: squares where enemy can capture our pieces
2. Add **forcing move detection**: check-like situations
3. Add **ability synergy signals**: pieces that benefit from RD/Interpose/etc.

#### Medium Priority
4. Add **piece mobility**: count of legal moves per piece
5. Add **attack value map**: expected damage if we attack each square
6. Simplify overlapping balance channels

#### Lower Priority
7. Consider positional memory (last N moves)
8. Add pawn structure analysis
9. Add king safety score (beyond just attack count)

---

## Questions for Review

1. ~~Should draw penalty be position-aware (use Dynamic Contempt)?~~ ✅ **YES - Implemented**
2. ~~Should combo bonus apply to Skirmish (Knight double-attacks)?~~ ✅ **YES - Skirmish has own reward now**
3. Is the repetition penalty (0.50/repeat) calibrated correctly?
4. ~~Should we add signals for "this move leads to repetition"?~~ ✅ **YES - Channel 30 + shuffle penalty**
5. ~~Are 62 channels enough, or should we expand?~~ ✅ **EXPANDED to 75 channels**
6. Should we use attention mechanisms to weight channels dynamically?

---

## Changes Made (Training Overhaul)

### Draw System
- **Removed**: 30-move no-damage draw rule (was causing 41% draws!)
- **Removed**: Black Rook +1 Interpose bonus
- **Added**: Exponential shuffle penalty (replaces hard draw cutoff)
- **Remaining draws**: Only insufficient material, threefold repetition, max turns

### Interpose Rework
- **Old**: Proactive selection of ally to protect (rarely used)
- **New**: Reactive interrupt - automatically triggers on orthogonal LOS (3 squares max)
- **Effect**: 50/50 damage split between target and Rook

### Reward System (via AbilityTracker)
- All ability usage now tracked per-side
- Each ability has dedicated reward formula
- King hunt rewards encourage aggression
- Shuffle penalty discourages passive play

### Expected Results
| Metric | Before | Target |
|--------|--------|--------|
| Draw rate | 41% | <10% |
| Game length | 188 turns | 80-100 turns |
| RD usage | ~10% | >50% |
| Interpose | ~5% | >30% |

---

*Document updated after training overhaul. Last update: 2026-01-13*
