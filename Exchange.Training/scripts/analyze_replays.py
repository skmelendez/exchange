#!/usr/bin/env python3
"""Analyze training replays to understand game dynamics."""

import json
import os
from pathlib import Path
from collections import defaultdict
import statistics
import argparse

def load_games(replays_dir):
    """Load all replay files."""
    all_games = []
    replays_path = Path(replays_dir)
    skipped_short = 0

    if not replays_path.exists():
        print(f"Replays directory not found: {replays_dir}")
        return []

    for iter_dir in sorted(replays_path.iterdir()):
        if not iter_dir.is_dir():
            continue
        try:
            iter_num = int(iter_dir.name.replace("iter_", ""))
        except:
            continue

        for game_file in iter_dir.glob("*.json"):
            try:
                with open(game_file) as f:
                    data = json.load(f)
                    
                    # Ensure moves key exists
                    if 'moves' not in data:
                        data['moves'] = []

                    # Filter outliers (games < 10 moves)
                    if len(data['moves']) < 10:
                        skipped_short += 1
                        continue

                    data['iteration'] = iter_num
                    data['file'] = str(game_file)
                    all_games.append(data)
            except:
                pass

    if skipped_short > 0:
        print(f"Excluded {skipped_short} games with < 10 moves (outliers/errors).")

    return all_games

def analyze_game_phases(game):
    """Analyze damage by early/mid/late game."""
    moves = game['moves']
    total_moves = len(moves)

    early_end = min(20, total_moves)
    mid_end = min(60, total_moves)

    phases = {
        'early': {'white_dmg': 0, 'black_dmg': 0, 'moves': 0},
        'mid': {'white_dmg': 0, 'black_dmg': 0, 'moves': 0},
        'late': {'white_dmg': 0, 'black_dmg': 0, 'moves': 0}
    }

    for i, move in enumerate(moves):
        dmg = move.get('damage', 0)
        team = move.get('team', 0)

        if i < early_end:
            phase = 'early'
        elif i < mid_end:
            phase = 'mid'
        else:
            phase = 'late'

        phases[phase]['moves'] += 1
        if team == 0:  # White
            phases[phase]['white_dmg'] += dmg
        else:  # Black
            phases[phase]['black_dmg'] += dmg

    return phases

def main():
    parser = argparse.ArgumentParser(description="Analyze training replays")
    parser.add_argument("--replays", default="runs/experiment/replays", help="Replays directory")
    parser.add_argument("--last", type=int, default=50, help="Only analyze last N iterations (default: 50)")
    args = parser.parse_args()

    all_games = load_games(args.replays)

    # Filter to last N iterations
    if all_games and args.last > 0:
        max_iter = max(g['iteration'] for g in all_games)
        min_iter = max(0, max_iter - args.last + 1)
        original_count = len(all_games)
        all_games = [g for g in all_games if g['iteration'] >= min_iter]
        print(f"Filtered to iterations {min_iter}-{max_iter} (last {args.last}): {len(all_games)} of {original_count} games")

    # Ensure each game dictionary has 'moves' key to avoid KeyError
    for g in all_games:
        if 'moves' not in g:
            g['moves'] = []

    if not all_games:
        print(f"No games found in {args.replays}")
        return

    print(f"Loaded {len(all_games)} games")
    print()

    # === OVERALL STATS ===
    white_wins = sum(1 for g in all_games if g.get('winner') == 0)
    black_wins = sum(1 for g in all_games if g.get('winner') == 1)
    draws = sum(1 for g in all_games if g.get('winner') is None)

    print("=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)
    print(f"White wins: {white_wins} ({white_wins/len(all_games)*100:.1f}%)")
    print(f"Black wins: {black_wins} ({black_wins/len(all_games)*100:.1f}%)")
    print(f"Draws: {draws} ({draws/len(all_games)*100:.1f}%)")
    print()

    # === DRAW CAUSE ANALYSIS ===
    if draws > 0:
        print("=" * 60)
        print("DRAW CAUSE BREAKDOWN")
        print("=" * 60)

        draw_causes = {
            'king_vs_king': 0,      # Only kings remaining
            'no_damage_30': 0,      # 30 moves without damage
            'threefold': 0,         # Threefold repetition
            'unknown': 0,           # Could not determine
        }

        for game in all_games:
            if game.get('winner') is not None:
                continue  # Not a draw

            final_state = game.get('final_state', {})

            # Check cause 1: King vs King (only kings remaining)
            pieces = final_state.get('pieces', [])
            alive_pieces = [p for p in pieces if p.get('currentHp', 0) > 0]
            non_king_alive = [p for p in alive_pieces if p.get('pieceType', 0) != 0]

            if len(non_king_alive) == 0 and len(alive_pieces) > 0:
                draw_causes['king_vs_king'] += 1
                continue

            # Check cause 2: 30 moves without damage
            moves_without_dmg = final_state.get('movesWithoutDamage', 0)
            if moves_without_dmg >= 30:
                draw_causes['no_damage_30'] += 1
                continue

            # Check cause 3: Threefold repetition (remaining case)
            # If we got here and it's a draw, it's likely threefold
            draw_causes['threefold'] += 1

        print(f"King vs King (insufficient material): {draw_causes['king_vs_king']:>5} ({draw_causes['king_vs_king']/draws*100:>5.1f}%)")
        print(f"30 moves without damage:              {draw_causes['no_damage_30']:>5} ({draw_causes['no_damage_30']/draws*100:>5.1f}%)")
        print(f"Threefold repetition:                 {draw_causes['threefold']:>5} ({draw_causes['threefold']/draws*100:>5.1f}%)")
        if draw_causes['unknown'] > 0:
            print(f"Unknown cause:                        {draw_causes['unknown']:>5} ({draw_causes['unknown']/draws*100:>5.1f}%)")
        print()

    # === GAME LENGTH ANALYSIS ===
    game_lengths = [len(g['moves']) for g in all_games]
    print(f"Game length: avg={statistics.mean(game_lengths):.1f}, min={min(game_lengths)}, max={max(game_lengths)}")
    print()

    # === DAMAGE BY PHASE ===
    print("=" * 60)
    print("DAMAGE BY GAME PHASE")
    print("=" * 60)

    all_phases = {'early': [], 'mid': [], 'late': []}
    for game in all_games:
        phases = analyze_game_phases(game)
        for phase in ['early', 'mid', 'late']:
            if phases[phase]['moves'] > 0:
                all_phases[phase].append(phases[phase])

    for phase in ['early', 'mid', 'late']:
        if all_phases[phase]:
            white_dmg = [p['white_dmg'] for p in all_phases[phase]]
            black_dmg = [p['black_dmg'] for p in all_phases[phase]]
            move_counts = [p['moves'] for p in all_phases[phase]]

            avg_white = statistics.mean(white_dmg)
            avg_black = statistics.mean(black_dmg)
            avg_moves = statistics.mean(move_counts)

            print(f"{phase.upper():6} ({avg_moves:.0f} moves avg):")
            print(f"  White damage: {avg_white:.1f} avg")
            print(f"  Black damage: {avg_black:.1f} avg")
            print(f"  Difference: {avg_white - avg_black:+.1f} (White advantage)")
            print()

    # === FIRST BLOOD ANALYSIS ===
    print("=" * 60)
    print("FIRST BLOOD (First damage dealt)")
    print("=" * 60)

    first_blood_white = 0
    first_blood_black = 0
    first_blood_turn = []

    for game in all_games:
        for i, move in enumerate(game['moves']):
            if move.get('damage', 0) > 0:
                if move['team'] == 0:
                    first_blood_white += 1
                else:
                    first_blood_black += 1
                first_blood_turn.append(i)
                break

    total_fb = first_blood_white + first_blood_black
    if total_fb > 0:
        print(f"White gets first blood: {first_blood_white} ({first_blood_white/total_fb*100:.1f}%)")
        print(f"Black gets first blood: {first_blood_black} ({first_blood_black/total_fb*100:.1f}%)")
        print(f"Average turn of first blood: {statistics.mean(first_blood_turn):.1f}")
    print()

    # === WIN RATE BY FIRST BLOOD ===
    print("=" * 60)
    print("WIN RATE BY FIRST BLOOD")
    print("=" * 60)

    fb_white_then_win = 0
    fb_white_then_lose = 0
    fb_black_then_win = 0
    fb_black_then_lose = 0

    for game in all_games:
        winner = game.get('winner')
        if winner is None:
            continue

        fb_team = None
        for move in game['moves']:
            if move.get('damage', 0) > 0:
                fb_team = move['team']
                break

        if fb_team == 0:  # White got first blood
            if winner == 0:
                fb_white_then_win += 1
            else:
                fb_white_then_lose += 1
        elif fb_team == 1:  # Black got first blood
            if winner == 1:
                fb_black_then_win += 1
            else:
                fb_black_then_lose += 1

    if fb_white_then_win + fb_white_then_lose > 0:
        wr = fb_white_then_win / (fb_white_then_win + fb_white_then_lose) * 100
        print(f"White gets first blood -> White wins: {wr:.1f}%")
    if fb_black_then_win + fb_black_then_lose > 0:
        wr = fb_black_then_win / (fb_black_then_win + fb_black_then_lose) * 100
        print(f"Black gets first blood -> Black wins: {wr:.1f}%")
    print()

    # === PIECE TYPE ANALYSIS ===
    print("=" * 60)
    print("DAMAGE BY PIECE TYPE")
    print("=" * 60)

    piece_names = ['King', 'Queen', 'Rook', 'Bishop', 'Knight', 'Pawn']
    piece_damage = defaultdict(lambda: {'white': 0, 'black': 0, 'count_w': 0, 'count_b': 0})

    for game in all_games:
        for move in game['moves']:
            dmg = move.get('damage', 0)
            if dmg > 0:
                pt = move.get('piece_type', 0)
                team = move.get('team', 0)
                if team == 0:
                    piece_damage[pt]['white'] += dmg
                    piece_damage[pt]['count_w'] += 1
                else:
                    piece_damage[pt]['black'] += dmg
                    piece_damage[pt]['count_b'] += 1

    for pt in range(6):
        pd = piece_damage[pt]
        total_dmg = pd['white'] + pd['black']
        total_attacks = pd['count_w'] + pd['count_b']
        if total_attacks > 0:
            avg_dmg = total_dmg / total_attacks
            print(f"{piece_names[pt]:8}: {total_attacks:5} attacks, {total_dmg:6} total damage, {avg_dmg:.1f} avg/attack")
            print(f"          White: {pd['count_w']:4} attacks ({pd['white']:5} dmg)  Black: {pd['count_b']:4} attacks ({pd['black']:5} dmg)")

    print()

    # === MOVE TYPE ANALYSIS ===
    print("=" * 60)
    print("MOVE TYPES (0=Move, 1=Attack, 2=MoveAttack, 3=Ability)")
    print("=" * 60)

    move_types = defaultdict(lambda: {'white': 0, 'black': 0, 'white_dmg': 0, 'black_dmg': 0})
    move_type_names = ['Move', 'Attack', 'MoveAttack', 'Ability']

    for game in all_games:
        for move in game['moves']:
            mt = move.get('move_type', 0)
            dmg = move.get('damage', 0)
            team = move.get('team', 0)
            if team == 0:
                move_types[mt]['white'] += 1
                move_types[mt]['white_dmg'] += dmg
            else:
                move_types[mt]['black'] += 1
                move_types[mt]['black_dmg'] += dmg

    for mt in range(4):
        m = move_types[mt]
        total = m['white'] + m['black']
        total_dmg = m['white_dmg'] + m['black_dmg']
        if total > 0:
            print(f"{move_type_names[mt]:10}: {total:6} total ({m['white']:5} W, {m['black']:5} B) | Damage: {total_dmg:6} ({m['white_dmg']:5} W, {m['black_dmg']:5} B)")

    print()

    # === ABILITY USAGE ===
    print("=" * 60)
    print("ABILITY USAGE")
    print("=" * 60)

    ability_names = ['RoyalDecree', 'Overextend', 'Interpose', 'Consecration', 'Skirmish', 'Advance']
    ability_usage = defaultdict(lambda: {'white': 0, 'black': 0, 'white_dmg': 0, 'black_dmg': 0})

    for game in all_games:
        for move in game['moves']:
            if move.get('move_type') == 3:  # Ability
                aid = move.get('ability_id')
                if aid is not None:
                    dmg = move.get('damage', 0)
                    team = move.get('team', 0)
                    if team == 0:
                        ability_usage[aid]['white'] += 1
                        ability_usage[aid]['white_dmg'] += dmg
                    else:
                        ability_usage[aid]['black'] += 1
                        ability_usage[aid]['black_dmg'] += dmg

    for aid in range(6):
        a = ability_usage[aid]
        total = a['white'] + a['black']
        if total > 0:
            total_dmg = a['white_dmg'] + a['black_dmg']
            print(f"{ability_names[aid]:12}: {total:5} uses ({a['white']:4} W, {a['black']:4} B) | Damage: {total_dmg:5}")

    print()

    # === ITERATION TRENDS ===
    print("=" * 60)
    print("TRENDS OVER TRAINING (every 10 iterations)")
    print("=" * 60)

    iter_stats = defaultdict(lambda: {'white_wins': 0, 'black_wins': 0, 'draws': 0, 'games': 0, 'white_dmg': 0, 'black_dmg': 0})

    for game in all_games:
        iter_bucket = (game['iteration'] // 10) * 10
        winner = game.get('winner')

        iter_stats[iter_bucket]['games'] += 1
        if winner == 0:
            iter_stats[iter_bucket]['white_wins'] += 1
        elif winner == 1:
            iter_stats[iter_bucket]['black_wins'] += 1
        else:
            iter_stats[iter_bucket]['draws'] += 1

        for move in game['moves']:
            dmg = move.get('damage', 0)
            if move['team'] == 0:
                iter_stats[iter_bucket]['white_dmg'] += dmg
            else:
                iter_stats[iter_bucket]['black_dmg'] += dmg

    print(f"{'Iter':>6} | {'Games':>5} | {'W Win%':>6} | {'B Win%':>6} | {'Draw%':>6} | {'W Dmg':>7} | {'B Dmg':>7} | {'Gap':>7}")
    print("-" * 72)

    for iter_bucket in sorted(iter_stats.keys()):
        s = iter_stats[iter_bucket]
        w_pct = s['white_wins'] / s['games'] * 100 if s['games'] > 0 else 0
        b_pct = s['black_wins'] / s['games'] * 100 if s['games'] > 0 else 0
        d_pct = s['draws'] / s['games'] * 100 if s['games'] > 0 else 0
        w_dmg = s['white_dmg'] / s['games'] if s['games'] > 0 else 0
        b_dmg = s['black_dmg'] / s['games'] if s['games'] > 0 else 0
        gap = w_dmg - b_dmg
        print(f"{iter_bucket:>6} | {s['games']:>5} | {w_pct:>5.1f}% | {b_pct:>5.1f}% | {d_pct:>5.1f}% | {w_dmg:>7.1f} | {b_dmg:>7.1f} | {gap:>+7.1f}")

    # === DETAILED PROGRESSION ANALYSIS ===
    print()
    print("=" * 60)
    print("DETAILED PROGRESSION OVER TRAINING")
    print("=" * 60)

    # Group games by iteration bucket (every 5 iterations for more granularity)
    bucket_size = 5
    iter_detailed = defaultdict(lambda: {
        'games': [],
        'game_lengths': [],
        'early_w_dmg': [], 'early_b_dmg': [],
        'mid_w_dmg': [], 'mid_b_dmg': [],
        'late_w_dmg': [], 'late_b_dmg': [],
        'first_blood_white': 0, 'first_blood_black': 0,
        'fb_turn': [],
        'piece_attacks': defaultdict(lambda: {'w': 0, 'b': 0}),
        'ability_uses': defaultdict(lambda: {'w': 0, 'b': 0}),
        'decisive_games': 0,
        # New stats
        'total_moves': 0,
        'damaging_moves': 0,
        'total_damage': 0,
        'king_damage_taken': {'w': 0, 'b': 0},  # damage TO each king
        'piece_kills': 0,  # total pieces eliminated
        'move_types': defaultdict(int),  # count by move type
        'fb_then_win': 0,  # games where first blood team won
        'fb_then_lose': 0,  # games where first blood team lost
        'damage_by_piece_type': defaultdict(int),  # total damage dealt BY each piece type
        # Consecration (Bishop heal) tracking - ability_id=3
        'consecration_turns': [],  # turn numbers when used
        'consecration_distances': [],  # distance from bishop to heal target
        'consecration_phases': {'early': 0, 'mid': 0, 'late': 0},
        # Royal Decree tracking - ability_id=0
        'royal_decree_turns': [],  # turn numbers when used
        'royal_decree_values': [],  # eval value at time of use
        'royal_decree_phases': {'early': 0, 'mid': 0, 'late': 0},
        # Overextend (Queen) tracking - ability_id=1
        'overextend_turns': [],
        'overextend_values': [],
        'overextend_damages': [],  # damage dealt with overextend
        'overextend_phases': {'early': 0, 'mid': 0, 'late': 0},
        # Interpose (Rook) tracking - ability_id=2
        'interpose_turns': [],
        'interpose_values': [],
        'interpose_phases': {'early': 0, 'mid': 0, 'late': 0},
        # Skirmish (Knight) tracking - ability_id=4
        'skirmish_turns': [],
        'skirmish_values': [],
        'skirmish_damages': [],  # damage dealt with skirmish
        'skirmish_phases': {'early': 0, 'mid': 0, 'late': 0},
        # Advance (Pawn) tracking - ability_id=5
        'advance_turns': [],
        'advance_values': [],
        'advance_phases': {'early': 0, 'mid': 0, 'late': 0},
        # Draw cause tracking
        'draw_king_vs_king': 0,
        'draw_no_damage_30': 0,
        'draw_threefold': 0,
        'total_draws': 0,
        # NEW: Interpose effectiveness tracking
        'interpose_damage_blocked': 0,  # total damage absorbed by Rook
        'interpose_activations': 0,  # times Interpose was activated
        'interpose_effective_uses': 0,  # times it actually blocked damage
        # NEW: Consecration effectiveness tracking
        'consecration_total_heal': 0,  # total HP healed
        'consecration_uses': 0,  # times used
        'consecration_heals': [],  # individual heal amounts
        # NEW: Pawn promotions
        'promotions_white': 0,
        'promotions_black': 0,
        # NEW: Royal Decree combo tracking (Gemini suggestion)
        'rd_turns_buffed_white': 0,  # turns where RD was active for white
        'rd_turns_buffed_black': 0,  # turns where RD was active for black
        'rd_attacks_while_buffed_white': 0,  # attacks made while white RD active
        'rd_attacks_while_buffed_black': 0,  # attacks made while black RD active
        'rd_damage_while_buffed_white': 0,  # total damage while white RD active
        'rd_damage_while_buffed_black': 0,  # total damage while black RD active
        'attacks_unbuffed_white': 0,  # attacks without RD
        'attacks_unbuffed_black': 0,
        'damage_unbuffed_white': 0,  # damage without RD
        'damage_unbuffed_black': 0,
    })

    for game in all_games:
        bucket = (game['iteration'] // bucket_size) * bucket_size
        d = iter_detailed[bucket]
        d['games'].append(game)
        d['game_lengths'].append(len(game['moves']))

        # Phase analysis
        phases = analyze_game_phases(game)
        d['early_w_dmg'].append(phases['early']['white_dmg'])
        d['early_b_dmg'].append(phases['early']['black_dmg'])
        d['mid_w_dmg'].append(phases['mid']['white_dmg'])
        d['mid_b_dmg'].append(phases['mid']['black_dmg'])
        d['late_w_dmg'].append(phases['late']['white_dmg'])
        d['late_b_dmg'].append(phases['late']['black_dmg'])

        # First blood
        for i, move in enumerate(game['moves']):
            if move.get('damage', 0) > 0:
                if move['team'] == 0:
                    d['first_blood_white'] += 1
                else:
                    d['first_blood_black'] += 1
                d['fb_turn'].append(i)
                break

        # Decisive games and draw tracking
        winner = game.get('winner')
        if winner is not None:
            d['decisive_games'] += 1
        else:
            # Track draw cause
            d['total_draws'] += 1
            final_state = game.get('final_state', {})

            # Check cause 1: King vs King
            pieces = final_state.get('pieces', [])
            alive_pieces = [p for p in pieces if p.get('currentHp', 0) > 0]
            non_king_alive = [p for p in alive_pieces if p.get('pieceType', 0) != 0]

            if len(non_king_alive) == 0 and len(alive_pieces) > 0:
                d['draw_king_vs_king'] += 1
            elif final_state.get('movesWithoutDamage', 0) >= 30:
                d['draw_no_damage_30'] += 1
            else:
                d['draw_threefold'] += 1

        # First blood conversion tracking
        fb_team = None
        for move in game['moves']:
            if move.get('damage', 0) > 0:
                fb_team = move['team']
                break
        if fb_team is not None and winner is not None:
            if fb_team == winner:
                d['fb_then_win'] += 1
            else:
                d['fb_then_lose'] += 1

        # Royal Decree buff state tracking (lasts 2 turns)
        white_rd_turns_remaining = 0
        black_rd_turns_remaining = 0

        # Piece and ability tracking
        for move in game['moves']:
            dmg = move.get('damage', 0)
            team = move.get('team', 0)
            team_key = 'w' if team == 0 else 'b'

            # Move type tracking
            mt = move.get('move_type', 0)
            d['move_types'][mt] += 1
            d['total_moves'] += 1

            if dmg > 0:
                pt = move.get('piece_type', 0)
                d['piece_attacks'][pt][team_key] += 1
                d['damaging_moves'] += 1
                d['total_damage'] += dmg
                d['damage_by_piece_type'][pt] += dmg

                # Track damage TO kings (target piece type would be needed, but we can infer from game end)
                # For now, track if attacking piece hit high-value target based on damage dealt

            # Track big hits (10+ damage, typically ability-boosted attacks)
            if dmg >= 10:
                d['piece_kills'] += 1

            if move.get('move_type') == 3:  # Ability
                aid = move.get('ability_id')
                if aid is not None:
                    d['ability_uses'][aid][team_key] += 1

                    turn = move.get('turn', i)
                    # Determine game phase
                    if turn < 20:
                        phase = 'early'
                    elif turn < 60:
                        phase = 'mid'
                    else:
                        phase = 'late'

                    # Track Consecration (Bishop heal, ability_id=3)
                    if aid == 3:
                        d['consecration_turns'].append(turn)
                        d['consecration_phases'][phase] += 1
                        # Calculate distance to heal target
                        ability_target = move.get('ability_target')
                        from_pos = move.get('from')
                        if ability_target and from_pos:
                            dist = abs(ability_target[0] - from_pos[0]) + abs(ability_target[1] - from_pos[1])
                            d['consecration_distances'].append(dist)

                    # Track Royal Decree (King, ability_id=0)
                    if aid == 0:
                        d['royal_decree_turns'].append(turn)
                        d['royal_decree_phases'][phase] += 1
                        value = move.get('value')
                        if value is not None:
                            d['royal_decree_values'].append(value)

                    # Track Overextend (Queen, ability_id=1)
                    if aid == 1:
                        d['overextend_turns'].append(turn)
                        d['overextend_phases'][phase] += 1
                        value = move.get('value')
                        if value is not None:
                            d['overextend_values'].append(value)
                        damage = move.get('damage', 0)
                        if damage > 0:
                            d['overextend_damages'].append(damage)

                    # Track Interpose (Rook, ability_id=2)
                    if aid == 2:
                        d['interpose_turns'].append(turn)
                        d['interpose_phases'][phase] += 1
                        value = move.get('value')
                        if value is not None:
                            d['interpose_values'].append(value)

                    # Track Skirmish (Knight, ability_id=4)
                    if aid == 4:
                        d['skirmish_turns'].append(turn)
                        d['skirmish_phases'][phase] += 1
                        value = move.get('value')
                        if value is not None:
                            d['skirmish_values'].append(value)
                        damage = move.get('damage', 0)
                        if damage > 0:
                            d['skirmish_damages'].append(damage)

                    # Track Advance (Pawn, ability_id=5)
                    if aid == 5:
                        d['advance_turns'].append(turn)
                        d['advance_phases'][phase] += 1
                        value = move.get('value')
                        if value is not None:
                            d['advance_values'].append(value)

                    # Track Interpose damage blocked (ability_id=2)
                    if aid == 2:
                        d['interpose_activations'] += 1

                    # Track Consecration healing (ability_id=3)
                    if aid == 3:
                        heal = move.get('consecration_heal', 0)
                        if heal > 0:
                            d['consecration_total_heal'] += heal
                            d['consecration_heals'].append(heal)
                            d['consecration_uses'] += 1

            # Track Interpose damage blocked from any attack
            interpose_blocked = move.get('interpose_blocked', 0)
            if interpose_blocked > 0:
                d['interpose_damage_blocked'] += interpose_blocked
                d['interpose_effective_uses'] += 1

            # Track pawn promotions
            if move.get('promotion', False):
                if team == 0:
                    d['promotions_white'] += 1
                else:
                    d['promotions_black'] += 1

            # Royal Decree combo tracking
            # Check if this move activates Royal Decree
            if move.get('move_type') == 3 and move.get('ability_id') == 0:
                if team == 0:
                    white_rd_turns_remaining = 2
                else:
                    black_rd_turns_remaining = 2

            # Track if current attack is buffed or unbuffed
            if dmg > 0:
                if team == 0:  # White attacking
                    if white_rd_turns_remaining > 0:
                        d['rd_attacks_while_buffed_white'] += 1
                        d['rd_damage_while_buffed_white'] += dmg
                    else:
                        d['attacks_unbuffed_white'] += 1
                        d['damage_unbuffed_white'] += dmg
                else:  # Black attacking
                    if black_rd_turns_remaining > 0:
                        d['rd_attacks_while_buffed_black'] += 1
                        d['rd_damage_while_buffed_black'] += dmg
                    else:
                        d['attacks_unbuffed_black'] += 1
                        d['damage_unbuffed_black'] += dmg

            # Track total buffed turns (for utilization calculation)
            if team == 0 and white_rd_turns_remaining > 0:
                d['rd_turns_buffed_white'] += 1
            elif team == 1 and black_rd_turns_remaining > 0:
                d['rd_turns_buffed_black'] += 1

            # Decrement RD turns after the team's move
            if team == 0 and white_rd_turns_remaining > 0:
                white_rd_turns_remaining -= 1
            elif team == 1 and black_rd_turns_remaining > 0:
                black_rd_turns_remaining -= 1

    # --- Game Length Progression ---
    print()
    print("GAME LENGTH PROGRESSION:")
    print(f"{'Iter':>6} | {'Avg Len':>8} | {'Min':>5} | {'Max':>5} | {'Decisive%':>9}")
    print("-" * 45)
    for bucket in sorted(iter_detailed.keys()):
        d = iter_detailed[bucket]
        if d['game_lengths']:
            avg_len = statistics.mean(d['game_lengths'])
            min_len = min(d['game_lengths'])
            max_len = max(d['game_lengths'])
            decisive_pct = d['decisive_games'] / len(d['games']) * 100 if d['games'] else 0
            print(f"{bucket:>6} | {avg_len:>8.1f} | {min_len:>5} | {max_len:>5} | {decisive_pct:>8.1f}%")

    # --- Phase Damage Progression ---
    print()
    print("EARLY GAME DAMAGE PROGRESSION (first 20 moves):")
    print(f"{'Iter':>6} | {'W Dmg':>7} | {'B Dmg':>7} | {'Gap':>7} | {'W Adv?':>6}")
    print("-" * 50)
    for bucket in sorted(iter_detailed.keys()):
        d = iter_detailed[bucket]
        if d['early_w_dmg']:
            w = statistics.mean(d['early_w_dmg'])
            b = statistics.mean(d['early_b_dmg'])
            gap = w - b
            adv = "W" if gap > 0.5 else ("B" if gap < -0.5 else "=")
            print(f"{bucket:>6} | {w:>7.1f} | {b:>7.1f} | {gap:>+7.1f} | {adv:>6}")

    print()
    print("MID GAME DAMAGE PROGRESSION (moves 20-60):")
    print(f"{'Iter':>6} | {'W Dmg':>7} | {'B Dmg':>7} | {'Gap':>7} | {'W Adv?':>6}")
    print("-" * 50)
    for bucket in sorted(iter_detailed.keys()):
        d = iter_detailed[bucket]
        if d['mid_w_dmg']:
            w = statistics.mean(d['mid_w_dmg'])
            b = statistics.mean(d['mid_b_dmg'])
            gap = w - b
            adv = "W" if gap > 1 else ("B" if gap < -1 else "=")
            print(f"{bucket:>6} | {w:>7.1f} | {b:>7.1f} | {gap:>+7.1f} | {adv:>6}")

    print()
    print("LATE GAME DAMAGE PROGRESSION (moves 60+):")
    print(f"{'Iter':>6} | {'W Dmg':>7} | {'B Dmg':>7} | {'Gap':>7} | {'W Adv?':>6}")
    print("-" * 50)
    for bucket in sorted(iter_detailed.keys()):
        d = iter_detailed[bucket]
        if d['late_w_dmg']:
            w = statistics.mean(d['late_w_dmg'])
            b = statistics.mean(d['late_b_dmg'])
            gap = w - b
            adv = "W" if gap > 2 else ("B" if gap < -2 else "=")
            print(f"{bucket:>6} | {w:>7.1f} | {b:>7.1f} | {gap:>+7.1f} | {adv:>6}")

    # --- First Blood Progression ---
    print()
    print("FIRST BLOOD PROGRESSION:")
    print(f"{'Iter':>6} | {'W FB%':>7} | {'B FB%':>7} | {'Avg Turn':>8}")
    print("-" * 40)
    for bucket in sorted(iter_detailed.keys()):
        d = iter_detailed[bucket]
        total_fb = d['first_blood_white'] + d['first_blood_black']
        if total_fb > 0:
            w_pct = d['first_blood_white'] / total_fb * 100
            b_pct = d['first_blood_black'] / total_fb * 100
            avg_turn = statistics.mean(d['fb_turn']) if d['fb_turn'] else 0
            print(f"{bucket:>6} | {w_pct:>6.1f}% | {b_pct:>6.1f}% | {avg_turn:>8.1f}")

    # --- Piece Usage Progression ---
    print()
    print("PIECE ATTACK DISTRIBUTION OVER TIME (% of total attacks):")
    piece_names = ['King', 'Queen', 'Rook', 'Bishop', 'Knight', 'Pawn']
    print(f"{'Iter':>6} | {'King':>6} | {'Queen':>6} | {'Rook':>6} | {'Bishop':>6} | {'Knight':>6} | {'Pawn':>6}")
    print("-" * 60)
    for bucket in sorted(iter_detailed.keys()):
        d = iter_detailed[bucket]
        total_attacks = sum(d['piece_attacks'][pt]['w'] + d['piece_attacks'][pt]['b'] for pt in range(6))
        if total_attacks > 0:
            pcts = []
            for pt in range(6):
                attacks = d['piece_attacks'][pt]['w'] + d['piece_attacks'][pt]['b']
                pcts.append(attacks / total_attacks * 100)
            print(f"{bucket:>6} | {pcts[0]:>5.1f}% | {pcts[1]:>5.1f}% | {pcts[2]:>5.1f}% | {pcts[3]:>5.1f}% | {pcts[4]:>5.1f}% | {pcts[5]:>5.1f}%")

    # --- Ability Usage Progression ---
    print()
    print("ABILITY USAGE OVER TIME (uses per game):")
    ability_names = ['RoyalDec', 'Overext', 'Interpose', 'Consec', 'Skirmish', 'Advance']
    print(f"{'Iter':>6} | {'RoyalD':>7} | {'OverEx':>7} | {'Interp':>7} | {'Consec':>7} | {'Skirm':>7} | {'Advanc':>7}")
    print("-" * 65)
    for bucket in sorted(iter_detailed.keys()):
        d = iter_detailed[bucket]
        num_games = len(d['games'])
        if num_games > 0:
            usages = []
            for aid in range(6):
                uses = d['ability_uses'][aid]['w'] + d['ability_uses'][aid]['b']
                usages.append(uses / num_games)
            print(f"{bucket:>6} | {usages[0]:>7.2f} | {usages[1]:>7.2f} | {usages[2]:>7.2f} | {usages[3]:>7.2f} | {usages[4]:>7.2f} | {usages[5]:>7.2f}")

    # --- Aggression Index ---
    print()
    print("AGGRESSION INDEX (% of moves that deal damage):")
    print(f"{'Iter':>6} | {'Aggr%':>7} | {'Dmg Moves':>10} | {'Total Moves':>12} | {'Dmg/Game':>9}")
    print("-" * 55)
    for bucket in sorted(iter_detailed.keys()):
        d = iter_detailed[bucket]
        num_games = len(d['games'])
        if d['total_moves'] > 0 and num_games > 0:
            aggr_pct = d['damaging_moves'] / d['total_moves'] * 100
            dmg_per_game = d['total_damage'] / num_games
            print(f"{bucket:>6} | {aggr_pct:>6.1f}% | {d['damaging_moves']:>10} | {d['total_moves']:>12} | {dmg_per_game:>9.1f}")

    # --- Damage Efficiency ---
    print()
    print("DAMAGE EFFICIENCY (avg damage per damaging move):")
    print(f"{'Iter':>6} | {'Dmg/Hit':>8} | {'Total Dmg':>10} | {'Hits':>8} | {'BigHits':>8} | {'Big%':>6}")
    print("-" * 62)
    print("(BigHits = hits dealing 10+ damage, typically ability-boosted)")
    for bucket in sorted(iter_detailed.keys()):
        d = iter_detailed[bucket]
        num_games = len(d['games'])
        if d['damaging_moves'] > 0:
            dmg_per_hit = d['total_damage'] / d['damaging_moves']
            big_hits_per_game = d['piece_kills'] / num_games if num_games > 0 else 0
            big_hit_pct = d['piece_kills'] / d['damaging_moves'] * 100 if d['damaging_moves'] > 0 else 0
            print(f"{bucket:>6} | {dmg_per_hit:>8.2f} | {d['total_damage']:>10} | {d['damaging_moves']:>8} | {big_hits_per_game:>8.1f} | {big_hit_pct:>5.1f}%")

    # --- First Blood Conversion Rate ---
    print()
    print("FIRST BLOOD CONVERSION (FB team win rate in decisive games):")
    print(f"{'Iter':>6} | {'FB->Win':>8} | {'FB->Lose':>9} | {'Conv%':>7} | {'Advantage':>10}")
    print("-" * 55)
    for bucket in sorted(iter_detailed.keys()):
        d = iter_detailed[bucket]
        total_fb_decisive = d['fb_then_win'] + d['fb_then_lose']
        if total_fb_decisive > 0:
            conv_pct = d['fb_then_win'] / total_fb_decisive * 100
            # Advantage over 50% baseline
            advantage = conv_pct - 50
            adv_str = f"+{advantage:.1f}%" if advantage > 0 else f"{advantage:.1f}%"
            print(f"{bucket:>6} | {d['fb_then_win']:>8} | {d['fb_then_lose']:>9} | {conv_pct:>6.1f}% | {adv_str:>10}")

    # --- Move Type Distribution Over Time ---
    print()
    print("MOVE TYPE DISTRIBUTION OVER TIME (% of total moves):")
    move_type_names = ['Move', 'Attack', 'MoveAtk', 'Ability']
    print(f"{'Iter':>6} | {'Move%':>7} | {'Attack%':>8} | {'MoveAtk%':>9} | {'Ability%':>9}")
    print("-" * 52)
    for bucket in sorted(iter_detailed.keys()):
        d = iter_detailed[bucket]
        if d['total_moves'] > 0:
            pcts = []
            for mt in range(4):
                pcts.append(d['move_types'][mt] / d['total_moves'] * 100)
            print(f"{bucket:>6} | {pcts[0]:>6.1f}% | {pcts[1]:>7.1f}% | {pcts[2]:>8.1f}% | {pcts[3]:>8.1f}%")

    # --- Damage by Piece Type Over Time ---
    print()
    print("DAMAGE OUTPUT BY PIECE TYPE OVER TIME (% of total damage):")
    piece_names = ['King', 'Queen', 'Rook', 'Bishop', 'Knight', 'Pawn']
    print(f"{'Iter':>6} | {'King':>6} | {'Queen':>6} | {'Rook':>6} | {'Bishop':>6} | {'Knight':>6} | {'Pawn':>6}")
    print("-" * 60)
    for bucket in sorted(iter_detailed.keys()):
        d = iter_detailed[bucket]
        total_dmg = sum(d['damage_by_piece_type'][pt] for pt in range(6))
        if total_dmg > 0:
            pcts = []
            for pt in range(6):
                pcts.append(d['damage_by_piece_type'][pt] / total_dmg * 100)
            print(f"{bucket:>6} | {pcts[0]:>5.1f}% | {pcts[1]:>5.1f}% | {pcts[2]:>5.1f}% | {pcts[3]:>5.1f}% | {pcts[4]:>5.1f}% | {pcts[5]:>5.1f}%")

    # --- Opening Speed (turns to first damage) ---
    print()
    print("OPENING SPEED (turns until first damage):")
    print(f"{'Iter':>6} | {'Avg Turn':>9} | {'Min':>5} | {'Max':>5} | {'Std Dev':>8}")
    print("-" * 45)
    for bucket in sorted(iter_detailed.keys()):
        d = iter_detailed[bucket]
        if d['fb_turn']:
            avg_turn = statistics.mean(d['fb_turn'])
            min_turn = min(d['fb_turn'])
            max_turn = max(d['fb_turn'])
            std_dev = statistics.stdev(d['fb_turn']) if len(d['fb_turn']) > 1 else 0
            print(f"{bucket:>6} | {avg_turn:>9.1f} | {min_turn:>5} | {max_turn:>5} | {std_dev:>8.1f}")

    # --- Draw Cause Progression ---
    print()
    print("=" * 70)
    print("DRAW CAUSE PROGRESSION")
    print("=" * 70)
    print()
    print("(Shows percentage breakdown of draw causes per iteration bucket)")
    print(f"{'Iter':>6} | {'Draws':>6} | {'Draw%':>6} | {'K vs K':>8} | {'No Dmg 30':>10} | {'Threefold':>10}")
    print("-" * 65)
    for bucket in sorted(iter_detailed.keys()):
        d = iter_detailed[bucket]
        num_games = len(d['games'])
        total_draws = d['total_draws']
        if num_games > 0:
            draw_pct = total_draws / num_games * 100
            if total_draws > 0:
                kvk_pct = d['draw_king_vs_king'] / total_draws * 100
                nodmg_pct = d['draw_no_damage_30'] / total_draws * 100
                three_pct = d['draw_threefold'] / total_draws * 100
                print(f"{bucket:>6} | {total_draws:>6} | {draw_pct:>5.1f}% | {kvk_pct:>7.1f}% | {nodmg_pct:>9.1f}% | {three_pct:>9.1f}%")
            else:
                print(f"{bucket:>6} | {total_draws:>6} | {draw_pct:>5.1f}% |      N/A |        N/A |        N/A")

    # --- Consecration (Bishop Heal) Analysis ---
    print()
    print("=" * 70)
    print("CONSECRATION (Bishop Heal) ANALYSIS")
    print("=" * 70)
    print()
    print("CONSECRATION USAGE BY GAME PHASE:")
    print(f"{'Iter':>6} | {'Total':>6} | {'Early':>7} | {'Mid':>7} | {'Late':>7} | {'Avg Turn':>9} | {'Avg Dist':>9}")
    print("-" * 70)
    for bucket in sorted(iter_detailed.keys()):
        d = iter_detailed[bucket]
        total = sum(d['consecration_phases'].values())
        if total > 0:
            early_pct = d['consecration_phases']['early'] / total * 100
            mid_pct = d['consecration_phases']['mid'] / total * 100
            late_pct = d['consecration_phases']['late'] / total * 100
            avg_turn = statistics.mean(d['consecration_turns']) if d['consecration_turns'] else 0
            avg_dist = statistics.mean(d['consecration_distances']) if d['consecration_distances'] else 0
            print(f"{bucket:>6} | {total:>6} | {early_pct:>6.1f}% | {mid_pct:>6.1f}% | {late_pct:>6.1f}% | {avg_turn:>9.1f} | {avg_dist:>9.2f}")

    print()
    print("CONSECRATION HEAL DISTANCE DISTRIBUTION (Manhattan distance to target):")
    print(f"{'Iter':>6} | {'Dist 1':>7} | {'Dist 2':>7} | {'Dist 3':>7} | {'Dist 4+':>7}")
    print("-" * 45)
    for bucket in sorted(iter_detailed.keys()):
        d = iter_detailed[bucket]
        if d['consecration_distances']:
            dist_counts = {1: 0, 2: 0, 3: 0, 4: 0}
            for dist in d['consecration_distances']:
                if dist >= 4:
                    dist_counts[4] += 1
                else:
                    dist_counts[dist] += 1
            total = len(d['consecration_distances'])
            print(f"{bucket:>6} | {dist_counts[1]/total*100:>6.1f}% | {dist_counts[2]/total*100:>6.1f}% | {dist_counts[3]/total*100:>6.1f}% | {dist_counts[4]/total*100:>6.1f}%")

    # --- Royal Decree (King) Analysis ---
    print()
    print("=" * 70)
    print("ROYAL DECREE (King) ANALYSIS")
    print("=" * 70)
    print()
    print("ROYAL DECREE USAGE BY GAME PHASE:")
    print(f"{'Iter':>6} | {'Total':>6} | {'Early':>7} | {'Mid':>7} | {'Late':>7} | {'Avg Turn':>9}")
    print("-" * 60)
    for bucket in sorted(iter_detailed.keys()):
        d = iter_detailed[bucket]
        total = sum(d['royal_decree_phases'].values())
        if total > 0:
            early_pct = d['royal_decree_phases']['early'] / total * 100
            mid_pct = d['royal_decree_phases']['mid'] / total * 100
            late_pct = d['royal_decree_phases']['late'] / total * 100
            avg_turn = statistics.mean(d['royal_decree_turns']) if d['royal_decree_turns'] else 0
            print(f"{bucket:>6} | {total:>6} | {early_pct:>6.1f}% | {mid_pct:>6.1f}% | {late_pct:>6.1f}% | {avg_turn:>9.1f}")

    print()
    print("ROYAL DECREE EVAL CONTEXT (value at time of use):")
    print("(Positive = winning, Negative = losing)")
    print(f"{'Iter':>6} | {'Avg Val':>8} | {'Winning':>8} | {'Losing':>8} | {'Neutral':>8} | {'Min':>7} | {'Max':>7}")
    print("-" * 70)
    for bucket in sorted(iter_detailed.keys()):
        d = iter_detailed[bucket]
        if d['royal_decree_values']:
            avg_val = statistics.mean(d['royal_decree_values'])
            winning = sum(1 for v in d['royal_decree_values'] if v > 0.1)
            losing = sum(1 for v in d['royal_decree_values'] if v < -0.1)
            neutral = len(d['royal_decree_values']) - winning - losing
            total = len(d['royal_decree_values'])
            min_val = min(d['royal_decree_values'])
            max_val = max(d['royal_decree_values'])
            print(f"{bucket:>6} | {avg_val:>+8.3f} | {winning/total*100:>7.1f}% | {losing/total*100:>7.1f}% | {neutral/total*100:>7.1f}% | {min_val:>+7.2f} | {max_val:>+7.2f}")

    # --- Overextend (Queen) Analysis ---
    print()
    print("=" * 70)
    print("OVEREXTEND (Queen) ANALYSIS")
    print("=" * 70)
    print()
    print("OVEREXTEND USAGE BY GAME PHASE:")
    print(f"{'Iter':>6} | {'Total':>6} | {'Early':>7} | {'Mid':>7} | {'Late':>7} | {'Avg Turn':>9} | {'Avg Dmg':>8}")
    print("-" * 70)
    for bucket in sorted(iter_detailed.keys()):
        d = iter_detailed[bucket]
        total = sum(d['overextend_phases'].values())
        if total > 0:
            early_pct = d['overextend_phases']['early'] / total * 100
            mid_pct = d['overextend_phases']['mid'] / total * 100
            late_pct = d['overextend_phases']['late'] / total * 100
            avg_turn = statistics.mean(d['overextend_turns']) if d['overextend_turns'] else 0
            avg_dmg = statistics.mean(d['overextend_damages']) if d['overextend_damages'] else 0
            print(f"{bucket:>6} | {total:>6} | {early_pct:>6.1f}% | {mid_pct:>6.1f}% | {late_pct:>6.1f}% | {avg_turn:>9.1f} | {avg_dmg:>8.1f}")

    print()
    print("OVEREXTEND EVAL CONTEXT (value at time of use):")
    print(f"{'Iter':>6} | {'Avg Val':>8} | {'Winning':>8} | {'Losing':>8} | {'Neutral':>8}")
    print("-" * 55)
    for bucket in sorted(iter_detailed.keys()):
        d = iter_detailed[bucket]
        if d['overextend_values']:
            avg_val = statistics.mean(d['overextend_values'])
            winning = sum(1 for v in d['overextend_values'] if v > 0.1)
            losing = sum(1 for v in d['overextend_values'] if v < -0.1)
            neutral = len(d['overextend_values']) - winning - losing
            total = len(d['overextend_values'])
            print(f"{bucket:>6} | {avg_val:>+8.3f} | {winning/total*100:>7.1f}% | {losing/total*100:>7.1f}% | {neutral/total*100:>7.1f}%")

    # --- Interpose (Rook) Analysis ---
    print()
    print("=" * 70)
    print("INTERPOSE (Rook) ANALYSIS")
    print("=" * 70)
    print()
    print("INTERPOSE USAGE BY GAME PHASE:")
    print(f"{'Iter':>6} | {'Total':>6} | {'Early':>7} | {'Mid':>7} | {'Late':>7} | {'Avg Turn':>9}")
    print("-" * 60)
    for bucket in sorted(iter_detailed.keys()):
        d = iter_detailed[bucket]
        total = sum(d['interpose_phases'].values())
        if total > 0:
            early_pct = d['interpose_phases']['early'] / total * 100
            mid_pct = d['interpose_phases']['mid'] / total * 100
            late_pct = d['interpose_phases']['late'] / total * 100
            avg_turn = statistics.mean(d['interpose_turns']) if d['interpose_turns'] else 0
            print(f"{bucket:>6} | {total:>6} | {early_pct:>6.1f}% | {mid_pct:>6.1f}% | {late_pct:>6.1f}% | {avg_turn:>9.1f}")

    print()
    print("INTERPOSE EVAL CONTEXT (value at time of use):")
    print(f"{'Iter':>6} | {'Avg Val':>8} | {'Winning':>8} | {'Losing':>8} | {'Neutral':>8}")
    print("-" * 55)
    for bucket in sorted(iter_detailed.keys()):
        d = iter_detailed[bucket]
        if d['interpose_values']:
            avg_val = statistics.mean(d['interpose_values'])
            winning = sum(1 for v in d['interpose_values'] if v > 0.1)
            losing = sum(1 for v in d['interpose_values'] if v < -0.1)
            neutral = len(d['interpose_values']) - winning - losing
            total = len(d['interpose_values'])
            print(f"{bucket:>6} | {avg_val:>+8.3f} | {winning/total*100:>7.1f}% | {losing/total*100:>7.1f}% | {neutral/total*100:>7.1f}%")

    # --- Skirmish (Knight) Analysis ---
    print()
    print("=" * 70)
    print("SKIRMISH (Knight) ANALYSIS")
    print("=" * 70)
    print()
    print("SKIRMISH USAGE BY GAME PHASE:")
    print(f"{'Iter':>6} | {'Total':>6} | {'Early':>7} | {'Mid':>7} | {'Late':>7} | {'Avg Turn':>9} | {'Avg Dmg':>8}")
    print("-" * 70)
    for bucket in sorted(iter_detailed.keys()):
        d = iter_detailed[bucket]
        total = sum(d['skirmish_phases'].values())
        if total > 0:
            early_pct = d['skirmish_phases']['early'] / total * 100
            mid_pct = d['skirmish_phases']['mid'] / total * 100
            late_pct = d['skirmish_phases']['late'] / total * 100
            avg_turn = statistics.mean(d['skirmish_turns']) if d['skirmish_turns'] else 0
            avg_dmg = statistics.mean(d['skirmish_damages']) if d['skirmish_damages'] else 0
            print(f"{bucket:>6} | {total:>6} | {early_pct:>6.1f}% | {mid_pct:>6.1f}% | {late_pct:>6.1f}% | {avg_turn:>9.1f} | {avg_dmg:>8.1f}")

    print()
    print("SKIRMISH EVAL CONTEXT (value at time of use):")
    print(f"{'Iter':>6} | {'Avg Val':>8} | {'Winning':>8} | {'Losing':>8} | {'Neutral':>8}")
    print("-" * 55)
    for bucket in sorted(iter_detailed.keys()):
        d = iter_detailed[bucket]
        if d['skirmish_values']:
            avg_val = statistics.mean(d['skirmish_values'])
            winning = sum(1 for v in d['skirmish_values'] if v > 0.1)
            losing = sum(1 for v in d['skirmish_values'] if v < -0.1)
            neutral = len(d['skirmish_values']) - winning - losing
            total = len(d['skirmish_values'])
            print(f"{bucket:>6} | {avg_val:>+8.3f} | {winning/total*100:>7.1f}% | {losing/total*100:>7.1f}% | {neutral/total*100:>7.1f}%")

    # --- Advance (Pawn) Analysis ---
    print()
    print("=" * 70)
    print("ADVANCE (Pawn) ANALYSIS")
    print("=" * 70)
    print()
    print("ADVANCE USAGE BY GAME PHASE:")
    print(f"{'Iter':>6} | {'Total':>6} | {'Early':>7} | {'Mid':>7} | {'Late':>7} | {'Avg Turn':>9}")
    print("-" * 60)
    for bucket in sorted(iter_detailed.keys()):
        d = iter_detailed[bucket]
        total = sum(d['advance_phases'].values())
        if total > 0:
            early_pct = d['advance_phases']['early'] / total * 100
            mid_pct = d['advance_phases']['mid'] / total * 100
            late_pct = d['advance_phases']['late'] / total * 100
            avg_turn = statistics.mean(d['advance_turns']) if d['advance_turns'] else 0
            print(f"{bucket:>6} | {total:>6} | {early_pct:>6.1f}% | {mid_pct:>6.1f}% | {late_pct:>6.1f}% | {avg_turn:>9.1f}")

    print()
    print("ADVANCE EVAL CONTEXT (value at time of use):")
    print(f"{'Iter':>6} | {'Avg Val':>8} | {'Winning':>8} | {'Losing':>8} | {'Neutral':>8}")
    print("-" * 55)
    for bucket in sorted(iter_detailed.keys()):
        d = iter_detailed[bucket]
        if d['advance_values']:
            avg_val = statistics.mean(d['advance_values'])
            winning = sum(1 for v in d['advance_values'] if v > 0.1)
            losing = sum(1 for v in d['advance_values'] if v < -0.1)
            neutral = len(d['advance_values']) - winning - losing
            total = len(d['advance_values'])
            print(f"{bucket:>6} | {avg_val:>+8.3f} | {winning/total*100:>7.1f}% | {losing/total*100:>7.1f}% | {neutral/total*100:>7.1f}%")

    # --- NEW: Interpose Effectiveness Analysis ---
    print()
    print("=" * 70)
    print("INTERPOSE EFFECTIVENESS (Rook damage absorption)")
    print("=" * 70)
    print()
    print("(Tracks how much damage Rook absorbed when protecting adjacent allies)")
    print(f"{'Iter':>6} | {'Activations':>11} | {'Effective':>9} | {'Eff%':>6} | {'Dmg Blocked':>11} | {'Avg Block':>9}")
    print("-" * 70)
    for bucket in sorted(iter_detailed.keys()):
        d = iter_detailed[bucket]
        activations = d['interpose_activations']
        effective = d['interpose_effective_uses']
        blocked = d['interpose_damage_blocked']
        if activations > 0:
            eff_pct = effective / activations * 100
            avg_block = blocked / effective if effective > 0 else 0
            print(f"{bucket:>6} | {activations:>11} | {effective:>9} | {eff_pct:>5.1f}% | {blocked:>11} | {avg_block:>9.1f}")

    # --- NEW: Consecration Effectiveness Analysis ---
    print()
    print("=" * 70)
    print("CONSECRATION EFFECTIVENESS (Bishop healing)")
    print("=" * 70)
    print()
    print("(Tracks healing provided by Bishop ability)")
    print(f"{'Iter':>6} | {'Uses':>6} | {'Total Heal':>10} | {'Avg Heal':>8} | {'Min':>5} | {'Max':>5}")
    print("-" * 60)
    for bucket in sorted(iter_detailed.keys()):
        d = iter_detailed[bucket]
        uses = d['consecration_uses']
        total_heal = d['consecration_total_heal']
        heals = d['consecration_heals']
        if uses > 0:
            avg_heal = total_heal / uses
            min_heal = min(heals) if heals else 0
            max_heal = max(heals) if heals else 0
            print(f"{bucket:>6} | {uses:>6} | {total_heal:>10} | {avg_heal:>8.1f} | {min_heal:>5} | {max_heal:>5}")

    # --- NEW: Pawn Promotion Analysis ---
    print()
    print("=" * 70)
    print("PAWN PROMOTIONS")
    print("=" * 70)
    print()
    print("(Tracks pawn promotions to Queen)")
    print(f"{'Iter':>6} | {'White':>6} | {'Black':>6} | {'Total':>6} | {'Per Game':>9}")
    print("-" * 50)
    for bucket in sorted(iter_detailed.keys()):
        d = iter_detailed[bucket]
        white = d['promotions_white']
        black = d['promotions_black']
        total = white + black
        num_games = len(d['games'])
        if num_games > 0:
            per_game = total / num_games
            print(f"{bucket:>6} | {white:>6} | {black:>6} | {total:>6} | {per_game:>9.2f}")

    # --- NEW: Royal Decree Combo Effectiveness ---
    print()
    print("=" * 80)
    print("ROYAL DECREE COMBO EFFECTIVENESS (Gemini Analysis)")
    print("=" * 80)
    print()
    print("Tracks whether the AI uses attacks while Royal Decree is active.")
    print("Buff Utilization = (Attacks while buffed) / (Turns buffed)")
    print("Damage Delta = (Avg damage while buffed) - (Avg damage unbuffed)")
    print("Target: Utilization >50%, Delta ~+2.0")
    print()
    print(f"{'Iter':>6} | {'Buff Turns':>10} | {'Buffed Atk':>10} | {'Util%':>7} | {'Avg Buff':>8} | {'Avg Unbuff':>10} | {'Delta':>7}")
    print("-" * 80)
    for bucket in sorted(iter_detailed.keys()):
        d = iter_detailed[bucket]
        # Combine white and black stats
        total_buff_turns = d['rd_turns_buffed_white'] + d['rd_turns_buffed_black']
        total_buff_attacks = d['rd_attacks_while_buffed_white'] + d['rd_attacks_while_buffed_black']
        total_buff_damage = d['rd_damage_while_buffed_white'] + d['rd_damage_while_buffed_black']
        total_unbuff_attacks = d['attacks_unbuffed_white'] + d['attacks_unbuffed_black']
        total_unbuff_damage = d['damage_unbuffed_white'] + d['damage_unbuffed_black']

        if total_buff_turns > 0:
            util_pct = total_buff_attacks / total_buff_turns * 100
            avg_buff_dmg = total_buff_damage / total_buff_attacks if total_buff_attacks > 0 else 0
            avg_unbuff_dmg = total_unbuff_damage / total_unbuff_attacks if total_unbuff_attacks > 0 else 0
            delta = avg_buff_dmg - avg_unbuff_dmg
            print(f"{bucket:>6} | {total_buff_turns:>10} | {total_buff_attacks:>10} | {util_pct:>6.1f}% | {avg_buff_dmg:>8.2f} | {avg_unbuff_dmg:>10.2f} | {delta:>+7.2f}")

    print()
    print("Interpretation:")
    print("- Util% < 20%: AI wastes Royal Decree (activates then doesn't attack)")
    print("- Delta near 0: Damage bonus may not be applying correctly")
    print("- Delta ~2.0: Buff is working as expected")


if __name__ == "__main__":
    main()
