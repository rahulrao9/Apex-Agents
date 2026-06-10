import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import defaultdict

NUM_SEEDS  = 5
REPLAY_FILES = [f"faction_war_seed_{i}.json" for i in range(1, NUM_SEEDS + 1)]
NUM_TEAMS  = 16
TEAM_SIZE  = 8

FACTION_MAP = {
    0: 'Pacifists',  1: 'Pacifists',
    2: 'Aggressors', 3: 'Aggressors',
    4: 'Bodyguards', 5: 'Bodyguards',
    6: 'Bullies',    7: 'Bullies',
    8: 'Distancers', 9: 'Distancers',
    10: 'Baseline', 11: 'Baseline', 12: 'Baseline',
    13: 'Baseline', 14: 'Baseline', 15: 'Baseline',
}

# Number of teams per faction (used to normalise)
FACTION_SIZES = defaultdict(int)
for fac in FACTION_MAP.values():
    FACTION_SIZES[fac] += 1


# ALGS-style non-linear placement point table (16-team BR)

PLACEMENT_POINTS = {
    1: 12, 2: 9, 3: 7, 4: 5, 5: 4, 6: 3,
    7: 2,  8: 2, 9: 1, 10: 1, 11: 1,
    12: 0, 13: 0, 14: 0, 15: 0, 16: 0,
}


# Scoring weights

KILL_PTS_PER_KILL  = 1
CONSISTENCY_WEIGHT = 0.3
WIN_BONUS          = 5
CV_PENALTY_WEIGHT  = 0.5


def get_team(pid):
    return (int(pid) - 1) // TEAM_SIZE


def extract_match_stats(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    packets = data if isinstance(data, list) else data.get('packets', [])

    team_death_tick  = np.zeros(NUM_TEAMS)
    # Track the latest PlayerDefeats value seen for each player pid
    player_defeats   = {}

    for tick_idx, packet in enumerate(packets):
        players = packet.get('player', {})
        for pid, p_data in players.items():
            tid = get_team(pid)
            if not (0 <= tid < NUM_TEAMS):
                continue

            # Survival tick — keep pushing forward while the player is present
            team_death_tick[tid] = tick_idx

            # PlayerDefeats is cumulative; just overwrite with latest value
            defeats = p_data.get('metrics', {}).get('PlayerDefeats', 0)
            player_defeats[pid] = int(defeats)

    # Sum each player's final defeat count into their team
    team_kills = np.zeros(NUM_TEAMS)
    for pid, defeats in player_defeats.items():
        tid = get_team(pid)
        if 0 <= tid < NUM_TEAMS:
            team_kills[tid] += defeats

    ranked_teams = np.argsort(team_death_tick)[::-1]
    placements   = np.zeros(NUM_TEAMS, dtype=int)
    for rank, tid in enumerate(ranked_teams):
        placements[tid] = rank + 1

    return placements, team_kills

def score_team(placements_across_games, kills_across_games):
    """Compute the full scoring-matrix score for a single team."""
    n = len(placements_across_games)

    # Non-linear placement pts
    raw_place = sum(PLACEMENT_POINTS.get(int(p), 0) for p in placements_across_games)

    # Kill pts
    raw_kills = float(np.sum(kills_across_games)) * KILL_PTS_PER_KILL

    # Win bonus
    raw_wins = sum(WIN_BONUS for p in placements_across_games if int(p) == 1)

    # Consistency bonus  (top-half rate × 3, so max ~3 pts)
    top_half_rate = np.mean([1 if p <= NUM_TEAMS / 2 else 0 for p in placements_across_games])
    raw_consist   = top_half_rate * CONSISTENCY_WEIGHT * 10

    # CV penalty
    arr = np.array(placements_across_games, dtype=float)
    cv  = arr.std() / arr.mean() if arr.mean() else 0.0
    raw_cv = cv * CV_PENALTY_WEIGHT * 10

    total = raw_place + raw_kills + raw_wins + raw_consist - raw_cv
    return dict(
        Total         = total,
        Placement_Pts = raw_place,
        Kill_Pts      = raw_kills,
        Win_Bonus     = raw_wins,
        Consistency   = raw_consist,
        CV_Penalty    = raw_cv,
        Top_Half_Rate = float(top_half_rate),
        Place_CV      = float(cv),
        Avg_Place     = float(arr.mean()),
        Avg_Kills     = float(np.mean(kills_across_games)),
        Wins          = int(sum(1 for p in placements_across_games if int(p) == 1)),
    )


def main():
    
    per_team = {tid: {'placements': [], 'kills': []} for tid in range(NUM_TEAMS)}

    print("Scoring Tournament Replays...")
    loaded = 0
    for file in REPLAY_FILES:
        if not os.path.exists(file):
            print(f"  [SKIP] {file} not found")
            continue
        loaded += 1
        placements, kills = extract_match_stats(file)
        for tid in range(NUM_TEAMS):
            per_team[tid]['placements'].append(int(placements[tid]))
            per_team[tid]['kills'].append(float(kills[tid]))

    if loaded == 0:
        print("\nNo replay files found. Generating synthetic demo data...")
        _inject_demo_data(per_team)

    
    # Compute per-team scores, then average within faction
    
    faction_team_scores = defaultdict(list)   # fac → [team_score_dict, ...]

    for tid, data in per_team.items():
        fac = FACTION_MAP[tid]
        ts  = score_team(data['placements'], data['kills'])
        faction_team_scores[fac].append(ts)

    leaderboard = []
    for fac, team_scores in faction_team_scores.items():
        keys = ['Total', 'Placement_Pts', 'Kill_Pts', 'Win_Bonus',
                'Consistency', 'CV_Penalty', 'Top_Half_Rate',
                'Place_CV', 'Avg_Place', 'Avg_Kills', 'Wins']
        # Average each metric across the faction's member teams
        agg = {k: float(np.mean([ts[k] for ts in team_scores])) for k in keys}
        agg['Faction'] = fac
        agg['N_Teams'] = len(team_scores)
        leaderboard.append(agg)

    leaderboard.sort(key=lambda x: x['Total'], reverse=True)

    
    # Console output
    
    W = 120
    print(f"\n{'=' * W}")
    print(f"{'NEURAL MMO FACTION WAR — MULTI-OBJECTIVE LEADERBOARD (per-team avg)':^{W}}")
    print(f"{'=' * W}")
    header = (f"{'Rank':<5} {'Faction':<13} {'Teams':>6} {'TOTAL':>7} │ "
              f"{'Place':>6} {'Kills':>6} {'Wins':>5} {'Consist':>8} {'CV Pen':>7} │ "
              f"{'Avg Plc':>8} {'Top½%':>7} {'Wins/tm':>8}")
    print(header)
    print("-" * W)

    for i, r in enumerate(leaderboard):
        print(
            f"{i+1:<5} {r['Faction']:<13} {r['N_Teams']:>6} {r['Total']:>7.1f} │ "
            f"{r['Placement_Pts']:>6.1f} {r['Kill_Pts']:>6.1f} {r['Win_Bonus']:>5.1f} "
            f"{r['Consistency']:>8.2f} {r['CV_Penalty']:>7.2f} │ "
            f"{r['Avg_Place']:>8.1f} {r['Top_Half_Rate']*100:>6.0f}% {r['Wins']:>8.2f}"
        )

    print(f"\nAll scores are the MEAN across a faction's member teams (fair vs unequal faction sizes).")
    print(f"Scoring: ALGS placement pts + {KILL_PTS_PER_KILL}/kill + {WIN_BONUS}×wins + consistency − CV penalty")
    print(f"{'=' * W}\n")

    
    # Visualisation
    
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 9))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax_main  = fig.add_subplot(gs[0, :])
    ax_scat  = fig.add_subplot(gs[1, 0])
    ax_cv    = fig.add_subplot(gs[1, 1])

    factions    = [r['Faction']        for r in leaderboard]
    bars_place  = [r['Placement_Pts']  for r in leaderboard]
    bars_kill   = [r['Kill_Pts']       for r in leaderboard]
    bars_win    = [r['Win_Bonus']      for r in leaderboard]
    bars_cons   = [r['Consistency']    for r in leaderboard]
    bars_cv_pen = [-r['CV_Penalty']    for r in leaderboard]

    x, w = np.arange(len(factions)), 0.6

    ax_main.bar(x, bars_place, w, label='Placement pts', color='#5B8DEF')
    ax_main.bar(x, bars_kill,  w, bottom=bars_place,
                label='Kill pts', color='#EF5350')
    ax_main.bar(x, bars_win, w,
                bottom=[a+b for a,b in zip(bars_place, bars_kill)],
                label='Win bonus', color='#FFD700')
    ax_main.bar(x, bars_cons, w,
                bottom=[a+b+c for a,b,c in zip(bars_place, bars_kill, bars_win)],
                label='Consistency', color='#66BB6A')
    ax_main.bar(x, bars_cv_pen, w, label='CV penalty (−)', color='#FF7043', alpha=0.7)

    ax_main.set_xticks(x)
    ax_main.set_xticklabels(factions, fontsize=9)
    ax_main.set_ylabel('Points (avg per member team)')
    ax_main.set_title('Multi-Objective Score Breakdown — avg per member team', fontsize=12, pad=10)
    ax_main.legend(loc='upper right', fontsize=8, ncol=5)
    ax_main.axhline(0, color='white', linewidth=0.4, alpha=0.3)
    ax_main.grid(axis='y', alpha=0.15)

    for i, r in enumerate(leaderboard):
        ax_main.text(i, r['Total'] + 0.2, f"{r['Total']:.1f}",
                     ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Scatter: consistency vs aggression
    top_half_rates = [r['Top_Half_Rate'] * 100 for r in leaderboard]
    avg_kills      = [r['Avg_Kills']           for r in leaderboard]
    totals         = [r['Total']               for r in leaderboard]
    vmin, vmax     = min(totals), max(totals)

    scatter_colors = plt.cm.plasma([(t - vmin) / max(vmax - vmin, 1e-9) for t in totals])
    ax_scat.scatter(avg_kills, top_half_rates, c=scatter_colors, s=80, zorder=3)
    for i, fac in enumerate(factions):
        ax_scat.annotate(fac, (avg_kills[i], top_half_rates[i]),
                         textcoords='offset points', xytext=(5, 3), fontsize=7)
    ax_scat.set_xlabel('Avg kills per game (per team)')
    ax_scat.set_ylabel('Top-half finish rate (%)')
    ax_scat.set_title('Consistency vs Aggression', fontsize=10)
    ax_scat.grid(alpha=0.15)
    plt.colorbar(
        plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=vmin, vmax=vmax)),
        ax=ax_scat
    ).set_label('Total score', fontsize=8)

    # CV chart
    cv_vals  = [r['Place_CV'] for r in leaderboard]
    med_cv   = float(np.median(cv_vals))
    bar_cols = ['#FF7043' if cv > med_cv else '#78909C' for cv in cv_vals]
    ax_cv.barh(factions[::-1], cv_vals[::-1], color=bar_cols[::-1])
    ax_cv.axvline(med_cv, color='white', linestyle='--', linewidth=0.8, alpha=0.5)
    ax_cv.set_xlabel('Coefficient of Variation (placement std / mean)')
    ax_cv.set_title('Placement Volatility (CV)', fontsize=10)
    ax_cv.grid(axis='x', alpha=0.15)
    ax_cv.legend(handles=[
        mpatches.Patch(color='#FF7043', label='Above-median volatility'),
        mpatches.Patch(color='#78909C', label='Below-median volatility'),
    ], fontsize=7, loc='lower right')

    fig.suptitle('Neural MMO Faction War — Tournament Analytics', fontsize=14, y=1.01)
    plt.tight_layout()

    os.makedirs('plots', exist_ok=True)
    out = 'plots/18_tournament_leaderboard.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"Exported {out}")

def _inject_demo_data(per_team):
    rng   = np.random.default_rng(42)
    tiers = {}
    for tid in range(NUM_TEAMS):
        fac = FACTION_MAP[tid]
        base = {'Aggressors': 0.85, 'Pacifists': 0.40,
                'Bodyguards': 0.70, 'Bullies': 0.65,
                'Distancers': 0.55, 'Baseline': 0.50}.get(fac, 0.50)
        tiers[tid] = base + rng.uniform(-0.1, 0.1)

    for _ in range(NUM_SEEDS):
        scores = {tid: rng.normal(tiers[tid], 0.2) for tid in range(NUM_TEAMS)}
        ranked = sorted(range(NUM_TEAMS), key=lambda t: scores[t], reverse=True)
        for rank, tid in enumerate(ranked):
            per_team[tid]['placements'].append(rank + 1)
            per_team[tid]['kills'].append(
                max(0.0, rng.normal(tiers[tid] * 5, 1.5))
            )


if __name__ == "__main__":
    main()