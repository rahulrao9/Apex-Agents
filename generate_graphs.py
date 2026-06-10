import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
from collections import defaultdict
from scipy.stats import gaussian_kde, entropy as scipy_entropy


#  CONFIGURATION

NUM_SEEDS    = 5
REPLAY_FILES = [f"faction_war_seed_{i}.json" for i in range(1, NUM_SEEDS + 1)]
OUTPUT_DIR   = 'plots/v4'
DPI          = 300
STYLE        = 'dark_background'

SKILLS = ['melee', 'range', 'mage', 'fishing', 'herbalism',
          'prospecting', 'carving', 'alchemy']


#  HELPERS


def safe_extract(value):
    if isinstance(value, dict):
        return value.get('val', 0)
    return value if value is not None else 0

def safe_skill(p_data, skill_name):
    for path in [('skills', skill_name, 'level'),
                 ('skills', skill_name),
                 (skill_name, 'level')]:
        node = p_data
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
        if node is not None:
            return safe_extract(node)
    return 1

def save(name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {path}")


#  DATA EXTRACTION


def extract(json_path):
    """Load one replay file and return the same rich data dict as the
    single-seed script.  Returns None on failure."""
    print(f"  Reading {json_path} ...")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return None

    packets = data if isinstance(data, list) else data.get('packets', [])
    if not packets:
        print(f"  ✗ No packets found in {json_path}")
        return None

    ticks, alive_counts = [], []
    avg_gold_series, avg_food_series, avg_water_series, entropy_series = [], [], [], []
    all_pos_x, all_pos_y = [], []
    death_causes         = defaultdict(lambda: defaultdict(int))
    combat_deaths_series = []
    resource_deaths_series = []

    agent_first_seen    = {}
    agent_last_seen     = {}
    agent_death_cause   = {}
    agent_terminal_skills = {}
    agent_terminal_gold = {}

    prev_players = {}

    for tick_idx, packet in enumerate(packets):
        players  = packet.get('player', {})
        curr_ids = set(players.keys())

        ticks.append(tick_idx)
        alive_counts.append(len(players))

        tick_gold, tick_food, tick_water = [], [], []
        pos_x_tick, pos_y_tick = [], []
        combat_deaths = resource_deaths = 0

        # ── deaths 
        for pid in set(prev_players.keys()) - curr_ids:
            last = prev_players[pid]
            food  = safe_extract(last.get('food',  last.get('resource', {}).get('food',  0)))
            water = safe_extract(last.get('water', last.get('resource', {}).get('water', 0)))
            if food <= 0:
                cause = 'starvation';   resource_deaths += 1
            elif water <= 0:
                cause = 'dehydration';  resource_deaths += 1
            else:
                cause = 'combat_or_fog'; combat_deaths  += 1
            death_causes[tick_idx][cause] += 1
            agent_death_cause[str(pid)]   = cause

        combat_deaths_series.append(combat_deaths)
        resource_deaths_series.append(resource_deaths)

        # ── players 
        for pid, p in players.items():
            pid_s     = str(pid)
            base_data = p.get('base', p)

            r = base_data.get('r')
            c = base_data.get('c')
            if r is None or c is None:
                pos = p.get('pos') or base_data.get('pos')
                if pos and len(pos) == 2:
                    r, c = pos[0], pos[1]
            if r is None or c is None:
                r = p.get('row') or base_data.get('row')
                c = p.get('col') or base_data.get('col')
            if r is not None and c is not None:
                pos_x_tick.append(int(c))
                pos_y_tick.append(int(r))
                all_pos_x.append(int(c))
                all_pos_y.append(int(r))

            gold  = safe_extract(p.get('gold',  p.get('base', {}).get('gold',  0)))
            food  = safe_extract(p.get('food',  p.get('resource', {}).get('food',  0)))
            water = safe_extract(p.get('water', p.get('resource', {}).get('water', 0)))
            tick_gold.append(gold);  tick_food.append(food);  tick_water.append(water)

            agent_first_seen.setdefault(pid_s, tick_idx)
            agent_last_seen[pid_s]      = tick_idx
            agent_terminal_gold[pid_s]  = gold
            agent_terminal_skills[pid_s] = {sk: safe_skill(p, sk) for sk in SKILLS}

        # ── spatial entropy 
        if pos_x_tick:
            hist, _ = np.histogramdd(np.column_stack([pos_x_tick, pos_y_tick]), bins=20)
            flat = hist.flatten(); flat = flat[flat > 0] / flat.sum()
            entropy_series.append(float(scipy_entropy(flat)))
        else:
            entropy_series.append(0.0)

        avg_gold_series.append (np.mean(tick_gold)  if tick_gold  else 0)
        avg_food_series.append (np.mean(tick_food)  if tick_food  else 0)
        avg_water_series.append(np.mean(tick_water) if tick_water else 0)
        prev_players = players

    lifetimes = {pid: agent_last_seen[pid] - agent_first_seen[pid]
                 for pid in agent_last_seen}

    return dict(
        ticks=ticks,
        alive_counts=alive_counts,
        avg_gold=avg_gold_series,
        avg_food=avg_food_series,
        avg_water=avg_water_series,
        entropy=entropy_series,
        all_pos_x=all_pos_x,
        all_pos_y=all_pos_y,
        death_causes=death_causes,
        combat_deaths=combat_deaths_series,
        resource_deaths=resource_deaths_series,
        lifetimes=lifetimes,
        terminal_skills=agent_terminal_skills,
        terminal_gold=agent_terminal_gold,
        agent_death_cause=agent_death_cause,
        SKILLS=SKILLS,
    )


#  AGGREGATION ENGINE


def aggregate_seeds(files):
    results = []
    for path in files:
        r = extract(path)
        if r is not None:
            results.append(r)

    if not results:
        print("ERROR: no valid replay files could be loaded.")
        return None

    n = len(results)
    print(f"\n  Aggregating {n}/{len(files)} seeds successfully loaded …")

    min_ticks = min(len(r['ticks']) for r in results)

    def ts_mean(key):
        return np.mean([r[key][:min_ticks] for r in results], axis=0).tolist()

    avg_alive          = ts_mean('alive_counts')
    avg_gold           = ts_mean('avg_gold')
    avg_food           = ts_mean('avg_food')
    avg_water          = ts_mean('avg_water')
    avg_entropy        = ts_mean('entropy')
    avg_combat_deaths  = ts_mean('combat_deaths')
    avg_resource_deaths= ts_mean('resource_deaths')

    pooled_death_causes = defaultdict(lambda: defaultdict(float))
    for r in results:
        for tick_idx, cause_dict in r['death_causes'].items():
            for cause, count in cause_dict.items():
                pooled_death_causes[tick_idx][cause] += count / n

    pooled_pos_x, pooled_pos_y = [], []
    for r in results:
        pooled_pos_x.extend(r['all_pos_x'])
        pooled_pos_y.extend(r['all_pos_y'])

    pooled_lifetimes       = {}
    pooled_terminal_skills = {}
    pooled_terminal_gold   = {}
    pooled_death_cause     = {}

    for seed_idx, r in enumerate(results):
        prefix = f"{seed_idx}_"
        for pid, v in r['lifetimes'].items():
            pooled_lifetimes[prefix + pid] = v
        for pid, v in r['terminal_skills'].items():
            pooled_terminal_skills[prefix + pid] = v
        for pid, v in r['terminal_gold'].items():
            pooled_terminal_gold[prefix + pid] = v
        for pid, v in r['agent_death_cause'].items():
            pooled_death_cause[prefix + pid] = v

    return dict(
        ticks=list(range(min_ticks)),
        alive_counts=avg_alive,
        avg_gold=avg_gold,
        avg_food=avg_food,
        avg_water=avg_water,
        entropy=avg_entropy,
        all_pos_x=pooled_pos_x,
        all_pos_y=pooled_pos_y,
        death_causes=pooled_death_causes,
        combat_deaths=avg_combat_deaths,
        resource_deaths=avg_resource_deaths,
        lifetimes=pooled_lifetimes,
        terminal_skills=pooled_terminal_skills,
        terminal_gold=pooled_terminal_gold,
        agent_death_cause=pooled_death_cause,
        SKILLS=results[0]['SKILLS'],
        n_seeds=n,
    )


def _seed_label(d):
    return f"(Averaged over {d.get('n_seeds', '?')} Seeds)"

def plot_heatmap(d):
    fig, ax = plt.subplots(figsize=(8, 8))
    h = ax.hist2d(d['all_pos_x'], d['all_pos_y'], bins=60, cmap='inferno')
    plt.colorbar(h[3], ax=ax, label='Agent-Tick Density')
    ax.set_title(f'Spatial Heatmap — Exploration vs Exploitation\n{_seed_label(d)}', fontsize=13, pad=12)
    ax.set_xlabel('Column (X)')
    ax.set_ylabel('Row (Y)')
    ax.invert_yaxis()
    save('01_spatial_heatmap.png')

def plot_survival(d):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(d['ticks'], d['alive_counts'], color='cyan', linewidth=2)
    ax.fill_between(d['ticks'], d['alive_counts'], color='cyan', alpha=0.12)
    ax.set_title(f'Agent Survival Curve\n{_seed_label(d)}', fontsize=13)
    ax.set_xlabel('Game Tick');  ax.set_ylabel('Agents Alive (avg)')
    ax.grid(alpha=0.25)
    save('02_survival_curve.png')

def plot_economy(d):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(d['ticks'], d['avg_gold'],  label='Avg Gold',  color='gold',       linewidth=2)
    ax.plot(d['ticks'], d['avg_food'],  label='Avg Food',  color='lightgreen', linewidth=2)
    ax.plot(d['ticks'], d['avg_water'], label='Avg Water', color='lightblue',  linewidth=2)
    ax.set_title(f'Resource Economy Over Time\n{_seed_label(d)}', fontsize=13)
    ax.set_xlabel('Game Tick');  ax.set_ylabel('Amount (avg)')
    ax.legend();  ax.grid(alpha=0.25)
    save('03_resource_economy.png')

def plot_death_causes(d):
    causes = ['starvation', 'dehydration', 'combat_or_fog']
    colors = ['#e07b39', '#4fc3f7', '#ef5350']
    ticks  = d['ticks']
    arrays = {c: np.zeros(len(ticks)) for c in causes}

    for tick_idx, tick_deaths in d['death_causes'].items():
        if tick_idx < len(ticks):
            for cause, count in tick_deaths.items():
                if cause in arrays:
                    arrays[cause][tick_idx] += count

    W = max(1, len(ticks) // 50)
    smoothed = {c: np.convolve(arrays[c], np.ones(W)/W, mode='same') for c in causes}

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.stackplot(ticks, [smoothed[c] for c in causes],
                 labels=['Starvation', 'Dehydration', 'Combat / Fog'],
                 colors=colors, alpha=0.85)
    ax.set_title(f'Cause of Death Over Time (Smoothed)\n{_seed_label(d)}', fontsize=13)
    ax.set_xlabel('Game Tick');  ax.set_ylabel('Deaths per Tick (smoothed, avg)')
    ax.legend(loc='upper left', fontsize=9);  ax.grid(alpha=0.2)
    save('04_death_causes.png')

def plot_radar(d):
    N      = len(SKILLS)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    all_sk = list(d['terminal_skills'].values())
    if not all_sk:
        return
    avg = [np.mean([ag.get(sk, 1) for ag in all_sk]) for sk in SKILLS] + [0]
    avg[-1] = avg[0]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, avg, color='cyan', linewidth=2)
    ax.fill(angles, avg, color='cyan', alpha=0.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([s.capitalize() for s in SKILLS], fontsize=11)
    ax.set_title(f'Average Terminal Skill Levels (Radar)\n{_seed_label(d)}', fontsize=13, pad=20)
    ax.grid(alpha=0.35)
    save('05_skill_radar.png')

def plot_skill_radar_by_death_cause(d):
    N      = len(SKILLS)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    palette = {'combat_or_fog': '#ef5350', 'starvation': '#e07b39',
               'dehydration': '#4fc3f7', 'other': '#78909c'}

    cause_groups = defaultdict(list)
    for pid, cause in d['agent_death_cause'].items():
        sk = d['terminal_skills'].get(pid)
        if sk:
            cause_groups[cause].append(sk)
    if not cause_groups:
        return

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for cause, agents in cause_groups.items():
        avg = [np.mean([ag.get(sk, 1) for ag in agents]) for sk in SKILLS]
        avg += avg[:1]
        label = "Combat/Fog" if cause == "combat_or_fog" else cause.capitalize()
        ax.plot(angles, avg, label=label, color=palette.get(cause, 'white'), linewidth=1.8)
        ax.fill(angles, avg, color=palette.get(cause, 'white'), alpha=0.07)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([s.capitalize() for s in SKILLS], fontsize=10)
    ax.set_title(f'Skill Profile by Cause of Death\n{_seed_label(d)}', fontsize=13, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)
    ax.grid(alpha=0.35)
    save('06_skill_radar_by_death.png')

def plot_lifetime_distribution(d):
    lifetimes = list(d['lifetimes'].values())
    if not lifetimes:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lifetimes, bins=60, color='cyan', edgecolor='none', alpha=0.8)
    if len(lifetimes) > 5:
        kde = gaussian_kde(lifetimes, bw_method=0.15)
        xs  = np.linspace(0, max(lifetimes), 400)
        ax2 = ax.twinx()
        ax2.plot(xs, kde(xs), color='orange', linewidth=2, label='KDE')
        ax2.set_ylabel('Density', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        ax2.legend(loc='upper right')
    ax.set_title(f'Agent Lifetime Distribution\n{_seed_label(d)}', fontsize=13)
    ax.set_xlabel('Survival Ticks');  ax.set_ylabel('Count (pooled)')
    ax.grid(alpha=0.2)
    save('07_lifetime_distribution.png')

def plot_efficiency_frontier(d):
    lifetimes = d['lifetimes'];  gold = d['terminal_gold']
    pids = list(lifetimes.keys())
    lts  = np.array([lifetimes[p]   for p in pids], dtype=float)
    gs   = np.array([gold.get(p, 0) for p in pids], dtype=float)
    deaths = [d['agent_death_cause'].get(p, 'other') for p in pids]
    palette = {'combat_or_fog': '#ef5350', 'starvation': '#e07b39',
               'dehydration': '#4fc3f7', 'other': '#78909c'}
    colors  = [palette.get(dc, '#78909c') for dc in deaths]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(lts, gs, c=colors, alpha=0.35, s=10, linewidths=0)

    # Pareto frontier
    sorted_idx = np.argsort(lts)
    max_g = -np.inf
    pareto_x, pareto_y = [], []
    for i in sorted_idx:
        if gs[i] > max_g:
            max_g = gs[i];  pareto_x.append(lts[i]);  pareto_y.append(gs[i])
    ax.plot(pareto_x, pareto_y, color='yellow', linewidth=1.5,
            linestyle='--', label='Pareto Frontier', zorder=5)

    handles = [mpatches.Patch(color=v,
                              label="Combat/Fog" if k == "combat_or_fog" else k.capitalize())
               for k, v in palette.items()]
    handles.append(plt.Line2D([0], [0], color='yellow', linestyle='--', label='Pareto Frontier'))
    ax.legend(handles=handles, fontsize=9)
    ax.set_title(f'Efficiency Frontier — Lifetime vs Terminal Gold\n{_seed_label(d)}', fontsize=13)
    ax.set_xlabel('Survival Ticks');  ax.set_ylabel('Terminal Gold')
    ax.grid(alpha=0.2)
    save('08_efficiency_frontier.png')

def plot_population_entropy(d):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(d['ticks'], d['entropy'], color='#a5d6a7', linewidth=1.5)
    ax.fill_between(d['ticks'], d['entropy'], color='#a5d6a7', alpha=0.15)
    ax.set_title(f'Population Spatial Entropy Over Time\n{_seed_label(d)}', fontsize=13)
    ax.set_xlabel('Game Tick');  ax.set_ylabel('Shannon Entropy (bits, avg)')
    ax.grid(alpha=0.25)
    ent_arr = np.array(d['entropy'])
    if len(ent_arr) > 10:
        peak_t = d['ticks'][int(np.argmax(ent_arr))]
        ax.axvline(peak_t, color='yellow', linestyle=':', linewidth=1.2, alpha=0.7)
        ax.text(peak_t + 2, max(ent_arr) * 0.95, 'Peak Exploration',
                color='yellow', fontsize=8)
    save('09_population_entropy.png')

def plot_combat_economy(d):
    W = max(1, len(d['ticks']) // 50)
    cs = np.convolve(d['combat_deaths'],   np.ones(W)/W, mode='same')
    rs = np.convolve(d['resource_deaths'], np.ones(W)/W, mode='same')
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(d['ticks'], cs, color='#ef5350', linewidth=2, label='Combat/Fog Deaths')
    ax.plot(d['ticks'], rs, color='#e07b39', linewidth=2, label='Resource Deaths')
    ax.fill_between(d['ticks'], cs, color='#ef5350', alpha=0.1)
    ax.fill_between(d['ticks'], rs, color='#e07b39', alpha=0.1)
    ax.set_title(f'Mortality: Combat vs Resource Rate (Smoothed)\n{_seed_label(d)}', fontsize=13)
    ax.set_xlabel('Game Tick');  ax.set_ylabel('Deaths per Tick (smoothed, avg)')
    ax.legend();  ax.grid(alpha=0.25)
    save('10_combat_engagement.png')

def plot_survival_by_death_cause(d):
    cause_lifetimes = defaultdict(list)
    for pid, cause in d['agent_death_cause'].items():
        lt = d['lifetimes'].get(pid)
        if lt is not None:
            cause_lifetimes[cause].append(lt)
    if not cause_lifetimes:
        return
    palette = {'combat_or_fog': '#ef5350', 'starvation': '#e07b39',
               'dehydration': '#4fc3f7', 'other': '#78909c'}
    causes  = list(cause_lifetimes.keys())
    means   = [np.mean(cause_lifetimes[c]) for c in causes]
    stds    = [np.std (cause_lifetimes[c]) for c in causes]
    colors  = [palette.get(c, '#78909c') for c in causes]
    labels  = ["Combat/Fog" if c == "combat_or_fog" else c.capitalize() for c in causes]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, means, yerr=stds, color=colors, capsize=5, edgecolor='none', alpha=0.85)
    ax.set_title(f'Mean Survival Ticks by Cause of Death\n{_seed_label(d)}', fontsize=13)
    ax.set_ylabel('Mean Ticks Survived (± std, pooled)');  ax.grid(axis='y', alpha=0.25)
    save('11_survival_by_cause.png')

def print_summary_stats(d):
    lifetimes = list(d['lifetimes'].values())
    print("\n" + "="*52)
    print(f"  SUMMARY STATISTICS  {_seed_label(d)}")
    print("="*52)
    print(f"  Seeds loaded         : {d.get('n_seeds', '?')}")
    print(f"  Total agents tracked : {len(lifetimes)}")
    print(f"  Ticks (min run)      : {len(d['ticks'])}")
    if lifetimes:
        print(f"  Mean lifetime        : {np.mean(lifetimes):.1f}")
        print(f"  Median lifetime      : {np.median(lifetimes):.1f}")
        print(f"  Std lifetime         : {np.std(lifetimes):.1f}")
        print(f"  Max lifetime         : {max(lifetimes)}")
    causes = defaultdict(int)
    for c in d['agent_death_cause'].values():
        causes[c] += 1
    print("  Death causes (pooled):")
    for c, n in sorted(causes.items(), key=lambda x: -x[1]):
        label = "Combat/Fog" if c == "combat_or_fog" else c.capitalize()
        print(f"    {label:<18}: {n}")
    all_sk = list(d['terminal_skills'].values())
    if all_sk:
        print("  Avg terminal skills (pooled):")
        avgs = {sk: np.mean([a.get(sk, 1) for a in all_sk]) for sk in SKILLS}
        for sk, v in sorted(avgs.items(), key=lambda x: -x[1]):
            print(f"    {sk:<18}: {v:.2f}")
    print("="*52)


#  MAIN


def main():
    plt.style.use(STYLE)

    print(f"\nLoading and aggregating {NUM_SEEDS} seeds …")
    d = aggregate_seeds(REPLAY_FILES)
    if d is None:
        return

    print("\nRendering graphs …")
    plot_heatmap(d)
    plot_survival(d)
    plot_economy(d)
    plot_death_causes(d)
    plot_radar(d)
    plot_skill_radar_by_death_cause(d)
    plot_lifetime_distribution(d)
    plot_efficiency_frontier(d)
    plot_population_entropy(d)
    plot_combat_economy(d)
    plot_survival_by_death_cause(d)
    print_summary_stats(d)

    print(f"\nDone! 11 multi-seed graphs saved to '{OUTPUT_DIR}/'.")

if __name__ == '__main__':
    main()