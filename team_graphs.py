import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
from collections import defaultdict


#  CONFIGURATION

NUM_SEEDS   = 5
REPLAY_FILES= [f"faction_war_seed_{i}.json" for i in range(1, NUM_SEEDS + 1)]
OUTPUT_DIR  = 'plots/v4'
DPI         = 300
STYLE       = 'dark_background'
NUM_TEAMS   = 16
TEAM_SIZE   = 8

SKILLS = ['melee', 'range', 'mage', 'fishing', 'herbalism', 'prospecting', 'carving', 'alchemy']

FACTION_MAP = {
    0: 'Pacifist', 1: 'Pacifist',
    2: 'Aggressive', 3: 'Aggressive',
    4: 'Bodyguards', 5: 'Bodyguards',
    6: 'Bullies', 7: 'Bullies',
    8: 'Distancers', 9: 'Distancers',
    10: 'Baseline', 11: 'Baseline', 12: 'Baseline', 13: 'Baseline', 14: 'Baseline', 15: 'Baseline'
}

FACTION_COLORS = {
    'Pacifist': '#4fc3f7',    
    'Aggressive': '#ef5350',  
    'Bodyguards': '#ffd54f',  
    'Bullies': '#ff9800',     
    'Distancers': '#ab47bc',  
    'Baseline': '#90a4ae'     
}


#  HELPERS

def safe_extract(value):
    if isinstance(value, dict): return value.get('val', 0)
    return value if value is not None else 0

def safe_skill(p_data, skill_name):
    paths = [('skills', skill_name, 'level'), ('skills', skill_name), (skill_name, 'level')]
    for path in paths:
        node = p_data
        for key in path:
            if isinstance(node, dict): node = node.get(key)
            else: node = None; break
        if node is not None: return safe_extract(node)
    return 1 

def get_team(pid):
    return (int(pid) - 1) // TEAM_SIZE

def save(name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {path}")

def get_legend_handles():
    return [mpatches.Patch(color=color, label=faction) for faction, color in FACTION_COLORS.items()]


#  DATA EXTRACTION (Single File)

def extract_single_seed(json_path):
    print(f"  Reading {json_path}...")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Failed to load {json_path}: {e}")
        return None
        
    packets = data if isinstance(data, list) else data.get('packets', [])
    num_ticks = len(packets)
    
    t_alive    = np.zeros((num_ticks, NUM_TEAMS))
    t_gold     = np.zeros((num_ticks, NUM_TEAMS))
    t_cohesion = np.zeros((num_ticks, NUM_TEAMS)) 
    
    unique_tiles  = {tid: set() for tid in range(NUM_TEAMS)}
    death_combat  = np.zeros(NUM_TEAMS)
    death_resource= np.zeros(NUM_TEAMS)
    terminal_skills = {tid: defaultdict(list) for tid in range(NUM_TEAMS)}
    
    prev_players = {}

    for tick_idx, packet in enumerate(packets):
        players = packet.get('player', {})
        curr_ids = set(players.keys())
        
        died_this_tick = set(prev_players.keys()) - curr_ids
        for pid in died_this_tick:
            tid = get_team(pid)
            if tid < 0 or tid >= NUM_TEAMS: continue
            last_state = prev_players[pid]
            food = safe_extract(last_state.get('food', last_state.get('resource', {}).get('food', 0)))
            water = safe_extract(last_state.get('water', last_state.get('resource', {}).get('water', 0)))
            if food <= 0 or water <= 0:
                death_resource[tid] += 1
            else:
                death_combat[tid] += 1

        team_positions = defaultdict(list)
        for pid, p in players.items():
            tid = get_team(pid)
            if tid < 0 or tid >= NUM_TEAMS: continue
                
            t_alive[tick_idx, tid] += 1
            t_gold[tick_idx, tid] += safe_extract(p.get('gold', p.get('base', {}).get('gold', 0)))

            base_data = p.get('base', p)
            r = base_data.get('r') or (p.get('pos', [None, None])[0] if p.get('pos') else None) or base_data.get('row')
            c = base_data.get('c') or (p.get('pos', [None, None])[1] if p.get('pos') else None) or base_data.get('col')
            
            if r is not None and c is not None:
                r, c = int(r), int(c)
                unique_tiles[tid].add((r, c))
                team_positions[tid].append((r, c))
            
            for sk in SKILLS:
                terminal_skills[tid][pid] = {sk: safe_skill(p, sk) for sk in SKILLS}

        for tid, positions in team_positions.items():
            if len(positions) > 1:
                centroid_r = np.mean([pos[0] for pos in positions])
                centroid_c = np.mean([pos[1] for pos in positions])
                spread = np.mean([abs(pos[0] - centroid_r) + abs(pos[1] - centroid_c) for pos in positions])
                t_cohesion[tick_idx, tid] = spread
            else:
                t_cohesion[tick_idx, tid] = 0 
                
        prev_players = players

    return num_ticks, t_alive, t_gold, t_cohesion, unique_tiles, death_combat, death_resource, terminal_skills


#  AGGREGATION ENGINE

def aggregate_seeds(files):
    print(f"Aggregating data across {len(files)} seeds...")
    
    ts_alive_list, ts_gold_list, ts_cohesion_list = [], [], []
    agg_exploration = np.zeros((len(files), NUM_TEAMS))
    agg_combat      = np.zeros((len(files), NUM_TEAMS))
    agg_resource    = np.zeros((len(files), NUM_TEAMS))
    agg_skills      = np.zeros((len(files), NUM_TEAMS))
    
    for i, path in enumerate(files):
        result = extract_single_seed(path)
        if result is None: continue
        
        num_ticks, t_alive, t_gold, t_cohesion, unique_tiles, death_combat, death_resource, terminal_skills = result
        
        ts_alive_list.append(t_alive)
        ts_gold_list.append(t_gold)
        ts_cohesion_list.append(t_cohesion)
        
        for tid in range(NUM_TEAMS):
            agg_exploration[i, tid] = len(unique_tiles[tid])
            agg_combat[i, tid]      = death_combat[tid]
            agg_resource[i, tid]    = death_resource[tid]
            
            team_skill_sum = 0
            if terminal_skills[tid]:
                for pid, sk_dict in terminal_skills[tid].items():
                    team_skill_sum += sum(sk_dict.values())
                agg_skills[i, tid] = team_skill_sum / len(terminal_skills[tid])

    # Truncate time-series arrays to match the shortest rollout to avoid numpy shape errors
    min_ticks = min([len(ts) for ts in ts_alive_list])
    
    avg_alive    = np.mean([ts[:min_ticks] for ts in ts_alive_list], axis=0)
    avg_gold     = np.mean([ts[:min_ticks] for ts in ts_gold_list], axis=0)
    avg_cohesion = np.mean([ts[:min_ticks] for ts in ts_cohesion_list], axis=0)
    
    # Average the cumulative bar-chart metrics
    avg_exploration = np.mean(agg_exploration, axis=0)
    avg_combat      = np.mean(agg_combat, axis=0)
    avg_resource    = np.mean(agg_resource, axis=0)
    avg_skills_f    = np.mean(agg_skills, axis=0)
    
    return min_ticks, avg_alive, avg_gold, avg_cohesion, avg_exploration, avg_combat, avg_resource, avg_skills_f


#  GRAPHS

def plot_survival_all(ticks, avg_alive):
    fig, ax = plt.subplots(figsize=(12, 6))
    for tid in range(NUM_TEAMS):
        color = FACTION_COLORS[FACTION_MAP[tid]]
        alpha = 0.8 if FACTION_MAP[tid] != 'Baseline' else 0.3
        ax.plot(range(ticks), avg_alive[:, tid], linewidth=2, color=color, alpha=alpha)
            
    ax.set_title('Team Survival Dynamics by Faction (Averaged across 5 Seeds)', fontsize=14)
    ax.set_xlabel('Game Tick', fontsize=12)
    ax.set_ylabel('Active Agents (Average)', fontsize=12)
    ax.legend(handles=get_legend_handles(), loc='lower left')
    ax.grid(alpha=0.2)
    save('12_faction_survival_multiseed.png')

def plot_wealth_all(ticks, avg_gold):
    fig, ax = plt.subplots(figsize=(12, 6))
    for tid in range(NUM_TEAMS):
        color = FACTION_COLORS[FACTION_MAP[tid]]
        alpha = 0.8 if FACTION_MAP[tid] != 'Baseline' else 0.3
        ax.plot(range(ticks), avg_gold[:, tid], linewidth=2, color=color, alpha=alpha)
            
    ax.set_title('Economy: Total Team Wealth Over Time (Averaged across 5 Seeds)', fontsize=14)
    ax.set_xlabel('Game Tick', fontsize=12)
    ax.set_ylabel('Total Team Gold (Average)', fontsize=12)
    ax.legend(handles=get_legend_handles(), loc='upper left')
    ax.grid(alpha=0.2)
    save('13_faction_economy_multiseed.png')

def plot_exploration(avg_exploration):
    fig, ax = plt.subplots(figsize=(10, 6))
    teams = np.arange(NUM_TEAMS)
    colors = [FACTION_COLORS[FACTION_MAP[tid]] for tid in teams]
    
    ax.bar(teams, avg_exploration, color=colors, alpha=0.85)
    ax.set_title('Exploration: Unique Tiles Visited (Averaged across 5 Seeds)', fontsize=14)
    ax.set_xlabel('Team ID', fontsize=12)
    ax.set_ylabel('Total Unique Tiles (Average)', fontsize=12)
    ax.set_xticks(teams)
    ax.legend(handles=get_legend_handles(), loc='upper left')
    ax.grid(axis='y', alpha=0.2)
    save('14_faction_exploration_multiseed.png')

def plot_cohesion(ticks, avg_cohesion):
    fig, ax = plt.subplots(figsize=(12, 6))
    for tid in range(NUM_TEAMS):
        smoothed = np.convolve(avg_cohesion[:, tid], np.ones(20)/20, mode='valid')
        color = FACTION_COLORS[FACTION_MAP[tid]]
        alpha = 0.8 if FACTION_MAP[tid] != 'Baseline' else 0.3
        ax.plot(range(len(smoothed)), smoothed, linewidth=2, color=color, alpha=alpha)
            
    ax.set_title('Team Cohesion (Averaged across 5 Seeds)', fontsize=14)
    ax.set_xlabel('Game Tick', fontsize=12)
    ax.set_ylabel('Average Distance to Team Center', fontsize=12)
    ax.legend(handles=get_legend_handles(), loc='upper left')
    ax.grid(alpha=0.2)
    save('15_faction_cohesion_multiseed.png')

def plot_mortality(avg_combat, avg_resource):
    fig, ax = plt.subplots(figsize=(12, 6))
    teams = np.arange(NUM_TEAMS)
    
    ax.bar(teams, avg_combat, label='Combat/Fog', color='#ef5350', alpha=0.85)
    ax.bar(teams, avg_resource, bottom=avg_combat, label='Starvation/Dehydration', color='#4fc3f7', alpha=0.85)
    
    ax.set_title('Mortality Breakdown by Team (Averaged across 5 Seeds)', fontsize=14)
    ax.set_xlabel('Team ID', fontsize=12)
    ax.set_ylabel('Total Deaths (Average)', fontsize=12)
    ax.set_xticks(teams)
    
    for tick_label, tid in zip(ax.get_xticklabels(), teams):
        tick_label.set_color(FACTION_COLORS[FACTION_MAP[tid]])
        tick_label.set_fontweight('bold')
        
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.2)
    save('16_faction_mortality_multiseed.png')

def plot_skills(avg_skills_f):
    fig, ax = plt.subplots(figsize=(12, 6))
    teams = np.arange(NUM_TEAMS)
    colors = [FACTION_COLORS[FACTION_MAP[tid]] for tid in teams]
    
    ax.bar(teams, avg_skills_f, color=colors, alpha=0.85)
    ax.set_title('Terminal Capability: Cumulative Skill Level (Averaged across 5 Seeds)', fontsize=14)
    ax.set_xlabel('Team ID', fontsize=12)
    ax.set_ylabel('Skill Level (Average)', fontsize=12)
    ax.set_xticks(teams)
    ax.legend(handles=get_legend_handles(), loc='upper left')
    ax.grid(axis='y', alpha=0.2)
    save('17_faction_skills_multiseed.png')

def main():
    plt.style.use(STYLE)
    result = aggregate_seeds(REPLAY_FILES)
    if result is None: return
    
    min_ticks, avg_alive, avg_gold, avg_cohesion, avg_exploration, avg_combat, avg_resource, avg_skills_f = result
    
    print("\nRendering averaged faction warfare graphs...")
    plot_survival_all(min_ticks, avg_alive)
    plot_wealth_all(min_ticks, avg_gold)
    plot_exploration(avg_exploration)
    plot_cohesion(min_ticks, avg_cohesion)
    plot_mortality(avg_combat, avg_resource)
    plot_skills(avg_skills_f)
    print("\nSuccess! Multi-seed aggregated graphs exported to the 'plots' folder.")

if __name__ == '__main__':
    main()