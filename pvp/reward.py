# this file is for reward shaping in the PVP competition. 
# It defines the RewardConfig class, which contains the configuration for the reward shaping, 
# and the RewardCalculator class, which computes the rewards based on the observations and the 
# translator's state. The reward shaping is designed to encourage survival, 
# combat, gathering, exploration, and teamwork while penalizing death and standing in poison zones.
"""
reward.py — Reward shaping for NeuralMMO multi-agent training.

Reward signals are deliberately sparse at the team level and shaped at the
individual level to guide agents toward survival, combat, gathering and
cooperative play without over-specifying a single strategy.

Reward components (per agent per step, unless stated otherwise):
  Survival: flat bonus for staying alive each step
  Health: shaped delta reward on HP change
  Food/Water: shaped delta reward on food/water change
  Combat: reward for killing NPCs and enemy players
  Gathering: reward for stepping onto herb tiles (herbalism)
  Exploration:reward for visiting new tiles (via fog / visit map)
  Teamwork:shared bonus when the whole team survives a milestone
  Poison: penalty for standing in the poison zone
  Death : one-time penalty on the step an agent dies
"""

import numpy as np

from const import (
    IDX_ENT_HEALTH, IDX_ENT_FOOD, IDX_ENT_WATER,
    IDX_ENT_ROW_INDEX, IDX_ENT_COL_INDEX,
    N_PLAYER_PER_TEAM, MAX_STEP,
    MAP_LEFT, MAP_RIGHT)


# 
# Weight configuration
# 
class RewardConfig:
    # Survival
    SURVIVAL_PER_STEP: float = 0.01          # small bonus each step alive

    # Health / resources
    HP_GAIN_SCALE: float = 0.002             # per HP point recovered
    HP_LOSS_SCALE: float = 0.002             # per HP point lost (penalty)
    FOOD_GAIN_SCALE: float = 0.001
    WATER_GAIN_SCALE: float = 0.001
    FOOD_LOW_THRESHOLD: float = 30.0         # below this, apply penalty
    FOOD_LOW_PENALTY: float = 0.005
    WATER_LOW_THRESHOLD: float = 30.0
    WATER_LOW_PENALTY: float = 0.005

    # Combat
    KILL_HOSTILE_NPC: float = 0.5            # hostile NPC kill
    KILL_NEUTRAL_NPC: float = 0.2            # neutral NPC kill (lower to discourage griefing)
    KILL_PASSIVE_NPC: float = 0.1            # passive NPC kill
    KILL_PLAYER: float = 1.0                 # enemy player kill

    # Gathering
    HERB_STEP_REWARD: float = 0.3            # stepping onto a herb tile

    # Exploration
    EXPLORE_NEW_TILE: float = 0.002          # per newly revealed tile

    # Team milestones (shared across the team, divided evenly)
    TEAM_ALIVE_MILESTONE_STEPS: tuple = (256, 512, 768)
    TEAM_MILESTONE_BONUS: float = 1.0        # total bonus, divided by team size

    # Poison zone
    POISON_PENALTY_SCALE: float = 0.02       # multiplied by poison strength value

    # Death
    DEATH_PENALTY: float = -1.0


# 
# Reward calculator
# 
class RewardCalculator:
# The RewardCalculator class computes the reward for each agent based on the current 
# observations and the translator's state. It maintains internal state to track previous 
# health, food, water levels, explored tiles, and team milestones. The compute method calculates 
# the reward for each agent according to the defined reward components and updates the internal 
# state accordingly.

    def __init__(self, cfg: RewardConfig = None):
        self.cfg = cfg or RewardConfig()
        self._prev_hp: list = None
        self._prev_food: list = None
        self._prev_water: list = None
        self._explored: np.ndarray = None   # (TERRAIN_SIZE+1, TERRAIN_SIZE+1) bool
        self._milestone_given: set = None
        self._terrain_size: int = 161       # TERRAIN_SIZE + 1

    # 
    # Public API
    # 

    def reset(self):
        self._prev_hp = [100.0] * N_PLAYER_PER_TEAM
        self._prev_food = [100.0] * N_PLAYER_PER_TEAM
        self._prev_water = [100.0] * N_PLAYER_PER_TEAM
        self._explored = np.zeros(
            (self._terrain_size, self._terrain_size), dtype=bool
        )
        self._milestone_given = set()

    def compute(self, translator, obs: dict, dones: dict) -> dict:

        cfg = self.cfg
        rewards = {i: 0.0 for i in range(N_PLAYER_PER_TEAM)}

        step = translator.curr_step

        for i in range(N_PLAYER_PER_TEAM):
            r = 0.0

            # -- Death penalty (applied once)
            if dones.get(i, False):
                r += cfg.DEATH_PENALTY
                rewards[i] += r
                # Reset bookkeeping so stale values don't bleed into next ep
                self._prev_hp[i] = 0.0
                self._prev_food[i] = 0.0
                self._prev_water[i] = 0.0
                continue  # dead agents get no other reward this step

            if i not in obs:
                continue  # already dead in a prior step

            ent = obs[i]['Entity']['Continuous'][0]
            curr_hp = float(ent[IDX_ENT_HEALTH])
            curr_food = float(ent[IDX_ENT_FOOD])
            curr_water = float(ent[IDX_ENT_WATER])

            # -- Survival bonus
            r += cfg.SURVIVAL_PER_STEP

            # -- Health delta
            hp_delta = curr_hp - self._prev_hp[i]
            if hp_delta > 0:
                r += hp_delta * cfg.HP_GAIN_SCALE
            else:
                r += hp_delta * cfg.HP_LOSS_SCALE  # negative * positive_scale = penalty

            # -- Food / water delta
            food_delta = curr_food - self._prev_food[i]
            water_delta = curr_water - self._prev_water[i]
            if food_delta > 0:
                r += food_delta * cfg.FOOD_GAIN_SCALE
            if water_delta > 0:
                r += water_delta * cfg.WATER_GAIN_SCALE
            if curr_food < cfg.FOOD_LOW_THRESHOLD:
                r -= cfg.FOOD_LOW_PENALTY
            if curr_water < cfg.WATER_LOW_THRESHOLD:
                r -= cfg.WATER_LOW_PENALTY

            # -- Update bookkeeping
            self._prev_hp[i] = curr_hp
            self._prev_food[i] = curr_food
            self._prev_water[i] = curr_water

            # -- Combat rewards 
            r += translator.npc_kill_num['h'][i] * cfg.KILL_HOSTILE_NPC
            r += translator.npc_kill_num['n'][i] * cfg.KILL_NEUTRAL_NPC
            r += translator.npc_kill_num['p'][i] * cfg.KILL_PASSIVE_NPC
            r += translator.player_kill_num[i] * cfg.KILL_PLAYER

            # -- Herb gathering 
            r += translator.step_onto_herb_cnt[i] * cfg.HERB_STEP_REWARD

            # -- Exploration ---
            row = int(ent[IDX_ENT_ROW_INDEX])
            col = int(ent[IDX_ENT_COL_INDEX])
            view_radius = 7
            r_lo = max(MAP_LEFT, row - view_radius)
            r_hi = min(MAP_RIGHT + 1, row + view_radius + 1)
            c_lo = max(MAP_LEFT, col - view_radius)
            c_hi = min(MAP_RIGHT + 1, col + view_radius + 1)
            newly_seen = (~self._explored[r_lo:r_hi, c_lo:c_hi]).sum()
            r += newly_seen * cfg.EXPLORE_NEW_TILE
            self._explored[r_lo:r_hi, c_lo:c_hi] = True

            # -- Poison zone penalty 
            poison_val = translator.poison_map[row, col]
            if poison_val > 0:
                r -= poison_val * cfg.POISON_PENALTY_SCALE

            rewards[i] += r

        #  #
        # 2. Team milestone bonuses
        #  #
        n_alive = sum(i in obs for i in range(N_PLAYER_PER_TEAM))
        per_agent_bonus = cfg.TEAM_MILESTONE_BONUS / N_PLAYER_PER_TEAM

        for milestone_step in cfg.TEAM_ALIVE_MILESTONE_STEPS:
            if step == milestone_step and milestone_step not in self._milestone_given:
                if n_alive == N_PLAYER_PER_TEAM:
                    for i in obs:
                        rewards[i] += per_agent_bonus
                    self._milestone_given.add(milestone_step)

        #  #
        # 3. Reset per-step counters on the translator so they don't
        #    accumulate across steps (kill counts are step-delta values).
        #  #
        _reset_step_counters(translator)

        return rewards


# 
# Internal helpers
# 

def _reset_step_counters(translator) -> None:
    """Zero out the step-level counters tracked inside the Translator."""
    for i in range(N_PLAYER_PER_TEAM):
        translator.player_kill_num[i] = 0
        translator.step_onto_herb_cnt[i] = 0
    for kind in 'pnh':
        for i in range(N_PLAYER_PER_TEAM):
            translator.npc_kill_num[kind][i] = 0


# 
# Convenience: normalise a reward dict into a flat numpy array
# 

def rewards_to_array(rewards: dict) -> np.ndarray:
    """Return rewards as a (N_PLAYER_PER_TEAM,) float32 array."""
    return np.array(
        [rewards.get(i, 0.0) for i in range(N_PLAYER_PER_TEAM)],
        dtype=np.float32,
    )