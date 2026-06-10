# this file is for the PVP Agent, which will be used in the PVP competition.
# It defines the NMMOAgent class, which implements the logic for controlling a team of
# agents in the NMMO environment. The agent uses a neural network for decision making,
# but also incorporates heuristic strategies based on the assigned faction for added diversity.
import torch
import torch.nn as nn
import numpy as np

from .const import N_PLAYER_PER_TEAM
from .model import NMMONet
from .util import tensorize_state, legal_mask

# ---------------------------------------------------------------------------
#  NMMO move-action indices
# ---------------------------------------------------------------------------
MOVE_NORTH = 0
MOVE_SOUTH = 1
MOVE_EAST  = 2
MOVE_WEST  = 3
MOVE_STAY  = 4

DETECTION_RADIUS = 4
HEURISTIC_OVERRIDE_RATE = 0.95

# --- Helper Directions ---
def _direction_towards(dr: int, dc: int) -> int:
    if abs(dr) >= abs(dc): return MOVE_SOUTH if dr > 0 else MOVE_NORTH
    else: return MOVE_EAST if dc > 0 else MOVE_WEST

def _direction_away(dr: int, dc: int) -> int:
    if abs(dr) >= abs(dc): return MOVE_NORTH if dr > 0 else MOVE_SOUTH
    else: return MOVE_WEST if dc > 0 else MOVE_EAST

# --- Strategy 1 & 2: Combat & Evasion ---
def _combat_heuristic(agent_obs: dict, strategy: str):
    entity_obs = np.asarray(agent_obs.get('Entity', []))
    if entity_obs.ndim != 2 or entity_obs.shape[1] < 5: return None

    best_dist, best_action = DETECTION_RADIUS + 1, None
    for row in entity_obs:
        is_player, is_alive, dr, dc = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        if is_player == 1 and is_alive == 1 and (dr != 0 or dc != 0):
            dist = abs(dr) + abs(dc)
            if dist < best_dist:
                best_dist = dist
                if strategy == 'evade': best_action = _direction_away(dr, dc)
                elif strategy == 'chase': best_action = _direction_towards(dr, dc)
    return best_action 

# --- Strategy 3: The Cowardly Bully ---
def _bully_heuristic(agent_obs: dict):
    entity_obs = np.asarray(agent_obs.get('Entity', []))
    if entity_obs.ndim != 2 or entity_obs.shape[1] < 5: return None

    enemies = []
    for row in entity_obs:
        is_player, is_alive, dr, dc = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        if is_player == 1 and is_alive == 1 and (dr != 0 or dc != 0):
            enemies.append((dr, dc))
            
    if len(enemies) == 0: return None
    elif len(enemies) == 1:
        # 1 enemy = Chase
        return _direction_towards(enemies[0][0], enemies[0][1])
    else:
        # Multiple enemies = Panic and Evade the closest
        return _direction_away(enemies[0][0], enemies[0][1])

# --- Strategy 4: Social Distancing (Anti-AoE) ---
def _social_distance_heuristic(agent_obs: dict):
    entity_obs = np.asarray(agent_obs.get('Entity', []))
    if entity_obs.ndim != 2: return None

    for row in entity_obs:
        is_player, is_alive, dr, dc = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        # If any player is exactly 1 tile away (or on the same tile but not us)
        if is_player == 1 and is_alive == 1 and (abs(dr) + abs(dc) <= 1) and (dr != 0 or dc != 0):
            return _direction_away(dr, dc)
    return None

# ===========================================================================
#  Base Agent
# ===========================================================================
class Agent:
    def __init__(self, use_gpu, *args, **kwargs):
        self.use_gpu = use_gpu
        self.device = torch.device('cuda') if use_gpu else torch.device('cpu')
        self.state_handler_dict = {}
        torch.set_num_threads(1)
        self.training_iter = 0

    def register_model(self, name, model):
        assert isinstance(model, nn.Module)
        self.state_handler_dict[name] = model

    def loads(self, agent_dict):
        self.training_iter = agent_dict['training_iter']
        for name, np_dict in agent_dict['model_dict'].items():
            model = self.state_handler_dict[name] 
            state_dict = {
                k: torch.as_tensor(v.copy(), device=self.device)
                for k, v in zip(model.state_dict().keys(), np_dict.values())
            }
            model.load_state_dict(state_dict)

# ===========================================================================
#  NMMOAgent — Faction Controller
# ===========================================================================
class NMMOAgent(Agent):
    def __init__(self, use_gpu, team_id="RealikunTeam-0"):
        super().__init__(use_gpu)
        self.net = NMMONet().to(self.device)
        self.register_model('net', self.net)

        self.hx = torch.zeros((N_PLAYER_PER_TEAM, self.net.n_lstm_hidden))
        self.cx = torch.zeros((N_PLAYER_PER_TEAM, self.net.n_lstm_hidden))

        # Assign faction based on Team ID
        self.team_id = team_id
        team_num = int(self.team_id.split('-')[1])
        
        if team_num < 2:   self.faction = 'pacifist'
        elif team_num < 4: self.faction = 'aggressive'
        elif team_num < 6: self.faction = 'bodyguards'
        elif team_num < 8: self.faction = 'bullies'
        elif team_num < 10: self.faction = 'distancers'
        else:              self.faction = 'baseline'

    @tensorize_state
    def infer(self, state, raw_obs=None, train=True):
        with torch.no_grad():
            logits, value, self.hx, self.cx = self.net.infer(state, self.hx, self.cx)
            logits = {k: legal_mask(logits[k], state['legal'][k]) for k in logits}
            dists = {k: torch.distributions.Categorical(logits=logits[k]) for k in logits}
            
            if train: actions = {k: dists[k].sample() for k in logits}
            else: actions = {k: dists[k].probs.argmax(dim=-1) for k in logits}

            action_keys = list(actions.keys())
            move_key    = action_keys[0]

            # Faction Override Logic
            if raw_obs is not None and self.faction != 'baseline':
                move_tensor = actions[move_key].clone()

                # --- Extract VIP position for Bodyguards ---
                vip_r, vip_c = None, None
                if self.faction == 'bodyguards':
                    vip_obs = raw_obs.get(0) or raw_obs.get('0')
                    if vip_obs:
                        base = vip_obs.get('base', vip_obs)
                        vip_r = base.get('r') or base.get('row')
                        vip_c = base.get('c') or base.get('col')

                for player_idx in range(N_PLAYER_PER_TEAM):
                    agent_obs = raw_obs.get(player_idx) or raw_obs.get(str(player_idx))
                    if agent_obs is None: continue

                    if np.random.random() > HEURISTIC_OVERRIDE_RATE: continue

                    forced_move = None
                    
                    if self.faction == 'pacifist':
                        forced_move = _combat_heuristic(agent_obs, 'evade')
                    elif self.faction == 'aggressive':
                        forced_move = _combat_heuristic(agent_obs, 'chase')
                    elif self.faction == 'bullies':
                        forced_move = _bully_heuristic(agent_obs)
                    elif self.faction == 'distancers':
                        forced_move = _social_distance_heuristic(agent_obs)
                    elif self.faction == 'bodyguards':
                        if player_idx != 0 and vip_r is not None and vip_c is not None:
                            agent_base = agent_obs.get('base', agent_obs)
                            my_r = agent_base.get('r') or agent_base.get('row')
                            my_c = agent_base.get('c') or agent_base.get('col')
                            if my_r is not None and my_c is not None:
                                dr, dc = vip_r - my_r, vip_c - my_c
                                # Regroup if separated by more than 3 tiles
                                if abs(dr) + abs(dc) > 3:
                                    forced_move = _direction_towards(dr, dc)
                    
                    if forced_move is not None:
                        move_tensor[player_idx] = forced_move

                actions[move_key] = move_tensor

            actions_list = [a.numpy() for a in actions.values()]

        return actions_list