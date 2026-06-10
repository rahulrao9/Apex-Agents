# this file is for training the PPO model for the PVP AI. 
# It defines the PPOTrainer class, which implements the training loop for the PPO algorithm. 
# The trainer interacts with a vectorized environment wrapper (VecNMMO) that manages multiple
#  parallel instances of the NMMO environment, collects rollouts, computes advantages, and performs
#  PPO updates on the policy network (NMMONet). The training loop also includes logging to 
# TensorBoard and checkpointing of model weights.
"""
train.py — PPO training loop for NeuralMMO multi-agent policy.

Architecture overview
---------------------
  - NMMONet: actor-critic network defined in model.py
  - Translator : obs/action translation defined in translator.py
  - RewardCalculator : shaped reward signals defined in reward.py
  - PPO: on-policy training with clipped surrogate objective, value function loss, and entropy bonus.

The training loop follows a standard on-policy rollout-then-update cycle:
  1. Collect `rollout_len` steps across all teams in the environment.
  2. Compute GAE advantages and value targets.
  3. Run `ppo_epochs` update passes over mini-batches of the rollout buffer.
  4. Repeat until `total_steps` is reached.

All hyper-parameters live in TrainConfig so they can be overridden from the
command line or a config file without touching this file.
"""

from __future__ import annotations

import argparse
import os
import time
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import nmmo

from const import N_PLAYER_PER_TEAM, N_TEAM, MAX_STEP
from model import NMMONet
from translator import Translator
from reward import RewardCalculator, RewardConfig, rewards_to_array



# Hyper-parameter container


@dataclass
class TrainConfig:
    # Environment
    num_envs: int = 4                   # parallel environment instances
    num_teams: int = N_TEAM             # teams per environment
    rollout_len: int = 128              # steps collected before each update
    total_steps: int = 500_000_000      # total env steps to train for

    # PPO
    ppo_epochs: int = 1
    num_minibatches: int = 1
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    norm_adv: bool = True
    clip_vloss: bool = True
    target_kl: Optional[float] = None  # early-stop PPO epoch if KL exceeds this

    # Optimiser
    lr: float = 1e-4
    anneal_lr: bool = True             # linear LR decay to zero

    # LSTM
    bptt_trunc_len: int = 16           # BPTT truncation length

    # Checkpointing / logging
    save_interval: int = 500           # save every N updates
    log_interval: int = 10             # TensorBoard log every N updates
    checkpoint_dir: str = "checkpoints"
    run_name: str = ""                 # auto-generated if empty
    seed: int = 42

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"



# Rollout buffer


class RolloutBuffer:
    """Stores one rollout of (state, action, reward, done, value, log_prob)."""

    def __init__(self, rollout_len: int, n_agents: int,
                 n_lstm_hidden: int, device: str):
        self.T = rollout_len
        self.A = n_agents           # total agents = num_envs * num_teams * team_size
        self.H = n_lstm_hidden
        self.device = device
        self._ptr = 0

        # These are filled step-by-step
        self.rewards   = torch.zeros(self.T, self.A, device=device)
        self.dones     = torch.zeros(self.T, self.A, device=device)
        self.values    = torch.zeros(self.T, self.A, device=device)
        self.log_probs = torch.zeros(self.T, self.A, device=device)
        self.actions   = {}     # head_name -> (T, A) int64 tensor; filled lazily
        self.states    = []     # list of length T; each entry is a dict of tensors

        # Computed after rollout
        self.advantages = None
        self.returns    = None

    def store(self, state, actions: dict, reward: torch.Tensor,
              done: torch.Tensor, value: torch.Tensor,
              log_prob: torch.Tensor):
        t = self._ptr
        self.states.append(state)
        self.rewards[t] = reward
        self.dones[t] = done
        self.values[t] = value
        self.log_probs[t] = log_prob
        for k, v in actions.items():
            if k not in self.actions:
                self.actions[k] = torch.zeros(self.T, self.A,
                                              dtype=torch.int64, device=self.device)
            self.actions[k][t] = v
        self._ptr += 1

    def compute_advantages(self, last_value: torch.Tensor,
                           last_done: torch.Tensor,
                           gamma: float, gae_lambda: float):
        """Compute generalised advantage estimates (GAE) in-place."""
        with torch.no_grad():
            adv = torch.zeros_like(self.rewards)
            lastgaelam = torch.zeros(self.A, device=self.device)
            for t in reversed(range(self.T)):
                if t == self.T - 1:
                    nextnonterminal = 1.0 - last_done.float()
                    nextvalues = last_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1].float()
                    nextvalues = self.values[t + 1]
                delta = (self.rewards[t]
                         + gamma * nextvalues * nextnonterminal
                         - self.values[t])
                lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                adv[t] = lastgaelam
            self.advantages = adv
            self.returns = adv + self.values

    def reset(self):
        self._ptr = 0
        self.states = []
        self.actions = {}
        self.rewards.zero_()
        self.dones.zero_()
        self.values.zero_()
        self.log_probs.zero_()
        self.advantages = None
        self.returns = None



# Helper: compute log-prob and entropy for multi-head categorical policy


def compute_logprob_and_entropy(logits: dict, actions: dict,
                                legal: dict, device: str):
    """
    Given per-head raw logits and the taken actions, returns:
        log_prob : (A,) sum of per-head log-probs
        entropy  : (A,) mean of per-head entropies
    """
    from torch.distributions import Categorical
    from util import legal_mask as apply_legal_mask

    log_probs = []
    entropies = []
    for name, logit in logits.items():
        # logit: (A, n_actions)
        if name in legal:
            lg = apply_legal_mask(logit, legal[name])
        else:
            lg = logit
        dist = Categorical(logits=lg)
        lp = dist.log_prob(actions[name])   # (A,)
        ent = dist.entropy()                # (A,)
        log_probs.append(lp)
        entropies.append(ent)

    log_prob = torch.stack(log_probs, dim=-1).sum(dim=-1)   # (A,)
    entropy  = torch.stack(entropies, dim=-1).mean(dim=-1)  # (A,)
    return log_prob, entropy



# Vectorised environment wrapper


class VecNMMO:
    """Thin wrapper around multiple nmmo.Env instances."""

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.envs: List[nmmo.Env] = []
        self.translators: List[List[Translator]] = []   # [env][team]
        self.reward_calcs: List[List[RewardCalculator]] = []

        for _ in range(cfg.num_envs):
            env = nmmo.Env()
            self.envs.append(env)
            team_translators = [Translator() for _ in range(cfg.num_teams)]
            team_rewards = [RewardCalculator() for _ in range(cfg.num_teams)]
            self.translators.append(team_translators)
            self.reward_calcs.append(team_rewards)

    def reset(self):
        all_obs = []
        for env_idx, env in enumerate(self.envs):
            raw_obs = env.reset()
            team_obs = self._split_by_team(raw_obs)
            env_state = []
            for team_idx in range(self.cfg.num_teams):
                t = self.translators[env_idx][team_idx]
                rc = self.reward_calcs[env_idx][team_idx]
                t_obs = team_obs[team_idx]
                t.reset(t_obs)
                rc.reset()
                state = t.trans_obs(t_obs)
                env_state.append(state)
            all_obs.append(env_state)
        return all_obs

    def step(self, all_actions):
        """
        all_actions: list[env] of list[team] of dict{head: np.array(team_size)}
        Returns (all_obs, all_rewards, all_dones)
        """
        all_obs, all_rewards, all_dones = [], [], []
        for env_idx, env in enumerate(self.envs):
            # Merge all team actions into a single raw_actions dict
            merged_raw = {}
            for team_idx in range(self.cfg.num_teams):
                t = self.translators[env_idx][team_idx]
                team_act = all_actions[env_idx][team_idx]
                raw = t.trans_action(team_act)
                merged_raw.update(raw)  # assumes disjoint agent keys
            raw_obs, _, raw_dones, _ = env.step(merged_raw)
            team_obs = self._split_by_team(raw_obs)
            team_dones = self._split_dones_by_team(raw_dones)
            env_obs, env_rewards, env_dones = [], [], []
            for team_idx in range(self.cfg.num_teams):
                t = self.translators[env_idx][team_idx]
                rc = self.reward_calcs[env_idx][team_idx]
                t_obs = team_obs[team_idx]
                t_dones = team_dones[team_idx]
                state = t.trans_obs(t_obs)
                rew_dict = rc.compute(t, t_obs, t_dones)
                rew_arr = rewards_to_array(rew_dict)
                env_obs.append(state)
                env_rewards.append(rew_arr)
                env_dones.append(t_dones)
            all_obs.append(env_obs)
            all_rewards.append(env_rewards)
            all_dones.append(env_dones)
        return all_obs, all_rewards, all_dones

    
    @staticmethod
    def _split_by_team(raw_obs: dict) -> List[dict]:
        """
        NeuralMMO uses agent IDs of the form (team_id * team_size + member_id).
        Returns a list[team] of dict{0..7: obs}.
        """
        from const import N_PLAYER_PER_TEAM
        teams: dict = defaultdict(dict)
        for agent_id, obs in raw_obs.items():
            team_idx = (agent_id - 1) // N_PLAYER_PER_TEAM
            member_idx = (agent_id - 1) % N_PLAYER_PER_TEAM
            teams[team_idx][member_idx] = obs
        return [teams[i] for i in range(N_TEAM)]

    @staticmethod
    def _split_dones_by_team(raw_dones: dict) -> List[dict]:
        from const import N_PLAYER_PER_TEAM
        teams: dict = defaultdict(dict)
        for agent_id, done in raw_dones.items():
            team_idx = (agent_id - 1) // N_PLAYER_PER_TEAM
            member_idx = (agent_id - 1) % N_PLAYER_PER_TEAM
            teams[team_idx][member_idx] = done
        return [teams[i] for i in range(N_TEAM)]



# Batch state helpers


def batch_states(all_obs: List[List[dict]], device: str) -> dict:
    """
    Stack observations from all envs, all teams into a single batched dict.
    Output shape: (batch, n_agents, ...) where batch = num_envs * num_teams.
    """
    flat = [team_state
            for env_states in all_obs
            for team_state in env_states]

    keys = flat[0].keys()
    batched = {}
    for k in keys:
        samples = [s[k] for s in flat]
        if isinstance(samples[0], dict):
            # e.g. 'legal' is a dict of arrays
            inner_keys = samples[0].keys()
            batched[k] = {
                ik: torch.FloatTensor(
                    np.stack([s[ik] for s in samples])
                ).to(device)
                for ik in inner_keys
            }
        else:
            batched[k] = torch.FloatTensor(np.stack(samples)).to(device)
    return batched


def batch_rewards(all_rewards, device: str) -> torch.Tensor:
    """Flatten env/team/agent rewards → (num_envs * num_teams, team_size)."""
    flat = [r for env_r in all_rewards for r in env_r]
    return torch.FloatTensor(np.stack(flat)).to(device)


def batch_dones(all_dones, n_agents: int, device: str) -> torch.Tensor:
    """Flatten done flags → (num_envs * num_teams, team_size)."""
    flat_dones = np.zeros((len(all_dones) * len(all_dones[0]), n_agents))
    idx = 0
    for env_d in all_dones:
        for team_d in env_d:
            for agent_i, done in team_d.items():
                flat_dones[idx, agent_i] = float(done)
            idx += 1
    return torch.FloatTensor(flat_dones).to(device)



# Sample actions from the policy


@torch.no_grad()
def sample_actions(logits: dict, legal: dict, device: str):
    """
    Sample from multi-head categorical distributions with legal masking.
    Returns:
        actions_dict : {head: (batch, team_size) int64}
        log_prob     : (batch * team_size,) float32
        entropy      : (batch * team_size,) float32
    """
    from torch.distributions import Categorical
    from util import legal_mask as apply_legal_mask

    actions_dict = {}
    log_probs = []
    entropies = []
    for name, logit in logits.items():
        bs, na, n_act = logit.shape
        logit_flat = logit.view(bs * na, n_act)
        if name in legal:
            legal_flat = legal[name].view(bs * na, n_act)
            logit_flat = apply_legal_mask(logit_flat, legal_flat)
        dist = Categorical(logits=logit_flat)
        act = dist.sample()                             # (bs*na,)
        log_probs.append(dist.log_prob(act))
        entropies.append(dist.entropy())
        actions_dict[name] = act.view(bs, na)

    log_prob = torch.stack(log_probs, dim=-1).sum(-1)   # (bs*na,)
    entropy  = torch.stack(entropies, dim=-1).mean(-1)  # (bs*na,)
    return actions_dict, log_prob.view(bs, na), entropy.view(bs, na)



# Main trainer


class PPOTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self._setup_seed()
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

        self.run_name = cfg.run_name or f"nmmo_{int(time.time())}"
        self.writer = SummaryWriter(
            log_dir=os.path.join("runs", self.run_name)
        )

        self.device = torch.device(cfg.device)
        self.model = NMMONet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr,
                                    eps=1e-5)

        self.vec_env = VecNMMO(cfg)

        # One pair of LSTM states per (env * team * agent)
        n_batches = cfg.num_envs * cfg.num_teams
        n_agents  = N_PLAYER_PER_TEAM
        H = self.model.n_lstm_hidden
        self.hx = torch.zeros(n_batches, n_agents, H, device=self.device)
        self.cx = torch.zeros(n_batches, n_agents, H, device=self.device)

        self.buffer = RolloutBuffer(
            rollout_len=cfg.rollout_len,
            n_agents=n_batches * n_agents,
            n_lstm_hidden=H,
            device=self.device,
        )

        self.global_step = 0
        self.update_count = 0

    
    def train(self):
        cfg = self.cfg
        n_batches = cfg.num_envs * cfg.num_teams

        all_obs = self.vec_env.reset()
        state = batch_states(all_obs, cfg.device)

        start_time = time.time()

        while self.global_step < cfg.total_steps:
            
            # 0. Learning-rate annealing
            
            if cfg.anneal_lr:
                frac = 1.0 - self.global_step / cfg.total_steps
                for pg in self.optimizer.param_groups:
                    pg["lr"] = frac * cfg.lr

            
            # 1. Collect rollout
            
            self.buffer.reset()
            ep_returns = []
            ep_lengths = []
            ep_return_acc = np.zeros(n_batches * N_PLAYER_PER_TEAM)

            for _ in range(cfg.rollout_len):
                with torch.no_grad():
                    logits, value, new_hx, new_cx = self.model(
                        state, self.hx, self.cx,
                        bptt_trunc_len=1,
                    )

                actions_dict, log_prob, _ = sample_actions(
                    logits, state['legal'], cfg.device
                )

                # Value: (batch, team_size) -> (batch * team_size,)
                value_flat = value.squeeze(-1)  # (batch,) from mean pool
                # Expand to per-agent for buffer storage
                value_flat_expanded = value_flat.unsqueeze(-1).expand(
                    -1, N_PLAYER_PER_TEAM).reshape(-1)

                # Convert sampled actions → per-env-team list for step
                all_raw_actions = self._unpack_actions(actions_dict,
                                                       n_batches)

                all_obs, all_rewards, all_dones = self.vec_env.step(
                    all_raw_actions
                )

                rewards_t = batch_rewards(all_rewards, cfg.device)
                dones_t   = batch_dones(all_dones, N_PLAYER_PER_TEAM,
                                        cfg.device)

                self.buffer.store(
                    state=state,
                    actions=actions_dict,
                    reward=rewards_t.view(-1),
                    done=dones_t.view(-1),
                    value=value_flat_expanded,
                    log_prob=log_prob.view(-1),
                )

                # Tracking
                ep_return_acc += rewards_t.cpu().numpy().reshape(-1)
                self.global_step += n_batches * N_PLAYER_PER_TEAM

                # Reset LSTM hidden for done agents
                done_mask = dones_t.unsqueeze(-1)          # (B, A, 1)
                self.hx = new_hx * (1.0 - done_mask)
                self.cx = new_cx * (1.0 - done_mask)

                state = batch_states(all_obs, cfg.device)

            # Bootstrap value for GAE
            with torch.no_grad():
                _, last_value, _, _ = self.model(
                    state, self.hx, self.cx, bptt_trunc_len=1
                )
            last_value_flat = last_value.squeeze(-1) \
                                        .unsqueeze(-1) \
                                        .expand(-1, N_PLAYER_PER_TEAM) \
                                        .reshape(-1)
            last_done_flat  = dones_t.view(-1)

            self.buffer.compute_advantages(
                last_value_flat, last_done_flat,
                cfg.gamma, cfg.gae_lambda
            )

            
            # 2. PPO update
            
            stats = self._ppo_update()
            self.update_count += 1

            
            # 3. Logging
            
            if self.update_count % cfg.log_interval == 0:
                sps = self.global_step / (time.time() - start_time)
                self.writer.add_scalar(
                    "charts/learning_rate",
                    self.optimizer.param_groups[0]["lr"], self.global_step
                )
                for k, v in stats.items():
                    self.writer.add_scalar(f"losses/{k}", v, self.global_step)
                self.writer.add_scalar(
                    "charts/SPS", sps, self.global_step
                )
                print(
                    f"[step {self.global_step:>12,}]  "
                    f"SPS={sps:.0f}  "
                    f"policy_loss={stats['policy_loss']:.4f}  "
                    f"value_loss={stats['value_loss']:.4f}  "
                    f"entropy={stats['entropy']:.4f}"
                )

            
            # 4. Checkpoint
            
            if self.update_count % cfg.save_interval == 0:
                self._save_checkpoint()

        self._save_checkpoint(final=True)
        self.writer.close()
        print("Training complete.")

    
    def _ppo_update(self) -> dict:
        cfg = self.cfg
        buf = self.buffer

        T = cfg.rollout_len
        A = buf.A

        # Flatten buffer tensors: (T*A,)
        b_rewards   = buf.rewards.view(-1)
        b_values    = buf.values.view(-1)
        b_log_probs = buf.log_probs.view(-1)
        b_actions   = {k: v.view(-1) for k, v in buf.actions.items()}
        b_adv       = buf.advantages.view(-1)
        b_returns   = buf.returns.view(-1)

        if cfg.norm_adv:
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        mb_size = (T * A) // cfg.num_minibatches
        indices = np.arange(T * A)

        stats = defaultdict(float)
        n_updates = 0

        for _ in range(cfg.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, T * A, mb_size):
                mb_idx = indices[start: start + mb_size]
                mb_idx_t = torch.from_numpy(mb_idx).long().to(self.device)

                # Retrieve matching states
                # For simplicity we re-run the forward pass on stored states.
                # In a production setup you would cache activations.
                mb_states = self._index_states(buf.states, mb_idx, T, A)
                mb_hx = torch.zeros(
                    len(mb_idx) // N_PLAYER_PER_TEAM,
                    N_PLAYER_PER_TEAM,
                    self.model.n_lstm_hidden,
                    device=self.device,
                )
                mb_cx = torch.zeros_like(mb_hx)

                logits, new_value, _, _ = self.model(
                    mb_states, mb_hx, mb_cx, bptt_trunc_len=1
                )

                legal_mb = mb_states.get('legal', {})
                mb_acts  = {k: v[mb_idx_t] for k, v in b_actions.items()}
                new_log_prob, entropy = compute_logprob_and_entropy(
                    logits, mb_acts, legal_mb, cfg.device
                )
                new_log_prob = new_log_prob.view(-1)
                entropy      = entropy.view(-1)

                new_value_flat = new_value.squeeze(-1) \
                                          .unsqueeze(-1) \
                                          .expand(-1, N_PLAYER_PER_TEAM) \
                                          .reshape(-1)

                logratio = new_log_prob - b_log_probs[mb_idx_t]
                ratio    = logratio.exp()

                mb_adv = b_adv[mb_idx_t]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(
                    ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                if cfg.clip_vloss:
                    v_clipped = b_values[mb_idx_t] + torch.clamp(
                        new_value_flat - b_values[mb_idx_t],
                        -cfg.clip_coef, cfg.clip_coef,
                    )
                    v_loss = torch.max(
                        (new_value_flat - b_returns[mb_idx_t]).pow(2),
                        (v_clipped     - b_returns[mb_idx_t]).pow(2),
                    ).mean() * 0.5
                else:
                    v_loss = (new_value_flat - b_returns[mb_idx_t]) \
                                 .pow(2).mean() * 0.5

                ent_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * ent_loss + cfg.vf_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()

                stats['policy_loss'] += pg_loss.item()
                stats['value_loss']  += v_loss.item()
                stats['entropy']     += ent_loss.item()
                stats['approx_kl']   += approx_kl.item()
                n_updates += 1

                if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                    break

        return {k: v / max(n_updates, 1) for k, v in stats.items()}

    
    def _unpack_actions(self, actions_dict: dict, n_batches: int) -> list:
        """
        Convert batched action tensors back into per-env-team-agent lists.
        Returns list[env] of list[team] of np.ndarray shape (4, team_size).
        """
        head_names = list(actions_dict.keys())
        # actions_dict[head]: (n_batches, team_size)
        stacked = np.stack(
            [actions_dict[h].cpu().numpy() for h in head_names], axis=0
        )  # (n_heads, n_batches, team_size)

        result = []
        batch_idx = 0
        for _ in range(self.cfg.num_envs):
            env_acts = []
            for _ in range(self.cfg.num_teams):
                # (n_heads, team_size) → transpose to (team_size, n_heads)
                team_act = stacked[:, batch_idx, :]  # (n_heads, team_size)
                env_acts.append(team_act)
                batch_idx += 1
            result.append(env_acts)
        return result

    
    @staticmethod
    def _index_states(states_list: list, flat_idx: np.ndarray,
                      T: int, A: int) -> dict:
        """
        Retrieve and stack a mini-batch of states from the stored rollout.
        flat_idx maps to (step_idx * A + agent_idx).
        This simplified version gathers one state dict per mini-batch element.
        """
        step_idx = flat_idx // A
        # Gather unique steps and build a mini-state by indexing tensors
        unique_steps = np.unique(step_idx)
        sampled = [states_list[s] for s in unique_steps]

        keys = sampled[0].keys()
        batched = {}
        for k in keys:
            samples = [s[k] for s in sampled]
            if isinstance(samples[0], dict):
                inner_keys = samples[0].keys()
                batched[k] = {
                    ik: torch.stack([s[ik] for s in samples], dim=0).squeeze(1)
                    for ik in inner_keys
                }
            else:
                batched[k] = torch.stack(samples, dim=0).squeeze(1)
        return batched

    
    def _save_checkpoint(self, final: bool = False):
        tag = "final" if final else f"step_{self.global_step}"
        path = os.path.join(self.cfg.checkpoint_dir,
                            f"{self.run_name}_{tag}.pt")
        torch.save({
            'global_step':   self.global_step,
            'update_count':  self.update_count,
            'model':         self.model.state_dict(),
            'optimizer':     self.optimizer.state_dict(),
            'config':        self.cfg,
        }, path)
        print(f"  → checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.global_step  = ckpt['global_step']
        self.update_count = ckpt['update_count']
        print(f"Loaded checkpoint from {path}  "
              f"(global_step={self.global_step})")

    
    def _setup_seed(self):
        s = self.cfg.seed
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)



# CLI entry point


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="NeuralMMO PPO trainer")
    cfg = TrainConfig()

    parser.add_argument("--num-envs", type=int, default=cfg.num_envs)
    parser.add_argument("--rollout-len", type=int, default=cfg.rollout_len)
    parser.add_argument("--total-steps", type=int, default=cfg.total_steps)
    parser.add_argument("--ppo-epochs", type=int, default=cfg.ppo_epochs)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--clip-coef", type=float, default=cfg.clip_coef)
    parser.add_argument("--ent-coef", type=float, default=cfg.ent_coef)
    parser.add_argument("--vf-coef", type=float, default=cfg.vf_coef)
    parser.add_argument("--gamma", type=float, default=cfg.gamma)
    parser.add_argument("--gae-lambda", type=float, default=cfg.gae_lambda)
    parser.add_argument("--bptt-trunc-len", type=int, default=cfg.bptt_trunc_len)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--device", type=str, default=cfg.device)
    parser.add_argument("--run-name", type=str, default=cfg.run_name)
    parser.add_argument("--checkpoint-dir", type=str, default=cfg.checkpoint_dir)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint to resume from.",
    )

    args = parser.parse_args()

    cfg.num_envs = args.num_envs
    cfg.rollout_len = args.rollout_len
    cfg.total_steps = args.total_steps
    cfg.ppo_epochs = args.ppo_epochs
    cfg.lr = args.lr
    cfg.clip_coef = args.clip_coef
    cfg.ent_coef = args.ent_coef
    cfg.vf_coef = args.vf_coef
    cfg.gamma = args.gamma
    cfg.gae_lambda = args.gae_lambda
    cfg.bptt_trunc_len = args.bptt_trunc_len
    cfg.seed = args.seed
    cfg.device = args.device
    cfg.run_name = args.run_name
    cfg.checkpoint_dir = args.checkpoint_dir

    return cfg, args.resume


def main():
    cfg, resume = parse_args()
    trainer = PPOTrainer(cfg)
    if resume:
        trainer.load_checkpoint(resume)
    trainer.train()


if __name__ == "__main__":
    main()