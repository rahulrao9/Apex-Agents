# this file is for evaluating a trained model on the NMMO environment. It loads a checkpoint,
# runs a specified number of evaluation episodes, and reports metrics such as mean survival time,
# total kills, and mean return. The evaluation can be run with optional rendering to visualize the
# environment during the episodes.
"""
evaluate.py — Load a trained checkpoint and run evaluation episodes.

Usage (run directly from the pvp/ directory):
    python evaluate.py model.pth
    python evaluate.py model.pth --num-episodes 5 --render
    python evaluate.py --checkpoint model.pth --device cuda
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

import nmmo

from const import N_PLAYER_PER_TEAM, N_TEAM
from model import NMMONet
from translator import Translator
from reward import RewardCalculator, rewards_to_array


class Evaluator:
    def __init__(self, checkpoint_path: str, device: str = "cpu",
                 render: bool = False):
        self.device = torch.device(device)
        self.render = render

        self.model = NMMONet().to(self.device)
        self._load(checkpoint_path)
        self.model.eval()

        self.env = nmmo.Env()
        self.translators = [Translator() for _ in range(N_TEAM)]
        self.reward_calcs = [RewardCalculator() for _ in range(N_TEAM)]

    def run(self, num_episodes: int = 1) -> dict:
        all_ep_stats = []
        for ep in range(num_episodes):
            stats = self._run_episode(ep)
            all_ep_stats.append(stats)
            print(f"Episode {ep+1:>3}: "
                  f"mean_survival={stats['mean_survival']:.1f}  "
                  f"total_kills={stats['total_kills']:.0f}  "
                  f"mean_return={stats['mean_return']:.3f}")

        summary = {
            k: float(np.mean([s[k] for s in all_ep_stats]))
            for k in all_ep_stats[0]
        }
        print("\n--- Evaluation summary ---")
        for k, v in summary.items():
            print(f"  {k}: {v:.4f}")
        return summary

    def _run_episode(self, ep_idx: int) -> dict:
        raw_obs = self.env.reset()
        team_obs_list = self._split_by_team(raw_obs)

        hx = torch.zeros(N_TEAM, N_PLAYER_PER_TEAM,
                         self.model.n_lstm_hidden, device=self.device)
        cx = torch.zeros_like(hx)

        for t_idx in range(N_TEAM):
            self.translators[t_idx].reset(team_obs_list[t_idx])
            self.reward_calcs[t_idx].reset()

        total_rewards = np.zeros((N_TEAM, N_PLAYER_PER_TEAM))
        survival_steps = np.zeros((N_TEAM, N_PLAYER_PER_TEAM))
        total_kills = 0.0
        done = False
        step = 0

        while not done:
            if self.render:
                self.env.render()

            # Build per-team state tensors
            all_states = []
            for t_idx in range(N_TEAM):
                state = self.translators[t_idx].trans_obs(team_obs_list[t_idx])
                all_states.append(state)

            merged_state = self._batch_states(all_states)

            with torch.no_grad():
                logits, _, new_hx, new_cx = self.model(
                    merged_state, hx, cx, bptt_trunc_len=1
                )

            # Greedy action selection (argmax)
            actions_dict = {
                name: lg.argmax(dim=-1)
                for name, lg in logits.items()
            }
            hx, cx = new_hx, new_cx

            merged_raw = {}
            flat_act = self._unpack_actions(actions_dict)
            for t_idx in range(N_TEAM):
                raw = self.translators[t_idx].trans_action(flat_act[t_idx])
                merged_raw.update(raw)

            raw_obs, _, raw_dones, raw_infos = self.env.step(merged_raw)
            team_obs_list = self._split_by_team(raw_obs)
            team_dones    = self._split_dones_by_team(raw_dones)

            for t_idx in range(N_TEAM):
                t = self.translators[t_idx]
                rc = self.reward_calcs[t_idx]
                rew_dict = rc.compute(t, team_obs_list[t_idx], team_dones[t_idx])
                rew_arr  = rewards_to_array(rew_dict)
                total_rewards[t_idx] += rew_arr
                for a_idx in range(N_PLAYER_PER_TEAM):
                    if not team_dones[t_idx].get(a_idx, False):
                        survival_steps[t_idx, a_idx] += 1
                total_kills += sum(
                    t.player_kill_num[a] + t.npc_kill_num['h'][a]
                    for a in range(N_PLAYER_PER_TEAM)
                )

            done = len(raw_obs) == 0 or step >= 1023
            step += 1

        return {
            'mean_survival': float(survival_steps.mean()),
            'total_kills':   float(total_kills),
            'mean_return':   float(total_rewards.mean()),
        }

    # ------------------------------------------------------------------
    def _load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)

        # Probe for model weights under several common key conventions:
        #   model_dict  (this checkpoint format)
        #   model       (our train.py format)
        #   net         (another common convention)
        #   bare dict   (weights saved directly)
        if isinstance(ckpt, dict):
            if 'model_dict' in ckpt:
                state_dict = ckpt['model_dict']
                extra = {k: v for k, v in ckpt.items() if k != 'model_dict'}
                if extra:
                    print(f"  checkpoint metadata: {extra}")
            elif 'model' in ckpt:
                state_dict = ckpt['model']
            elif 'net' in ckpt:
                state_dict = ckpt['net']
            else:
                state_dict = ckpt  # assume bare state_dict
        else:
            raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")

        self.model.load_state_dict(state_dict, strict=True)
        print(f"Loaded model from {path}")

    @staticmethod
    def _split_by_team(raw_obs: dict):
        from collections import defaultdict
        teams: dict = defaultdict(dict)
        for agent_id, obs in raw_obs.items():
            t_idx = (agent_id - 1) // N_PLAYER_PER_TEAM
            m_idx = (agent_id - 1) % N_PLAYER_PER_TEAM
            teams[t_idx][m_idx] = obs
        return [teams[i] for i in range(N_TEAM)]

    @staticmethod
    def _split_dones_by_team(raw_dones: dict):
        from collections import defaultdict
        teams: dict = defaultdict(dict)
        for agent_id, done in raw_dones.items():
            t_idx = (agent_id - 1) // N_PLAYER_PER_TEAM
            m_idx = (agent_id - 1) % N_PLAYER_PER_TEAM
            teams[t_idx][m_idx] = done
        return [teams[i] for i in range(N_TEAM)]

    def _batch_states(self, states_list: list) -> dict:
        import torch
        keys = states_list[0].keys()
        batched = {}
        for k in keys:
            samples = [s[k] for s in states_list]
            if isinstance(samples[0], dict):
                inner_keys = samples[0].keys()
                batched[k] = {
                    ik: torch.FloatTensor(
                        np.stack([s[ik] for s in samples])
                    ).to(self.device)
                    for ik in inner_keys
                }
            else:
                batched[k] = torch.FloatTensor(
                    np.stack(samples)
                ).to(self.device)
        return batched

    def _unpack_actions(self, actions_dict: dict) -> list:
        head_names = list(actions_dict.keys())
        stacked = np.stack(
            [actions_dict[h].cpu().numpy() for h in head_names], axis=0
        )  # (n_heads, N_TEAM, team_size)
        result = []
        for t_idx in range(N_TEAM):
            result.append(stacked[:, t_idx, :])  # (n_heads, team_size)
        return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NeuralMMO evaluator")
    # Accept checkpoint as positional OR --checkpoint so both work:
    #   python evaluate.py model.pth
    #   python evaluate.py --checkpoint model.pth
    parser.add_argument("checkpoint_pos",  type=str, nargs="?", default=None,
                        metavar="CHECKPOINT",
                        help="Path to checkpoint (positional)")
    parser.add_argument("--checkpoint",    type=str, default=None,
                        help="Path to checkpoint (keyword form)")
    parser.add_argument("--num-episodes",  type=int, default=3)
    parser.add_argument("--device",        type=str, default="cpu")
    parser.add_argument("--render",        action="store_true")
    args = parser.parse_args()

    ckpt_path = args.checkpoint_pos or args.checkpoint
    if ckpt_path is None:
        parser.error("Provide a checkpoint: python evaluate.py model.pth")

    evaluator = Evaluator(
        checkpoint_path=ckpt_path,
        device=args.device,
        render=args.render,
    )
    evaluator.run(num_episodes=args.num_episodes)


if __name__ == "__main__":
    main()