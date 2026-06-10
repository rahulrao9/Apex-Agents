# NeuralMMO Multi-Agent RL — Training Instructions

A PPO-based multi-agent reinforcement learning system for NeuralMMO.

---

## Project structure

```
your_package/
├── __init__.py
├── const.py          # game constants
├── util.py           # utility functions & decorators
├── model.py          # NMMONet actor-critic (ResNet + Transformer + LSTM)
├── translator.py     # obs/action translation
├── reward.py         # reward shaping
├── train.py          # PPO training loop  ← new
└── evaluate.py       # evaluation / inference  ← new
```

---

## Quick start

### 1. Training

Run from the **repo root** (the directory that contains `your_package/`):

```bash
python -m your_package.train
```

All hyper-parameters have sensible defaults. Common overrides:

```bash
python -m your_package.train \
    --num-envs 8          \   # parallel environments
    --rollout-len 128     \   # steps per rollout before update
    --total-steps 500000000 \ # total environment steps
    --lr 1e-4             \
    --clip-coef 0.2       \
    --ent-coef 0.01       \
    --vf-coef 0.5         \
    --gamma 0.99          \
    --gae-lambda 0.95     \
    --bptt-trunc-len 16   \
    --seed 42             \
    --device cuda         \   # or "cpu"
    --run-name my_run     \
    --checkpoint-dir checkpoints
```

Checkpoints are saved to `checkpoints/` every 500 updates (configurable via `--save-interval` in `TrainConfig`).

### 2. Resume training

```bash
python -m your_package.train \
    --resume checkpoints/my_run_step_1000000.pt
```

### 3. Monitor with TensorBoard

```bash
tensorboard --logdir runs/
```

Open `http://localhost:6006` in your browser.

---

## Evaluation

Run a trained model for a fixed number of episodes and print summary stats:

```bash
python -m your_package.evaluate \
    --checkpoint checkpoints/my_run_final.pt \
    --num-episodes 5 \
    --device cpu
```

Add `--render` to render the environment visually (requires a display):

```bash
python -m your_package.evaluate \
    --checkpoint checkpoints/my_run_final.pt \
    --render
```

Output per episode:

```
Episode   1: mean_survival=612.3  total_kills=14.0  mean_return=18.421
Episode   2: mean_survival=589.7  total_kills=11.0  mean_return=16.843
...
--- Evaluation summary ---
  mean_survival: 601.0000
  total_kills:   12.5000
  mean_return:   17.6320
```

---

## Reward shaping

Reward weights live in `reward.RewardConfig` and can be adjusted without touching any other file:

| Component           | Default | Description                                              |
|---------------------|---------|----------------------------------------------------------|
| `SURVIVAL_PER_STEP` | 0.01    | Flat bonus for staying alive each step                   |
| `HP_GAIN_SCALE`     | 0.002   | Reward per HP recovered                                  |
| `HP_LOSS_SCALE`     | 0.002   | Penalty per HP lost                                      |
| `KILL_HOSTILE_NPC`  | 0.5     | Reward for killing a hostile NPC                         |
| `KILL_NEUTRAL_NPC`  | 0.2     | Reward for killing a neutral NPC (lower to limit griefing)|
| `KILL_PASSIVE_NPC`  | 0.1     | Reward for killing a passive NPC                         |
| `KILL_PLAYER`       | 1.0     | Reward for killing an enemy player                       |
| `HERB_STEP_REWARD`  | 0.3     | Reward for stepping onto a herb tile                     |
| `EXPLORE_NEW_TILE`  | 0.002   | Reward per newly revealed tile                           |
| `TEAM_MILESTONE_BONUS` | 1.0 | Shared bonus when full team survives milestone steps     |
| `POISON_PENALTY_SCALE` | 0.02 | Penalty multiplied by poison zone strength              |
| `DEATH_PENALTY`     | -1.0   | One-time penalty on death                                |

To override at runtime, instantiate `RewardConfig` manually before creating `RewardCalculator`:

```python
from your_package.reward import RewardConfig, RewardCalculator

cfg = RewardConfig()
cfg.KILL_PLAYER = 2.0     # reward player kills more heavily
cfg.EXPLORE_NEW_TILE = 0.005

calc = RewardCalculator(cfg)
```

---

## Key hyper-parameters

| Parameter         | Default   | Description                                   |
|-------------------|-----------|-----------------------------------------------|
| `num_envs`        | 4         | Parallel environments                         |
| `rollout_len`     | 128       | Steps per rollout                             |
| `ppo_epochs`      | 1         | PPO update passes per rollout                 |
| `num_minibatches` | 1         | Mini-batches per PPO epoch                    |
| `clip_coef`       | 0.2       | PPO clipping ε                                |
| `ent_coef`        | 0.01      | Entropy bonus coefficient                     |
| `vf_coef`         | 0.5       | Value-loss coefficient                        |
| `gamma`           | 0.99      | Discount factor                               |
| `gae_lambda`      | 0.95      | GAE λ                                         |
| `bptt_trunc_len`  | 16        | LSTM BPTT truncation length                   |
| `lr`              | 1e-4      | Adam learning rate                            |
| `anneal_lr`       | True      | Linear LR decay to 0 over `total_steps`       |
| `max_grad_norm`   | 0.5       | Gradient clipping norm                        |

---

## Tips

- Start with `--num-envs 4` on a single GPU. Increase to 8–16 if VRAM allows.
- A `bptt_trunc_len` of 16 is a good balance between sequence richness and memory. Increase for harder tasks; decrease if OOM.
- The `ent_coef` decay can be implemented by adding a linear schedule in `train.py` if exploration collapses early.
- TensorBoard's `losses/approx_kl` is the best single indicator of training stability — values consistently above 0.02 suggest the learning rate or `clip_coef` is too high.