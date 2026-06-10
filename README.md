# APEX AGENTS (Neural MMO) 

<p align="center">
  <img src="https://github.com/rahulrao9/Apex-Agents/raw/main/imgs/map.png" width="45%" alt="Map" />
  <img src="https://github.com/rahulrao9/Apex-Agents/raw/main/imgs/fight.gif" width="45%" alt="Fight" />
</p>

Apex Agents is a Multi-Agent Reinforcement Learning (MARL) experiment built on the Neural MMO 1.6 environment. It evaluates a Neuro-Symbolic hybrid architecture by injecting explicit, rule-based heuristics over a frozen deep-learning baseline (Realikun).

By isolating the movement action head from the combat targeting head via probabilistic gating, this project demonstrates how symbolic logic can radically alter multi-agent dynamics. This architecture yields emergent strategies—such as flawless ranged kiting—and distinct economic efficiencies without the computational overhead of network fine-tuning.

The environment culminates in an esports-style Battle Royale tournament, utilizing a multi-objective scoring matrix inspired by the Apex Legends Global Series (ALGS) to definitively rank the neuro-symbolic variants.

This repository supports two workflows:

1. **Visualize existing replay files** (`.lzma` / generated replays)
2. **Generate new tournament simulations**, analytics, and leaderboards

---

# Option 1: Visualize Existing Replays

If you only want to watch previously generated matches, you do **not** need to install the full tournament framework.

## Prerequisites

* Python 3.8+
* Neural MMO 1.6 Unity Client
* `websockets`

---

## 1. Download the Unity Client

Download the Neural MMO 1.6 Unity Client:

https://github.com/NeuralMMO/client/archive/refs/tags/v1.6.zip

Extract the archive and launch:

```text
UnityClient/neural-mmo.exe
```

Keep the Unity client running before starting the replay streamer.

---

## 2. Install Replay Streaming Dependency

```bash
pip install websockets
```

---

## 3. Stream a Replay

Place the replay file (`.lzma`) in the same directory as `stream_replay.py`.

Then run:

```bash
python stream_replay.py
```

If your replay file is stored elsewhere, update the replay path inside `stream_replay.py`.

The replay will be streamed directly to the Unity Client for visualization.

---

# Option 2: Generate New Tournament Simulations

This framework supports:

* Multi-seed tournament evaluation
* Faction-based combat simulations
* Replay generation
* Statistical analytics
* Tournament leaderboard scoring

---

## Repository Structure

```text
NMMO-1.6_FINAL/
├── neurips2022nmmo/            # Local NMMO environment wrapper
├── plots/                      # Generated analytics
├── pvp/
│   ├── ai.py
│   ├── agent.py
│   ├── model.pth
│   ├── translator.py
│   └── ...
├── vec_noise-1.1.4/
├── requirements.txt
├── run.py
├── stream_replay.py
├── submission.py
├── generate_graphs.py
├── team_graphs.py
└── tourny_leaderboard.py
```

---

# Installation

## 1. Create a Conda Environment

Due to compatibility constraints between legacy Gym and newer NumPy releases, Python 3.9 is required.

```bash
conda create -n nmmo-factions python=3.9 -y
conda activate nmmo-factions
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Important**

Do **not** run:

```bash
pip install ./vec_noise-1.1.4
pip install ./neurips2022nmmo
```

These modules are imported locally and should remain source directories.

---

# Running the Tournament

## Step 1 — Generate Multi-Seed Rollouts

```bash
python run.py
```

This launches multiple independent tournament simulations using different random seeds.

Example outputs:

```text
faction_war_seed_1.json
faction_war_seed_2.json
faction_war_seed_3.json
faction_war_seed_4.json
faction_war_seed_5.json
```

Each replay contains a complete match history and can later be visualized using the Unity Client.

---

## Step 2 — Generate Analytics

```bash
python team_graphs.py
```

Generated analytics are stored in:

```text
plots/
```

Typical outputs include:

* Survival curves
* Team cohesion metrics
* Exploration statistics
* Skill progression plots
* Mortality analysis

---

## Step 3 — Compute Tournament Rankings

```bash
python tourny_leaderboard.py
```

The leaderboard combines:

* Survival placement
* Kill count
* Wealth accumulation
* Cross-seed consistency
* Placement volatility penalties

---

# Replay Visualization

After generating simulations, replay files can be streamed to the Unity Client.

Start the Unity Client first, then run:

```bash
python stream_replay.py faction_war_seed_1.json
```

or configure the replay path directly inside `stream_replay.py`.

---

# Typical Workflow

```bash
# Activate environment
conda activate nmmo-factions

# Generate tournament rollouts
python run.py

# Generate analytics
python team_graphs.py

# Compute final rankings
python tourny_leaderboard.py

# Visualize a replay
python stream_replay.py
```

---

# Notes

* Python 3.9 is required for tournament generation.
* The Unity Client is only required for replay visualization.
* Replays are generated automatically by `run.py`.
* Analytics scripts assume all tournament runs completed successfully.
* Generated figures are saved automatically in the `plots/` directory.
