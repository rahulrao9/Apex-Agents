# This script serves as the Master Manager for running multiple isolated rollouts of the NMMO 
# environment, each with a different random seed. It spawns separate subprocesses to execute 
# the matches, ensuring that each rollout is completely independent and does not interfere with 
# others. After each match, it immediately extracts the replay data from the LZMA file into a JSON 
# format, making it ready for analysis by team_analysis.py. This approach guarantees clean isolation 
# between runs while still allowing for efficient orchestration from a single entry point.
import json
import lzma
import os
import sys
import subprocess
from neurips2022nmmo import CompetitionConfig, submission, RollOut

def run_single_seed(seed):
    """Worker function: Runs a single match and extracts it."""
    file_name_base = f"faction_war_seed_{seed}"
    print(f"\n{'='*60}", flush=True)
    print(f"  STARTING MULTI-SEED ROLLOUT: {seed}", flush=True)
    print(f"{'='*60}", flush=True)

    # 1. Initialize config
    config = CompetitionConfig()
    config.HORIZON = 1024
    config.SAVE_REPLAY = file_name_base

    # 2. Load the teams
    teams = []
    for i in range(16):
        team = submission.get_team_from_submission(
            submission_path=".",
            team_id=f"RealikunTeam-{i}",
            env_config=config,
        )
        teams.append(team)

    # 3. Run the match
    ro = RollOut(config, teams, parallel=False, show_progress=True)
    ro.run(n_episode=1, render=False)

    # 4. Decompress the LZMA file immediately
    lzma_file = f"{file_name_base}.lzma"
    json_file = f"{file_name_base}.json"

    print(f"Extracting {lzma_file} to {json_file}...", flush=True)
    try:
        with lzma.open(lzma_file, "rt", encoding="utf-8") as f_in:
            data = json.load(f_in)
        with open(json_file, "w", encoding="utf-8") as f_out:
            json.dump(data, f_out, indent=2)
        print(f"Success! {json_file} is ready.", flush=True)
    except Exception as e:
        print(f"Extraction failed for seed {seed}: {e}", flush=True)


if __name__ == "__main__":
    # If the script is called with a specific seed argument, run just that seed
    if len(sys.argv) > 1 and sys.argv[1].startswith("--seed="):
        seed_val = int(sys.argv[1].split("=")[1])
        run_single_seed(seed_val)
        sys.exit(0)  # Force clean exit so Twisted reactor doesn't block

    # Otherwise, act as the Master Manager and spawn 5 separate subprocesses
    else:
        NUM_SEEDS = 5
        print(f"Master Process: Orchestrating {NUM_SEEDS} isolated rollouts...", flush=True)

        for s in range(1, NUM_SEEDS + 1):
            print(f"\nMaster: Launching seed {s}...", flush=True)
            result = subprocess.run(
                [sys.executable, __file__, f"--seed={s}"],
                env={**os.environ, "PYTHONUNBUFFERED": "1"}
            )
            if result.returncode != 0:
                print(f"WARNING: Seed {s} exited with code {result.returncode}", flush=True)
            else:
                print(f"Master: Seed {s} completed cleanly.", flush=True)

        print("\nAll 5 Multi-Seed Rollouts Complete! Ready for team_analysis.py.", flush=True)