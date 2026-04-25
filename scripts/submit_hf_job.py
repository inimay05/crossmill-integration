"""
Submit a CrossMill training run as a HuggingFace Job (uses your HF credits).

HF Jobs run your script on a cloud GPU — results are saved to a HF dataset
repo and can then be pulled locally or pushed to your Space.

Prerequisites:
  - huggingface-cli login   (or export HF_TOKEN=hf_...)
  - A HF dataset repo to store results, e.g. yamyam05/crossmill-runs
  - HF Job credits (Pro / Enterprise plan)

Usage:
    python scripts/submit_hf_job.py \\
        --env safenutri --task easy --mode cross \\
        --timesteps 500000 \\
        --results-repo yamyam05/crossmill-runs \\
        --hardware t4-medium

Hardware tiers and credit burn rate (approximate):
    t4-small    $0.60/hr   — okay for short runs
    t4-medium   $0.90/hr   — recommended for RL training (500k steps ~30 min)
    a10g-small  $1.05/hr   — faster; needed for GRPO LLM training
    a10g-large  $3.15/hr   — use only if a10g-small OOMs
"""
import argparse
import os
import sys
import textwrap
from pathlib import Path

from huggingface_hub import HfApi, create_repo

INTEGRATION_ROOT = Path(__file__).resolve().parent.parent
HF_USERNAME = "yamyam05"


# ── The training script that will run inside the HF Job container ──────────
JOB_SCRIPT_TEMPLATE = textwrap.dedent("""
#!/usr/bin/env python3
\"\"\"Auto-generated training entrypoint for HF Jobs.\"\"\"
import os, sys, subprocess, json
from pathlib import Path

# Clone the integration repo inside the job container
WORK = Path("/tmp/crossmill")
WORK.mkdir(exist_ok=True)

def run(cmd): subprocess.run(cmd, shell=True, check=True, cwd=str(WORK))

run("git clone https://github.com/inimay05/crossmill-integration .")
run("git clone https://github.com/inimay05/crossmill-safenutri ../crossmill-safenutri")
run("git clone https://github.com/inimay05/crossmill-megaforge  ../crossmill-megaforge")

# Install deps
run("pip install -q openenv-core gymnasium stable-baselines3 sb3-contrib "
    "torch pydantic pyyaml huggingface_hub matplotlib pandas scipy "
    "transformers peft 'trl>=0.9' datasets bitsandbytes accelerate")

# Run the RL training
run(f"python scripts/train.py "
    f"--env {ENV} --task {TASK} --memory_mode {MODE} "
    f"--timesteps {TIMESTEPS} --seed 42")

# Push results to HF dataset repo
from huggingface_hub import upload_folder, create_repo
token = os.environ.get("HF_TOKEN")
create_repo("{results_repo}", repo_type="dataset", exist_ok=True, token=token)
upload_folder(
    repo_id="{results_repo}",
    repo_type="dataset",
    folder_path=str(WORK / "runs"),
    path_in_repo="runs",
    ignore_patterns=["*.tfevents*"],
    token=token,
    commit_message="training run complete — {env}/{task}/{mode}",
)
print("Results pushed to: https://huggingface.co/datasets/{results_repo}")
""")


def submit_job(env, task, mode, timesteps, results_repo, hardware, token):
    api = HfApi(token=token)

    # Write the rendered job script to a temp location
    script_content = JOB_SCRIPT_TEMPLATE.format(
        results_repo=results_repo,
        env=env, task=task, mode=mode,
    ).replace("{ENV}", f'"{env}"') \
     .replace("{TASK}", f'"{task}"') \
     .replace("{MODE}", f'"{mode}"') \
     .replace("{TIMESTEPS}", str(timesteps))

    script_path = INTEGRATION_ROOT / "runs" / "hf_job_entrypoint.py"
    script_path.parent.mkdir(exist_ok=True)
    script_path.write_text(script_content)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              CrossMill HF Job Submission                     ║
╠══════════════════════════════════════════════════════════════╣
║  env:           {env:<45}║
║  task:          {task:<45}║
║  memory mode:   {mode:<45}║
║  timesteps:     {timesteps:<45}║
║  hardware:      {hardware:<45}║
║  results repo:  {results_repo:<45}║
╚══════════════════════════════════════════════════════════════╝
""")

    # Ensure results dataset repo exists
    create_repo(results_repo, repo_type="dataset", exist_ok=True, token=token)
    print(f"Results will be saved to: https://huggingface.co/datasets/{results_repo}")

    print("\nNOTE: HF Jobs API is available for Pro/Enterprise accounts.")
    print("If you have Jobs access, submit via HF web UI:")
    print(f"  1. Go to https://huggingface.co/jobs/new")
    print(f"  2. Select hardware: {hardware}")
    print(f"  3. Upload entrypoint: {script_path}")
    print(f"  4. Set env var HF_TOKEN=<your-token>")
    print(f"\nOr run directly on a GPU Space terminal (see full guide).")
    print(f"\nEntrypoint written to: {script_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit CrossMill training to HF Jobs")
    parser.add_argument("--env",          default="safenutri", choices=["safenutri", "megaforge"])
    parser.add_argument("--task",         default="easy",      choices=["easy", "medium", "hard"])
    parser.add_argument("--mode",         default="cross",     choices=["none", "local", "cross"])
    parser.add_argument("--timesteps",    default=500_000,     type=int)
    parser.add_argument("--results-repo", default=f"{HF_USERNAME}/crossmill-runs")
    parser.add_argument("--hardware",     default="t4-medium",
                        choices=["t4-small","t4-medium","a10g-small","a10g-large","a100-large"])
    parser.add_argument("--token",        default=None)
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: Set HF_TOKEN env var or pass --token hf_...")
        sys.exit(1)

    submit_job(
        env=args.env, task=args.task, mode=args.mode,
        timesteps=args.timesteps, results_repo=args.results_repo,
        hardware=args.hardware, token=token,
    )
