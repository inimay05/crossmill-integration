"""
After training completes, call this script to sync new runs/ results
back to the HF Space so the Results Dashboard updates.

Usage (from crossmill-integration with venv active):
    python scripts/push_results_to_space.py --space yamyam05/crossmill
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, upload_file, upload_folder

INTEGRATION_ROOT = Path(__file__).resolve().parent.parent


def push_results(space_repo_id: str, token: str | None = None):
    api = HfApi(token=token)

    summary_files = sorted(glob.glob(str(INTEGRATION_ROOT / "runs/*/summary_*.json")))
    if not summary_files:
        print("No summary JSON files found in runs/ — nothing to push.")
        return

    print(f"Pushing results to Space: {space_repo_id}")

    # Push entire runs/ subtree (JSON + PNG; ZIP excluded by pattern below)
    upload_folder(
        repo_id=space_repo_id,
        repo_type="space",
        folder_path=str(INTEGRATION_ROOT / "runs"),
        path_in_repo="runs",
        ignore_patterns=["*.zip", "*.tfevents*"],
        token=token,
        commit_message="sync training results from local run",
    )
    print("  runs/ synced")

    # Print a quick summary to console
    for path in summary_files:
        with open(path) as f:
            d = json.load(f)
        print(
            f"  {d['env']:12s} | {d['task_id']:6s} | {d['memory_mode']:5s} | "
            f"post_score={d.get('post_score','?')}"
        )

    print(f"\nDone. Space will rebuild at: https://huggingface.co/spaces/{space_repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True, help="HF Space repo ID, e.g. yamyam05/crossmill")
    parser.add_argument("--token", default=None, help="HF token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    push_results(args.space, token=token)
