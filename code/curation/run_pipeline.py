"""
run_pipeline.py

Runs the three-script dataset curation pipeline in order.

Usage: python run_pipeline.py

Scripts executed in sequence:
  1. curate_wvs_evs.py           - Extract, curate, and tidy WVS/EVS questions
  2. add_understanding_society.py - Append Understanding Society questions
  3. finalize_dataset.py          - Generate scale variants and combine with OpinionQA
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
SCRIPTS = [
    "curate_wvs_evs.py",
    "add_understanding_society.py",
    "finalize_dataset.py",
]


def main():
    for i, script in enumerate(SCRIPTS, start=1):
        script_path = SCRIPTS_DIR / script
        print(f"\n{'=' * 60}")
        print(f"Step {i}/{len(SCRIPTS)}: {script}")
        print(f"{'=' * 60}")
        result = subprocess.run([sys.executable, str(script_path)], check=False)
        if result.returncode != 0:
            print(
                f"\nPipeline aborted: {script} exited with code {result.returncode}."
            )
            sys.exit(result.returncode)
    print(f"\n{'=' * 60}")
    print("Pipeline complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
