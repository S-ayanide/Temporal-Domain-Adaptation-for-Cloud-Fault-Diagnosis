"""
Run cross-domain pipeline: prepare Google→Alibaba, then train Table 1.

Does not modify data/processed/ (within-Alibaba setup). Output lives in
data/processed_google_alibaba/ unless you pass --processed-dir to the prep step.
"""

import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--skip-prep",
        action="store_true",
        help="Only run training (data already prepared)",
    )
    ap.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed_google_alibaba",
        help="Relative to updated_research/; passed to prep + train",
    )
    args = ap.parse_args()
    proc = BASE_DIR / args.processed_dir

    if not args.skip_prep:
        r = subprocess.run(
            [
                sys.executable,
                str(BASE_DIR / "00_prepare_data_google_alibaba.py"),
                "--processed-dir",
                str(proc),
            ],
            cwd=str(BASE_DIR),
        )
        if r.returncode != 0:
            sys.exit(r.returncode)

    r = subprocess.run(
        [
            sys.executable,
            str(BASE_DIR / "01_train_all_models.py"),
            "--processed-dir",
            str(proc),
        ],
        cwd=str(BASE_DIR),
    )
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
