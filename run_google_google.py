"""Prepare within-Google tensors and run Table 1 training."""

import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-prep", action="store_true")
    ap.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed_google_google",
    )
    args = ap.parse_args()
    proc = BASE_DIR / args.processed_dir

    if not args.skip_prep:
        r = subprocess.run(
            [
                sys.executable,
                str(BASE_DIR / "00_prepare_data_google_google.py"),
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
