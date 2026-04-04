"""
Master script — runs all steps in sequence.
Run with:  nohup python run_all.py > logs/run_all.log 2>&1 &
"""

import logging, subprocess, sys, time
from pathlib import Path

BASE_DIR = Path(__file__).parent
LOG_DIR  = BASE_DIR/"logs"; LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_DIR/"run_all.log"),
              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

STEPS = [
    ("Step 0/5: Data Preparation",               "00_prepare_data.py"),
    ("Step 1/5: Train All Models (Table 1)",      "01_train_all_models.py"),
    ("Step 2/5: Label Scarcity (Figure 2)",       "02_experiment_label_scarcity.py"),
    ("Step 3/5: Class Imbalance (Figure 3)",      "03_experiment_class_imbalance.py"),
    ("Step 4/5: Heterogeneous Nodes (Figure 4)",  "04_experiment_heterogeneous_nodes.py"),
    ("Step 5/5: Ablation Study (Figure 5)",       "05_ablation_study.py"),
]


def run():
    total_start = time.time()
    for label, script in STEPS:
        logger.info("="*65)
        logger.info(f"  {label}")
        logger.info("="*65)
        t0 = time.time()
        result = subprocess.run(
            [sys.executable, str(BASE_DIR/script)],
            cwd=str(BASE_DIR), check=False
        )
        elapsed = (time.time()-t0)/60
        if result.returncode != 0:
            logger.error(f"  FAILED ({elapsed:.1f} min)  —  "
                         f"return code {result.returncode}")
            logger.error("  Stopping. Fix the error and re-run, or "
                         "run the script individually.")
            sys.exit(1)
        logger.info(f"  Done in {elapsed:.1f} min\n")

    total = (time.time()-total_start)/60
    logger.info("="*65)
    logger.info(f"  All steps complete in {total:.1f} min.")
    logger.info("  Results → updated_research/results/")
    logger.info("="*65)


if __name__ == "__main__":
    run()
