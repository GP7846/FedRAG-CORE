#!/usr/bin/env python
"""
VIPER — single entry point for the complete experiment suite.

Usage:
    python main.py                 # full run (~4-5 hours, tune sizes in src/config.py)
    python main.py --smoke-test    # ~1-2 minute dry run on tiny synthetic data,
                                    # to confirm the whole pipeline works before
                                    # committing to the full run
    python main.py --only exp1,exp3   # run a subset (comma-separated exp1..exp7)

Every experiment is wrapped in its own try/except: if one experiment
fails (e.g. a dataset download hiccup that even the synthetic fallback
couldn't cover), the remaining experiments still run and you still get
partial results instead of nothing.
"""

import argparse
import os
import sys
import time
import logging
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("--smoke-test", action="store_true", help="tiny fast dry run to validate the pipeline")
parser.add_argument("--only", type=str, default=None, help="comma-separated subset, e.g. exp1,exp3")
args = parser.parse_args()

if args.smoke_test:
    os.environ["VIPER_SMOKE_TEST"] = "1"

# config.py reads VIPER_SMOKE_TEST at import time, so the env var above
# MUST be set before any `src.*` module is imported.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(config.LOGS_DIR, "run_log.txt")),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("viper.main")

EXPERIMENTS = [
    ("exp1", "src.experiments.exp1_main_comparison", "Table 1 - Main Comparison"),
    ("exp2", "src.experiments.exp2_poison_ratio", "Table 2 - Poison Ratio Sweep"),
    ("exp3", "src.experiments.exp3_ablation", "Table 3 - Ablation Study"),
    ("exp4", "src.experiments.exp4_adaptive_attack", "Table 4 - Adaptive Attack"),
    ("exp5", "src.experiments.exp5_scalability", "Table 5 - Scalability"),
    ("exp6", "src.experiments.exp6_retrieval_quality", "Table 6 - Retrieval Quality"),
    ("exp7", "src.experiments.exp7_computational_cost", "Table 7 - Computational Cost"),
]


def main():
    only = set(args.only.split(",")) if args.only else None
    mode = "SMOKE TEST" if config.SMOKE_TEST else "FULL RUN"
    logger.info(f"=================== VIPER {mode} STARTING ===================")
    logger.info(f"Datasets: {config.DATASETS} | doc_limit={config.DATASET_DOC_LIMIT} "
                f"query_limit={config.DATASET_QUERY_LIMIT} | seeds={config.RANDOM_SEEDS}")

    overall_start = time.time()
    summary = []

    for key, module_path, title in EXPERIMENTS:
        if only and key not in only:
            continue
        logger.info(f"----- {title} ({key}) -----")
        t0 = time.time()
        try:
            module = __import__(module_path, fromlist=["run"])
            df = module.run()
            elapsed = time.time() - t0
            logger.info(f"{key} DONE in {elapsed/60:.1f} min -> {len(df)} rows")
            summary.append((key, title, "OK", f"{elapsed/60:.1f} min", len(df)))
        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f"{key} FAILED after {elapsed/60:.1f} min: {e}")
            logger.error(traceback.format_exc())
            summary.append((key, title, "FAILED", f"{elapsed/60:.1f} min", str(e)))

    total_elapsed = time.time() - overall_start
    logger.info("=================== VIPER RUN COMPLETE ===================")
    logger.info(f"Total time: {total_elapsed/60:.1f} minutes")
    logger.info("Summary:")
    for key, title, status, elapsed, extra in summary:
        logger.info(f"  {key:6s} {title:35s} {status:8s} {elapsed:10s} {extra}")

    with open(os.path.join(config.RESULTS_DIR, "run_summary.txt"), "w") as f:
        f.write(f"VIPER run summary ({mode})\n")
        f.write(f"Total time: {total_elapsed/60:.1f} minutes\n\n")
        for key, title, status, elapsed, extra in summary:
            f.write(f"{key:6s} {title:35s} {status:8s} {elapsed:10s} {extra}\n")

    logger.info(f"All CSV tables written to: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
