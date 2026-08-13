"""
Table 1 — Main Result: VIPER vs. 6 baselines, across all 3 datasets.
Output: results/table1_main_comparison.csv
"""
import logging
import numpy as np
import pandas as pd

from .. import config, harness

logger = logging.getLogger("viper.exp1")


def run():
    rows = []
    for dataset in config.DATASETS:
        for method in config.BASELINE_METHODS:
            asr_list, f1_list, fpr_list, prec_list, rec_list, lat_list = [], [], [], [], [], []
            for seed in config.RANDOM_SEEDS:
                fed = harness.build_federation(dataset, seed=seed)
                fed = harness.inject_poison(fed, attack_type="standard", seed=seed)
                flags, elapsed_ms = harness.run_defense(method, fed, seed=seed)
                res = harness.evaluate(fed, flags)
                asr_list.append(res["asr"]); f1_list.append(res["f1"])
                fpr_list.append(res["fpr"]); prec_list.append(res["precision"])
                rec_list.append(res["recall"])
                lat_list.append(elapsed_ms / max(1, len(fed["vectors"])))
            rows.append({
                "dataset": dataset, "method": method,
                "ASR_mean": np.mean(asr_list), "ASR_std": np.std(asr_list),
                "F1_mean": np.mean(f1_list), "F1_std": np.std(f1_list),
                "FPR_mean": np.mean(fpr_list), "FPR_std": np.std(fpr_list),
                "Precision_mean": np.mean(prec_list), "Recall_mean": np.mean(rec_list),
                "Latency_ms_per_vector_mean": np.mean(lat_list),
            })
            logger.info(f"[exp1] {dataset} / {method} -> ASR={np.mean(asr_list):.3f} F1={np.mean(f1_list):.3f}")

    df = pd.DataFrame(rows)
    out_path = f"{config.RESULTS_DIR}/table1_main_comparison.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"[exp1] saved -> {out_path}")
    return df
