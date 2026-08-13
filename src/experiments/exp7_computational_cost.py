"""
Table 7 — Computational Cost: per-vector latency and throughput of every
defense method, proving VIPER's overhead is practical.
Output: results/table7_computational_cost.csv
"""
import logging
import numpy as np
import pandas as pd

from .. import config, harness

logger = logging.getLogger("viper.exp7")
PRIMARY_DATASET = config.DATASETS[0]


def run():
    rows = []
    for method in config.BASELINE_METHODS:
        lat_ms_list, throughput_list = [], []
        for seed in config.RANDOM_SEEDS:
            fed = harness.build_federation(PRIMARY_DATASET, seed=seed)
            fed = harness.inject_poison(fed, attack_type="standard", seed=seed)
            flags, elapsed_ms = harness.run_defense(method, fed, seed=seed)
            n = len(fed["vectors"])
            lat_ms_list.append(elapsed_ms / max(1, n))
            throughput_list.append(n / (elapsed_ms / 1000.0) if elapsed_ms > 0 else float("inf"))
        rows.append({
            "method": method,
            "Latency_ms_per_vector_mean": np.mean(lat_ms_list),
            "Throughput_vectors_per_sec_mean": np.mean([t for t in throughput_list if np.isfinite(t)]) if any(np.isfinite(throughput_list)) else float("inf"),
        })
        logger.info(f"[exp7] {method} -> {np.mean(lat_ms_list):.4f} ms/vector")

    df = pd.DataFrame(rows)
    out_path = f"{config.RESULTS_DIR}/table7_computational_cost.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"[exp7] saved -> {out_path}")
    return df
