"""
Table 4 — Adaptive Attacker: attacker knows VIPER exists and tries to
evade it (gaussian noise, cluster mimicking, low-confidence bridging).
Output: results/table4_adaptive_attack.csv
"""
import logging
import numpy as np
import pandas as pd

from .. import config, harness

logger = logging.getLogger("viper.exp4")
PRIMARY_DATASET = config.DATASETS[0]


def run():
    rows = []
    for attack_type in config.ATTACK_TYPES:
        for method in ["no_defense", "viper"]:
            asr_list, f1_list = [], []
            for seed in config.RANDOM_SEEDS:
                fed = harness.build_federation(PRIMARY_DATASET, seed=seed)
                fed = harness.inject_poison(fed, attack_type=attack_type, seed=seed)
                flags, _ = harness.run_defense(method, fed, seed=seed)
                res = harness.evaluate(fed, flags)
                asr_list.append(res["asr"]); f1_list.append(res["f1"])
            rows.append({
                "attack_type": attack_type, "method": method,
                "ASR_mean": np.mean(asr_list), "ASR_std": np.std(asr_list),
                "F1_mean": np.mean(f1_list),
            })
            logger.info(f"[exp4] {attack_type} / {method} -> ASR={np.mean(asr_list):.3f}")

    df = pd.DataFrame(rows)
    out_path = f"{config.RESULTS_DIR}/table4_adaptive_attack.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"[exp4] saved -> {out_path}")
    return df
