"""
Table 2 — Attack Strength Variation: how ASR/F1 change as the malicious
client's poison ratio increases, No-Defense vs VIPER.
Output: results/table2_poison_ratio.csv
"""
import logging
import numpy as np
import pandas as pd

from .. import config, harness

logger = logging.getLogger("viper.exp2")
PRIMARY_DATASET = config.DATASETS[0]  # medqa; kept to one dataset to respect the time budget


def run():
    rows = []
    for ratio in config.POISON_RATIO_SWEEP:
        for method in ["no_defense", "viper"]:
            asr_list, f1_list = [], []
            for seed in config.RANDOM_SEEDS:
                fed = harness.build_federation(PRIMARY_DATASET, poison_ratio=ratio, seed=seed)
                fed = harness.inject_poison(fed, attack_type="standard", seed=seed)
                flags, _ = harness.run_defense(method, fed, seed=seed)
                res = harness.evaluate(fed, flags)
                asr_list.append(res["asr"]); f1_list.append(res["f1"])
            rows.append({
                "dataset": PRIMARY_DATASET, "poison_ratio": ratio, "method": method,
                "ASR_mean": np.mean(asr_list), "ASR_std": np.std(asr_list),
                "F1_mean": np.mean(f1_list), "F1_std": np.std(f1_list),
            })
            logger.info(f"[exp2] ratio={ratio} / {method} -> ASR={np.mean(asr_list):.3f}")

    df = pd.DataFrame(rows)
    out_path = f"{config.RESULTS_DIR}/table2_poison_ratio.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"[exp2] saved -> {out_path}")
    return df
