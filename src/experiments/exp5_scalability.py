"""
Table 5 — Scalability: does VIPER hold up as the federation grows from
5 to 50 clients (with proportionally more malicious clients)?
Output: results/table5_scalability.csv
"""
import logging
import numpy as np
import pandas as pd

from .. import config, harness

logger = logging.getLogger("viper.exp5")
PRIMARY_DATASET = config.DATASETS[0]


def run():
    rows = []
    for num_clients in config.SCALABILITY_CLIENT_COUNTS:
        num_malicious = max(1, round(0.1 * num_clients))
        asr_list, f1_list, lat_list = [], [], []
        for seed in config.RANDOM_SEEDS:
            fed = harness.build_federation(PRIMARY_DATASET, num_clients=num_clients,
                                             num_malicious=num_malicious, seed=seed)
            fed = harness.inject_poison(fed, attack_type="standard", seed=seed)
            flags, elapsed_ms = harness.run_defense("viper", fed, num_malicious_guess=num_malicious, seed=seed)
            res = harness.evaluate(fed, flags)
            asr_list.append(res["asr"]); f1_list.append(res["f1"])
            lat_list.append(elapsed_ms / max(1, len(fed["vectors"])))
        rows.append({
            "num_clients": num_clients, "num_malicious": num_malicious,
            "ASR_mean": np.mean(asr_list), "F1_mean": np.mean(f1_list),
            "Latency_ms_per_vector_mean": np.mean(lat_list),
        })
        logger.info(f"[exp5] clients={num_clients} -> ASR={np.mean(asr_list):.3f} F1={np.mean(f1_list):.3f}")

    df = pd.DataFrame(rows)
    out_path = f"{config.RESULTS_DIR}/table5_scalability.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"[exp5] saved -> {out_path}")
    return df
