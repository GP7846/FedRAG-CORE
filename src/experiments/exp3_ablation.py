"""
Table 3 — Ablation: proves every VIPER/C.O.R.E component contributes.
Output: results/table3_ablation.csv
"""
import logging
import numpy as np
import pandas as pd

from .. import config, harness, embedder
from ..defenses import viper as viper_defense
from .. import metrics as metrics_mod

logger = logging.getLogger("viper.exp3")
PRIMARY_DATASET = config.DATASETS[0]


def run():
    rows = []
    for config_name, components in config.ABLATION_CONFIGS.items():
        asr_list, fpr_list, f1_list, recall5_list = [], [], [], []
        for seed in config.RANDOM_SEEDS:
            fed = harness.build_federation(PRIMARY_DATASET, seed=seed)
            fed = harness.inject_poison(fed, attack_type="standard", seed=seed)

            flags, _info = viper_defense.run_viper(
                fed["vectors"], fed["client_ids"], components=components,
            )
            res = harness.evaluate(fed, flags)
            asr_list.append(res["asr"])
            fpr_list.append(res["fpr"])
            f1_list.append(res["f1"])

            keep_mask = ~flags
            filtered_vectors = fed["vectors"][keep_mask]
            filtered_ids = [fed["doc_ids"][i] for i in range(len(fed["doc_ids"])) if keep_mask[i]]
            q_texts = [q["text"] for q in fed["queries"]]
            q_rel = [q["relevant_doc_id"] for q in fed["queries"]]
            if q_texts and len(filtered_vectors) > 0:
                q_vecs = embedder.embed_texts(q_texts)
                recall5 = metrics_mod.recall_at_k(q_vecs, q_rel, filtered_vectors, filtered_ids, k=5)
            else:
                recall5 = 0.0
            recall5_list.append(recall5)

        rows.append({
            "config": config_name,
            "centroid": components.get("centroid", False),
            "trap": components.get("trap", False),
            "core": components.get("core", False),
            "ASR_mean": np.mean(asr_list),
            "FPR_mean": np.mean(fpr_list),
            "F1_mean": np.mean(f1_list),
            "Recall@5_mean": np.mean(recall5_list),
        })
        logger.info(f"[exp3] {config_name} -> ASR={np.mean(asr_list):.3f} F1={np.mean(f1_list):.3f}")

    df = pd.DataFrame(rows)
    out_path = f"{config.RESULTS_DIR}/table3_ablation.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"[exp3] saved -> {out_path}")
    return df
