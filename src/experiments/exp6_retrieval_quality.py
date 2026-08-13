"""
Table 6 — Retrieval Quality on clean queries: proves VIPER does not
degrade normal RAG performance while filtering out poison.
Output: results/table6_retrieval_quality.csv
"""
import logging
import numpy as np
import pandas as pd

from .. import config, harness, embedder, metrics as metrics_mod

logger = logging.getLogger("viper.exp6")


def run():
    rows = []
    for dataset in config.DATASETS:
        for method in ["no_defense", "viper"]:
            r1_l, r5_l, r10_l, mrr_l, ndcg_l = [], [], [], [], []
            for seed in config.RANDOM_SEEDS:
                fed = harness.build_federation(dataset, seed=seed)
                fed = harness.inject_poison(fed, attack_type="standard", seed=seed)
                flags, _ = harness.run_defense(method, fed, seed=seed)
                keep_mask = ~flags
                filtered_vectors = fed["vectors"][keep_mask]
                filtered_ids = [fed["doc_ids"][i] for i in range(len(fed["doc_ids"])) if keep_mask[i]]

                q_texts = [q["text"] for q in fed["queries"]]
                q_rel = [q["relevant_doc_id"] for q in fed["queries"]]
                if not q_texts:
                    continue
                q_vecs = embedder.embed_texts(q_texts)
                r1_l.append(metrics_mod.recall_at_k(q_vecs, q_rel, filtered_vectors, filtered_ids, k=1))
                r5_l.append(metrics_mod.recall_at_k(q_vecs, q_rel, filtered_vectors, filtered_ids, k=5))
                r10_l.append(metrics_mod.recall_at_k(q_vecs, q_rel, filtered_vectors, filtered_ids, k=10))
                mrr_l.append(metrics_mod.mrr_at_k(q_vecs, q_rel, filtered_vectors, filtered_ids, k=10))
                ndcg_l.append(metrics_mod.ndcg_at_k(q_vecs, q_rel, filtered_vectors, filtered_ids, k=10))

            rows.append({
                "dataset": dataset, "method": method,
                "Recall@1_mean": np.mean(r1_l) if r1_l else 0.0,
                "Recall@5_mean": np.mean(r5_l) if r5_l else 0.0,
                "Recall@10_mean": np.mean(r10_l) if r10_l else 0.0,
                "MRR@10_mean": np.mean(mrr_l) if mrr_l else 0.0,
                "NDCG@10_mean": np.mean(ndcg_l) if ndcg_l else 0.0,
            })
            logger.info(f"[exp6] {dataset} / {method} -> Recall@5={np.mean(r5_l) if r5_l else 0.0:.3f}")

    df = pd.DataFrame(rows)
    out_path = f"{config.RESULTS_DIR}/table6_retrieval_quality.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"[exp6] saved -> {out_path}")
    return df
