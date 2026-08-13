"""
Shared scaffolding used by every experiment script:
  1. build a federated corpus (N clients, 1+ malicious) for a dataset
  2. craft & inject poison vectors from the malicious client
  3. run any of the 7 detection methods (no_defense/krum/fltrust/
     isolation_forest/lof/strip/viper) over ALL newly-submitted vectors
  4. compute the full metric set

Keeping this in one place is what lets exp1..exp7 stay short and keeps
the numbers across tables internally consistent.
"""

import numpy as np

from . import config, embedder, federation, attacks, metrics
from .defenses import baselines as base_defenses
from .defenses import viper as viper_defense


def build_federation(dataset_name, num_clients=None, poison_ratio=None,
                      num_malicious=1, seed=42, doc_limit=None, query_limit=None):
    from . import data_loader
    num_clients = num_clients or config.DEFAULT_NUM_CLIENTS
    poison_ratio = config.DEFAULT_POISON_RATIO if poison_ratio is None else poison_ratio

    docs, queries = data_loader.load_dataset_by_name(dataset_name, doc_limit, query_limit)
    client_docs = federation.partition_across_clients(docs, num_clients, seed=seed)
    malicious_ids = federation.designate_malicious(num_clients, num_malicious, seed=seed)

    all_texts, all_client_ids, all_doc_ids = [], [], []
    for cid, cdocs in client_docs.items():
        for d in cdocs:
            all_texts.append(d["text"])
            all_client_ids.append(cid)
            all_doc_ids.append(d["id"])

    all_vectors = embedder.embed_texts(all_texts)

    return {
        "docs": docs, "queries": queries,
        "texts": all_texts, "client_ids": np.array(all_client_ids),
        "doc_ids": all_doc_ids, "vectors": all_vectors,
        "malicious_ids": malicious_ids, "poison_ratio": poison_ratio,
    }


def inject_poison(fed, attack_type="standard", seed=42):
    """Crafts poison vectors from the malicious client(s) and appends them
    to the federation's vector DB. Returns updated fed dict plus poison
    bookkeeping (trigger query vectors used, is_poison mask, etc.)."""
    rng_seed = seed
    n_total_honest = len(fed["vectors"])
    n_poison = max(1, int(fed["poison_ratio"] * n_total_honest))

    payload_vecs = embedder.embed_texts(config.MALICIOUS_PAYLOADS)

    # sample trigger queries the attacker targets (subset of the query set)
    queries = fed["queries"]
    n_trig = min(n_poison, len(queries)) if len(queries) > 0 else n_poison
    n_trig = max(n_trig, 1)
    trig_subset = queries[:n_trig] if len(queries) >= n_trig else (
        queries * (n_trig // max(1, len(queries)) + 1)
    )[:n_trig]
    trigger_texts = [q["text"] for q in trig_subset] if trig_subset else ["What is the recommended treatment?"] * n_poison
    trigger_vecs = embedder.embed_texts(trigger_texts)
    # repeat/truncate to exactly n_poison
    if len(trigger_vecs) < n_poison:
        reps = int(np.ceil(n_poison / len(trigger_vecs)))
        trigger_vecs = np.tile(trigger_vecs, (reps, 1))[:n_poison]
    else:
        trigger_vecs = trigger_vecs[:n_poison]

    clean_sample = fed["vectors"][np.random.RandomState(seed).choice(
        len(fed["vectors"]), size=min(200, len(fed["vectors"])), replace=False)]

    poison_vecs, meta = attacks.craft_poison_batch(
        trigger_vecs, payload_vecs, attack_type=attack_type,
        clean_vectors=clean_sample, seed=rng_seed,
    )

    malicious_client_id = sorted(fed["malicious_ids"])[0] if fed["malicious_ids"] else 0
    poison_texts = [config.MALICIOUS_PAYLOADS[m["payload_idx"]] for m in meta]

    full_vectors = np.vstack([fed["vectors"], poison_vecs])
    full_texts = fed["texts"] + poison_texts
    full_client_ids = np.concatenate([fed["client_ids"], np.full(len(poison_vecs), malicious_client_id)])
    full_doc_ids = fed["doc_ids"] + [f"poison_{i}" for i in range(len(poison_vecs))]
    is_poison = np.concatenate([np.zeros(len(fed["vectors"]), dtype=bool), np.ones(len(poison_vecs), dtype=bool)])

    out = dict(fed)
    out.update({
        "vectors": full_vectors, "texts": full_texts,
        "client_ids": full_client_ids, "doc_ids": full_doc_ids,
        "is_poison": is_poison, "trigger_vecs": trigger_vecs,
        "n_poison": n_poison,
    })
    return out


def run_defense(method, fed, num_malicious_guess=1, seed=42):
    """Returns (flags[bool over ALL vectors], scores or None, elapsed_ms)."""
    vectors = fed["vectors"]
    client_ids = fed["client_ids"]
    n = len(vectors)

    t0 = metrics.LatencyTimer()
    with t0:
        if method == "no_defense":
            flags = np.zeros(n, dtype=bool)
        elif method == "krum":
            flags, _ = base_defenses.flag_by_krum(vectors, num_malicious_guess=num_malicious_guess)
        elif method == "fltrust":
            honest_mask = fed["client_ids"] != sorted(fed["malicious_ids"])[0]
            trusted_root = vectors[honest_mask][:50] if honest_mask.sum() > 0 else vectors[:50]
            flags, _ = base_defenses.flag_by_fltrust(vectors, trusted_root)
        elif method == "isolation_forest":
            flags, _ = base_defenses.flag_by_isolation_forest(vectors, contamination=min(0.3, max(0.01, fed["n_poison"] / n)))
        elif method == "lof":
            flags, _ = base_defenses.flag_by_lof(vectors, contamination=min(0.3, max(0.01, fed["n_poison"] / n)))
        elif method == "strip":
            ref_pool = vectors[np.random.RandomState(seed).choice(n, size=min(100, n), replace=False)]
            flags, _ = base_defenses.flag_by_strip(vectors, ref_pool, seed=seed)
        elif method == "viper":
            flags, _info = viper_defense.run_viper(
                vectors, client_ids
            )
        else:
            raise ValueError(f"Unknown defense method: {method}")
    return flags, t0.elapsed_ms


def evaluate(fed, flags, k=None):
    """Given post-defense flags, compute ASR (on the FILTERED db) and detection metrics."""
    k = k or config.TOP_K_RETRIEVAL
    keep_mask = ~flags
    filtered_vectors = fed["vectors"][keep_mask]
    filtered_is_poison = fed["is_poison"][keep_mask]

    asr = metrics.attack_success_rate(fed["trigger_vecs"], filtered_vectors, filtered_is_poison, k=k)
    det = metrics.detection_metrics(fed["is_poison"], flags)
    return {"asr": asr, **det}
