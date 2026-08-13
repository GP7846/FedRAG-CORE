import time
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def cosine_topk(query_vec, db_vectors, k=5):
    """db_vectors assumed L2-normalized. Returns indices of top-k by cosine sim."""
    if len(db_vectors) == 0:
        return np.array([], dtype=int)
    sims = db_vectors @ query_vec
    k = min(k, len(db_vectors))
    idx = np.argpartition(-sims, k - 1)[:k]
    return idx[np.argsort(-sims[idx])]


def attack_success_rate(trigger_vecs, db_vectors, db_is_poison, k=5):
    """Fraction of trigger queries for which >=1 poison vector is retrieved in top-k."""
    if len(trigger_vecs) == 0 or len(db_vectors) == 0:
        return 0.0
    hits = 0
    for tv in trigger_vecs:
        top_idx = cosine_topk(tv, db_vectors, k=k)
        if np.any(db_is_poison[top_idx]):
            hits += 1
    return hits / len(trigger_vecs)


def detection_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if len(y_true) == 0:
        return dict(precision=0.0, recall=0.0, f1=0.0, fpr=0.0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    neg = np.sum(y_true == 0)
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fpr = fp / neg if neg > 0 else 0.0
    return dict(precision=float(precision), recall=float(recall), f1=float(f1), fpr=float(fpr))


def recall_at_k(query_vecs, relevant_ids, db_vectors, db_ids, k=5):
    if len(query_vecs) == 0:
        return 0.0
    hits = 0
    for qv, rel_id in zip(query_vecs, relevant_ids):
        top_idx = cosine_topk(qv, db_vectors, k=k)
        top_ids = [db_ids[i] for i in top_idx]
        if rel_id in top_ids:
            hits += 1
    return hits / len(query_vecs)


def mrr_at_k(query_vecs, relevant_ids, db_vectors, db_ids, k=10):
    if len(query_vecs) == 0:
        return 0.0
    rr_sum = 0.0
    for qv, rel_id in zip(query_vecs, relevant_ids):
        top_idx = cosine_topk(qv, db_vectors, k=k)
        top_ids = [db_ids[i] for i in top_idx]
        if rel_id in top_ids:
            rank = top_ids.index(rel_id) + 1
            rr_sum += 1.0 / rank
    return rr_sum / len(query_vecs)


def ndcg_at_k(query_vecs, relevant_ids, db_vectors, db_ids, k=10):
    if len(query_vecs) == 0:
        return 0.0
    total = 0.0
    for qv, rel_id in zip(query_vecs, relevant_ids):
        top_idx = cosine_topk(qv, db_vectors, k=k)
        top_ids = [db_ids[i] for i in top_idx]
        dcg = 0.0
        for rank, did in enumerate(top_ids, start=1):
            if did == rel_id:
                dcg += 1.0 / np.log2(rank + 1)
        idcg = 1.0  # single relevant doc -> ideal DCG = 1/log2(2) = 1
        total += dcg / idcg
    return total / len(query_vecs)


class LatencyTimer:
    def __init__(self):
        self.t0 = None
        self.elapsed_ms = None

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self.t0) * 1000.0
