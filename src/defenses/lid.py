"""
Local Intrinsic Dimensionality (LID) anomaly scorer.

Poisoned vectors that force two unrelated semantic regions together tend
to sit in locally higher-dimensional ("more spread out") neighborhoods
than genuine, topically coherent vectors. LID captures this via the
growth rate of the k-NN distance distribution and, unlike raw cosine
distance, does not collapse in high-dimensional embedding spaces.

    LID(v) = - ( (1/k) * sum_i log( r_i(v) / r_k(v) ) )^-1

Reliable with as few as ~50 vectors, which matters since a small
federation may only have a few hundred vectors total.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_lid(vectors, k=20):
    n = len(vectors)
    k_eff = max(2, min(k, n - 1))
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean").fit(vectors)
    distances, _ = nbrs.kneighbors(vectors)
    distances = distances[:, 1:]  # drop self (distance 0)

    lid_scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        d = distances[i]
        d = d[d > 1e-12]
        if len(d) < 2:
            lid_scores[i] = 0.0
            continue
        r_k = d[-1]
        ratios = np.clip(d / r_k, 1e-12, 1.0)
        mean_log = np.mean(np.log(ratios))
        lid_scores[i] = -1.0 / mean_log if mean_log < 0 else 0.0
    return lid_scores


def flag_by_lid(vectors, k=20, percentile=90):
    """Returns (flags[bool], scores[float]). Flags True = suspected poison."""
    scores = compute_lid(vectors, k=k)
    if len(scores) == 0:
        return np.array([], dtype=bool), scores
    thresh = np.percentile(scores, percentile)
    flags = scores >= thresh
    return flags, scores
