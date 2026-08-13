"""
Baseline detectors VIPER is compared against.

Honesty note (also in README): Krum / FLTrust were designed for FL
gradient updates, and STRIP was designed for input-perturbation backdoor
detection in classifiers. There is no off-the-shelf embedding-space
version of these, so each is adapted below to operate on raw vectors —
this is exactly the "existing defenses don't transfer" gap VIPER's paper
argues, and the adaptations are the fairest-possible transfer of each
method's core idea so the comparison is not a straw man.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors


def krum_scores(vectors, num_malicious_guess=1):
    """Classic Krum: score(v) = sum of distances to its n-f-2 nearest
    neighbors. Higher score = more likely malicious (further from the
    honest majority)."""
    n = len(vectors)
    f = max(1, min(num_malicious_guess, n - 2))
    k = max(1, n - f - 2)
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(vectors)
    dists, _ = nbrs.kneighbors(vectors)
    scores = dists[:, 1:].sum(axis=1)  # exclude self
    return scores


def flag_by_krum(vectors, num_malicious_guess=1, percentile=90):
    if len(vectors) == 0:
        return np.array([], dtype=bool), np.array([])
    scores = krum_scores(vectors, num_malicious_guess)
    thresh = np.percentile(scores, percentile)
    return scores >= thresh, scores


def flag_by_fltrust(vectors, trusted_root_vectors, percentile=10):
    """FLTrust-like: score by cosine similarity to a small trusted
    reference set held by the server. Low trust score = suspicious.
    Flag the BOTTOM percentile (least trusted)."""
    if len(vectors) == 0:
        return np.array([], dtype=bool), np.array([])
    root_centroid = trusted_root_vectors.mean(axis=0)
    root_centroid = root_centroid / (np.linalg.norm(root_centroid) + 1e-12)
    sims = vectors @ root_centroid
    thresh = np.percentile(sims, percentile)
    flags = sims <= thresh
    return flags, sims


def flag_by_isolation_forest(vectors, contamination=0.1, seed=42):
    if len(vectors) < 5:
        return np.zeros(len(vectors), dtype=bool), np.zeros(len(vectors))
    clf = IsolationForest(contamination=contamination, random_state=seed)
    pred = clf.fit_predict(vectors)  # -1 = anomaly
    scores = -clf.score_samples(vectors)  # higher = more anomalous
    return pred == -1, scores


def flag_by_lof(vectors, contamination=0.1, n_neighbors=20):
    n = len(vectors)
    if n < 5:
        return np.zeros(n, dtype=bool), np.zeros(n)
    nn = max(2, min(n_neighbors, n - 1))
    clf = LocalOutlierFactor(n_neighbors=nn, contamination=contamination)
    pred = clf.fit_predict(vectors)  # -1 = anomaly
    scores = -clf.negative_outlier_factor_
    return pred == -1, scores


def flag_by_strip(vectors, clean_reference_pool, n_trials=10, percentile=90, seed=0):
    """STRIP-like: perturb each candidate by blending with random clean
    reference vectors and measure the ENTROPY of which reference vector
    ends up nearest after blending. Genuine, well-anchored vectors keep
    a stable nearest neighbor across trials (low entropy). A poison
    vector straddling two concepts flips its nearest neighbor
    unpredictably (high entropy) -> flagged."""
    rng = np.random.RandomState(seed)
    n = len(vectors)
    if n == 0 or len(clean_reference_pool) == 0:
        return np.zeros(n, dtype=bool), np.zeros(n)
    entropies = np.zeros(n)
    pool = clean_reference_pool
    for i in range(n):
        v = vectors[i]
        nn_choices = []
        for _ in range(n_trials):
            ref = pool[rng.randint(0, len(pool))]
            blended = 0.5 * v + 0.5 * ref
            blended = blended / (np.linalg.norm(blended) + 1e-12)
            sims = pool @ blended
            nn_choices.append(int(np.argmax(sims)))
        _, counts = np.unique(nn_choices, return_counts=True)
        p = counts / counts.sum()
        entropies[i] = -np.sum(p * np.log(p + 1e-12))
    thresh = np.percentile(entropies, percentile)
    return entropies >= thresh, entropies
