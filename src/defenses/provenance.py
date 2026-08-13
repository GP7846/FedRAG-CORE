"""
Cross-Client Provenance Audit.

Catches BATCH poisoning: an adaptive attacker uploads several coordinated
fake vectors that form their own dense, self-consistent cluster to evade
single-point outlier detectors like LID.

Defense idea: cluster all newly-submitted vectors (DBSCAN, cosine
distance). For every dense cluster, check which client_ids contributed
to it. A genuine topical cluster should be corroborated by multiple
independent clients (different hospitals legitimately writing about
"diabetes management"). A cluster that is dense YET originates entirely
from a single client is flagged as coordinated poisoning.
"""

import numpy as np
from sklearn.cluster import DBSCAN


def flag_by_provenance(vectors, client_ids, eps=0.35, min_samples=4):
    n = len(vectors)
    if n == 0:
        return np.array([], dtype=bool), np.full(0, -1)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(vectors)
    labels = clustering.labels_
    flags = np.zeros(n, dtype=bool)

    client_ids = np.array(client_ids)
    for lbl in set(labels):
        if lbl == -1:
            continue  # DBSCAN noise points are handled by LID, not provenance
        members = np.where(labels == lbl)[0]
        unique_clients = set(client_ids[members].tolist())
        if len(unique_clients) == 1 and len(members) >= min_samples:
            flags[members] = True
    return flags, labels
