"""
VIPER Defense: F.I.X + T.R.A.P + C.O.R.E
Novel contribution: C.O.R.E (Correlated Residual Echo)
Speed: Pure NumPy SVD (10x faster than sklearn PCA)
Threshold: Bounded Dynamic MAD (adapts per dataset, floor=0.45)
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from .. import config

_CORE_CACHE = {}


def _compute_core_components(vectors, client_ids):
    n = len(vectors)
    nbrs = NearestNeighbors(n_neighbors=min(80, n-1), metric='cosine').fit(vectors)
    dists, indices = nbrs.kneighbors(vectors)

    residuals = np.zeros(n)
    r_norms = np.zeros((n, vectors.shape[1]))

    for i in range(n):
        nn_idx = indices[i, 1:]
        diff_client = nn_idx[client_ids[nn_idx] != client_ids[i]][:50]
        if len(diff_client) < 10:
            diff_client = nn_idx[:50]
        neighbors = vectors[diff_client]
        v = vectors[i]

        # NumPy SVD (10x faster than sklearn PCA)
        mean_vec = np.mean(neighbors, axis=0)
        centered_neighbors = neighbors - mean_vec
        centered_v = v - mean_vec
        _, _, Vt = np.linalg.svd(centered_neighbors, full_matrices=False)
        k = min(10, len(neighbors) - 1)
        Vt_k = Vt[:k]
        reconstructed_centered = (centered_v @ Vt_k.T) @ Vt_k
        residual = centered_v - reconstructed_centered
        residuals[i] = np.linalg.norm(residual)
        r_norms[i] = residual / (np.linalg.norm(residual) + 1e-12)

    # C.O.R.E: cross-residual correlation matrix
    residual_sims = r_norms @ r_norms.T
    np.fill_diagonal(residual_sims, -1)
    core_scores = np.max(residual_sims, axis=1)

    # Centroid scores
    centroid = vectors.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
    cos_sims = vectors @ centroid

    return residuals, core_scores, cos_sims


def run_viper(vectors, client_ids, texts=None, embed_fn=None,
              components=None, vote_threshold=None):
    n = len(vectors)
    if n < 10:
        return np.zeros(n, dtype=bool), {}

    components = components or {"centroid": True, "trap": True, "core": True}

    # Cache per unique vector set
    cache_key = (id(vectors), vectors.shape)
    if cache_key not in _CORE_CACHE:
        _CORE_CACHE.clear()
        _CORE_CACHE[cache_key] = _compute_core_components(vectors, client_ids)
    residuals, core_scores, cos_sims = _CORE_CACHE[cache_key]

    flags = np.zeros(n, dtype=bool)
    info = {}

    # C.O.R.E — Bounded Dynamic MAD threshold
    if components.get("core", False):
        median_core = np.median(core_scores)
        mad_core = np.median(np.abs(core_scores - median_core))
        dynamic_thresh = median_core + 5.0 * mad_core
        final_threshold = max(0.45, dynamic_thresh)
        core_flags = core_scores > final_threshold
        flags |= core_flags
        info["core_flags"] = core_flags
        info["core_scores"] = core_scores

    # T.R.A.P — MAD on reconstruction error
    if components.get("trap", False):
        median_res = np.median(residuals)
        mad_res = np.median(np.abs(residuals - median_res))
        trap_flags = residuals > median_res + 3.0 * mad_res
        flags |= trap_flags
        info["trap_flags"] = trap_flags

    # Centroid — extreme outlier gatekeeper
    if components.get("centroid", False):
        centroid_flags = cos_sims <= np.percentile(cos_sims, 8)
        flags |= centroid_flags
        info["centroid_flags"] = centroid_flags

    info["votes"] = flags.astype(int)
    return flags, info
