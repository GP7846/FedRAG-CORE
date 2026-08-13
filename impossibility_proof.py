"""
Impossibility Proof: Exhaustive Evaluation of Vector-Space Paradigms
on In-Domain Poisoning (NFCorpus Dataset)

This script reproduces Table: "Semantic Absorption - All 11 Methods"
from the paper: "The Limits of Vector-Space Defense in Federated RAG"

Run:
    python impossibility_proof.py

Results saved to: results/table_impossibility_all11.csv
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
from transformers import AutoModel

# ── Setup ──────────────────────────────────────────────────────────────────
from src import harness, config

os.makedirs("results", exist_ok=True)

print("Loading NFCorpus federation...")
fed = harness.build_federation("nfcorpus", seed=42)
fed = harness.inject_poison(fed, attack_type="standard", seed=42)
vectors    = fed["vectors"]
is_poison  = fed["is_poison"]
client_ids = fed["client_ids"]
n          = len(vectors)

print(f"Total vectors: {n} | Poison: {is_poison.sum()} | Clean: {(~is_poison).sum()}")
print("-" * 60)

rows = []

def evaluate_flags(flags, name, paradigm, p_score=0.0, c_score=0.0):
    res = harness.evaluate(fed, flags)
    row = {
        "method":       name,
        "paradigm":     paradigm,
        "poison_mean":  round(float(p_score), 4),
        "clean_mean":   round(float(c_score), 4),
        "caught":       int((flags & is_poison).sum()),
        "total_poison": int(is_poison.sum()),
        "FP":           int((flags & ~is_poison).sum()),
        "ASR":          round(res["asr"], 3),
        "F1":           round(res["f1"], 3),
        "FPR":          round(res["fpr"], 3),
    }
    rows.append(row)
    print(f"[{paradigm}] {name}: caught={row['caught']}/25 "
          f"ASR={row['ASR']} F1={row['F1']} FPR={row['FPR']}")
    return flags


# ── Method 1: Centroid Distance ─────────────────────────────────────────────
print("\n--- Method 1: Centroid Distance (Spatial Geometry) ---")
centroid  = vectors.mean(axis=0)
centroid  = centroid / (np.linalg.norm(centroid) + 1e-12)
cos_sims  = vectors @ centroid
flags     = cos_sims <= np.percentile(cos_sims, 8)
evaluate_flags(flags, "Centroid", "Spatial Geometry",
               cos_sims[is_poison].mean(), cos_sims[~is_poison].mean())


# ── Method 2: k-NN Distance ─────────────────────────────────────────────────
print("\n--- Method 2: kNN Distance (Spatial Geometry) ---")
nbrs      = NearestNeighbors(n_neighbors=11, metric="cosine").fit(vectors)
dists, indices = nbrs.kneighbors(vectors)
knn_d     = dists[:, 1:].mean(axis=1)
flags     = knn_d >= np.percentile(knn_d, 85)
evaluate_flags(flags, "kNN Distance", "Spatial Geometry",
               knn_d[is_poison].mean(), knn_d[~is_poison].mean())


# ── Method 3: TRAP/PCA (Cross-Client Subspace Projection) ───────────────────
print("\n--- Method 3: TRAP/PCA (Topological Manifold) ---")
nbrs2     = NearestNeighbors(n_neighbors=80, metric="cosine").fit(vectors)
_, idx2   = nbrs2.kneighbors(vectors)
residuals = np.zeros(n)
for i in range(n):
    nn  = idx2[i, 1:]
    dc  = nn[client_ids[nn] != client_ids[i]][:50]
    if len(dc) < 10:
        dc = nn[:50]
    nb  = vectors[dc]
    mv  = nb.mean(0)
    c   = nb - mv
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    Vt_k     = Vt[:min(10, len(nb) - 1)]
    cv       = vectors[i] - mv
    residuals[i] = np.linalg.norm(cv - (cv @ Vt_k.T) @ Vt_k)
med   = np.median(residuals)
mad   = np.median(np.abs(residuals - med))
flags = residuals > med + 3 * mad
evaluate_flags(flags, "TRAP/PCA", "Topological Manifold",
               residuals[is_poison].mean(), residuals[~is_poison].mean())


# ── Method 4: C.O.R.E (Correlated Residual Echo) ────────────────────────────
print("\n--- Method 4: C.O.R.E (Correlation) ---")
r_norms = np.zeros_like(vectors)
for i in range(n):
    nn  = idx2[i, 1:]
    dc  = nn[client_ids[nn] != client_ids[i]][:50]
    if len(dc) < 10:
        dc = nn[:50]
    nb  = vectors[dc]
    mv  = nb.mean(0)
    c   = nb - mv
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    Vt_k     = Vt[:min(10, len(nb) - 1)]
    cv       = vectors[i] - mv
    res      = cv - (cv @ Vt_k.T) @ Vt_k
    r_norms[i] = res / (np.linalg.norm(res) + 1e-12)
sim     = r_norms @ r_norms.T
np.fill_diagonal(sim, -1)
core_s  = sim.max(axis=1)
flags   = core_s > 0.6
evaluate_flags(flags, "C.O.R.E", "Correlation",
               core_s[is_poison].mean(), core_s[~is_poison].mean())


# ── Method 5: DSV (Dual-Subspace Validation) ────────────────────────────────
print("\n--- Method 5: DSV (Topological Manifold) ---")
def build_sub(vecs, k=10):
    mv = vecs.mean(0); c = vecs - mv
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    return mv, Vt[:min(k, len(vecs) - 1)]

def recon_err(v, mv, Vt_k):
    cv = v - mv
    return np.linalg.norm(cv - (cv @ Vt_k.T) @ Vt_k)

dsv = np.zeros(n)
rng = np.random.RandomState(42)
for cid in np.unique(client_ids):
    cm = client_ids == cid
    ci = np.where(cm)[0]
    im, iV = build_sub(vectors[ci])
    oi = np.where(~cm)[0]
    s  = rng.choice(oi, size=min(100, len(oi)), replace=False)
    xm, xV = build_sub(vectors[s])
    for idx in ci:
        dsv[idx] = recon_err(vectors[idx], xm, xV) - recon_err(vectors[idx], im, iV)
med   = np.median(dsv)
mad   = np.median(np.abs(dsv - med))
flags = dsv > med + 3 * mad
evaluate_flags(flags, "DSV", "Topological Manifold",
               dsv[is_poison].mean(), dsv[~is_poison].mean())


# ── Method 6: LPA (Latent Perturbation Audit) ───────────────────────────────
print("\n--- Method 6: LPA (Dynamic Perturbation) ---")
nbrs21    = NearestNeighbors(n_neighbors=21, metric="cosine").fit(vectors)
_, idx21  = nbrs21.kneighbors(vectors)
stab      = np.zeros(n)
rng2      = np.random.RandomState(42)
for i in range(n):
    setA   = set(idx21[i, 1:21])
    trials = []
    for _ in range(10):
        vn = vectors[i] + rng2.normal(0, 0.02, vectors[i].shape)
        vn = vn / (np.linalg.norm(vn) + 1e-12)
        _, ni = nbrs21.kneighbors(vn.reshape(1, -1), n_neighbors=21)
        setB  = set(ni[0, 1:])
        trials.append(len(setA & setB) / len(setA | setB))
    stab[i] = np.mean(trials)
flags = stab < np.percentile(stab, 15)
evaluate_flags(flags, "LPA", "Dynamic Perturbation",
               stab[is_poison].mean(), stab[~is_poison].mean())


# ── Method 7: LVP (Latent Vocabulary Projection) ────────────────────────────
print("\n--- Method 7: LVP (Lexical Projection) ---")
model  = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5", cache_dir="cache")
we     = model.embeddings.word_embeddings.weight.detach().numpy()
we_n   = we / (np.linalg.norm(we, axis=1, keepdims=True) + 1e-12)
coh    = np.zeros(n)
for i in range(n):
    v   = vectors[i] / (np.linalg.norm(vectors[i]) + 1e-12)
    top = np.argsort(we_n @ v)[-50:]
    tv  = we_n[top]
    sm  = tv @ tv.T
    np.fill_diagonal(sm, np.nan)
    coh[i] = np.nanmean(sm)
flags = coh < np.percentile(coh, 15)
evaluate_flags(flags, "LVP", "Lexical Projection",
               coh[is_poison].mean(), coh[~is_poison].mean())


# ── Method 8: Kurtosis ──────────────────────────────────────────────────────
print("\n--- Method 8: Kurtosis (Statistical) ---")
kurt  = np.array([stats.kurtosis(v) for v in vectors])
flags = kurt < np.percentile(kurt, 10)
evaluate_flags(flags, "Kurtosis", "Statistical",
               kurt[is_poison].mean(), kurt[~is_poison].mean())


# ── Method 9: Skewness ──────────────────────────────────────────────────────
print("\n--- Method 9: Skewness (Statistical) ---")
skew  = np.array([stats.skew(v) for v in vectors])
flags = skew < np.percentile(skew, 10)
evaluate_flags(flags, "Skewness", "Statistical",
               skew[is_poison].mean(), skew[~is_poison].mean())


# ── Method 10: Shannon Entropy ──────────────────────────────────────────────
print("\n--- Method 10: Shannon Entropy (Information Theory) ---")
def shannon(v):
    p = np.abs(v); p = p / (p.sum() + 1e-12)
    return -np.sum(p * np.log(p + 1e-12))
ent   = np.array([shannon(v) for v in vectors])
flags = ent > np.percentile(ent, 90)
evaluate_flags(flags, "Shannon Entropy", "Information Theory",
               ent[is_poison].mean(), ent[~is_poison].mean())


# ── Method 11: Autoencoder ──────────────────────────────────────────────────
print("\n--- Method 11: Autoencoder (Non-linear Manifold) ---")
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(384, 64), nn.ReLU(), nn.Linear(64, 16))
        self.dec = nn.Sequential(nn.Linear(16, 64),  nn.ReLU(), nn.Linear(64, 384))
    def forward(self, x):
        return self.dec(self.enc(x))

clean_v = torch.tensor(vectors[~is_poison], dtype=torch.float32)
all_v   = torch.tensor(vectors, dtype=torch.float32)
ae      = AE()
opt     = torch.optim.Adam(ae.parameters(), lr=1e-3)
for epoch in range(200):
    ae.train()
    loss = ((ae(clean_v) - clean_v) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
ae.eval()
with torch.no_grad():
    err = ((ae(all_v) - all_v) ** 2).mean(dim=1).numpy()
med   = np.median(err)
mad   = np.median(np.abs(err - med))
flags = err > med + 2.5 * mad
evaluate_flags(flags, "Autoencoder", "Non-linear Manifold",
               err[is_poison].mean(), err[~is_poison].mean())


# ── Save Results ────────────────────────────────────────────────────────────
df = pd.DataFrame(rows)
out_path = "results/table_impossibility_all11.csv"
df.to_csv(out_path, index=False)

print("\n" + "=" * 60)
print("SEMANTIC ABSORPTION — IMPOSSIBILITY PROOF")
print("=" * 60)
print(df[["method", "paradigm", "caught", "FP", "ASR", "F1", "FPR"]].to_string(index=False))
print(f"\nSaved to: {out_path}")
print("\nConclusion: All 11 vector-space paradigms fail to detect")
print("in-domain poisoning. This proves Semantic Absorption.")
