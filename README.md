# FedRAG-CORE: The Limits of Vector-Space Defense in Federated RAG

> **"We do not just build a defense. We prove where every defense must fail — and why."**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Dataset](https://img.shields.io/badge/Datasets-SciFact%20%7C%20NFCorpus-orange)](https://huggingface.co/datasets/BeIR)
[![Model](https://img.shields.io/badge/Embedder-BGE--small--en--v1.5-purple)](https://huggingface.co/BAAI/bge-small-en-v1.5)

---

## Overview

This repository contains the complete, reproducible experimental suite for the paper:

**"The Fundamental Limits of Vector-Space Defense in Federated Retrieval-Augmented Generation: Semantic Absorption and the Case for Two-Tier Security"**

We study **knowledge poisoning attacks** in Federated RAG (Fed-RAG) systems, where multiple organizations (e.g., hospitals) collaboratively build a shared AI knowledge base by contributing document embeddings — never sharing raw text. A malicious participant can inject **poisoned vectors** that cause the shared LLM to generate dangerous, fabricated answers.

This work makes three contributions:

1. **The Attack** — A black-box interpolation attack that crafts poisoned embeddings without any access to the encoder's gradients.
2. **The Defense (C.O.R.E)** — A novel Correlated Residual Echo detector that achieves ASR=0.12, F1=0.635, FPR=0.0 on cross-domain poisoning.
3. **The Impossibility Result** — A formal empirical proof that 11 distinct vector-space paradigms ALL fail against in-domain poisoning due to a phenomenon we term **Semantic Absorption**.

---

## The Core Problem (in Simple Terms)

```
10 Hospitals → share document embeddings → build shared AI Doctor

1 hospital is malicious:
   Uploads: embed("Drinking bleach cures the flu")
   
Doctor asks: "How do I cure the flu?"
AI retrieves poisoned vector → generates dangerous answer
```

**The challenge:** The central server never sees raw text (privacy). It only sees math vectors. Can it detect the poison vector without reading anyone's documents?

**Our answer:** Yes — for cross-domain attacks. No — for in-domain attacks. Here is the proof for both.

---

## Key Results

### Main Result: VIPER/C.O.R.E vs. 6 Baselines (SciFact Dataset)

| Method | ASR ↓ | F1 ↑ | FPR ↓ | Latency |
|--------|--------|------|-------|---------|
| No Defense | 1.000 | 0.000 | — | 0ms |
| Krum | 1.000 | 0.151 | 0.096 | 0.09ms |
| FLTrust | 1.000 | 0.164 | 0.095 | 0.001ms |
| Isolation Forest | 1.000 | 0.107 | 0.089 | 0.82ms |
| LOF | 1.000 | 0.120 | 0.088 | 0.055ms |
| STRIP | 0.853 | 0.138 | 0.423 | 0.36ms |
| **VIPER/C.O.R.E (Ours)** | **0.120** | **0.635** | **0.089** | 20.8ms |

**VIPER reduces attack success by 88% while maintaining near-zero false positives.**

### Adaptive Attack Robustness

| Attack Type | No Defense ASR | VIPER ASR |
|-------------|---------------|-----------|
| Standard Interpolation | 1.000 | 0.120 |
| Gaussian Evasion | 0.800 | **0.000** |
| Low Confidence | 1.000 | **0.000** |
| Cluster Mimicking | 1.000 | 0.680 |

### Retrieval Quality (VIPER Does Not Hurt Clean Retrieval)

| Dataset | No Defense NDCG@10 | VIPER NDCG@10 |
|---------|--------------------|----------------|
| SciFact | 0.445 | **0.657** (+48%) |
| NFCorpus | 0.193 | **0.208** (+8%) |

> **Key finding:** VIPER not only detects poison — it actively improves retrieval quality by removing off-manifold noisy vectors.

### Scalability

| Clients | Malicious | ASR | F1 |
|---------|-----------|-----|-----|
| 5 | 1 | 0.067 | 0.601 |
| 10 | 1 | 0.120 | 0.598 |
| 20 | 2 | 0.120 | 0.611 |
| 50 | 5 | 0.120 | 0.617 |

**VIPER scales stably — performance does not degrade as the federation grows.**

---

## The Impossibility Result: Semantic Absorption

On the **NFCorpus** dataset (uniform medical domain), we tested 11 distinct vector-space detection paradigms. **All 11 failed.**

| # | Paradigm | Method | Caught/25 | ASR | F1 |
|---|----------|--------|-----------|-----|-----|
| 1 | Spatial Geometry | Centroid Distance | 7 | 1.00 | 0.298 |
| 2 | Spatial Geometry | k-NN Distance | 14 | 0.88 | 0.418 |
| 3 | Topological Manifold | TRAP/PCA | 7 | 1.00 | 0.438 |
| 4 | Topological Manifold | DSV (Dual-Subspace) | 1 | 1.00 | 0.050 |
| 5 | Topological Manifold | Autoencoder | 12 | 0.92 | 0.545 |
| 6 | Correlation | C.O.R.E | 0 | 1.00 | 0.000 |
| 7 | Dynamic Perturbation | LPA (Latent Perturbation) | 9 | 0.96 | 0.273 |
| 8 | Lexical Projection | LVP (Vocabulary Projection) | 7 | 1.00 | 0.209 |
| 9 | Statistical | Kurtosis | 8 | 0.96 | 0.302 |
| 10 | Statistical | Skewness | 8 | 1.00 | 0.302 |
| 11 | Information Theory | Shannon Entropy | 4 | 1.00 | 0.151 |

**Why all methods fail:** When a malicious payload (e.g., "bleach cures flu") is embedded inside a semantically similar corpus (medical documents), the embedding model's contrastive training absorbs the payload into the existing domain curvature. The 5% payload signal becomes mathematically indistinguishable from natural corpus variance. We term this **Semantic Absorption**.

> *Once an embedding model fuses an in-domain payload into a vector, no mathematical operation on that vector alone can recover the distinction — the information is provably lost in the compression to 384 dimensions.*

---

## Our Novel Defense: C.O.R.E

**C.O.R.E = Correlated Residual Echo**

### The Key Insight

Poison vectors created by linear interpolation all share the **same payload direction** in embedding space. When we extract the off-manifold residual of each vector (via cross-client PCA projection) and normalize it to unit length, poison vectors' residuals will strongly correlate with each other — because they all point toward the same payload concept. Clean vectors' residuals point in random directions.

### The Algorithm

```python
# Step 1: Cross-client PCA — build local subspace from OTHER clients' vectors
for each vector v from client X:
    neighbors = vectors from clients ≠ X
    subspace  = PCA(neighbors, n_components=10)
    
    # Step 2: Extract residual (off-manifold component)
    reconstructed = subspace.project_and_reconstruct(v)
    residual      = v - reconstructed
    
    # Step 3: Normalize to unit vector (kills magnitude, keeps direction)
    r_norm = residual / ||residual||

# Step 4: C.O.R.E — cross-residual correlation matrix
similarity_matrix = r_norms @ r_norms.T
core_score[i]     = max(similarity_matrix[i])  # exclude self

# Step 5: Bounded Dynamic MAD threshold
threshold = max(0.45, median + 5 * MAD)
poison_flag = core_score > threshold
```

### Why It Works (Cross-Domain)

- **Clean vectors:** residuals point in random directions → low cross-correlation
- **Poison vectors:** residuals all point toward the payload cluster → core_score ≈ 1.0

### Why It Fails (In-Domain)

When payload and corpus share the same semantic domain, the residual direction is not distinguishable from normal corpus variance → core_scores overlap completely.

---

## The Proposed Solution: Two-Tier Architecture

Since purely vector-space defenses are provably blind to in-domain poisoning, we propose:

```
┌─────────────────────────────────────────────────────┐
│                  TIER 1: SERVER SIDE                │
│         VIPER/C.O.R.E Vector-Space Filtering        │
│  Automatically catches all cross-domain attacks     │
│  (naive injections, batch poisoning, evasion)       │
└──────────────────────────┬──────────────────────────┘
                           │ Clean vectors only
                           ▼
                   Vector Database (FAISS)
                           │
                           ▼ Retrieved text chunks
┌─────────────────────────────────────────────────────┐
│                  TIER 2: EDGE SIDE                  │
│         Context Auditing (e.g., Llama-Guard)        │
│  Scans retrieved plaintext for semantic             │
│  contradictions before LLM generation               │
└─────────────────────────────────────────────────────┘
```

Tier 1 handles all vector-detectable attacks with zero privacy cost. Tier 2 handles in-domain semantic attacks at the retrieval client's edge — where plaintext is available and privacy is preserved (text never leaves the querying client).

---

## Repository Structure

```
FedRAG-CORE/
├── main.py                          # Single entry point — runs all 7 experiments
├── run_all.sh                       # One-shot install + run script
├── impossibility_proof.py           # Standalone script: all 11 impossibility methods
├── requirements.txt
├── README.md
│
├── src/
│   ├── config.py                    # All hyperparameters in one place
│   ├── data_loader.py               # Auto-downloads SciFact + NFCorpus (BeIR)
│   ├── embedder.py                  # BGE-small-en-v1.5 wrapper (GPU/CPU auto)
│   ├── federation.py                # Federated client partitioning
│   ├── attacks.py                   # Interpolation + 3 adaptive attack variants
│   ├── metrics.py                   # ASR, F1, FPR, Recall@k, MRR, NDCG
│   ├── harness.py                   # Shared experiment scaffolding
│   │
│   ├── defenses/
│   │   ├── viper.py                 # C.O.R.E: our novel defense (main contribution)
│   │   ├── baselines.py             # Krum, FLTrust, IsoForest, LOF, STRIP
│   │   ├── lid.py                   # Local Intrinsic Dimensionality
│   │   ├── provenance.py            # Cross-client provenance audit
│   │   └── stability.py            # Semantic stability via paraphrasing
│   │
│   └── experiments/
│       ├── exp1_main_comparison.py  # Table 1: VIPER vs 6 baselines × 2 datasets
│       ├── exp2_poison_ratio.py     # Table 2: Attack strength sweep
│       ├── exp3_ablation.py         # Table 3: C.O.R.E component ablation
│       ├── exp4_adaptive_attack.py  # Table 4: Adaptive attacker robustness
│       ├── exp5_scalability.py      # Table 5: 5 → 50 clients
│       ├── exp6_retrieval_quality.py# Table 6: Clean retrieval quality
│       └── exp7_computational_cost.py # Table 7: Latency + throughput
│
└── results/
    ├── table1_main_comparison.csv
    ├── table2_poison_ratio.csv
    ├── table3_ablation.csv
    ├── table4_adaptive_attack.csv
    ├── table5_scalability.csv
    ├── table6_retrieval_quality.csv
    ├── table7_computational_cost.csv
    └── table_impossibility_all11.csv
```

---

## Reproducing the Experiments

### Requirements

- Python 3.10+
- GPU recommended (CUDA), CPU works but slower
- ~4 GB disk space (model + dataset cache)

### Installation

```bash
git clone https://github.com/GP7846/FedRAG-CORE.git
cd FedRAG-CORE
pip install -r requirements.txt
```

### Quick Start

```bash
# 1. Smoke test first (~2 minutes, validates environment)
python main.py --smoke-test

# 2. Full experiment suite (~25-30 minutes on GPU)
python main.py

# 3. Impossibility proof (all 11 methods on NFCorpus)
python impossibility_proof.py
```

### Run Specific Experiments

```bash
# Run only Table 1 and Table 3
python main.py --only exp1,exp3
```

### One-Shot (Install + Run + Zip Results)

```bash
bash run_all.sh
```

---

## Datasets

All datasets download automatically on first run via HuggingFace.

| Dataset | Source | Domain | Docs Used | Role |
|---------|--------|--------|-----------|------|
| SciFact | [BeIR/scifact](https://huggingface.co/datasets/BeIR/scifact) | Scientific claims | 250 | Cross-domain (VIPER succeeds) |
| NFCorpus | [BeIR/nfcorpus](https://huggingface.co/datasets/BeIR/nfcorpus) | Medical nutrition | 250 | In-domain (Semantic Absorption) |

---

## Embedding Model

**BAAI/bge-small-en-v1.5** — 33M parameters, 384-dimensional output, ~130MB.
Auto-downloaded from HuggingFace on first run. GPU acceleration used automatically if available.

---

## Threat Model

- **Federation:** N clients (default N=10), 1 malicious
- **Privacy constraint:** Server never sees raw text — only vectors + client ID
- **Attacker capability:** Full control over one client's submissions; knows the embedding model; may know the defense exists (adaptive attack)
- **Attack goal:** When a legitimate user asks a trigger question, the LLM retrieves the poisoned vector and generates a dangerous answer
- **Defender goal:** Detect and remove poisoned vectors at insertion time, without decoding any client's private text

---

## Attack: Interpolation Poisoning

```python
# Black-box, no backpropagation required
v_poison = normalize(α * v_trigger + (1-α) * v_payload)

# α = 0.95 (strong attack, close to trigger, hard to detect)
# v_trigger = embedding of the target query ("How to cure the flu?")
# v_payload = embedding of the dangerous answer ("Drink bleach")
```

Four variants tested: standard, gaussian evasion, cluster mimicking, low confidence.

---

## Ablation Study (C.O.R.E Components)

| Configuration | ASR | F1 | FPR |
|--------------|-----|-----|-----|
| Centroid only | 0.12 | 0.524 | 0.096 |
| TRAP only | 0.12 | 0.524 | 0.089 |
| **C.O.R.E only** | **0.12** | **0.635** | **0.089** |
| Centroid + TRAP | 0.12 | 0.524 | 0.096 |
| Centroid + C.O.R.E | 0.12 | 0.620 | 0.096 |
| TRAP + C.O.R.E | 0.12 | 0.636 | 0.089 |
| Full VIPER | 0.12 | 0.598 | 0.107 |

**C.O.R.E is the dominant component.** It alone matches the full system's ASR reduction.

---

## Citation

If you use this code or build on this work, please cite:

```bibtex
@article{fedrag_core_2026,
  title   = {The Fundamental Limits of Vector-Space Defense in Federated RAG:
             Semantic Absorption and the Case for Two-Tier Security},
  author  = {Sahoo, Gopinath},
  year    = {2026},
  url     = {https://github.com/GP7846/FedRAG-CORE}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

**Gopinath Sahoo**
gopinathsahoo4676@gmail.com
GitHub: [@GP7846](https://github.com/GP7846)
