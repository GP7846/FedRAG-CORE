# VIPER — Knowledge Poisoning Attacks & Defense in Federated RAG

Complete, runnable experiment suite: attack + defense + 6 baselines +
7 result tables, on 3 public benchmark datasets, no manual dataset
download, no API keys required.

## What this actually is (read this first)

This is a real, working research pipeline — every number in the output
CSVs is computed from an actual run, nothing is hardcoded or faked.
A few honest notes on design choices, so you can defend them to
reviewers instead of being surprised by a question:

- **No Docker/gRPC.** Kaggle kernels don't allow privileged containers,
  and gRPC-over-Docker adds fragile networking that risks a mid-run
  crash. Federation is simulated by partitioning data across client IDs
  — the server still only ever sees vectors + a client_id, never raw
  text, which is exactly the threat model the paper argues about. If
  you run on vast.ai and want literal process isolation, you can wrap
  `src/federation.py` calls in separate containers later without
  changing any algorithm.
- **No Groq/OpenAI API.** Everything runs locally (BGE-small embedding
  model, auto-downloaded from HuggingFace). Attack Success Rate (ASR)
  is measured the standard way used in RAG-poisoning literature
  (e.g. PoisonedRAG-style evaluation): *does the poisoned vector get
  retrieved in the top-k for the trigger query?* This is a more
  reproducible and much faster metric than generating full LLM answers,
  and it's what most Q1 RAG-security papers actually report.
- **Baselines are adaptations, not literal reproductions.** Krum and
  FLTrust were built for FL gradient updates, and STRIP for classifier
  input-perturbation. Each is adapted here to operate directly on
  embeddings — see docstrings in `src/defenses/baselines.py`. This is
  itself the paper's point: existing defenses don't transfer cleanly.
- **Datasets auto-download** from HuggingFace on first run (MedQA,
  BEIR/NFCorpus, BEIR/MS MARCO). If a download fails for any reason
  (no internet, HF hiccup), that dataset silently falls back to a
  deterministic synthetic medical corpus so the whole run doesn't die —
  check `logs/data_warnings.log` to see if this happened.
- **Homomorphic encryption (the "E" in VIPER) is NOT implemented here**
  — it's a cite-only theoretical extension (FRAG protocol / TenSEAL),
  as discussed. Implementing real HE would blow the 4-5 hour budget.

## Quick start (Kaggle / vast.ai / any Linux box with a GPU)

```bash
bash run_all.sh
```

This installs dependencies, runs a 1-2 minute smoke test (tiny
synthetic data, catches environment problems fast), then runs the full
suite (~4-5 hours) and zips the results.

To run manually:

```bash
pip install -r requirements.txt
python main.py --smoke-test      # sanity check first, always
python main.py                   # full run
python main.py --only exp1,exp3  # just specific tables
```

## Output

Seven CSVs land in `results/`:

| File | Content |
|---|---|
| `table1_main_comparison.csv` | VIPER vs 6 baselines x 3 datasets |
| `table2_poison_ratio.csv` | ASR/F1 as attack strength increases |
| `table3_ablation.csv` | Contribution of LID / Provenance / Stability |
| `table4_adaptive_attack.csv` | Robustness vs an attacker who knows VIPER |
| `table5_scalability.csv` | 5 → 50 clients |
| `table6_retrieval_quality.csv` | Proof VIPER doesn't hurt clean retrieval |
| `table7_computational_cost.csv` | Latency / throughput per method |

Plus `results/run_summary.txt` and `logs/run_log.txt` (full run log,
timestamps per experiment, any dataset fallback warnings).

## Tuning runtime

Everything is controlled from `src/config.py`:

- `DATASET_DOC_LIMIT` / `DATASET_QUERY_LIMIT` — corpus size per dataset
- `RANDOM_SEEDS` — how many repeated trials per config (more = more
  statistically solid, slower)
- `SCALABILITY_CLIENT_COUNTS`, `POISON_RATIO_SWEEP` — sweep granularity

Defaults are sized to comfortably fit 4-5 hours on a single GPU. If
your instance is CPU-only or slower, drop `DATASET_DOC_LIMIT` to ~150
and `RANDOM_SEEDS` to `[42]` (1 seed) first — rerun the smoke test to
confirm, then launch the full run.

## Project layout

```
VIPER/
├── main.py                        # single entry point
├── run_all.sh                     # install + smoke test + full run + zip
├── requirements.txt
├── src/
│   ├── config.py                  # every tunable knob
│   ├── data_loader.py             # auto-download + synthetic fallback
│   ├── embedder.py                # BGE-small wrapper
│   ├── federation.py              # client partitioning
│   ├── attacks.py                 # interpolation + adaptive attacks
│   ├── metrics.py                 # ASR, P/R/F1/FPR, Recall@k, MRR, NDCG
│   ├── harness.py                 # shared build/inject/defend/evaluate logic
│   ├── defenses/
│   │   ├── lid.py                 # Local Intrinsic Dimensionality
│   │   ├── provenance.py          # cross-client cluster audit
│   │   ├── stability.py           # insertion-time paraphrase stability
│   │   ├── viper.py               # combines the three above (voting)
│   │   └── baselines.py           # Krum, FLTrust, IsoForest, LOF, STRIP
│   └── experiments/
│       └── exp1..exp7             # one script per results table
├── results/                       # CSV output (created at runtime)
└── logs/                          # run_log.txt, data_warnings.log
```

## Next step after this run

Numbers in `results/*.csv` are what go into your paper's tables and
figures — plotting code was intentionally left out per your request
(you said you'll generate figures yourself from the numerical CSVs).
