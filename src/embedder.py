"""
Thin wrapper around a lightweight sentence-transformer (BGE-small, ~33M
params, ~130MB fp32). Auto-downloads from HuggingFace on first use.
Falls back to a fast deterministic hash-embedding if sentence-transformers
or torch cannot be imported/downloaded at all (keeps the pipeline alive
on a badly configured machine, e.g. total offline smoke testing).
"""

import os
import hashlib
import logging
import numpy as np

from . import config

logger = logging.getLogger("viper.embedder")

_model = None
_backend = None  # "st" or "hash"


def _l2_normalize(mat):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return mat / norms


def _get_model():
    global _model, _backend
    if _model is not None:
        return _model, _backend
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer(config.EMBED_MODEL_NAME, device=device, cache_folder=config.CACHE_DIR)
        _backend = "st"
        logger.info(f"Loaded embedding model '{config.EMBED_MODEL_NAME}' on {device}.")
    except Exception as e:
        logger.warning(f"Falling back to hash-embedding backend (reason: {e}).")
        _model = None
        _backend = "hash"
    return _model, _backend


def _hash_embed(texts, dim=config.EMBED_DIM):
    """Deterministic bag-of-hashed-ngrams embedding. Only used if the real
    embedding model truly cannot be loaded (no internet / no torch)."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        tokens = t.lower().split()
        for tok in tokens:
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if (h // dim) % 2 == 0 else -1.0
            out[i, idx] += sign
    return _l2_normalize(out)


def embed_texts(texts, batch_size=None, normalize=True):
    """texts: list[str] -> np.ndarray [N, D] float32, L2-normalized by default."""
    if len(texts) == 0:
        return np.zeros((0, config.EMBED_DIM), dtype=np.float32)
    batch_size = batch_size or config.EMBED_BATCH_SIZE
    model, backend = _get_model()
    if backend == "st":
        vecs = model.encode(
            texts, batch_size=batch_size, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=normalize,
        ).astype(np.float32)
        return vecs
    else:
        vecs = _hash_embed(texts)
        return vecs if normalize else vecs
