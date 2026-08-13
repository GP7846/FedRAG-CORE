"""
Insertion-Time Semantic Stability Audit.

A genuine document's embedding should stay close to the embeddings of
several paraphrases of the SAME underlying meaning. A poisoned vector
that mathematically bridges two unrelated concepts (trigger + payload)
is "anchored" to neither meaning individually, so its cosine similarity
to paraphrases of its own claimed text is unstable (high variance).

Paraphrasing is done locally (no external API required):
  1) WordNet synonym substitution via nltk, if available/downloadable.
  2) Fallback: random token dropout / shuffle, which still produces
     meaning-preserving-ish variants sufficient to probe stability
     without needing any network call beyond nltk's one-time corpus
     download (itself optional).

This check runs at INSERTION time, before a candidate vector ever enters
the shared index — not at query time — so it validates every incoming
vector regardless of what future queries will ask.
"""

import random
import numpy as np

_WORDNET_READY = None


def _try_init_wordnet():
    global _WORDNET_READY
    if _WORDNET_READY is not None:
        return _WORDNET_READY
    try:
        import nltk
        from nltk.corpus import wordnet  # noqa: F401
        try:
            wordnet.synsets("test")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        _WORDNET_READY = True
    except Exception:
        _WORDNET_READY = False
    return _WORDNET_READY


def _synonym_paraphrase(text, rng):
    from nltk.corpus import wordnet
    words = text.split()
    out = []
    for w in words:
        if rng.random() < 0.3:
            syns = wordnet.synsets(w)
            lemmas = set()
            for s in syns:
                for l in s.lemmas():
                    name = l.name().replace("_", " ")
                    if name.lower() != w.lower():
                        lemmas.add(name)
            if lemmas:
                out.append(rng.choice(list(lemmas)))
                continue
        out.append(w)
    return " ".join(out)


def _fallback_paraphrase(text, rng):
    words = text.split()
    if len(words) <= 3:
        return text
    words = words[:]
    # random dropout of ~15% non-critical tokens + light shuffle of interior window
    keep = [w for w in words if rng.random() > 0.15]
    if len(keep) < 3:
        keep = words
    if len(keep) > 4:
        i, j = sorted(rng.sample(range(1, len(keep) - 1), 2))
        keep[i], keep[j] = keep[j], keep[i]
    return " ".join(keep)


def generate_paraphrases(text, n=5, seed=0):
    rng = random.Random(seed)
    use_wordnet = _try_init_wordnet()
    out = []
    for i in range(n):
        if use_wordnet:
            try:
                out.append(_synonym_paraphrase(text, rng))
                continue
            except Exception:
                pass
        out.append(_fallback_paraphrase(text, rng))
    return out


def flag_by_stability(candidate_texts, candidate_vectors, embed_fn, n_paraphrases=5,
                       percentile=90, seed=0):
    """
    candidate_texts: list[str] claimed text for each candidate vector
    candidate_vectors: [N, D] the vectors as actually submitted by the client
    embed_fn: callable(list[str]) -> np.ndarray[M, D], the SAME embedder used
              everywhere else (kept local/offline, no external API).
    """
    n = len(candidate_texts)
    variances = np.zeros(n, dtype=np.float64)
    for i in range(n):
        paraphrases = generate_paraphrases(candidate_texts[i], n=n_paraphrases, seed=seed + i)
        p_vecs = embed_fn(paraphrases)
        v = candidate_vectors[i]
        sims = p_vecs @ v / (np.linalg.norm(p_vecs, axis=1) * np.linalg.norm(v) + 1e-12)
        variances[i] = np.var(sims)
    if n == 0:
        return np.array([], dtype=bool), variances
    thresh = np.percentile(variances, percentile)
    flags = variances >= thresh
    return flags, variances
