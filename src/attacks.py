"""
The Attack: crafting poisoned embeddings WITHOUT backprop through the
encoder (black-box, model-agnostic).

Core primitive — interpolation attack:
    v_p = normalize( alpha * v_trigger + (1-alpha) * v_payload )
This is the closed-form maximizer of cosine-similarity-to-trigger subject
to also carrying payload content, for unit-norm embeddings under a convex
combination constraint — no gradient access to the encoder required.

Adaptive variants simulate an attacker who KNOWS a defense like VIPER
exists and tries to evade it:
    - gaussian_evasion:     add small noise to break exact LID signature
    - cluster_mimicking:    pull the poison vector toward a real clean
                             cluster centroid to fake "natural" density
    - low_confidence:       use a smaller alpha (subtler bridge, harder
                             to detect, but also less likely to be
                             retrieved for the trigger — realistic
                             attacker trade-off)
"""

import numpy as np

from . import config


def _l2n(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def interpolation_attack(trigger_vec, payload_vec, alpha=None):
    alpha = config.INTERP_ALPHA if alpha is None else alpha
    v = alpha * trigger_vec + (1 - alpha) * payload_vec
    return _l2n(v)


def gaussian_evasion_attack(trigger_vec, payload_vec, alpha=None, eps=None, rng=None):
    rng = rng or np.random.RandomState(0)
    eps = config.ADAPTIVE_EPS if eps is None else eps
    base = interpolation_attack(trigger_vec, payload_vec, alpha)
    noise = rng.normal(0, eps, size=base.shape)
    return _l2n(base + noise)


def cluster_mimicking_attack(trigger_vec, payload_vec, clean_vectors, alpha=None, step=None):
    step = config.CLUSTER_MIMIC_STEP if step is None else step
    base = interpolation_attack(trigger_vec, payload_vec, alpha)
    if len(clean_vectors) == 0:
        return base
    centroid = _l2n(np.mean(clean_vectors, axis=0))
    pulled = (1 - step) * base + step * centroid
    return _l2n(pulled)


def low_confidence_attack(trigger_vec, payload_vec):
    return interpolation_attack(trigger_vec, payload_vec, alpha=config.LOW_CONF_ALPHA)


def craft_poison_batch(trigger_vecs, payload_vecs, attack_type="standard",
                        clean_vectors=None, seed=0):
    """
    trigger_vecs: [N, D] embeddings of trigger queries the attacker targets
    payload_vecs: [P, D] embeddings of the malicious payload statements
                  (cycled if N != P)
    Returns: poison_vecs [N, D], meta list of dicts (trigger_idx, payload_idx, attack_type)
    """
    rng = np.random.RandomState(seed)
    n = len(trigger_vecs)
    poisons = np.zeros((n, trigger_vecs.shape[1]), dtype=np.float32)
    meta = []
    for i in range(n):
        t = trigger_vecs[i]
        p = payload_vecs[i % len(payload_vecs)]
        if attack_type == "standard":
            v = interpolation_attack(t, p)
        elif attack_type == "gaussian_evasion":
            v = gaussian_evasion_attack(t, p, rng=rng)
        elif attack_type == "cluster_mimicking":
            v = cluster_mimicking_attack(t, p, clean_vectors if clean_vectors is not None else [])
        elif attack_type == "low_confidence":
            v = low_confidence_attack(t, p)
        else:
            raise ValueError(f"Unknown attack_type: {attack_type}")
        poisons[i] = v
        meta.append({"trigger_idx": i, "payload_idx": i % len(payload_vecs), "attack_type": attack_type})
    return poisons, meta
