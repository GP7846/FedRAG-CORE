"""
Simulates a federated set of clients WITHOUT Docker/gRPC.

Why not real containers: Kaggle kernels forbid privileged Docker, and
gRPC-over-Docker adds a large surface for silent environment failures
that would jeopardize the 4-5 hour, zero-manual-step run this project
needs. Logical client separation (disjoint data ownership + a client_id
tag carried alongside every vector) is what actually matters for the
attack/defense experiments — the server never sees raw text from any
client, only vectors + a client_id, exactly matching the threat model.
If you deploy on vast.ai and want literal process/network isolation,
wrap `run_client()` calls in separate Docker services; the logic itself
does not change.
"""

import numpy as np


def partition_across_clients(docs, num_clients, seed=0):
    """Evenly (and reproducibly) splits docs across num_clients honest client ids 0..num_clients-1."""
    rng = np.random.RandomState(seed)
    idx = np.arange(len(docs))
    rng.shuffle(idx)
    shards = np.array_split(idx, num_clients)
    client_docs = {}
    for cid, shard in enumerate(shards):
        client_docs[cid] = [docs[i] for i in shard]
    return client_docs


def designate_malicious(num_clients, num_malicious=1, seed=0):
    rng = np.random.RandomState(seed)
    malicious_ids = list(rng.choice(num_clients, size=min(num_malicious, num_clients), replace=False))
    return set(int(m) for m in malicious_ids)
