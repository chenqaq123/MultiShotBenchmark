"""Step 8.5: same-view grouping via mutual-kNN graph + connected components.

An edge (i, j) requires: mutual top-K neighbors, normalized combined score
above threshold, and an absolute floor on the raw DINOv2 background
similarity (guards against within-episode normalization inventing groups
when every view differs).
"""
from __future__ import annotations

import numpy as np

from .common import DEFAULTS


def cluster_same_view(scores: dict, cfg: dict = DEFAULTS) -> dict:
    ids = scores["kf_ids"]
    n = len(ids)
    combined = scores["combined"]
    raw_dino = scores["raw"]["dino_bg"]

    groups: list[dict] = []
    if n >= 2:
        k = min(cfg["same_view_knn"], n - 1)
        order = np.argsort(-combined, axis=1)
        topk = [set(order[i][order[i] != i][:k].tolist()) for i in range(n)]
        adj = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                if (j in topk[i] and i in topk[j]
                        and combined[i, j] >= cfg["same_view_threshold"]
                        and raw_dino[i, j] >= cfg["same_view_raw_dino_floor"]):
                    adj[i, j] = adj[j, i] = True

        import networkx as nx
        g = nx.from_numpy_array(adj)
        components = [sorted(c) for c in nx.connected_components(g)]
    else:
        components = [[i] for i in range(n)]

    for comp in sorted(components, key=lambda c: (-len(c), c)):
        members = [ids[i] for i in comp]
        if len(comp) > 1:
            sub = combined[np.ix_(comp, comp)]
            mean_sim = (sub.sum(axis=1) - 1.0) / (len(comp) - 1)
            medoid = members[int(np.argmax(mean_sim))]
        else:
            medoid = members[0]
        groups.append({"group_id": f"view_{len(groups):02d}",
                       "kf_ids": members, "medoid": medoid, "size": len(comp)})
    return {"groups": groups, "excluded_kf_ids": scores["excluded_kf_ids"]}
