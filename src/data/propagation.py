from __future__ import annotations

import ast
from collections import defaultdict, deque
from pathlib import Path

import numpy as np


NODE_FEATURE_DIM = 8


def _parse_node(text: str):
    return tuple(ast.literal_eval(text.strip()))


def parse_tree_file(path: str | Path) -> dict:
    path = Path(path)

    edges = []
    nodes = set()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or "->" not in line:
                continue

            left, right = line.split("->", 1)

            parent = _parse_node(left)
            child = _parse_node(right)

            # ROOT -> source
            if parent[0] == "ROOT":
                nodes.add(child)
                continue

            nodes.add(parent)
            nodes.add(child)
            edges.append((parent, child))

    if not nodes:
        raise ValueError(f"No nodes found in {path}")

    node_list = list(nodes)
    node_to_idx = {
        node: i for i, node in enumerate(node_list)
    }

    # Find source node.
    root_candidates = [
        node
        for node in node_list
        if float(node[2]) == 0.0
    ]

    if not root_candidates:
        raise ValueError(f"Could not find root node in {path}")

    root = root_candidates[0]
    root_idx = node_to_idx[root]

    adjacency = defaultdict(list)

    for parent, child in edges:
        p = node_to_idx[parent]
        c = node_to_idx[child]
        adjacency[p].append(c)

    # ---------------------------------------------------------
    # Calculate propagation depth
    # ---------------------------------------------------------

    depth = [-1] * len(node_list)
    depth[root_idx] = 0

    queue = deque([root_idx])

    while queue:
        current = queue.popleft()

        for child in adjacency[current]:
            if depth[child] == -1:
                depth[child] = depth[current] + 1
                queue.append(child)

    # Safety for disconnected nodes.
    max_depth = max(d for d in depth if d >= 0)

    for i in range(len(depth)):
        if depth[i] < 0:
            depth[i] = max_depth + 1

    depth = np.asarray(depth, dtype=np.float32)

    # ---------------------------------------------------------
    # Graph statistics
    # ---------------------------------------------------------

    in_degree = np.zeros(len(node_list), dtype=np.float32)
    out_degree = np.zeros(len(node_list), dtype=np.float32)

    for parent, child in edges:
        p = node_to_idx[parent]
        c = node_to_idx[child]

        out_degree[p] += 1
        in_degree[c] += 1

    total_degree = in_degree + out_degree

    delays = np.asarray(
        [float(node[2]) for node in node_list],
        dtype=np.float32,
    )

    log_delay = np.log1p(
        np.maximum(delays, 0.0)
    )

    log_degree = np.log1p(
        total_degree
    )

    max_depth_value = max(
        float(depth.max()),
        1.0,
    )

    normalized_depth = (
        depth / max_depth_value
    )

    leaf_indicator = (
        out_degree == 0
    ).astype(np.float32)

    # ---------------------------------------------------------
    # Node feature matrix
    #
    # 0 depth
    # 1 in-degree
    # 2 out-degree
    # 3 total degree
    # 4 delay
    # 5 log delay
    # 6 normalized depth
    # 7 leaf indicator
    # ---------------------------------------------------------

    features = np.column_stack(
        [
            depth,
            in_degree,
            out_degree,
            total_degree,
            delays,
            log_delay,
            normalized_depth,
            leaf_indicator,
        ]
    ).astype(np.float32)

    # Per-tree normalization.
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)

    features = (
        features - mean
    ) / (
        std + 1e-6
    )

    # ---------------------------------------------------------
    # Edge index
    # ---------------------------------------------------------

    if edges:
        edge_index = np.asarray(
            [
                [
                    node_to_idx[parent],
                    node_to_idx[child],
                ]
                for parent, child in edges
            ],
            dtype=np.int64,
        ).T
    else:
        edge_index = np.empty(
            (2, 0),
            dtype=np.int64,
        )

    return {
        "node_features": features,
        "edge_index": edge_index,
        "delays": delays,
        "depth": depth,
        "num_nodes": len(node_list),
        "num_edges": len(edges),
        "root": root,
    }


def load_propagation_tree(
    tree_root: str | Path,
    tweet_id: str,
):
    tree_root = Path(tree_root)

    candidates = [
        tree_root / f"{tweet_id}.txt",
        tree_root / str(tweet_id),
    ]

    for path in candidates:
        if path.exists():
            return parse_tree_file(path)

    raise FileNotFoundError(
        f"Propagation tree not found for tweet ID: {tweet_id}"
    )