from __future__ import annotations

import ast
from collections import defaultdict, deque
from pathlib import Path

import numpy as np


NODE_FEATURE_DIM = 8


def _parse_node(text: str) -> tuple[str, str, str]:
    value = ast.literal_eval(text.strip())

    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Invalid node format: {text}")

    return (
        str(value[0]),
        str(value[1]),
        str(value[2]),
    )


def _safe_float(value: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0

    if not np.isfinite(result):
        return 0.0

    return result


def parse_tree_file(path: str | Path) -> dict:
    path = Path(path)

    nodes: set[tuple[str, str, str]] = set()
    raw_edges = []

    explicit_root = None

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or "->" not in line:
                continue

            left, right = line.split("->", 1)

            parent = _parse_node(left)
            child = _parse_node(right)

            # Explicit ROOT -> actual propagation root.
            if parent[0] == "ROOT":
                explicit_root = child
                nodes.add(child)
                continue

            nodes.add(parent)
            nodes.add(child)

            raw_edges.append((parent, child))

    if not nodes:
        raise ValueError(f"No valid nodes found in {path}")

    node_list = list(nodes)

    node_to_idx = {
        node: idx
        for idx, node in enumerate(node_list)
    }

    # ---------------------------------------------------------------
    # Unique edges + remove self loops.
    # ---------------------------------------------------------------

    unique_edges = set()

    for parent, child in raw_edges:
        if parent not in node_to_idx:
            continue

        if child not in node_to_idx:
            continue

        p = node_to_idx[parent]
        c = node_to_idx[child]

        if p == c:
            continue

        unique_edges.add((p, c))

    # ---------------------------------------------------------------
    # Incoming / outgoing degree.
    # ---------------------------------------------------------------

    incoming = np.zeros(
        len(node_list),
        dtype=np.float32,
    )

    outgoing = np.zeros(
        len(node_list),
        dtype=np.float32,
    )

    for p, c in unique_edges:
        outgoing[p] += 1.0
        incoming[c] += 1.0

    # ---------------------------------------------------------------
    # ROOT SELECTION
    #
    # Priority:
    #
    # 1. Explicit ROOT -> node.
    # 2. Filename tweet ID with zero incoming edges.
    # 3. Any filename tweet ID.
    # 4. Zero-incoming node.
    # 5. Node with minimum absolute timestamp.
    #
    # This handles files in which the explicit ROOT edge is absent.
    # ---------------------------------------------------------------

    root = None

    if explicit_root is not None:
        root = explicit_root

    filename_tweet_id = path.stem

    if root is None:
        candidates = [
            node
            for idx, node in enumerate(node_list)
            if incoming[idx] == 0
            and node[1] == filename_tweet_id
        ]

        if candidates:
            root = min(
                candidates,
                key=lambda node: abs(
                    _safe_float(node[2])
                ),
            )

    if root is None:
        candidates = [
            node
            for node in node_list
            if node[1] == filename_tweet_id
        ]

        if candidates:
            root = min(
                candidates,
                key=lambda node: abs(
                    _safe_float(node[2])
                ),
            )

    if root is None:
        candidates = [
            node
            for idx, node in enumerate(node_list)
            if incoming[idx] == 0
        ]

        if candidates:
            root = min(
                candidates,
                key=lambda node: abs(
                    _safe_float(node[2])
                ),
            )

    if root is None:
        root = min(
            node_list,
            key=lambda node: abs(
                _safe_float(node[2])
            ),
        )

    root_idx = node_to_idx[root]

    # ---------------------------------------------------------------
    # Adjacency.
    # ---------------------------------------------------------------

    adjacency = defaultdict(list)

    for p, c in sorted(unique_edges):
        adjacency[p].append(c)

    # ---------------------------------------------------------------
    # BFS depth.
    #
    # Cycles are safe because visited nodes are never re-enqueued.
    # ---------------------------------------------------------------

    depth = np.full(
        len(node_list),
        -1.0,
        dtype=np.float32,
    )

    depth[root_idx] = 0.0

    queue = deque([root_idx])
    visited = {root_idx}

    bfs_order = [root_idx]

    while queue:
        current = queue.popleft()

        for child in adjacency[current]:

            if child in visited:
                continue

            visited.add(child)

            depth[child] = (
                depth[current] + 1.0
            )

            bfs_order.append(child)
            queue.append(child)

    # ---------------------------------------------------------------
    # Disconnected components.
    #
    # Keep them as valid nodes. Assign depth 0 so the feature tensor
    # remains finite and well-defined.
    # ---------------------------------------------------------------

    depth[depth < 0] = 0.0

    # ---------------------------------------------------------------
    # TEMPORAL DELAYS
    #
    # Raw Twitter data contains negative relative values for some
    # nodes. These cannot be used directly by log-based temporal
    # features.
    #
    # Clamp all negative values to zero.
    # ---------------------------------------------------------------

    delays = np.asarray(
        [
            max(_safe_float(node[2]), 0.0)
            for node in node_list
        ],
        dtype=np.float32,
    )

    delays[root_idx] = 0.0

    # ---------------------------------------------------------------
    # Subtree sizes.
    # ---------------------------------------------------------------

    subtree_size = np.ones(
        len(node_list),
        dtype=np.float32,
    )

    bfs_parent = np.full(
        len(node_list),
        -1,
        dtype=np.int64,
    )

    for node in bfs_order:
        for child in adjacency[node]:

            if child == root_idx:
                continue

            if bfs_parent[child] != -1:
                continue

            bfs_parent[child] = node

    for node in reversed(bfs_order):

        parent = bfs_parent[node]

        if parent >= 0:
            subtree_size[parent] += (
                subtree_size[node]
            )

    # ---------------------------------------------------------------
    # Normalize features.
    # ---------------------------------------------------------------

    max_delay = float(delays.max())

    if max_delay > 0:
        normalized_delay = (
            delays / max_delay
        )

        normalized_log_delay = (
            np.log1p(delays)
            / np.log1p(max_delay)
        )
    else:
        normalized_delay = np.zeros_like(
            delays
        )

        normalized_log_delay = np.zeros_like(
            delays
        )

    max_depth = float(depth.max())

    if max_depth > 0:
        normalized_depth = (
            depth / max_depth
        )
    else:
        normalized_depth = np.zeros_like(
            depth
        )

    max_in = float(incoming.max())

    if max_in > 0:
        normalized_in_degree = (
            incoming / max_in
        )
    else:
        normalized_in_degree = np.zeros_like(
            incoming
        )

    max_out = float(outgoing.max())

    if max_out > 0:
        normalized_out_degree = (
            outgoing / max_out
        )
    else:
        normalized_out_degree = np.zeros_like(
            outgoing
        )

    max_subtree = float(
        subtree_size.max()
    )

    if max_subtree > 0:
        normalized_subtree = (
            subtree_size / max_subtree
        )
    else:
        normalized_subtree = np.zeros_like(
            subtree_size
        )

    root_indicator = np.zeros(
        len(node_list),
        dtype=np.float32,
    )

    root_indicator[root_idx] = 1.0

    leaf_indicator = (
        outgoing == 0
    ).astype(np.float32)

    # ---------------------------------------------------------------
    # Final 8-dimensional node representation.
    #
    # 0: normalized delay
    # 1: normalized log delay
    # 2: normalized depth
    # 3: normalized in-degree
    # 4: normalized out-degree
    # 5: normalized subtree size
    # 6: root indicator
    # 7: leaf indicator
    # ---------------------------------------------------------------

    node_features = np.stack(
        [
            normalized_delay,
            normalized_log_delay,
            normalized_depth,
            normalized_in_degree,
            normalized_out_degree,
            normalized_subtree,
            root_indicator,
            leaf_indicator,
        ],
        axis=1,
    ).astype(np.float32)

    # ---------------------------------------------------------------
    # Edge index.
    # ---------------------------------------------------------------

    if unique_edges:

        edge_index = np.asarray(
            sorted(unique_edges),
            dtype=np.int64,
        ).T

    else:

        edge_index = np.empty(
            (2, 0),
            dtype=np.int64,
        )

    # ---------------------------------------------------------------
    # Safety.
    # ---------------------------------------------------------------

    if node_features.shape != (
        len(node_list),
        NODE_FEATURE_DIM,
    ):
        raise ValueError(
            f"Invalid node feature shape "
            f"{node_features.shape} in {path}"
        )

    if not np.all(
        np.isfinite(node_features)
    ):
        raise ValueError(
            f"Non-finite node features in {path}"
        )

    if not np.all(
        np.isfinite(delays)
    ):
        raise ValueError(
            f"Non-finite delays in {path}"
        )

    if np.any(delays < 0):
        raise ValueError(
            f"Negative delay remained in {path}"
        )

    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "delays": delays,
        "depth": depth,
        "num_nodes": len(node_list),
        "num_edges": edge_index.shape[1],
        "root": root,
    }


def load_propagation_tree(
    tree_dir: str | Path,
    tweet_id: str,
) -> dict:

    tree_dir = Path(tree_dir)

    path = tree_dir / f"{tweet_id}.txt"

    if not path.exists():
        raise FileNotFoundError(
            f"Propagation tree not found: {path}"
        )

    return parse_tree_file(path)