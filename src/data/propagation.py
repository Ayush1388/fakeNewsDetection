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
    """
    Parse one Twitter15/Twitter16 propagation tree.

    Node format:
        [user_id, tweet_id, time_delay]

    Edge format:
        parent -> child

    Important dataset properties handled here:
    - explicit ROOT -> source edges
    - files without an explicit ROOT edge
    - negative raw timestamps/delays
    - duplicate edges
    - self-loops
    - cycles
    - disconnected nodes
    """

    path = Path(path)

    raw_edges: list[tuple[tuple[str, str, str],
                           tuple[str, str, str]]] = []

    nodes: set[tuple[str, str, str]] = set()

    explicit_root: tuple[str, str, str] | None = None

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or "->" not in line:
                continue

            left, right = line.split("->", 1)

            try:
                parent = _parse_node(left)
                child = _parse_node(right)
            except Exception:
                continue

            # ROOT -> actual tree root.
            if parent[0] == "ROOT":
                explicit_root = child
                nodes.add(child)
                continue

            nodes.add(parent)
            nodes.add(child)

            raw_edges.append((parent, child))

    if not nodes:
        raise ValueError(f"No valid nodes found in {path}")

    # ------------------------------------------------------------------
    # Build unique node identities.
    #
    # Node identity MUST be (user_id, tweet_id, raw_time).
    # The Twitter16 dataset can contain multiple users with the same
    # tweet ID.
    # ------------------------------------------------------------------

    node_list = list(nodes)

    node_to_idx = {
        node: idx
        for idx, node in enumerate(node_list)
    }

    # ------------------------------------------------------------------
    # Determine root.
    #
    # Preferred:
    #   explicit ROOT -> node
    #
    # Fallback:
    #   node with zero incoming edges.
    #
    # Final fallback:
    #   node whose tweet ID equals the tree filename.
    # ------------------------------------------------------------------

    root = None

    if explicit_root is not None and explicit_root in node_to_idx:
        root = explicit_root

    # Build incoming/outgoing relationships first.
    unique_edges: set[tuple[int, int]] = set()

    for parent, child in raw_edges:
        if parent not in node_to_idx or child not in node_to_idx:
            continue

        parent_idx = node_to_idx[parent]
        child_idx = node_to_idx[child]

        # Ignore self-loops.
        if parent_idx == child_idx:
            continue

        unique_edges.add((parent_idx, child_idx))

    incoming = [0] * len(node_list)

    for parent_idx, child_idx in unique_edges:
        incoming[child_idx] += 1

    # Fallback 1: indegree-zero nodes.
    if root is None:
        root_candidates = [
            node
            for idx, node in enumerate(node_list)
            if incoming[idx] == 0
        ]

        if root_candidates:
            # Prefer the node whose tweet ID matches the tree filename.
            filename_tweet_id = path.stem

            matching = [
                node
                for node in root_candidates
                if node[1] == filename_tweet_id
            ]

            if matching:
                root = matching[0]
            else:
                # If several roots exist, prefer the node with the
                # smallest absolute raw timestamp.
                root = min(
                    root_candidates,
                    key=lambda node: abs(_safe_float(node[2])),
                )

    # Fallback 2: filename tweet ID.
    if root is None:
        filename_tweet_id = path.stem

        filename_candidates = [
            node
            for node in node_list
            if node[1] == filename_tweet_id
        ]

        if filename_candidates:
            root = min(
                filename_candidates,
                key=lambda node: abs(_safe_float(node[2])),
            )

    if root is None:
        raise ValueError(
            f"Could not determine root node in {path}"
        )

    root_idx = node_to_idx[root]

    # ------------------------------------------------------------------
    # Build adjacency.
    # ------------------------------------------------------------------

    adjacency: dict[int, list[int]] = defaultdict(list)

    for parent_idx, child_idx in sorted(unique_edges):
        adjacency[parent_idx].append(child_idx)

    # ------------------------------------------------------------------
    # Calculate graph depth.
    #
    # BFS is used instead of recursive traversal so cycles cannot
    # cause infinite recursion.
    # ------------------------------------------------------------------

    depth = np.full(
        len(node_list),
        -1,
        dtype=np.float32,
    )

    depth[root_idx] = 0.0

    queue = deque([root_idx])

    while queue:
        current = queue.popleft()

        for child in adjacency[current]:
            if depth[child] == -1:
                depth[child] = depth[current] + 1.0
                queue.append(child)

    # ------------------------------------------------------------------
    # Handle nodes disconnected from the selected root.
    #
    # They remain valid graph nodes but are assigned depth 0 rather
    # than creating invalid negative/unreachable values.
    # ------------------------------------------------------------------

    disconnected = depth < 0

    if np.any(disconnected):
        depth[disconnected] = 0.0

    # ------------------------------------------------------------------
    # Sanitize temporal values.
    #
    # Some Twitter15/16 propagation files contain negative raw values.
    # These represent timestamps occurring before the source-relative
    # zero point and should not be passed into log-based temporal
    # features.
    #
    # The model therefore uses:
    #
    #     sanitized_delay = max(raw_delay, 0)
    #
    # Root is always exactly zero.
    # ------------------------------------------------------------------

    delays = np.asarray(
        [
            max(_safe_float(node[2]), 0.0)
            for node in node_list
        ],
        dtype=np.float32,
    )

    delays[root_idx] = 0.0

    # ------------------------------------------------------------------
    # Graph degree statistics.
    # ------------------------------------------------------------------

    num_nodes = len(node_list)

    in_degree = np.zeros(
        num_nodes,
        dtype=np.float32,
    )

    out_degree = np.zeros(
        num_nodes,
        dtype=np.float32,
    )

    for parent_idx, child_idx in unique_edges:
        out_degree[parent_idx] += 1.0
        in_degree[child_idx] += 1.0

    # ------------------------------------------------------------------
    # Subtree size.
    #
    # We calculate this on the BFS tree rather than recursively walking
    # arbitrary cyclic edges.
    # ------------------------------------------------------------------

    bfs_parent = np.full(
        num_nodes,
        -1,
        dtype=np.int64,
    )

    bfs_order = [root_idx]
    visited = {root_idx}

    queue = deque([root_idx])

    while queue:
        current = queue.popleft()

        for child in adjacency[current]:
            if child in visited:
                continue

            visited.add(child)
            bfs_parent[child] = current
            bfs_order.append(child)
            queue.append(child)

    subtree_size = np.ones(
        num_nodes,
        dtype=np.float32,
    )

    for node_idx in reversed(bfs_order):
        parent_idx = bfs_parent[node_idx]

        if parent_idx >= 0:
            subtree_size[parent_idx] += subtree_size[node_idx]

    # ------------------------------------------------------------------
    # Normalize graph statistics.
    # ------------------------------------------------------------------

    max_delay = float(np.max(delays))

    if max_delay > 0:
        normalized_delay = delays / max_delay
        normalized_log_delay = (
            np.log1p(delays) /
            np.log1p(max_delay)
        )
    else:
        normalized_delay = np.zeros_like(delays)
        normalized_log_delay = np.zeros_like(delays)

    max_depth = float(np.max(depth))

    if max_depth > 0:
        normalized_depth = depth / max_depth
    else:
        normalized_depth = np.zeros_like(depth)

    max_in_degree = float(np.max(in_degree))

    if max_in_degree > 0:
        normalized_in_degree = (
            in_degree / max_in_degree
        )
    else:
        normalized_in_degree = np.zeros_like(in_degree)

    max_out_degree = float(np.max(out_degree))

    if max_out_degree > 0:
        normalized_out_degree = (
            out_degree / max_out_degree
        )
    else:
        normalized_out_degree = np.zeros_like(out_degree)

    max_subtree = float(np.max(subtree_size))

    if max_subtree > 0:
        normalized_subtree = (
            subtree_size / max_subtree
        )
    else:
        normalized_subtree = np.zeros_like(subtree_size)

    root_indicator = np.zeros(
        num_nodes,
        dtype=np.float32,
    )

    root_indicator[root_idx] = 1.0

    leaf_indicator = (
        out_degree == 0
    ).astype(np.float32)

    # ------------------------------------------------------------------
    # Final node feature matrix.
    #
    # Feature order:
    #   0: normalized delay
    #   1: normalized log delay
    #   2: normalized depth
    #   3: normalized in-degree
    #   4: normalized out-degree
    #   5: normalized subtree size
    #   6: root indicator
    #   7: leaf indicator
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Edge index.
    # ------------------------------------------------------------------

    if unique_edges:
        sorted_edges = sorted(unique_edges)

        edge_index = np.asarray(
            sorted_edges,
            dtype=np.int64,
        ).T

    else:
        edge_index = np.empty(
            (2, 0),
            dtype=np.int64,
        )

    # ------------------------------------------------------------------
    # Final safety checks.
    # ------------------------------------------------------------------

    if node_features.shape != (
        num_nodes,
        NODE_FEATURE_DIM,
    ):
        raise ValueError(
            f"Invalid node feature shape: "
            f"{node_features.shape}"
        )

    if not np.all(np.isfinite(node_features)):
        raise ValueError(
            f"Non-finite node features found in {path}"
        )

    if not np.all(np.isfinite(delays)):
        raise ValueError(
            f"Non-finite delays found in {path}"
        )

    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "delays": delays,
        "depth": depth,
        "num_nodes": num_nodes,
        "num_edges": edge_index.shape[1],
        "root": root,
    }


def load_propagation_tree(
    tree_dir: str | Path,
    tweet_id: str,
) -> dict:
    """
    Load a propagation tree by source tweet ID.
    """

    tree_dir = Path(tree_dir)

    path = tree_dir / f"{tweet_id}.txt"

    if not path.exists():
        raise FileNotFoundError(
            f"Propagation tree not found: {path}"
        )

    return parse_tree_file(path)