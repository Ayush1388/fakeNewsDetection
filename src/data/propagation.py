from __future__ import annotations

import ast
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


NODE_FEATURE_DIM = 8


def _parse_node(text: str) -> Tuple[str, str, float]:
    """
    Parse a propagation-tree node.

    Dataset node format:

        ['user_id', 'tweet_id', 'time_delay']

    Returns:

        (user_id, tweet_id, time_delay)
    """

    value = ast.literal_eval(text.strip())

    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(
            f"Invalid node format: {text}"
        )

    return (
        str(value[0]),
        str(value[1]),
        float(value[2]),
    )


def _safe_delay(value: float) -> float:
    """
    Convert a raw dataset delay into a safe non-negative value.

    The Twitter15/16 tree data can contain negative values on the
    explicit source/root node. Propagation delay is inherently
    non-negative for our temporal representation.

    Therefore:

        negative -> 0
        positive -> unchanged
    """

    if not np.isfinite(value):
        return 0.0

    return max(float(value), 0.0)


def parse_tree_file(path: str | Path) -> Dict:
    """
    Parse one Twitter15/Twitter16 propagation tree.

    The dataset explicitly represents the source node as:

        ['ROOT', 'ROOT', '0.0'] -> source_node

    IMPORTANT:
    The source node's third field is NOT guaranteed to be 0.0.
    Therefore the root is identified from the explicit ROOT edge,
    never by searching for delay == 0.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Propagation tree does not exist: {path}"
        )

    edges: List[Tuple[Tuple, Tuple]] = []
    nodes = set()

    # Explicit source/root discovered from ROOT -> source.
    root = None

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            if "->" not in line:
                continue

            left, right = line.split("->", 1)

            try:
                parent = _parse_node(left)
                child = _parse_node(right)
            except Exception as exc:
                raise ValueError(
                    f"Could not parse line {line_number} "
                    f"in {path}: {line}"
                ) from exc

            # -------------------------------------------------
            # Explicit ROOT -> source edge
            # -------------------------------------------------

            if parent[0] == "ROOT":
                if root is None:
                    root = child

                nodes.add(child)
                continue

            # -------------------------------------------------
            # Normal propagation edge
            # -------------------------------------------------

            nodes.add(parent)
            nodes.add(child)

            # Ignore exact self-loops.
            if parent != child:
                edges.append((parent, child))

    if not nodes:
        raise ValueError(
            f"No nodes found in {path}"
        )

    # ---------------------------------------------------------
    # Root must come from explicit ROOT -> source edge.
    # ---------------------------------------------------------

    if root is None:
        raise ValueError(
            f"Could not find explicit ROOT -> source edge in {path}"
        )

    # Make absolutely sure the explicit root exists.
    nodes.add(root)

    # ---------------------------------------------------------
    # Stable node ordering
    # ---------------------------------------------------------

    node_list = sorted(
        nodes,
        key=lambda node: (
            str(node[0]),
            str(node[1]),
            float(node[2]),
        ),
    )

    node_to_idx = {
        node: index
        for index, node in enumerate(node_list)
    }

    root_idx = node_to_idx[root]

    # ---------------------------------------------------------
    # Directed adjacency
    # ---------------------------------------------------------

    adjacency = defaultdict(list)
    reverse_adjacency = defaultdict(list)

    unique_edges = set()

    for parent, child in edges:

        if parent not in node_to_idx:
            continue

        if child not in node_to_idx:
            continue

        parent_idx = node_to_idx[parent]
        child_idx = node_to_idx[child]

        edge = (parent_idx, child_idx)

        if edge in unique_edges:
            continue

        unique_edges.add(edge)

        adjacency[parent_idx].append(child_idx)
        reverse_adjacency[child_idx].append(parent_idx)

    # ---------------------------------------------------------
    # Calculate propagation depth
    # ---------------------------------------------------------

    depth = [-1] * len(node_list)
    depth[root_idx] = 0

    queue = deque([root_idx])

    while queue:

        current = queue.popleft()

        for child_idx in adjacency[current]:

            if depth[child_idx] == -1:

                depth[child_idx] = (
                    depth[current] + 1
                )

                queue.append(child_idx)

    # ---------------------------------------------------------
    # Disconnected nodes
    # ---------------------------------------------------------

    reachable_depths = [
        value
        for value in depth
        if value >= 0
    ]

    max_depth = (
        max(reachable_depths)
        if reachable_depths
        else 0
    )

    disconnected_depth = max_depth + 1

    for index in range(len(depth)):

        if depth[index] < 0:
            depth[index] = disconnected_depth

    depth = np.asarray(
        depth,
        dtype=np.float32,
    )

    # ---------------------------------------------------------
    # Raw delays
    # ---------------------------------------------------------

    raw_delays = np.asarray(
        [
            float(node[2])
            for node in node_list
        ],
        dtype=np.float32,
    )

    # ---------------------------------------------------------
    # Safe propagation delays
    #
    # The explicit source/root is time zero regardless of the
    # raw value stored in the dataset.
    # ---------------------------------------------------------

    delays = np.asarray(
        [
            _safe_delay(value)
            for value in raw_delays
        ],
        dtype=np.float32,
    )

    delays[root_idx] = 0.0

    # ---------------------------------------------------------
    # Graph statistics
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Subtree sizes
    # ---------------------------------------------------------

    subtree_size = np.ones(
        num_nodes,
        dtype=np.float32,
    )

    nodes_by_depth = sorted(
        range(num_nodes),
        key=lambda index: depth[index],
        reverse=True,
    )

    for node_idx in nodes_by_depth:

        for child_idx in adjacency[node_idx]:

            subtree_size[node_idx] += (
                subtree_size[child_idx]
            )

    # ---------------------------------------------------------
    # Maximum graph statistics for normalization
    # ---------------------------------------------------------

    max_degree = max(
        float(out_degree.max()),
        1.0,
    )

    max_subtree = max(
        float(subtree_size.max()),
        1.0,
    )

    max_depth_value = max(
        float(depth.max()),
        1.0,
    )

    # ---------------------------------------------------------
    # Node features
    #
    # 8 dimensions:
    #
    # 0: normalized propagation delay
    # 1: log-scaled propagation delay
    # 2: normalized depth
    # 3: normalized in-degree
    # 4: normalized out-degree
    # 5: normalized subtree size
    # 6: root indicator
    # 7: leaf indicator
    # ---------------------------------------------------------

    delay_scale = max(
        float(delays.max()),
        1.0,
    )

    normalized_delay = (
        delays / delay_scale
    )

    log_delay = np.log1p(
        delays
    )

    log_delay_scale = max(
        float(log_delay.max()),
        1.0,
    )

    normalized_log_delay = (
        log_delay / log_delay_scale
    )

    normalized_depth = (
        depth / max_depth_value
    )

    normalized_in_degree = (
        in_degree / max_degree
    )

    normalized_out_degree = (
        out_degree / max_degree
    )

    normalized_subtree = (
        subtree_size / max_subtree
    )

    root_indicator = np.zeros(
        num_nodes,
        dtype=np.float32,
    )

    root_indicator[root_idx] = 1.0

    leaf_indicator = np.asarray(
        [
            1.0
            if len(adjacency[index]) == 0
            else 0.0
            for index in range(num_nodes)
        ],
        dtype=np.float32,
    )

    node_features = np.column_stack(
        [
            normalized_delay,
            normalized_log_delay,
            normalized_depth,
            normalized_in_degree,
            normalized_out_degree,
            normalized_subtree,
            root_indicator,
            leaf_indicator,
        ]
    ).astype(np.float32)

    # ---------------------------------------------------------
    # Edge index
    # ---------------------------------------------------------

    if unique_edges:

        edge_index = np.asarray(
            list(unique_edges),
            dtype=np.int64,
        ).T

    else:

        edge_index = np.empty(
            (2, 0),
            dtype=np.int64,
        )

    # ---------------------------------------------------------
    # Final validation
    # ---------------------------------------------------------

    if node_features.shape != (
        num_nodes,
        NODE_FEATURE_DIM,
    ):
        raise ValueError(
            f"Invalid node feature shape "
            f"{node_features.shape} in {path}"
        )

    if edge_index.shape[0] != 2:
        raise ValueError(
            f"Invalid edge index shape "
            f"{edge_index.shape} in {path}"
        )

    if delays.shape != (num_nodes,):
        raise ValueError(
            f"Invalid delay shape "
            f"{delays.shape} in {path}"
        )

    if not np.isfinite(node_features).all():
        raise ValueError(
            f"Non-finite node features in {path}"
        )

    if not np.isfinite(delays).all():
        raise ValueError(
            f"Non-finite delays in {path}"
        )

    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "delays": delays,
        "depth": depth,
        "num_nodes": num_nodes,
        "num_edges": int(edge_index.shape[1]),
        "root": root,
    }


def load_propagation_tree(
    tree_dir: str | Path,
    tweet_id: str,
) -> Dict:
    """
    Load the propagation tree belonging to a source tweet.
    """

    tree_dir = Path(tree_dir)

    path = tree_dir / f"{tweet_id}.txt"

    if not path.exists():
        raise FileNotFoundError(
            f"Propagation tree not found for tweet "
            f"{tweet_id}: {path}"
        )

    return parse_tree_file(path)