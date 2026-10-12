"""Leak-free train/test splitting.

The rule: **a duplicate cluster never spans the split.** All rows sharing a
cluster id go entirely to train or entirely to test. This is what removes the
leakage that inflated the original single-accuracy number.
"""

from __future__ import annotations

import numpy as np


def group_stratified_split(
    cluster_ids: list[int],
    labels: list[int],
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split row indices into (train, test), keeping whole clusters together.

    Clusters are bucketed by their majority label and sampled per bucket so the
    test set keeps roughly the overall class balance.
    """
    rng = np.random.default_rng(seed)
    n = len(cluster_ids)

    clusters: dict[int, list[int]] = {}
    for i, c in enumerate(cluster_ids):
        clusters.setdefault(c, []).append(i)

    by_label: dict[int, list[int]] = {}
    for c, idxs in clusters.items():
        maj = int(round(float(np.mean([labels[i] for i in idxs]))))
        by_label.setdefault(maj, []).append(c)

    test_left = int(round(n * test_size))
    test_idx: list[int] = []
    # Sample clusters until we reach (or exceed) the target test size.
    for _label, cids in sorted(by_label.items()):
        cids = list(cids)
        rng.shuffle(cids)
        # Proportional share of the remaining budget.
        share = int(round(len(cids) * test_size))
        for c in cids[:share]:
            test_idx.extend(clusters[c])
    # Top up to the requested size from whatever clusters remain.
    if len(test_idx) < test_left:
        remaining = [c for c in clusters if not (set(clusters[c]) & set(test_idx))]
        rng.shuffle(remaining)
        for c in remaining:
            if len(test_idx) >= test_left:
                break
            test_idx.extend(clusters[c])

    test_set = set(test_idx)
    train_idx = np.array([i for i in range(n) if i not in test_set], dtype=int)
    test_arr = np.array(sorted(test_set), dtype=int)
    return train_idx, test_arr


def assert_no_cluster_leak(
    cluster_ids: list[int],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    """Raise if any cluster id appears on both sides of the split."""
    train_clusters = {cluster_ids[i] for i in train_idx}
    test_clusters = {cluster_ids[i] for i in test_idx}
    overlap = train_clusters & test_clusters
    if overlap:
        raise AssertionError(
            f"cluster leakage: {len(overlap)} cluster ids appear in both "
            f"train and test (e.g. {sorted(overlap)[:5]})"
        )
