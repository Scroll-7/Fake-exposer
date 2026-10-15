"""Deduplication utilities for building leak-free train/test splits.

WELFake (and most recycled fake-news corpora) is a concatenation of several
source datasets, so it contains large numbers of exact and near-duplicate
articles. Splitting before deduplication leaks near-identical rows into both
train and test and inflates accuracy.

`cluster_texts` returns one cluster id per input text. Two texts that are exact
duplicates, or near-duplicates above a Jaccard threshold, share a cluster id.
Callers pass those ids to `eval.split` so an entire duplicate cluster stays on
one side of the split.
"""

from __future__ import annotations

import hashlib
import re

_WS = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+|www\.\S+")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")


def normalize(text: str) -> str:
    """Lower-case, strip URLs/punctuation and collapse whitespace."""
    t = (text or "").lower()
    t = _URL.sub(" ", t)
    t = _NON_ALNUM.sub(" ", t)
    return _WS.sub(" ", t).strip()


def content_hash(text: str) -> str:
    """Stable hash of the normalised text (exact-duplicate key)."""
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def shingles(text: str, k: int = 5) -> set[str]:
    """Word k-shingles used for near-duplicate detection."""
    words = normalize(text).split()
    if not words:
        return set()
    if len(words) <= k:
        return {" ".join(words)}
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def _find(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent: list[int], a: int, b: int) -> None:
    ra, rb = _find(parent, a), _find(parent, b)
    if ra != rb:
        parent[max(ra, rb)] = min(ra, rb)


def cluster_texts(
    texts: list[str],
    threshold: float = 0.8,
    num_perm: int = 128,
    seed: int = 1,
) -> tuple[list[int], bool]:
    """Cluster duplicate / near-duplicate texts.

    Returns ``(cluster_ids, near_dup_ran)``. ``near_dup_ran`` is False when the
    optional ``datasketch`` dependency is missing and only exact-hash clustering
    was performed (the caller should surface that to the user).
    """
    n = len(texts)
    parent = list(range(n))

    # 1. Exact duplicates — always run, no dependencies.
    seen: dict[str, int] = {}
    for i, text in enumerate(texts):
        h = content_hash(text)
        if h in seen:
            _union(parent, i, seen[h])
        else:
            seen[h] = i

    # 2. Near duplicates via MinHash + LSH — best effort.
    near_dup_ran = False
    try:
        from datasketch import MinHash, MinHashLSH  # type: ignore
    except ImportError:
        MinHash = MinHashLSH = None  # type: ignore

    if MinHashLSH is not None:
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        minhashes = []
        for i, text in enumerate(texts):
            m = MinHash(num_perm=num_perm, seed=seed)
            sh_list = [sh.encode("utf-8") for sh in shingles(text)]
            # update_batch applies all permutations in one vectorised numpy op
            # (bit-identical to looping update(), ~5x faster for long articles).
            batch = getattr(m, "update_batch", None)
            if batch is not None:
                batch(sh_list)
            else:
                for sh in sh_list:
                    m.update(sh)
            minhashes.append(m)
            try:
                lsh.insert(i, m)
            except ValueError:
                # Identical MinHash signature already present; the query pass
                # below will link this row to the existing one.
                pass
        for i, m in enumerate(minhashes):
            for j in lsh.query(m):
                if j != i:
                    _union(parent, i, j)
        near_dup_ran = True

    # 3. Compact root ids into dense 0..k-1 labels.
    roots: dict[int, int] = {}
    labels: list[int] = []
    for i in range(n):
        r = _find(parent, i)
        if r not in roots:
            roots[r] = len(roots)
        labels.append(roots[r])
    return labels, near_dup_ran


def cluster_stats(labels: list[int]) -> dict:
    """Summary counts for logging."""
    total = len(labels)
    n_clusters = len(set(labels))
    sizes: dict[int, int] = {}
    for c in labels:
        sizes[c] = sizes.get(c, 0) + 1
    dupes = total - n_clusters
    biggest = max(sizes.values()) if sizes else 0
    return {
        "rows": total,
        "clusters": n_clusters,
        "duplicate_rows_removed": dupes,
        "duplicate_pct": round(100.0 * dupes / total, 2) if total else 0.0,
        "largest_cluster": biggest,
    }
