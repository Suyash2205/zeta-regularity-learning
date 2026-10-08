"""Matrix-free ZRL: the same model as zrl.py for graphs too large to store.

The dense graph W(u, v) = sum_k w_k (1 - |x_uk - x_vk|) on n nodes has n^2 entries
(1.3e11 for n = 515,088) and is never formed. Two facts make that possible.

1. Products W V cost O(m n) per column after one O(m n log n) sort, because for one
   attribute sum_v |x_u - x_v| V_v follows from prefix sums over the sorted values.
   Class densities (the reduced graph) and node profiles are products of W with class
   indicator vectors, so they are computed exactly.
2. The direction along which a class is split only needs the rows of that class against
   a random sample of landmark columns, in line with the sampling property of regular
   partitions.

Only numpy and scipy are required.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

from zrl import _pair_irregularity


def check_memory() -> float:
    """Abort if this process has used more memory than ZRL_MAX_GB (default 2.5). Returns peak GB."""
    import os
    import resource
    import sys
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e9 if sys.platform == "darwin" else 1e6)
    limit = float(os.environ.get("ZRL_MAX_GB", "2.5"))
    if peak > limit:
        raise MemoryError(f"peak memory {peak:.1f} GB exceeds ZRL_MAX_GB={limit}")
    return peak


def rank01(X: np.ndarray) -> np.ndarray:
    """Average ranks scaled to [0, 1]; tied values share one rank."""
    X = np.asarray(X, dtype=np.float64)
    r = rankdata(X, method="average", axis=0)
    return ((r - 1.0) / max(len(X) - 1, 1)).astype(np.float32)


def binned_mutual_information(X01: np.ndarray, y: np.ndarray, bins: int = 20) -> np.ndarray:
    """Mutual information (nats) between each rank-scaled attribute and a binary label."""
    y = np.asarray(y).astype(int)
    py = np.array([1 - y.mean(), y.mean()])
    out = np.zeros(X01.shape[1])
    for j in range(X01.shape[1]):
        b = np.minimum((X01[:, j] * bins).astype(int), bins - 1)
        joint = np.zeros((bins, 2))
        np.add.at(joint, (b, y), 1.0)
        joint /= joint.sum()
        px = joint.sum(axis=1, keepdims=True)
        nz = joint > 0
        out[j] = (joint[nz] * np.log(joint[nz] / (px @ py[None, :])[nz])).sum()
    return out


class L1Graph:
    """Implicit dense graph W(u, v) = sum_k w_k (1 - |x_uk - x_vk|), W(u, u) = 0."""

    def __init__(self, X01: np.ndarray, w: np.ndarray, threads: int = 4):
        self.n = X01.shape[0]
        self.active = np.flatnonzero(w > 0)
        self.w = np.asarray(w, dtype=np.float64)[self.active]
        self.X = np.ascontiguousarray(X01[:, self.active], dtype=np.float32)
        self.order = [np.argsort(self.X[:, j], kind="stable").astype(np.int32) for j in range(self.X.shape[1])]
        self.xs = [self.X[o, j].astype(np.float64) for j, o in enumerate(self.order)]
        self.total = float(self.w.sum())
        self.threads = threads

    def _one(self, j: int, V: np.ndarray) -> np.ndarray:
        """w_j * K_j V for the kernel K_j(u, v) = 1 - |x_uj - x_vj|."""
        o, xs = self.order[j], self.xs[j][:, None]
        Vs = V[o]
        P = np.cumsum(Vs, axis=0)
        Q = np.cumsum(xs * Vs, axis=0)
        T, Tq = P[-1], Q[-1]
        # sum_v |x_u - x_v| V_v = x_u (2P - T) - (2Q - Tq); tied values contribute zero either way
        res = T - (xs * (2.0 * P - T) - (2.0 * Q - Tq))
        out = np.empty_like(res)
        out[o] = res
        return self.w[j] * out

    def matmat(self, V: np.ndarray, chunk: int = 8) -> np.ndarray:
        """Exact W @ V for an (n, r) matrix V."""
        V = np.asarray(V, dtype=np.float64)
        out = np.zeros_like(V)
        with ThreadPoolExecutor(self.threads) as pool:
            for c in range(0, V.shape[1], chunk):
                Vc = V[:, c:c + chunk]
                acc = np.zeros_like(Vc)
                for part in pool.map(lambda j: self._one(j, Vc), range(len(self.w))):
                    acc += part
                out[:, c:c + chunk] = acc - self.total * Vc  # remove the diagonal W(u, u)
        return out

    def block(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Dense W[rows][:, cols] as float32.

        Attributes are split across threads only when ZRL_PARALLEL_BLOCK=1, because every
        thread then holds its own (rows, cols) buffer; leave it off on small machines.
        """
        import os

        def part(js):
            B = np.zeros((len(rows), len(cols)), dtype=np.float32)
            for j in js:
                B += np.float32(self.w[j]) * (1.0 - np.abs(self.X[rows, j][:, None] - self.X[cols, j][None, :]))
            return B

        workers = self.threads if os.environ.get("ZRL_PARALLEL_BLOCK") == "1" else 1
        groups = [g for g in np.array_split(np.arange(len(self.w)), workers) if len(g)]
        if len(groups) == 1:
            B = part(groups[0])
        else:
            with ThreadPoolExecutor(workers) as pool:
                parts = list(pool.map(part, groups))
            B = parts[0]
            for extra in parts[1:]:
                B += extra
        B[rows[:, None] == cols[None, :]] = 0.0
        return B


@dataclass
class ScalePartition:
    labels: np.ndarray
    R: np.ndarray               # exact reduced graph
    profile: np.ndarray         # exact (n, k) node profiles
    deviation: np.ndarray
    index: float
    irregular_fraction: float   # estimated on a stratified sample of nodes
    mse: float                  # estimated on random node pairs
    history: list
    timings: dict = None        # seconds per stage: sort, refine, summary, checks

    @property
    def k(self) -> int:
        return len(self.R)


def exact_summary(G: L1Graph, labels: np.ndarray, chunk: int = 16):
    """Exact reduced graph R, node profiles and index for a labelling.

    Works through the class indicator columns a few at a time, so the largest
    temporaries are (n, chunk) float64 and the (n, k) float32 profile.
    """
    n, k = G.n, int(labels.max()) + 1
    sizes = np.bincount(labels, minlength=k).astype(np.float64)
    profile = np.empty((n, k), dtype=np.float32)
    S = np.zeros((k, k))
    for c in range(0, k, chunk):
        cols = np.arange(c, min(c + chunk, k))
        E = (labels[:, None] == cols[None, :]).astype(np.float64)
        P = G.matmat(E, chunk=chunk)                  # P[u, j] = sum of W(u, v) over v in class cols[j]
        for j in range(len(cols)):
            S[:, cols[j]] = np.bincount(labels, weights=P[:, j], minlength=k)
        profile[:, cols] = P / np.maximum(sizes[cols][None, :] - E, 1.0)
        check_memory()
    pairs = np.outer(sizes, sizes) - np.diag(sizes)
    R = S / np.maximum(pairs, 1.0)
    share = sizes / n
    dev2 = np.zeros(n)
    for c in range(0, k, chunk):
        cols = slice(c, min(c + chunk, k))
        dev2 += (((profile[:, cols] - R[:, cols][labels].astype(np.float32)) ** 2) * share[cols][None, :]).sum(axis=1)
    index = float((np.outer(sizes, sizes) * R**2).sum() / n**2)
    return R, profile, np.sqrt(dev2), index


def sampled_checks(G: L1Graph, labels: np.ndarray, R: np.ndarray, eps: float, rng, per_class: int = 100, pairs: int = 2_000_000):
    """Share of irregular class pairs on a stratified node sample, and reconstruction MSE on random pairs."""
    k = len(R)
    sample = np.concatenate([rng.choice(np.flatnonzero(labels == c), min(per_class, int((labels == c).sum())), replace=False)
                             for c in range(k)])
    Ws = G.block(sample, sample)
    ls = labels[sample]
    classes = [np.flatnonzero(ls == c) for c in range(k)]
    bad = total = 0
    for i in range(k):
        for j in range(i + 1, k):
            d = float(Ws[np.ix_(classes[i], classes[j])].mean())
            bad += _pair_irregularity(Ws, classes[i], classes[j], d, eps, rng) > eps
            total += 1
    u, v = rng.integers(0, G.n, pairs), rng.integers(0, G.n, pairs)
    keep = u != v
    u, v = u[keep], v[keep]
    w_uv = np.zeros(len(u))
    for j, wj in enumerate(G.w):
        w_uv += wj * (1.0 - np.abs(G.X[u, j] - G.X[v, j]))
    mse = float(((w_uv - R[labels[u], labels[v]]) ** 2).mean())
    return (bad / total if total else 1.0), mse


def split_labels(G: L1Graph, labels: np.ndarray, landmarks: np.ndarray, max_bytes: float = 6e8) -> np.ndarray:
    """One refinement round: split every class at the median of its leading left singular vector.

    The vector is computed from the rows of the class against the landmark columns,
    centred on the class-to-class block means.
    """
    k = int(labels.max()) + 1
    lm_class = labels[landmarks]
    new = np.empty_like(labels)
    for c in range(k):
        A = np.flatnonzero(labels == c)
        cols = landmarks if len(A) * len(landmarks) * 4 <= max_bytes else landmarks[: int(max_bytes / (4 * len(A)))]
        cc = lm_class[: len(cols)]
        M = G.block(A, cols)
        for b in np.unique(cc):
            M[:, cc == b] -= M[:, cc == b].mean()
        gram = (M.T @ M).astype(np.float64)
        v = np.linalg.eigh(gram)[1][:, -1]
        u = M @ v.astype(np.float32)
        upper = u > np.median(u)
        if upper.sum() == 0 or upper.sum() == len(A):
            upper = np.arange(len(A)) % 2 == 0
        new[A] = 2 * c + upper
    return new


def fit(X01: np.ndarray, w: np.ndarray, k: int = 64, eps: float = 0.05, n_landmarks: int = 512, seed: int = 0,
        labels: np.ndarray | None = None, threads: int = 4, checks: bool = True) -> ScalePartition:
    """Regularity partition with k classes (a power of two) of the implicit graph, with exact summaries.

    Pass `labels` to summarise a partition obtained elsewhere (k-means, random) with the same machinery.
    """
    import time
    rng = np.random.default_rng(seed)
    t0 = time.time()
    G = L1Graph(X01, w, threads=threads)
    t1 = time.time()
    history = []
    if labels is None:
        labels = np.zeros(G.n, dtype=np.int32)
        landmarks = rng.choice(G.n, min(n_landmarks, G.n), replace=False)  # drawn once, reused in every round
        while labels.max() + 1 < k:
            labels = split_labels(G, labels, landmarks)
            history.append(int(labels.max()) + 1)
    t2 = time.time()
    R, profile, deviation, index = exact_summary(G, labels)
    t3 = time.time()
    irregular, mse = sampled_checks(G, labels, R, eps, rng) if checks else (float("nan"), float("nan"))
    t4 = time.time()
    check_memory()
    timings = {"sort": t1 - t0, "refine": t2 - t1, "summary": t3 - t2, "checks": t4 - t3}
    return ScalePartition(labels, R, profile, deviation, index, irregular, mse, history, timings)
