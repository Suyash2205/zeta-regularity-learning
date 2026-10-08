"""ZRL: zeta-weighted attributes + regularity-lemma partition for dense graphs.

Pipeline
    X (n nodes x m attributes)
      -> rank attributes by an unsupervised relevance score
      -> zeta weights by rank          w_k = (k - 1 + q)^-s / sum_j (j - 1 + q)^-s
      -> dense weighted graph          W_uv = sum_k w_k * (1 - |x_uk - x_vk|)
      -> regularity partition          equitable classes, reduced graph R
      -> node profile                  density of each node towards every class,
                                       and its deviation from its own class

s = 0 gives equal weights (the plain average); larger s concentrates weight on
the top-ranked attributes. q shifts the ranking (Hurwitz / Zipf-Mandelbrot form).
Only numpy is required.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------- zeta weights
def zeta_weights(m: int, s: float = 1.0, q: float = 1.0) -> np.ndarray:
    """Weights for ranks 1..m from the truncated Hurwitz zeta series.

    sum_{k=1..m} (k - 1 + q)^-s = zeta(s, q) - zeta(s, q + m) for s > 1, so the
    weights are the first m terms of zeta(s, q), renormalised to sum to 1.
    """
    if q <= 0:
        raise ValueError("q must be positive")
    terms = (np.arange(m) + q) ** (-float(s))
    return terms / terms.sum()


def scale01(X: np.ndarray) -> np.ndarray:
    """Rank-transform every column to [0, 1] (robust to heavy tails)."""
    X = np.asarray(X, dtype=float)
    ranks = X.argsort(axis=0).argsort(axis=0).astype(float)
    return ranks / max(len(X) - 1, 1)


def attribute_relevance(X: np.ndarray, method: str = "entropy", bins: int = 20) -> np.ndarray:
    """Unsupervised relevance score per attribute (higher = more informative).

    entropy  : 1 - normalised Shannon entropy of the min-max scaled histogram
               (the divergence degree of the entropy weight method)
    variance : variance of the min-max scaled attribute
    """
    X = np.asarray(X, dtype=float)
    lo, hi = X.min(axis=0), X.max(axis=0)
    Z = (X - lo) / np.where(hi > lo, hi - lo, 1.0)
    if method == "variance":
        return Z.var(axis=0)
    if method != "entropy":
        raise ValueError(f"unknown relevance method: {method}")
    score = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        p = np.histogram(Z[:, j], bins=bins, range=(0.0, 1.0))[0] / len(Z)
        p = p[p > 0]
        score[j] = 1.0 + (p * np.log(p)).sum() / np.log(bins)
    return score


def attribute_weights(X, s=1.0, q=1.0, method="entropy", order=None) -> np.ndarray:
    """Zeta weight for every attribute, in the original column order."""
    m = np.asarray(X).shape[1]
    if order is None:
        order = np.argsort(-attribute_relevance(X, method), kind="stable")
    w = np.empty(m)
    w[np.asarray(order)] = zeta_weights(m, s, q)
    return w


# --------------------------------------------------------------- dense graph
def weighted_graph(X01: np.ndarray, w: np.ndarray) -> np.ndarray:
    """W_uv = sum_k w_k (1 - |x_uk - x_vk|), entries in [0, 1], zero diagonal."""
    n = len(X01)
    W = np.zeros((n, n), dtype=np.float32)
    for k in np.flatnonzero(w > 0):
        col = X01[:, k].astype(np.float32)
        W += np.float32(w[k]) * (1.0 - np.abs(col[:, None] - col[None, :]))
    np.fill_diagonal(W, 0.0)
    return W


# ------------------------------------------------------- regularity partition
def _top_singular(M: np.ndarray, rng, iters: int = 25):
    """Leading singular triple of M by power iteration."""
    v = rng.standard_normal(M.shape[1])
    v /= np.linalg.norm(v) + 1e-12
    u = np.zeros(M.shape[0])
    sigma = 0.0
    for _ in range(iters):
        u = M @ v
        sigma = np.linalg.norm(u)
        if sigma < 1e-12:
            break
        u /= sigma
        v = M.T @ u
        v /= np.linalg.norm(v) + 1e-12
    return sigma, u, v


def _block_density(W, classes):
    k = len(classes)
    R = np.zeros((k, k))
    for i, A in enumerate(classes):
        for j, B in enumerate(classes):
            if i == j:
                R[i, j] = W[np.ix_(A, A)].sum() / max(len(A) * (len(A) - 1), 1)
            else:
                R[i, j] = W[np.ix_(A, B)].mean()
    return R


def _pair_irregularity(W, A, B, d, eps, rng):
    """Largest density deviation found on witness subsets of an (A, B) pair.

    Witnesses come from the signs of the leading singular vectors of the centred
    block, restricted to subsets of at least eps|A| and eps|B| vertices. A value
    above eps shows the pair is not eps-regular; a value below eps means no
    witness was found (it is a heuristic check, not a certificate).
    """
    M = W[np.ix_(A, B)].astype(float) - d
    _, u, v = _top_singular(M, rng)
    worst = 0.0
    for X in (u > 0, u <= 0):
        if X.sum() < eps * len(A):
            continue
        for Y in (v > 0, v <= 0):
            if Y.sum() < eps * len(B):
                continue
            worst = max(worst, abs(M[np.ix_(X, Y)].mean()))
    return worst


@dataclass
class RegularPartition:
    labels: np.ndarray        # class of every node
    R: np.ndarray             # reduced graph: k x k matrix of pair densities
    irregular_fraction: float  # share of class pairs with a witness of irregularity
    index: float              # mean-square density (the lemma's potential function)
    history: list             # one dict per refinement round

    @property
    def k(self) -> int:
        return len(self.R)


def regularity_partition(W, eps=0.1, k_min=4, k_max=32, min_size=8, seed=0) -> RegularPartition:
    """Equitable partition refined until at most an eps share of pairs is irregular.

    k_min is the lemma's lower bound on the number of classes.

    Follows the constructive proof of Szemeredi's lemma: test every pair, and
    while too many are irregular, refine. Every class is split in two at the
    median of the leading left singular vector of its centred rows, which keeps
    the partition equitable and raises the index.
    """
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    labels = np.zeros(n, dtype=int)
    history = []
    while True:
        classes = [np.flatnonzero(labels == c) for c in range(labels.max() + 1)]
        k = len(classes)
        R = _block_density(W, classes)
        sizes = np.array([len(c) for c in classes], dtype=float)
        index = float((np.outer(sizes, sizes) * R**2).sum() / n**2)
        pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
        bad = sum(
            _pair_irregularity(W, classes[i], classes[j], R[i, j], eps, rng) > eps
            for i, j in pairs
        )
        frac = bad / len(pairs) if pairs else 1.0
        history.append({"k": k, "index": index, "irregular_fraction": frac})
        done = frac <= eps and k >= k_min
        if done or 2 * k > k_max or sizes.min() < 2 * min_size:
            return RegularPartition(labels, R, frac, index, history)
        new = np.empty(n, dtype=int)
        for c, A in enumerate(classes):
            M = W[A].astype(float) - R[c][labels][None, :]
            _, u, _ = _top_singular(M, rng)
            upper = u > np.median(u)
            if upper.sum() == 0 or upper.sum() == len(A):  # degenerate direction
                upper = np.arange(len(A)) % 2 == 0
            new[A] = 2 * c + upper
        labels = new


# -------------------------------------------------------------- node profiles
def node_profiles(W, part: RegularPartition):
    """Per-node density towards every class, and deviation from its own class.

    In an eps-regular pair all but a small share of vertices have density
    towards the other class close to the pair density, so a large deviation
    marks a node that does not behave like its class.
    """
    n, k = W.shape[0], part.k
    onehot = np.zeros((n, k))
    onehot[np.arange(n), part.labels] = 1.0
    sizes = onehot.sum(axis=0)
    own = sizes[None, :] - onehot  # a node is not its own neighbour
    profile = (W.astype(float) @ onehot) / np.maximum(own, 1.0)
    expected = part.R[part.labels]
    share = sizes / n
    deviation = np.sqrt((((profile - expected) ** 2) * share[None, :]).sum(axis=1))
    return profile, deviation


def reconstruction_error(W, part: RegularPartition) -> float:
    """Mean squared error of replacing W by the reduced graph."""
    approx = part.R[np.ix_(part.labels, part.labels)]
    diff = W.astype(float) - approx
    np.fill_diagonal(diff, 0.0)
    n = W.shape[0]
    return float((diff**2).sum() / (n * (n - 1)))


def fit(X, s=1.0, q=1.0, eps=0.1, k_min=4, k_max=32, relevance="entropy", order=None, seed=0):
    """Run the full pipeline. Returns a dict with weights, graph, partition and profiles."""
    w = attribute_weights(X, s=s, q=q, method=relevance, order=order)
    W = weighted_graph(scale01(X), w)
    part = regularity_partition(W, eps=eps, k_min=k_min, k_max=k_max, seed=seed)
    profile, deviation = node_profiles(W, part)
    return {"weights": w, "W": W, "partition": part, "profile": profile,
            "deviation": deviation, "mse": reconstruction_error(W, part)}
