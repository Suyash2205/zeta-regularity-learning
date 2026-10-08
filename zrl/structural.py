"""Structural attributes of every account, computed from the transaction graph itself.

Usage: python structural.py <transactions.csv> <accounts.parquet> <out.parquet>

Adds, per account: PageRank on the payment graph and on its reverse, the
non-backtracking walk centrality of the undirected graph, the share of partners
that pay back (reciprocity), and the mean degree of the partners.

The non-backtracking centrality solves M(t) x = (1 - t^2) 1 with the deformed graph
Laplacian M(t) = I - tA + t^2 (D - I), the matrix whose determinant appears in the
Ihara-Bass formula for the graph zeta function.
"""
import sys

import numpy as np
import polars as pl
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg, eigs

src, acc_path, out = sys.argv[1:4]
COLS = ["ts", "from_bank", "from_acct", "to_bank", "to_acct", "amt_recv", "cur_recv", "amt_paid", "cur_paid", "fmt", "label"]
edges = (
    pl.scan_csv(src, has_header=True, new_columns=COLS, schema_overrides={"from_bank": pl.Utf8, "to_bank": pl.Utf8})
    .select(src=pl.col("from_bank") + "_" + pl.col("from_acct"), dst=pl.col("to_bank") + "_" + pl.col("to_acct"))
    .filter(pl.col("src") != pl.col("dst"))
    .group_by(["src", "dst"]).agg(n=pl.len())
    .collect()
)
acc = pl.read_parquet(acc_path).sort("account")
index = {a: i for i, a in enumerate(acc["account"].to_list())}
n = len(index)
rows = np.fromiter((index[a] for a in edges["src"].to_list()), dtype=np.int64, count=edges.height)
cols = np.fromiter((index[a] for a in edges["dst"].to_list()), dtype=np.int64, count=edges.height)
A = sp.csr_matrix((np.ones(edges.height), (rows, cols)), shape=(n, n))  # directed, unweighted
print(f"accounts {n}, directed edges {A.nnz}", flush=True)


def pagerank(M, damping=0.85, iters=60):
    """Power iteration; M[i, j] = 1 if i pays j."""
    out_deg = np.asarray(M.sum(axis=1)).ravel()
    inv = np.divide(1.0, out_deg, out=np.zeros_like(out_deg), where=out_deg > 0)
    P = sp.diags(inv) @ M
    r = np.full(n, 1.0 / n)
    for _ in range(iters):
        dangling = r[out_deg == 0].sum()
        r = damping * (P.T @ r + dangling / n) + (1 - damping) / n
    return r


pr_in = pagerank(A)        # importance as a receiver of money
pr_out = pagerank(A.T.tocsr())  # importance as a sender of money

U = ((A + A.T) > 0).astype(np.float64).tocsr()  # undirected simple graph
deg = np.asarray(U.sum(axis=1)).ravel()

# Radius of convergence: 1 / spectral radius of the companion matrix [[A, I - D], [I, 0]],
# which has the same non-trivial spectrum as the non-backtracking (Hashimoto) matrix.
def companion(v):
    a, b = v[:n], v[n:]
    return np.concatenate([U @ a + (1.0 - deg) * b, a])


rho = float(abs(eigs(LinearOperator((2 * n, 2 * n), matvec=companion, dtype=np.float64), k=1, which="LM",
                     return_eigenvectors=False, tol=1e-4, maxiter=500)[0]))
t = 0.5 / rho
M = sp.identity(n, format="csr") - t * U + (t * t) * sp.diags(deg - 1.0)
nbt, info = cg(M, np.full(n, 1.0 - t * t), rtol=1e-10, maxiter=2000)
print(f"non-backtracking spectral radius {rho:.2f}, t = {t:.5f}, cg info {info}, "
      f"residual {np.linalg.norm(M @ nbt - (1 - t * t)):.2e}", flush=True)

recip = np.asarray(A.multiply(A.T).sum(axis=1)).ravel() / np.maximum(np.asarray(A.sum(axis=1)).ravel(), 1.0)
nbr_deg = (U @ deg) / np.maximum(deg, 1.0)

structural = pl.DataFrame({
    "account": acc["account"], "pagerank_in": pr_in, "pagerank_out": pr_out, "nbt_centrality": nbt,
    "reciprocity": recip, "partner_degree": nbr_deg, "degree": deg,
})
acc.join(structural, on="account").write_parquet(out)
print(structural.select(pl.exclude("account")).describe())
