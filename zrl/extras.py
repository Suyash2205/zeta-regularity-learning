"""Supplementary experiments: scalability and robustness to noise attributes.

Usage: python extras.py <accounts_v2.parquet> <out_dir> scale|noise
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import zrl  # noqa: E402
import zrl_scale as zs  # noqa: E402

src, out_dir = sys.argv[1], Path(sys.argv[2])
tasks = set(sys.argv[3:4]) or {"scale", "noise"}
THREADS = int(__import__("os").environ.get("ZRL_THREADS", "4"))
out_dir.mkdir(parents=True, exist_ok=True)

acc = pl.read_parquet(src).sort("account")
FEATS = [c for c in acc.columns if c not in ("account", "is_laundering", "n_tx")]
y_all = acc["is_laundering"].to_numpy().astype(int)
X_all = acc.select(FEATS).to_numpy()
m = len(FEATS)


def zeta_w(order, s, total):
    w = np.zeros(total)
    w[order] = zrl.zeta_weights(total, s, 1.0)
    return w


if "scale" in tasks:
    # Time and peak memory of one complete fit as the number of accounts grows. Every run is a fresh
    # process, so peak memory belongs to that run alone; each size is repeated to show the variation.
    import subprocess
    rows = []
    sizes = [v for v in (10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000) if v < len(X_all)] + [len(X_all)]
    for n in sizes:
        for rep in range(int(__import__("os").environ.get("ZRL_SCALE_REPEATS", "3"))):
            res = subprocess.run([sys.executable, __file__, src, str(out_dir), "one_scale_run", str(n), str(rep)],
                                 capture_output=True, text=True, check=True)
            rows.append(json.loads(res.stdout.strip().splitlines()[-1]))
            print(rows[-1], flush=True)
            pl.DataFrame(rows).write_csv(out_dir / "scalability.csv")

if "one_scale_run" in tasks:
    n, rep = int(sys.argv[4]), int(sys.argv[5])
    idx = np.sort(np.random.default_rng(rep).choice(len(X_all), n, replace=False))
    X01 = zs.rank01(X_all[idx])
    order = np.argsort(-zs.binned_mutual_information(X01, y_all[idx]), kind="stable")
    w = zeta_w(order, 1.0, m)
    t = time.time()
    part = zs.fit(X01, w, k=64, seed=rep, threads=THREADS)
    total = time.time() - t
    G = zs.L1Graph(X01, w, threads=THREADS)
    t = time.time()
    G.matmat(np.ones((n, 1)))
    t_mv = time.time() - t
    print(json.dumps({"n": n, "rep": rep, "edges": n * (n - 1) // 2, "fit_seconds": total, "matvec_seconds": t_mv,
                      **{f"{k}_seconds": v for k, v in part.timings.items()}, "peak_rss_gb": zs.check_memory(),
                      "dense_float32_gb": n * n * 4 / 1e9, "mse": part.mse, "irregular_fraction": part.irregular_fraction}))
    sys.exit(0)

if "noise" in tasks:
    # Append pure-noise attributes: zeta weights should hold up, equal weights should degrade.
    rng = np.random.default_rng(1)
    pos = np.flatnonzero(y_all == 1)
    neg = rng.choice(np.flatnonzero(y_all == 0), 100_000 - len(pos), replace=False)
    idx = np.sort(np.concatenate([pos, neg]))  # every laundering account plus a random sample of the rest
    y = y_all[idx]
    rows = []
    for extra in (0, m // 2, m, 2 * m):
        X01 = zs.rank01(np.column_stack([X_all[idx], rng.random((len(idx), extra))]))
        total = m + extra
        for fold, (train, test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(X01, y)):
            order = np.argsort(-zs.binned_mutual_information(X01[train], y[train]), kind="stable")
            for s in (0.0, 1.0):
                part = zs.fit(X01, zeta_w(order, s, total), k=64, seed=0, threads=THREADS, checks=False)
                F = np.column_stack([part.profile, part.deviation])
                clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", solver="newton-cholesky"))
                p = clf.fit(F[train], y[train]).predict_proba(F[test])[:, 1]
                rows.append({"noise_attributes": extra, "method": f"ZRL s={s:g}", "fold": fold,
                             "auc": roc_auc_score(y[test], p), "ap": average_precision_score(y[test], p),
                             "weight_on_noise": float(zeta_w(order, s, total)[m:].sum())})
            gb = HistGradientBoostingClassifier(max_iter=200, random_state=fold).fit(X01[train], y[train])
            p = gb.predict_proba(X01[test])[:, 1]
            rows.append({"noise_attributes": extra, "method": "gradient boosting", "fold": fold,
                         "auc": roc_auc_score(y[test], p), "ap": average_precision_score(y[test], p), "weight_on_noise": float("nan")})
        pl.DataFrame(rows).write_csv(out_dir / "noise.csv")
        print("noise attributes", extra, "done", flush=True)
    json.dump({"n": len(idx), "positives": int(y.sum())}, open(out_dir / "noise_meta.json", "w"))
