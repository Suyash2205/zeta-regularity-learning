"""Cross-validated evaluation of matrix-free ZRL on every account of a transaction file.

Design: 20% of the accounts (stratified) form a ranking set whose labels fix the attribute
order; the other 80% are split into five folds. Models train on four folds plus the ranking
set and are scored on the fifth.

Usage: python experiment_scale.py <accounts_v2.parquet> <out_dir> <full|main> [repeats] [threads]

Stratified five-fold cross-validation over all accounts, repeated. In each fold the
attribute ranking and every supervised model see training labels only; the graph and
the partition use no labels. "main" runs the main configuration and the comparison
methods; "full" adds the parameter sweeps and the ablations (first repeat only).
Results are written after every fold. Set ZRL_MAX_GB to the memory this run may use.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import zrl  # noqa: E402
import zrl_scale as zs  # noqa: E402

src, out_dir, mode = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
REPEATS = int(sys.argv[4]) if len(sys.argv) > 4 else 1
THREADS = int(sys.argv[5]) if len(sys.argv) > 5 else 4
out_dir.mkdir(parents=True, exist_ok=True)
MAIN = {"s": 1.0, "q": 1.0, "k": 64}
STRUCTURAL = ["pagerank_in", "pagerank_out", "nbt_centrality", "reciprocity", "partner_degree", "degree"]

acc = pl.read_parquet(src).sort("account")
FEATS = [c for c in acc.columns if c not in ("account", "is_laundering", "n_tx", "n_flagged")]
y = acc["is_laundering"].to_numpy().astype(int)
X_raw = acc.select(FEATS).to_numpy()
X01 = zs.rank01(X_raw)
n, m = X01.shape
ALL = list(range(m))
BEHAVIOURAL = [i for i, f in enumerate(FEATS) if f not in STRUCTURAL]
NO_NBT = [i for i, f in enumerate(FEATS) if f != "nbt_centrality"]
ENTROPY_ORDER = np.argsort(-zrl.attribute_relevance(X_raw, "entropy"), kind="stable")  # raw values: ranks are uniform
del acc, X_raw
print(f"accounts {n}, attributes {m}, laundering {y.sum()} ({y.mean():.3%}), mode {mode}, repeats {REPEATS}", flush=True)


def metrics(score, test):
    yt = y[test]
    ranked = np.argsort(-score)
    out = {"auc": float(roc_auc_score(yt, score)), "ap": float(average_precision_score(yt, score))}
    for pct in (1, 5, 10):  # recall among the highest-scored share of held-out accounts
        out[f"recall_top{pct}"] = float(yt[ranked[: max(1, len(test) * pct // 100)]].sum() / yt.sum())
    return out


def logit(F, train, test):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", solver="newton-cholesky", max_iter=200))
    return clf.fit(F[train], y[train]).predict_proba(F[test])[:, 1]


def class_risk(labels, train, k):
    pos = np.bincount(labels[train], weights=y[train], minlength=k)
    tot = np.bincount(labels[train], minlength=k)
    return (pos + y[train].mean()) / (tot + 1.0)


def weights_for(order, s, q, cols):
    """Zeta weights on the attributes `cols`, ranked by `order` (positions within cols); zero elsewhere."""
    w = np.zeros(m)
    w[np.asarray(cols)[order]] = zrl.zeta_weights(len(cols), s, q)
    return w


rows, structure, cache = [], [], {}
fold_of = np.full((REPEATS, n), -1, dtype=np.int8)   # fold of every account per repeat; -1 = ranking set
predictions = {}                                      # first-repeat held-out scores of the main methods
SAVE_PREDICTIONS = n < 1_000_000
t0 = time.time()


def keep(name, score, test, rep):
    if SAVE_PREDICTIONS and rep == 0:
        predictions.setdefault(name, np.full(n, np.nan, dtype=np.float32))[test] = score


def fitted(w, k, tag, labels=None):
    """Fit once per distinct weight vector; keep only what scoring needs."""
    key = (tag, k, tuple(np.round(w, 12)))
    if key not in cache:
        t = time.time()
        p = zs.fit(X01, w, k=k, seed=0, labels=labels, threads=THREADS)
        cache[key] = {"F": np.column_stack([p.profile, p.deviation.astype(np.float32)]), "labels": p.labels, "k": p.k,
                      "stats": {"classes": p.k, "irregular_fraction": p.irregular_fraction, "mse": p.mse, "index": p.index,
                                "seconds": time.time() - t}}
    return cache[key]


def evaluate(name, sweep, value, fit, train, test, rep, fold, cfg):
    scores = (("profile_logit", logit(fit["F"], train, test)),
              ("class_risk", class_risk(fit["labels"], train, fit["k"])[fit["labels"][test]]),
              ("deviation", fit["F"][test, -1]))
    if name == "ZRL" and sweep == "main":
        keep("zrl_profile_logit", scores[0][1], test, rep)
    for kind, score in scores:
        rows.append({"method": f"{name}/{kind}", "sweep": sweep, "value": str(value), "rep": rep, "fold": fold, **cfg, **metrics(score, test)})
    structure.append({"method": name, "sweep": sweep, "value": str(value), "rep": rep, "fold": fold, **cfg, **fit["stats"]})


def plain(name, value, score, test, rep, fold):
    rows.append({"method": name, "sweep": "baseline", "value": value, "rep": rep, "fold": fold, **MAIN, **metrics(score, test)})


# A fixed 20% of the accounts is set aside to rank the attributes. It is never part of a test fold,
# so the weights use no test labels, and one attribute order serves every fold.
from sklearn.model_selection import train_test_split  # noqa: E402

eval_idx, rank_idx = train_test_split(np.arange(n), test_size=0.2, stratify=y, random_state=0)
mi = zs.binned_mutual_information(X01[rank_idx], y[rank_idx])
order = np.argsort(-mi, kind="stable")
w_main = weights_for(order, MAIN["s"], MAIN["q"], ALL)
print("ranking set", len(rank_idx), "evaluation set", len(eval_idx), "top attributes", [FEATS[i] for i in order[:6]], flush=True)

for rep in range(REPEATS):
    for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=rep).split(eval_idx, y[eval_idx])):
        train, test = np.concatenate([eval_idx[tr], rank_idx]), eval_idx[te]
        fold_of[rep, test] = fold
        evaluate("ZRL", "main", "main", fitted(w_main, MAIN["k"], "zrl"), train, test, rep, fold, MAIN)

        # Other partitions of the same zeta-weighted graph, summarised exactly by the same machinery.
        km = MiniBatchKMeans(MAIN["k"], n_init=3, batch_size=8192, random_state=0).fit_predict(X01 * np.sqrt(w_main)).astype(np.int32)
        evaluate("k-means partition", "partition", "kmeans", fitted(w_main, MAIN["k"], "kmeans", km), train, test, rep, fold, MAIN)
        rnd = (np.random.default_rng(0).permutation(n) % MAIN["k"]).astype(np.int32)
        evaluate("random equal-sized partition", "partition", "random", fitted(w_main, MAIN["k"], "random", rnd), train, test, rep, fold, MAIN)
        # Equal weights: the plain average that the zeta weights replace.
        evaluate("ZRL (equal weights)", "weights", "equal", fitted(np.full(m, 1.0 / m), MAIN["k"], "zrl"), train, test, rep, fold, {**MAIN, "s": 0.0})

        # Models that do not use the graph.
        lr_score = logit(X01, train, test)
        plain("logistic regression (all attributes)", "lr", lr_score, test, rep, fold)
        keep("logistic_regression", lr_score, test, rep)
        gb = HistGradientBoostingClassifier(max_iter=200, random_state=fold).fit(X01[train], y[train])
        gb_score = gb.predict_proba(X01[test])[:, 1]
        plain("gradient boosting (all attributes)", "gb", gb_score, test, rep, fold)
        keep("gradient_boosting", gb_score, test, rep)
        gb = HistGradientBoostingClassifier(max_iter=200, random_state=fold).fit(X01[train][:, BEHAVIOURAL], y[train])
        plain("gradient boosting (behavioural attributes)", "gb_behavioural", gb.predict_proba(X01[test][:, BEHAVIOURAL])[:, 1], test, rep, fold)
        sub = np.random.default_rng(fold).choice(train, min(100_000, len(train)), replace=False)
        iso = IsolationForest(n_estimators=200, random_state=fold).fit(X01[sub])
        plain("isolation forest (no labels)", "iforest", -iso.score_samples(X01[test]), test, rep, fold)

        if mode == "full" and rep == 0:
            for s in (0.5, 1.5, 2.0, 3.0):
                evaluate("ZRL", "s", s, fitted(weights_for(order, s, 1.0, ALL), MAIN["k"], "zrl"), train, test, rep, fold, {**MAIN, "s": s})
            for q in (2.0, 4.0):
                evaluate("ZRL", "q", q, fitted(weights_for(order, 1.0, q, ALL), MAIN["k"], "zrl"), train, test, rep, fold, {**MAIN, "q": q})
            for k in (16, 32, 128, 256):
                evaluate("ZRL", "k", k, fitted(w_main, k, "zrl"), train, test, rep, fold, {**MAIN, "k": k})
            # Unsupervised ranking: no labels anywhere in the weights.
            evaluate("ZRL (entropy ranking)", "ranking", "entropy",
                     fitted(weights_for(ENTROPY_ORDER, 1.0, 1.0, ALL), MAIN["k"], "zrl"), train, test, rep, fold, MAIN)
            # Attribute ablations.
            for name, cols in (("behavioural attributes only", BEHAVIOURAL), ("without non-backtracking centrality", NO_NBT)):
                sub_order = np.argsort(-mi[cols], kind="stable")
                evaluate(f"ZRL ({name})", "attributes", name, fitted(weights_for(sub_order, 1.0, 1.0, cols), MAIN["k"], "zrl"),
                         train, test, rep, fold, MAIN)

        pl.DataFrame(rows, infer_schema_length=None).write_csv(out_dir / "scores.csv")
        pl.DataFrame(structure, infer_schema_length=None).write_csv(out_dir / "structure.csv")
        print(f"rep {rep} fold {fold} done, {len(cache)} fits cached, top attributes {[FEATS[i] for i in order[:4]]}, "
              f"peak {zs.check_memory():.1f} GB, {time.time() - t0:.0f}s", flush=True)

np.savez_compressed(out_dir / "folds.npz", fold_of=fold_of, label=y.astype(np.int8))
if predictions:
    np.savez_compressed(out_dir / "predictions_rep0.npz", **{k: v.astype(np.float16) for k, v in predictions.items()})

# Descriptive output for the main configuration (same weights and partition as in the folds).
w = w_main
part = zs.fit(X01, w, k=MAIN["k"], seed=0, threads=THREADS)
k = part.k
sizes = np.bincount(part.labels, minlength=k)
class_mean = np.vstack([X01[part.labels == c].mean(axis=0) for c in range(k)])
top_dev = np.argsort(-part.deviation)[: n // 100]
np.savez_compressed(out_dir / "full_fit.npz", R=part.R, weights=w, relevance=mi, sizes=sizes,
                    positives=np.bincount(part.labels, weights=y, minlength=k), class_mean=class_mean)
json.dump({"features": FEATS, "n": n, "m": m, "positives": int(y.sum()), "main": MAIN, "edges": n * (n - 1) // 2,
           "irregular_fraction": part.irregular_fraction, "mse": part.mse, "index": part.index,
           "rmse_bound_r6": float(np.pi / np.sqrt(6)), "deviation_top1pct_positives": int(y[top_dev].sum()),
           "ranking_set": int(len(rank_idx)), "evaluation_set": int(len(eval_idx)),
           "threads": THREADS, "timings": part.timings,
           "top_attributes": [FEATS[i] for i in order[:8]]}, open(out_dir / "meta.json", "w"), indent=1)
print("done", f"{time.time() - t0:.0f}s", flush=True)
