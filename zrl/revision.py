"""Experiments added in revision: the label-free main configuration and its robustness.

Usage: python revision.py <accounts.parquet> <out_dir> <task> [args]

Tasks
  unsup <repeats>   label-free main configuration (entropy ranking) on every fold, against the label-ranked
                    variant, a k-means partition and a random partition of the same graph; class concentration
  seeds             variation over five landmark seeds
  strict            the same evaluation with a stricter label (at least two flagged transactions)
  tuned             gradient boosting and logistic regression with settings chosen on a validation split
  gnn <trans.csv>   an account-level graph neural network (GraphSAGE with mean aggregation)
  weights           zeta weights against other decreasing weight laws when noise attributes are added
  temporal <labels_later.parquet>   train on the past, predict accounts first flagged later

Splits are those of experiment_scale.py: a stratified 20% ranking set, then stratified five-fold
cross-validation on the rest; models train on four folds plus the ranking set.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import zrl  # noqa: E402
import zrl_scale as zs  # noqa: E402

src, out_dir, task = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
out_dir.mkdir(parents=True, exist_ok=True)
THREADS = int(__import__("os").environ.get("ZRL_THREADS", "4"))
K, S_EXP, Q = 64, 1.0, 1.0

acc = pl.read_parquet(src).sort("account")
FEATS = [c for c in acc.columns if c not in ("account", "is_laundering", "n_tx", "n_flagged")]
y = acc["is_laundering"].to_numpy().astype(int)
n_flagged = acc["n_flagged"].to_numpy() if "n_flagged" in acc.columns else None
X_raw = acc.select(FEATS).to_numpy()
X01 = zs.rank01(X_raw)
n, m = X01.shape
entropy_relevance = zrl.attribute_relevance(X_raw, "entropy")
ENT_ORDER = np.argsort(-entropy_relevance, kind="stable")
print(f"accounts {n}, attributes {m}, laundering {y.sum()} ({y.mean():.3%}), task {task}", flush=True)


def law_weights(order, law, total=None):
    """Weights by rank under a named law; `order` lists attributes from most to least relevant."""
    total = total or len(order)
    ranks = np.arange(1, total + 1, dtype=float)
    raw = {"zeta": ranks ** (-S_EXP), "equal": np.ones(total), "geometric": 0.85 ** (ranks - 1), "linear": total - ranks + 1}[law]
    w = np.zeros(total)
    w[order] = raw / raw.sum()
    return w


def metrics(score, truth):
    ranked = np.argsort(-score)
    out = {"auc": float(roc_auc_score(truth, score)), "ap": float(average_precision_score(truth, score))}
    for pct in (1, 5, 10):
        out[f"recall_top{pct}"] = float(truth[ranked[: max(1, len(truth) * pct // 100)]].sum() / max(truth.sum(), 1))
    return out


def logit(F, target, train, test, C=1.0):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", solver="newton-cholesky", max_iter=200))
    return clf.fit(F[train], target[train]).predict_proba(F[test])[:, 1]


def class_risk(labels, target, train, k):
    pos = np.bincount(labels[train], weights=target[train], minlength=k)
    tot = np.bincount(labels[train], minlength=k)
    return (pos + target[train].mean()) / (tot + 1.0)


def folds(target, repeats):
    """The splits of experiment_scale.py: fixed ranking set, then stratified five-fold on the rest."""
    eval_idx, rank_idx = train_test_split(np.arange(n), test_size=0.2, stratify=target, random_state=0)
    for rep in range(repeats):
        for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=rep).split(eval_idx, target[eval_idx])):
            yield rep, fold, np.concatenate([eval_idx[tr], rank_idx]), eval_idx[te], rank_idx


def concentration(labels, target):
    """How unevenly the positives are spread over the classes (descriptive, all accounts)."""
    k = int(labels.max()) + 1
    size = np.bincount(labels, minlength=k).astype(float)
    pos = np.bincount(labels, weights=target, minlength=k)
    rate = pos / np.maximum(size, 1)
    order = np.argsort(-rate)
    cum_acc = np.cumsum(size[order]) / size.sum()
    cum_pos = np.cumsum(pos[order]) / pos.sum()
    share = lambda frac: float(np.interp(frac, np.concatenate([[0.0], cum_acc]), np.concatenate([[0.0], cum_pos])))  # noqa: E731
    base = target.mean()
    return {"share_top12": share(0.125), "share_top25": share(0.25), "share_top50": share(0.5), "max_lift": float(rate.max() / base),
            "classes_zero": int((pos == 0).sum()), "classes_ge_2x": int((rate >= 2 * base).sum()),
            "smallest_class": int(size.min()), "largest_class": int(size.max())}


def rmse_with_error(G, labels, R, pairs=2_000_000, seed=1):
    """Reconstruction error on random pairs, with its standard error (delta method on the mean squared error)."""
    rng = np.random.default_rng(seed)
    u, v = rng.integers(0, G.n, pairs), rng.integers(0, G.n, pairs)
    keep = u != v
    u, v = u[keep], v[keep]
    w_uv = np.zeros(len(u))
    for j, wj in enumerate(G.w):
        w_uv += wj * (1.0 - np.abs(G.X[u, j] - G.X[v, j]))
    sq = (w_uv - R[labels[u], labels[v]]) ** 2
    mse, se_mse = sq.mean(), sq.std(ddof=1) / np.sqrt(len(sq))
    return float(np.sqrt(mse)), float(se_mse / (2 * np.sqrt(mse)))


def summarise(w, labels=None, seed=0):
    p = zs.fit(X01, w, k=K, seed=seed, labels=labels, threads=THREADS)
    return p, np.column_stack([p.profile, p.deviation.astype(np.float32)])


def score_partition(rows, name, part, F, target, repeats, extra=None):
    for rep, fold, train, test, _ in folds(target, repeats):
        risk = class_risk(part.labels, target, train, part.k)
        for kind, sc in (("profile_logit", logit(F, target, train, test)), ("class_risk", risk[part.labels[test]]), ("deviation", F[test, -1])):
            rows.append({"method": f"{name}/{kind}", "rep": rep, "fold": fold, **(extra or {}), **metrics(sc, target[test])})


t0 = time.time()

# ------------------------------------------------------------------------------------------------ unsup
if task == "unsup":
    repeats = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    w_ent = law_weights(ENT_ORDER, "zeta")
    _, _, _, _, rank_idx = next(folds(y, 1))
    mi = zs.binned_mutual_information(X01[rank_idx], y[rank_idx])
    w_mi = law_weights(np.argsort(-mi, kind="stable"), "zeta")
    rows, conc, stats, parts = [], [], [], {}
    G = zs.L1Graph(X01, w_ent, threads=THREADS)
    variants = [("ZRL-U", w_ent, None), ("ZRL-MI (label-ranked)", w_mi, None),
                ("k-means partition", w_ent, "kmeans"), ("random partition", w_ent, "random")]
    for name, w, kind in variants:
        labels = None
        if kind == "kmeans":
            labels = MiniBatchKMeans(K, n_init=3, batch_size=8192, random_state=0).fit_predict(X01 * np.sqrt(w)).astype(np.int32)
        elif kind == "random":
            labels = (np.random.default_rng(0).permutation(n) % K).astype(np.int32)
        part, F = summarise(w, labels)
        score_partition(rows, name, part, F, y, repeats)
        Gw = G if w is w_ent else zs.L1Graph(X01, w, threads=THREADS)
        rmse, se = rmse_with_error(Gw, part.labels, part.R)
        stats.append({"method": name, "rmse": rmse, "rmse_se": se, "irregular_fraction": part.irregular_fraction, "index": part.index,
                      **(part.timings or {})})
        conc.append({"method": name, **concentration(part.labels, y)})
        parts[name.split(" ")[0].lower().replace("-", "_")] = part.labels.astype(np.int16)
        pl.DataFrame(rows).write_csv(out_dir / "scores_rev.csv")
        pl.DataFrame(stats).write_csv(out_dir / "structure_rev.csv")
        pl.DataFrame(conc).write_csv(out_dir / "concentration.csv")
        print(f"{name} done, {time.time() - t0:.0f}s", flush=True)
        if name == "ZRL-U":  # descriptive output for the class table and the reduced-graph figure
            sizes = np.bincount(part.labels, minlength=K)
            np.savez_compressed(out_dir / "full_fit_rev.npz", R=part.R, weights=w, relevance=entropy_relevance, sizes=sizes,
                                positives=np.bincount(part.labels, weights=y, minlength=K),
                                class_mean=np.vstack([X01[part.labels == c].mean(axis=0) for c in range(K)]))
    np.savez_compressed(out_dir / "partitions_rev.npz", **parts)
    json.dump({"features": FEATS, "n": n, "m": m, "positives": int(y.sum()), "entropy_order": [FEATS[i] for i in ENT_ORDER],
               "mi_order": [FEATS[i] for i in np.argsort(-mi, kind="stable")],
               "relevance_ties": int(len(entropy_relevance) - len(np.unique(entropy_relevance))),
               "top_weights": [[FEATS[i], float(w_ent[i])] for i in ENT_ORDER[:8]]}, open(out_dir / "meta_rev.json", "w"), indent=1)

# ------------------------------------------------------------------------------------------------ seeds
elif task == "seeds":
    w_ent = law_weights(ENT_ORDER, "zeta")
    G = zs.L1Graph(X01, w_ent, threads=THREADS)
    rows, per_seed = [], []
    for seed in range(5):
        part, F = summarise(w_ent, seed=seed)
        before = len(rows)
        score_partition(rows, "ZRL-U", part, F, y, 1, {"seed": seed})
        aucs = [r["auc"] for r in rows[before:] if r["method"].endswith("profile_logit")]
        rmse, se = rmse_with_error(G, part.labels, part.R)
        per_seed.append({"seed": seed, "rmse": rmse, "rmse_se": se, "irregular_fraction": part.irregular_fraction, "index": part.index,
                         "profile_auc": float(np.mean(aucs)), **concentration(part.labels, y)})
        pl.DataFrame(per_seed).write_csv(out_dir / "seeds.csv")
        pl.DataFrame(rows).write_csv(out_dir / "seeds_scores.csv")
        print(f"seed {seed} done, {time.time() - t0:.0f}s", flush=True)

# ------------------------------------------------------------------------------------------------ strict
elif task == "strict":
    y2 = (n_flagged >= 2).astype(int)
    print(f"strict label: {y2.sum()} accounts ({y2.mean():.3%})", flush=True)
    part, F = summarise(law_weights(ENT_ORDER, "zeta"))
    rows = []
    score_partition(rows, "ZRL-U", part, F, y2, 1)
    for rep, fold, train, test, _ in folds(y2, 1):
        rows.append({"method": "logistic regression", "rep": rep, "fold": fold, **metrics(logit(X01, y2, train, test), y2[test])})
        gb = HistGradientBoostingClassifier(max_iter=200, random_state=fold).fit(X01[train], y2[train])
        rows.append({"method": "gradient boosting", "rep": rep, "fold": fold, **metrics(gb.predict_proba(X01[test])[:, 1], y2[test])})
    pl.DataFrame(rows).write_csv(out_dir / "strict.csv")
    json.dump({"positives": int(y2.sum()), "share": float(y2.mean()), **{"conc_" + k: v for k, v in concentration(part.labels, y2).items()}},
              open(out_dir / "strict_meta.json", "w"))

# ------------------------------------------------------------------------------------------------ tuned
elif task == "tuned":
    rng = np.random.default_rng(0)
    grid = [{"learning_rate": float(rng.choice([0.03, 0.05, 0.1, 0.2])), "max_leaf_nodes": int(rng.choice([15, 31, 63, 127])),
             "min_samples_leaf": int(rng.choice([20, 50, 100, 200])), "l2_regularization": float(rng.choice([0.0, 0.1, 1.0])),
             "class_weight": [None, "balanced"][int(rng.integers(0, 2))]} for _ in range(16)]
    splits = list(folds(y, 1))
    _, _, train0, _, _ = splits[0]
    fit_idx, val_idx = train_test_split(train0, test_size=0.2, stratify=y[train0], random_state=0)
    trials = []
    for i, cfg in enumerate(grid):
        gb = HistGradientBoostingClassifier(max_iter=500, early_stopping=True, random_state=0, **cfg).fit(X01[fit_idx], y[fit_idx])
        trials.append({"model": "gradient boosting", **cfg, "class_weight": str(cfg["class_weight"]),
                       "val_ap": float(average_precision_score(y[val_idx], gb.predict_proba(X01[val_idx])[:, 1])), "iterations": int(gb.n_iter_)})
        print(trials[-1], flush=True)
    best = grid[int(np.argmax([t["val_ap"] for t in trials]))]
    lr_trials = [{"C": C, "val_ap": float(average_precision_score(y[val_idx], logit(X01, y, fit_idx, val_idx, C=C)))} for C in (0.01, 0.1, 1.0, 10.0)]
    best_c = max(lr_trials, key=lambda t: t["val_ap"])["C"]
    rows = []
    for rep, fold, train, test, _ in splits:
        gb = HistGradientBoostingClassifier(max_iter=500, early_stopping=True, random_state=fold, **best).fit(X01[train], y[train])
        rows.append({"method": "gradient boosting (tuned)", "rep": rep, "fold": fold, **metrics(gb.predict_proba(X01[test])[:, 1], y[test])})
        rows.append({"method": "logistic regression (tuned)", "rep": rep, "fold": fold, **metrics(logit(X01, y, train, test, C=best_c), y[test])})
        pl.DataFrame(rows).write_csv(out_dir / "tuned.csv")
    json.dump({"gradient_boosting_trials": trials, "gradient_boosting_best": {**best, "class_weight": str(best["class_weight"])},
               "logistic_trials": lr_trials, "logistic_best_C": best_c,
               "protocol": "16 random configurations and 4 values of C, chosen by average precision on a 20% validation split of the "
                           "training part of the first fold; the chosen settings are then used in all five folds"},
              open(out_dir / "tuned_meta.json", "w"), indent=1)

# ------------------------------------------------------------------------------------------------ gnn
elif task == "gnn":
    import torch

    torch.manual_seed(0)
    torch.set_num_threads(THREADS)
    COLS = ["ts", "from_bank", "from_acct", "to_bank", "to_acct", "amt_recv", "cur_recv", "amt_paid", "cur_paid", "fmt", "label"]
    edges = (pl.scan_csv(sys.argv[4], has_header=True, new_columns=COLS, schema_overrides={"from_bank": pl.Utf8, "to_bank": pl.Utf8})
             .select(src=pl.col("from_bank") + "_" + pl.col("from_acct"), dst=pl.col("to_bank") + "_" + pl.col("to_acct"))
             .filter(pl.col("src") != pl.col("dst")).unique().collect())
    index = {a: i for i, a in enumerate(acc["account"].to_list())}
    s_idx = torch.tensor([index[a] for a in edges["src"].to_list()])
    d_idx = torch.tensor([index[a] for a in edges["dst"].to_list()])

    def mean_adj(rows_, cols_):
        deg = torch.bincount(rows_, minlength=n).clamp(min=1).float()
        return torch.sparse_coo_tensor(torch.stack([rows_, cols_]), 1.0 / deg[rows_], (n, n)).coalesce()

    A_in, A_out = mean_adj(d_idx, s_idx), mean_adj(s_idx, d_idx)  # mean over payers, mean over payees
    Xt = torch.tensor(X01, dtype=torch.float32)

    class Sage(torch.nn.Module):
        def __init__(self, d, h=64):
            super().__init__()
            self.l1, self.l2, self.out = torch.nn.Linear(3 * d, h), torch.nn.Linear(3 * h, h), torch.nn.Linear(h, 1)
            self.drop = torch.nn.Dropout(0.2)

        def forward(self, x):
            h = torch.relu(self.l1(torch.cat([x, torch.sparse.mm(A_in, x), torch.sparse.mm(A_out, x)], 1)))
            h = self.drop(h)
            h = torch.relu(self.l2(torch.cat([h, torch.sparse.mm(A_in, h), torch.sparse.mm(A_out, h)], 1)))
            return self.out(self.drop(h)).squeeze(1)

    rows = []
    for rep, fold, train, test, _ in folds(y, 1):
        fit_idx, val_idx = train_test_split(train, test_size=0.1, stratify=y[train], random_state=fold)
        torch.manual_seed(fold)
        model = Sage(m)
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        yt = torch.tensor(y, dtype=torch.float32)
        pos_weight = torch.tensor((1 - y[fit_idx].mean()) / y[fit_idx].mean())
        best_ap, best_scores, best_epoch = -1.0, None, 0
        for epoch in range(1, 201):
            model.train()
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(Xt)[fit_idx], yt[fit_idx], pos_weight=pos_weight)
            loss.backward()
            opt.step()
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    scores = model(Xt).numpy()
                ap = average_precision_score(y[val_idx], scores[val_idx])
                if ap > best_ap:
                    best_ap, best_scores, best_epoch = ap, scores.copy(), epoch
        rows.append({"method": "GraphSAGE (account level)", "rep": rep, "fold": fold, "best_epoch": best_epoch, **metrics(best_scores[test], y[test])})
        pl.DataFrame(rows).write_csv(out_dir / "gnn.csv")
        print(rows[-1], f"{time.time() - t0:.0f}s", flush=True)
    json.dump({"torch": torch.__version__, "edges": int(len(s_idx)), "architecture": "2 GraphSAGE layers with separate mean aggregation "
               "over payers and payees, hidden size 64, dropout 0.2, Adam lr 0.01, weight decay 5e-4, 200 full-batch epochs, class-weighted "
               "loss, model state chosen by average precision on 10% of the training accounts"}, open(out_dir / "gnn_meta.json", "w"), indent=1)

# ------------------------------------------------------------------------------------------------ weights
elif task == "weights":
    rng = np.random.default_rng(1)
    pos = np.flatnonzero(y == 1)
    idx = np.sort(np.concatenate([pos, rng.choice(np.flatnonzero(y == 0), 100_000 - len(pos), replace=False)]))
    ys = y[idx]
    rows = []
    for extra in (0, m, 2 * m):
        Xr = np.column_stack([X_raw[idx], rng.random((len(idx), extra))])
        Xs = zs.rank01(Xr)
        order = np.argsort(-zrl.attribute_relevance(Xr, "entropy"), kind="stable")  # no labels
        noise_rank = float(np.mean(np.argsort(order)[m:])) + 1 if extra else float("nan")
        for law in ("equal", "zeta", "geometric", "linear"):
            w = law_weights(order, law, m + extra)
            p = zs.fit(Xs, w, k=K, seed=0, threads=THREADS, checks=False)
            F = np.column_stack([p.profile, p.deviation])
            for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(Xs, ys)):
                clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", solver="newton-cholesky"))
                sc = clf.fit(F[tr], ys[tr]).predict_proba(F[te])[:, 1]
                rows.append({"noise_attributes": extra, "law": law, "fold": fold, "auc": roc_auc_score(ys[te], sc),
                             "ap": average_precision_score(ys[te], sc), "weight_on_noise": float(w[m:].sum()),
                             "effective_attributes": float(1 / (w**2).sum()), "mean_rank_of_noise": noise_rank})
            pl.DataFrame(rows).write_csv(out_dir / "weights.csv")
            print(f"noise {extra} law {law} done, {time.time() - t0:.0f}s", flush=True)

# ------------------------------------------------------------------------------------------------ temporal
elif task == "temporal":
    later = pl.read_parquet(sys.argv[4])  # account, flagged_later
    y_later = acc.join(later, on="account", how="left")["flagged_later"].fill_null(0).to_numpy().astype(int)
    candidates = np.flatnonzero(y == 0)          # not flagged in the past period
    target = y_later[candidates]
    train = np.arange(n)                         # every account of the past period, with its past label
    print(f"past accounts {n}, flagged in the past {y.sum()}, candidates {len(candidates)}, first flagged later {target.sum()} "
          f"({target.mean():.3%})", flush=True)
    part, F = summarise(law_weights(ENT_ORDER, "zeta"))
    scores = {"ZRL-U profiles": logit(F, y, train, candidates), "ZRL-U deviation (no labels)": F[candidates, -1],
              "logistic regression": logit(X01, y, train, candidates),
              "gradient boosting": HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X01, y).predict_proba(X01[candidates])[:, 1]}
    rng = np.random.default_rng(0)
    rows = []
    for name, sc in scores.items():
        boot = []
        for _ in range(500):
            b = rng.integers(0, len(candidates), len(candidates))
            if target[b].sum() > 0:
                boot.append(roc_auc_score(target[b], sc[b]))
        rows.append({"method": name, **metrics(sc, target), "auc_lo": float(np.percentile(boot, 2.5)), "auc_hi": float(np.percentile(boot, 97.5))})
        print(rows[-1], flush=True)
    pl.DataFrame(rows).write_csv(out_dir / "temporal.csv")
    json.dump({"past_accounts": n, "flagged_past": int(y.sum()), "candidates": int(len(candidates)), "flagged_later": int(target.sum()),
               **{"conc_" + k: v for k, v in concentration(part.labels[candidates], target).items()}}, open(out_dir / "temporal_meta.json", "w"))

else:
    sys.exit(f"unknown task {task}")
print("done", f"{time.time() - t0:.0f}s", flush=True)
