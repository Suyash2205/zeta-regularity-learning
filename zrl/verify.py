"""Check the published results without rerunning the experiments.

Usage
  python zrl/verify.py quick
      Checksums of every result file, the paper's tables and summary rebuilt from the result files, and every
      number quoted in the paper looked up in that summary. About ten seconds; needs no data.
  python zrl/verify.py dumps <hi|li|hm> [repeats]
      Takes the saved class assignment of every partition (results/results_*/partitions_rev.npz) and the account
      attributes, and recomputes from them the reduced graph, the reconstruction error, the class concentration,
      the irregularity sample and the cross-validated AUC and AP of the profile classifier, class risk and deviation,
      then compares each with the published value. Skips the slow refinement that produced the partitions.
      repeats: how many of the three fold repeats to recompute (default 1). HI-Small takes about five minutes
      and 3 GB of memory; LI-Small about seven minutes and 4 GB; HI-Medium needs about 16 GB.
  python zrl/verify.py spot <hi|li> <piece> [args]
      Reruns one piece from scratch and compares it with the published value. Pieces:
        partition            the ZRL-U partition, from the raw attributes (about 2.5 minutes on HI-Small)
        seed <0-4>           the partition with another landmark seed (HI-Small; Table of seeds)
        kmeans               the k-means partition
        balanced             the balanced k-means partition (HI-Small)
        fold <rep> <fold>    one test fold of logistic regression, gradient boosting (default and tuned) and, on
                             HI-Small, spline logistic regression, each trained from scratch
      The ZRL partitions are not bit-identical across machines (different BLAS libraries round differently), so
      a rerun is compared with tolerances: on the machine used for this check, 99.96% of the accounts of
      HI-Small landed in the same class and the error agreed to six decimals.

The account attributes are rebuilt from the IBM transactions files by ./reproduce.sh, or downloaded ready-made
from the GitHub release of this repository (results/DATA.sha256 lists them; verify.sh fetches them).
Exit status is non-zero if any comparison fails.
"""
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
TOL = {"auc": 0.002, "ap": 0.005, "rmse": 2e-4, "share": 1e-6, "index": 1e-5, "R": 1e-4, "irregular": 0.01,
       # the profile classifier, per fold and for the mean over the folds. Measured on a laptop and a Linux cloud
       # machine (see VERIFY.md): up to 0.013 AUC in one fold, 0.0044 for the mean of five folds.
       "auc_profile": 0.015, "ap_profile": 0.025, "auc_mean": 0.005, "ap_mean": 0.01}
failures = []


def check(name, got, want, tol):
    ok = got == want if tol is None else abs(float(got) - float(want)) <= tol
    if not ok:
        failures.append(name)
    shown = f"{got}" if tol is None else f"{float(got):.6g} vs {float(want):.6g}"
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: {shown}", flush=True)
    return ok


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def finish():
    if failures:
        sys.exit(f"\n{len(failures)} check(s) failed: {failures}")
    print("\nall checks passed")


# ---------------------------------------------------------------------------------------------------- quick
def canonical(x):
    """Lists whose order carries no meaning (rows of a group-by) are compared as sorted lists."""
    if isinstance(x, dict):
        return {k: canonical(v) for k, v in x.items()}
    if isinstance(x, list):
        items = [canonical(v) for v in x]
        return sorted(items, key=lambda v: json.dumps(v, sort_keys=True)) if items and isinstance(items[0], dict) else items
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    return x


def lookup(summary, path):
    node = summary
    for part in re.findall(r"\[[^\]]*\]|[^/]+", path):
        if part.startswith("["):
            key, value = part[1:-1].split("=", 1)
            node = next(r for r in node if str(r[key]) == value)
        elif isinstance(node, list):
            node = node[int(part)]
        else:
            node = node[part]
    return node


def quick():
    print("1. checksums of the result files (results/MANIFEST.sha256)")
    lines = (RES / "MANIFEST.sha256").read_text().split("\n")
    bad = [f for h, f in (ln.split("  ", 1) for ln in lines if ln) if not (RES / f).exists() or sha(RES / f) != h]
    check(f"{len(lines) - 1} files unchanged", len(bad), 0, None)
    for f in bad[:10]:
        print("     changed or missing:", f)

    print("2. tables and summary rebuilt from the result files by zrl/report.py")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, str(ROOT / "zrl" / "report.py"), str(RES), tmp], check=True, capture_output=True)
        check("tables.tex identical to results/paper/tables.tex", (Path(tmp) / "tables.tex").read_bytes() == (RES / "paper" / "tables.tex").read_bytes(), True, None)
        rebuilt = json.load(open(Path(tmp) / "summary.json"))
    published = json.load(open(RES / "paper" / "summary.json"))
    check("summary.json identical to results/paper/summary.json (row order aside)", canonical(rebuilt) == canonical(published), True, None)

    print("3. numbers quoted in the paper, looked up in the rebuilt summary (results/paper/paper_numbers.csv)")
    rows = list(csv.DictReader(open(RES / "paper" / "paper_numbers.csv")))
    wrong = 0
    for r in rows:
        decimals = len(r["printed"].split(".")[1]) if "." in r["printed"] else 0
        value = float(lookup(rebuilt, r["source"])) * float(r["scale"])
        if f"{value:.{decimals}f}" != r["printed"]:
            wrong += 1
            print(f"     {r['claim']}: paper says {r['printed']}, results give {value:.{decimals + 2}f}")
    check(f"{len(rows)} quoted numbers match the results", wrong, 0, None)
    finish()


# ---------------------------------------------------------------------------------------------------- data
def load(short):
    """Import revision.py with the 'load' task: the same data loading and helpers as the published run."""
    path = ROOT / "data" / f"{short}_accounts.parquet"
    want = dict(reversed(ln.split("  ", 1)) for ln in (RES / "DATA.sha256").read_text().split("\n") if ln)
    if not path.exists():
        sys.exit(f"{path} is missing: run ./verify.sh {sys.argv[1]} {short}, which downloads it, or ./reproduce.sh, which builds it")
    got = sha(path)
    print(f"attributes {path.name}: sha256 {got[:16]}...")
    if f"data/{path.name}" in want and want[f"data/{path.name}"] != got:
        print("  note: this file differs from the published one; results may differ slightly")
    sys.argv = ["revision.py", str(path), tempfile.mkdtemp(), "load"]
    sys.path.insert(0, str(ROOT / "zrl"))
    import revision
    return revision


def read_csv(path):
    import polars as pl
    return pl.read_csv(path, infer_schema_length=None)


# ---------------------------------------------------------------------------------------------------- dumps
PARTITIONS = {"zrl_u": "ZRL-U", "zrl_mi": "ZRL-MI (label-ranked)", "k_means": "k-means partition", "random": "random partition"}


def dumps(short, repeats):
    import numpy as np
    t0 = time.time()
    R = load(short)
    zs = R.zs
    d = RES / f"results_{short}"
    meta = json.load(open(d / "meta_rev.json"))
    print("1. data and splits")
    check("accounts", R.n, meta["n"], None)
    check("laundering accounts", int(R.y.sum()), meta["positives"], None)
    check("attribute order (entropy)", [R.FEATS[i] for i in R.ENT_ORDER], meta["entropy_order"], None)
    saved = np.load(d / "folds.npz")
    check("labels identical to the saved ones", bool((saved["label"] == R.y).all()), True, None)
    for rep, fold, train, test, rank in R.folds(R.y, saved["fold_of"].shape[0]):
        if not (saved["fold_of"][rep, test] == fold).all() or not (saved["fold_of"][rep, rank] == -1).all():
            check(f"fold {rep}/{fold} identical to folds.npz", False, True, None)
            break
    else:
        check(f"all {saved['fold_of'].shape[0] * 5} test folds identical to folds.npz", True, True, None)

    w_ent = R.law_weights(R.ENT_ORDER, "zeta")
    _, _, _, _, rank_idx = next(R.folds(R.y, 1))
    mi = zs.binned_mutual_information(R.X01[rank_idx], R.y[rank_idx])
    check("attribute order (mutual information)", [R.FEATS[i] for i in np.argsort(-mi, kind="stable")], meta["mi_order"], None)
    w_mi = R.law_weights(np.argsort(-mi, kind="stable"), "zeta")
    full = np.load(d / "full_fit_rev.npz")
    check("zeta weights", float(np.abs(full["weights"] - w_ent).max()), 0.0, 1e-12)

    parts = dict(np.load(d / "partitions_rev.npz"))
    structure, conc = read_csv(d / "structure_rev.csv"), read_csv(d / "concentration.csv")
    scores = read_csv(d / "scores_rev.csv")
    if short == "hi" and (RES / "results_rev" / "hi" / "fair_partition.npz").exists():
        parts["balanced_k_means"] = np.load(RES / "results_rev" / "hi" / "fair_partition.npz")["balanced_k_means"]
        PARTITIONS["balanced_k_means"] = "balanced k-means partition"
        fr = RES / "results_rev" / "hi"
        import polars as pl
        structure = pl.concat([structure.select("method", "rmse", "irregular_fraction", "index"),
                               read_csv(fr / "fair_structure.csv").select("method", "rmse", "irregular_fraction", "index")])
        conc = pl.concat([conc, read_csv(fr / "fair_concentration.csv")])
        scores = pl.concat([scores, read_csv(fr / "fair.csv").filter(pl.col("method").str.starts_with("balanced"))], how="diagonal")

    for step, (key, labels) in enumerate(parts.items(), start=2):
        name = PARTITIONS[key]
        print(f"{step}. {name}: summary of the saved partition ({time.time() - t0:.0f}s)")
        labels = labels.astype(np.int32)
        w = w_mi if key == "zrl_mi" else w_ent
        p = zs.fit(R.X01, w, k=R.K, labels=labels, threads=R.THREADS, checks=False)
        G = zs.L1Graph(R.X01, w, threads=R.THREADS)
        srow = structure.filter(structure["method"] == name).row(0, named=True)
        check("reconstruction error (RMSE)", R.rmse_with_error(G, labels, p.R)[0], srow["rmse"], TOL["rmse"])
        check("index", p.index, srow["index"], TOL["index"])
        # The published fit drew the 512 landmarks from the same generator before the irregularity sample.
        rng = np.random.default_rng(0)
        if key in ("zrl_u", "zrl_mi"):
            rng.choice(R.n, min(512, R.n), replace=False)
        check("share of irregular class pairs (sampled)", zs.sampled_checks(G, labels, p.R, 0.05, rng)[0], srow["irregular_fraction"], TOL["irregular"])
        crow = conc.filter(conc["method"] == name).row(0, named=True)
        got = R.concentration(labels, R.y)
        check("class concentration (8 figures)", all(abs(got[k] - crow[k]) <= TOL["share"] for k in got), True, None)
        if key == "zrl_u":
            check("reduced graph R (largest entry difference)", float(np.abs(p.R - full["R"]).max()), 0.0, TOL["R"])
            check("class sizes", np.bincount(labels, minlength=R.K).tolist(), full["sizes"].tolist(), None)
        F = np.column_stack([p.profile, p.deviation.astype(np.float32)])
        rows = []
        R.score_partition(rows, name, p, F, R.y, repeats)
        # The profile classifier is a logistic regression whose solver often stops at its iteration limit, and
        # where it stops depends on the BLAS library and threads; it is compared with the wider tolerance.
        for r in rows:
            r["loose"] = r["method"].endswith("profile_logit")
        for loose in (False, True):
            sel = [r for r in rows if r["loose"] == loose]
            if not sel:
                continue
            diff = {m: [r[m] - scores.filter((scores["method"] == r["method"]) & (scores["rep"] == r["rep"]) & (scores["fold"] == r["fold"]))[m][0]
                        for r in sel] for m in ("auc", "ap")}
            label = f"{len(sel)} fold scores" + (" of the profile classifier" if loose else " of class risk and deviation")
            suffix = "_profile" if loose else ""
            for m in ("auc", "ap"):
                check(f"{m.upper()} of {label}, largest difference", max(map(abs, diff[m])), 0.0, TOL[m + suffix])
                if loose:
                    check(f"{m.upper()} of {label}, mean difference", float(np.mean(diff[m])), 0.0, TOL[m + "_mean"])
        del p, F, G
        print(f"   peak memory so far {zs.check_memory():.1f} GB", flush=True)

    pred = d / "predictions_rep0.npz"
    if pred.exists():
        print(f"{step + 1}. AUC of the first-round baselines from their saved predictions (repeat 0)")
        P, old = np.load(pred), read_csv(d / "scores.csv")
        for key, name in (("logistic_regression", "logistic regression (all attributes)"), ("gradient_boosting", "gradient boosting (all attributes)")):
            worst = 0.0
            for fold in range(5):
                test = np.flatnonzero(saved["fold_of"][0] == fold)
                pub = old.filter((old["method"] == name) & (old["rep"] == 0) & (old["fold"] == fold))["auc"][0]
                worst = max(worst, abs(R.metrics(P[key][test].astype(np.float64), R.y[test])["auc"] - pub))
            check(f"{name}, largest fold difference (scores stored in half precision)", worst, 0.0, TOL["auc"])
    print(f"finished in {time.time() - t0:.0f}s")
    finish()


# ---------------------------------------------------------------------------------------------------- spot
def spot(short, piece, args):
    import numpy as np
    import polars as pl
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import adjusted_rand_score
    t0 = time.time()
    R = load(short)
    zs = R.zs
    d = RES / f"results_{short}"
    w_ent = R.law_weights(R.ENT_ORDER, "zeta")

    def compare_partition(p, saved, name, structure_row):
        same = float((p.labels == saved).mean())
        print(f"  accounts in the same class as the saved partition: {same:.4%} (adjusted Rand index {adjusted_rand_score(saved, p.labels):.4f})")
        rmse = R.rmse_with_error(zs.L1Graph(R.X01, w_ent, threads=R.THREADS), p.labels, p.R)[0]
        check(f"{name} RMSE", rmse, structure_row["rmse"], 5e-4)
        return rmse

    if piece in ("partition", "kmeans", "balanced"):
        key = {"partition": "zrl_u", "kmeans": "k_means", "balanced": "balanced_k_means"}[piece]
        if piece == "partition":
            p, _ = R.summarise(w_ent)
        else:
            if piece == "kmeans":
                from sklearn.cluster import MiniBatchKMeans
                labels = MiniBatchKMeans(R.K, n_init=3, batch_size=8192, random_state=0).fit_predict(R.X01 * np.sqrt(w_ent)).astype(np.int32)
            else:
                labels = R.balanced_kmeans(w_ent)
            p, _ = R.summarise(w_ent, labels)
        print(f"rebuilt in {time.time() - t0:.0f}s")
        if piece == "balanced":
            saved = np.load(RES / "results_rev" / "hi" / "fair_partition.npz")["balanced_k_means"]
            srow = read_csv(RES / "results_rev" / "hi" / "fair_structure.csv").row(0, named=True)
            scores = read_csv(RES / "results_rev" / "hi" / "fair.csv")
        else:
            saved = np.load(d / "partitions_rev.npz")[key]
            srow = read_csv(d / "structure_rev.csv").filter(pl.col("method") == PARTITIONS[key]).row(0, named=True)
            scores = read_csv(d / "scores_rev.csv")
        name = srow["method"]
        compare_partition(p, saved, name, srow)
        pub = scores.filter((pl.col("method") == f"{name}/profile_logit") & (pl.col("rep") == 0))["auc"].mean()
        F = np.column_stack([p.profile, p.deviation.astype(np.float32)])
        got = float(np.mean([R.metrics(R.logit(F, R.y, tr, te), R.y[te])["auc"] for _, _, tr, te, _ in R.folds(R.y, 1)]))
        check(f"{name} profile AUC, mean of the five folds of repeat 0", got, pub, TOL["auc_mean"])
    elif piece == "seed":
        seed = int(args[0])
        p, F = R.summarise(w_ent, seed=seed)
        print(f"rebuilt in {time.time() - t0:.0f}s")
        row = read_csv(RES / "results_rev" / "seeds.csv").filter(pl.col("seed") == seed).row(0, named=True)
        check(f"seed {seed} RMSE", R.rmse_with_error(zs.L1Graph(R.X01, w_ent, threads=R.THREADS), p.labels, p.R)[0], row["rmse"], 5e-4)
        got = float(np.mean([R.metrics(R.logit(F, R.y, tr, te), R.y[te])["auc"] for _, _, tr, te, _ in R.folds(R.y, 1)]))
        check(f"seed {seed} profile AUC (five folds)", got, row["profile_auc"], TOL["auc_mean"])
        check(f"seed {seed} laundering share in the top quarter", R.concentration(p.labels, R.y)["share_top25"], row["share_top25"], 0.01)
    elif piece == "fold":
        rep, fold = int(args[0]), int(args[1])
        _, _, train, test, _ = list(R.folds(R.y, rep + 1))[rep * 5 + fold]
        old = read_csv(d / "scores.csv")
        pub = lambda frame, method: frame.filter((pl.col("method") == method) & (pl.col("rep") == rep) & (pl.col("fold") == fold))["auc"][0]  # noqa: E731
        check("logistic regression, default settings", R.metrics(R.logit(R.X01, R.y, train, test), R.y[test])["auc"],
              pub(old, "logistic regression (all attributes)"), TOL["auc"])
        gb = HistGradientBoostingClassifier(max_iter=200, random_state=fold).fit(R.X01[train], R.y[train])
        check("gradient boosting, default settings", R.metrics(gb.predict_proba(R.X01[test])[:, 1], R.y[test])["auc"],
              pub(old, "gradient boosting (all attributes)"), TOL["auc"])
        rev = RES / "results_rev" / short
        if rep == 0 and (rev / "tuned.csv").exists():
            tm = json.load(open(rev / "tuned_meta.json"))
            best = {**tm["gradient_boosting_best"], "class_weight": None if tm["gradient_boosting_best"]["class_weight"] == "None" else "balanced"}
            tuned = read_csv(rev / "tuned.csv")
            gb = HistGradientBoostingClassifier(max_iter=500, early_stopping=True, random_state=fold, **best).fit(R.X01[train], R.y[train])
            check("gradient boosting, tuned settings", R.metrics(gb.predict_proba(R.X01[test])[:, 1], R.y[test])["auc"],
                  pub(tuned, "gradient boosting (tuned)"), TOL["auc"])
            check("logistic regression, tuned C", R.metrics(R.logit(R.X01, R.y, train, test, C=tm["logistic_best_C"]), R.y[test])["auc"],
                  pub(tuned, "logistic regression (tuned)"), TOL["auc"])
        if (rev / "fair.csv").exists():
            from sklearn.preprocessing import SplineTransformer
            S_ = SplineTransformer(n_knots=5, degree=3, knots="uniform", include_bias=False).fit(np.vstack([np.zeros(R.m), np.ones(R.m)])).transform(R.X01).astype(np.float32)
            check("logistic regression, spline-expanded attributes", R.metrics(R.logit(S_, R.y, train, test), R.y[test])["auc"],
                  pub(read_csv(rev / "fair.csv"), "logistic regression (spline-expanded attributes)"), TOL["auc"])
    else:
        sys.exit(f"unknown piece {piece}; see the usage at the top of zrl/verify.py")
    print(f"finished in {time.time() - t0:.0f}s, peak memory {zs.check_memory():.1f} GB")
    finish()


def manifest():
    """Maintainers: record the checksum of every result file (run after the results change)."""
    files = sorted(f for f in RES.rglob("*") if f.is_file() and f.name != "MANIFEST.sha256" and f.name != ".DS_Store")
    (RES / "MANIFEST.sha256").write_text("".join(f"{sha(f)}  {f.relative_to(RES).as_posix()}\n" for f in files))
    print(f"wrote results/MANIFEST.sha256 ({len(files)} files)")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("quick", "dumps", "spot", "manifest"):
        sys.exit(__doc__)
    mode = sys.argv[1]
    if mode == "manifest":
        manifest()
    elif mode == "quick":
        quick()
    elif mode == "dumps":
        dumps(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1)
    else:
        spot(sys.argv[2], sys.argv[3], sys.argv[4:])
