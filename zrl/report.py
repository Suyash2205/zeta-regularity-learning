"""Turn the result files into the paper's LaTeX tables, figures and a summary JSON.

Usage: python report.py <results_dir> <out_dir>

Reads results_hi, results_li, results_hm (scores.csv from experiment_scale.py and scores_rev.csv from
revision.py), results_extra and results_rev. Missing pieces are skipped. Light: small files only.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
import zrl  # noqa: E402

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
DATASETS = [("hi", "HI-Small"), ("li", "LI-Small"), ("hm", "HI-Medium")]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#2a78d6", "#0d366b"])
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def read(path, **kw):
    return pl.read_csv(path, infer_schema_length=None, **kw) if Path(path).exists() else None


old = {k: read(src / f"results_{k}" / "scores.csv", schema_overrides={"value": pl.Utf8}) for k, _ in DATASETS}
rev = {k: read(src / f"results_{k}" / "scores_rev.csv") for k, _ in DATASETS}
extra = {k: {t: read(src / "results_rev" / k / f"{t}.csv") for t in ("tuned", "gnn", "strict", "temporal", "weights")} for k, _ in DATASETS}
present = [k for k, _ in DATASETS if rev[k] is not None and old[k] is not None]
NAME = dict(DATASETS)
summary = {"datasets": {}, "folds": {}}
tex = {}

# label, where it comes from, method id
METHODS = [
    ("Gradient boosting, tuned", "tuned", "gradient boosting (tuned)"),
    ("Gradient boosting, default settings", "old", "gradient boosting (all attributes)"),
    ("GraphSAGE on the transaction graph", "gnn", "GraphSAGE (account level)"),
    ("Logistic regression, tuned", "tuned", "logistic regression (tuned)"),
    ("Logistic regression, default settings", "old", "logistic regression (all attributes)"),
    ("ZRL-U profiles (label-free summary)", "rev", "ZRL-U/profile_logit"),
    ("ZRL-MI profiles (label-ranked weights)", "rev", "ZRL-MI (label-ranked)/profile_logit"),
    ("ZRL profiles, equal weights", "old", "ZRL (equal weights)/profile_logit"),
    ("Profiles on a $k$-means partition", "rev", "k-means partition/profile_logit"),
    ("Profiles on a random partition", "rev", "random partition/profile_logit"),
    ("ZRL-U class risk", "rev", "ZRL-U/class_risk"),
    ("$k$-means class risk", "rev", "k-means partition/class_risk"),
    ("ZRL-U deviation (no labels)", "rev", "ZRL-U/deviation"),
    ("Isolation forest (no labels)", "old", "isolation forest (no labels)"),
]
BREAK_AFTER = {4, 9, 11}
METRICS = [("auc", "AUC"), ("ap", "AP"), ("recall_top1", "R@1\\%"), ("recall_top5", "R@5\\%")]
MAIN = "ZRL-U profiles (label-free summary)"


def rows_of(key, origin, method):
    df = {"old": old[key], "rev": rev[key]}.get(origin)
    if origin in ("tuned", "gnn"):
        df = extra[key][origin]
    if df is None:
        return None
    d = df.filter(pl.col("method") == method)
    if origin == "old" and "sweep" in d.columns and method.startswith("ZRL (equal"):
        d = d.filter(pl.col("sweep") == "weights")
    return d.sort(["rep", "fold"]) if d.height else None


def corrected_t(a, b):
    """Corrected resampled t-test for repeated k-fold results (test/train ratio 0.25)."""
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    if n < 2 or d.var(ddof=1) == 0:
        return float(d.mean()), float("nan"), int((d > 0).sum()), n
    t = d.mean() / np.sqrt((1 / n + 0.25) * d.var(ddof=1))
    return float(d.mean()), float(2 * stats.t.sf(abs(t), n - 1)), int((d > 0).sum()), n


def ptxt(p):
    return "--" if np.isnan(p) else ("$<.001$" if p < 0.001 else f"{p:.3f}".lstrip("0"))


def fit_width(block, width):
    return block.replace("\\begin{tabular}", "\\resizebox{" + width + "}{!}{\\begin{tabular}").replace("\\end{tabular}", "\\end{tabular}}")


# ------------------------------------------------------------------ main table
lines = ["\\begin{table*}[t]", "\\caption{Detection of laundering accounts on held-out accounts (mean over test folds, standard deviation in parentheses)}",
         "\\label{tab:main}", "\\centering", "\\footnotesize", "\\begin{tabular}{@{}l" + "cccc" * len(present) + "@{}}", "\\toprule",
         " & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{NAME[k]}}}" for k in present) + " \\\\",
         " ".join(f"\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(present))),
         "Method & " + " & ".join(" & ".join(m for _, m in METRICS) for _ in present) + " \\\\", "\\midrule"]
for i, (label, origin, method) in enumerate(METHODS):
    cells = []
    for k in present:
        d = rows_of(k, origin, method)
        summary["datasets"].setdefault(k, {})[label] = None if d is None else {
            **{m: [float(d[m].mean()), float(d[m].std() or 0.0)] for m, _ in METRICS}, "folds": d.height}
        for m, _ in METRICS:
            cells.append("--" if d is None else (f"{d[m].mean():.3f} ({d[m].std() or 0.0:.3f})" if m == "auc" else f"{d[m].mean():.3f}"))
    lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    if i in BREAK_AFTER:
        lines.append("\\addlinespace")
lines += ["\\bottomrule", "\\end{tabular}",
          "\\par\\vspace{3pt}\\parbox{0.98\\textwidth}{\\footnotesize \\emph{Note.} AUC = area under the ROC curve; AP = average precision; R@1\\% "
          "and R@5\\% = share of the held-out laundering accounts among the 1\\% and 5\\% highest-scored held-out accounts. Fifteen test folds "
          "on HI-Small and LI-Small and five on HI-Medium, except the tuned models and GraphSAGE, which use the five folds of the first "
          "repeat and were not run on HI-Medium. Profiles are scored by a logistic regression on the 64-class profile and the deviation; "
          "class risk scores an account by the laundering rate of its class among training accounts.}", "\\end{table*}"]
tex["tab:main"] = fit_width("\n".join(lines), "\\textwidth")
summary["folds"] = {k: int(rev[k].select(pl.struct("rep", "fold").n_unique()).item()) for k in present}

# ------------------------------------------------------- paired significance
lines = ["\\begin{table}[t]", "\\caption{Paired comparisons with ZRL-U profiles (AUC), corrected resampled $t$-test}", "\\label{tab:sig}",
         "\\centering", "\\footnotesize", "\\begin{tabular}{@{}llrrr@{}}", "\\toprule", "Data & Compared method & Diff. & Wins & $p$ \\\\", "\\midrule"]
summary["paired"] = {}
for k in present:
    main_df = rows_of(k, "rev", "ZRL-U/profile_logit")
    for label, origin, method in METHODS:
        if label == MAIN or "class risk" in label or "deviation" in label or "Isolation" in label:
            continue
        d = rows_of(k, origin, method)
        if d is None:
            continue
        j = main_df.join(d, on=["rep", "fold"], suffix="_o")
        diff, p, wins, n = corrected_t(j["auc"].to_numpy(), j["auc_o"].to_numpy())
        summary["paired"].setdefault(k, {})[label] = {"diff": diff, "p": p, "wins": wins, "n": n}
        short = label.replace(" on the transaction graph", "").replace(" settings", "").replace(" (label-ranked weights)", "")
        lines.append(f"{NAME[k]} & {short} & {diff:+.3f} & {wins}/{n} & {ptxt(p)} \\\\")
    lines.append("\\addlinespace")
lines += ["\\bottomrule", "\\end{tabular}",
          "\\par\\vspace{3pt}\\parbox{0.98\\columnwidth}{\\footnotesize \\emph{Note.} Diff.\\ is the AUC of ZRL-U profiles minus that of the "
          "compared method on the same test folds; Wins counts the folds in which ZRL-U profiles are ahead.}", "\\end{table}"]
tex["tab:sig"] = fit_width("\n".join(lines), "\\columnwidth")

# ----------------------------------------------- summary accuracy and concentration
lines = ["\\begin{table}[t]", "\\caption{Accuracy of 64-class summaries and concentration of laundering accounts in the classes}",
         "\\label{tab:structure}", "\\centering", "\\footnotesize", "\\begin{tabular}{@{}llrrrrr@{}}", "\\toprule",
         " & & & & \\multicolumn{2}{c}{Share of laundering in} & \\\\", "\\cmidrule(lr){5-6}",
         "Data & Partition & RMSE & Irreg. & top 12.5\\% & top 25\\% & Max.\\ lift \\\\", "\\midrule"]
summary["structure"] = {}
LAB = {"ZRL-U": "Regularity (ZRL-U)", "ZRL-MI (label-ranked)": "Regularity (ZRL-MI)", "k-means partition": "$k$-means", "random partition": "Random"}
for k in present:
    st, co = read(src / f"results_{k}" / "structure_rev.csv"), read(src / f"results_{k}" / "concentration.csv")
    if st is None or co is None:
        continue
    for method in ("ZRL-U", "ZRL-MI (label-ranked)", "k-means partition", "random partition"):
        s_, c_ = st.filter(pl.col("method") == method).to_dicts(), co.filter(pl.col("method") == method).to_dicts()
        if not s_ or not c_:
            continue
        s_, c_ = s_[0], c_[0]
        summary["structure"].setdefault(k, {})[method] = {**{x: s_[x] for x in ("rmse", "rmse_se", "irregular_fraction", "index")}, **c_}
        lines.append(f"{NAME[k]} & {LAB[method]} & {s_['rmse']:.4f} & {s_['irregular_fraction']:.3f} & {c_['share_top12'] * 100:.1f}\\% & "
                     f"{c_['share_top25'] * 100:.1f}\\% & {c_['max_lift']:.1f} \\\\")
    lines.append("\\addlinespace")
lines += ["\\bottomrule", "\\end{tabular}",
          "\\par\\vspace{3pt}\\parbox{0.98\\columnwidth}{\\footnotesize \\emph{Note.} RMSE = root-mean-square error of replacing each edge "
          "weight by its class-pair density, estimated on two million random pairs (standard error below $2\\times10^{-5}$). Irreg.\\ = share "
          "of class pairs found irregular at $\\varepsilon=0.05$ on 100 sampled accounts per class. Concentration: classes are ordered by "
          "laundering rate and the share of all laundering accounts in the first 12.5\\% and 25\\% of accounts is reported (descriptive, all "
          "accounts); a random partition gives 12.5\\% and 25\\%. Max.\\ lift = highest class rate over the overall rate.}", "\\end{table}"]
tex["tab:structure"] = fit_width("\n".join(lines), "\\columnwidth")

# ------------------------------------------------------------------ seeds
seeds = read(src / "results_rev" / "seeds.csv")
if seeds is not None:
    cols = [("rmse", "RMSE", 4), ("irregular_fraction", "Irregular pairs", 3), ("profile_auc", "Profile AUC (5 folds)", 4),
            ("share_top25", "Laundering in top 25\\%", 3), ("max_lift", "Max.\\ lift", 2)]
    summary["seeds"] = {c: {"mean": float(seeds[c].mean()), "sd": float(seeds[c].std()), "min": float(seeds[c].min()), "max": float(seeds[c].max())}
                        for c, _, _ in cols}
    summary["seeds"]["n"] = seeds.height
    lines = ["\\begin{table}[t]", f"\\caption{{Variation over {seeds.height} landmark seeds (HI-Small, ZRL-U)}}", "\\label{tab:seeds}", "\\centering",
             "\\footnotesize", "\\begin{tabular}{@{}lrrrr@{}}", "\\toprule", "Quantity & Mean & SD & Min & Max \\\\", "\\midrule"]
    for c, lab, dg in cols:
        v = summary["seeds"][c]
        lines.append(f"{lab} & {v['mean']:.{dg}f} & {v['sd']:.{dg}f} & {v['min']:.{dg}f} & {v['max']:.{dg}f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    tex["tab:seeds"] = "\n".join(lines)

# ------------------------------------------------------------------ temporal
if any(extra[k]["temporal"] is not None for k in present):
    lines = ["\\begin{table}[t]", "\\caption{Time-based test: accounts first flagged after the cut-off, scored with attributes and labels from before it}",
             "\\label{tab:temporal}", "\\centering", "\\footnotesize", "\\begin{tabular}{@{}llrrr@{}}", "\\toprule",
             "Data & Method & AUC (95\\% interval) & AP & R@5\\% \\\\", "\\midrule"]
    summary["temporal"] = {}
    for k in present:
        t = extra[k]["temporal"]
        if t is None:
            continue
        meta = json.load(open(src / "results_rev" / k / "temporal_meta.json"))
        summary["temporal"][k] = {"meta": meta, "rows": t.to_dicts()}
        for r in t.to_dicts():
            lab = r["method"].replace(" (no labels)", "")
            lines.append(f"{NAME[k]} & {lab[0].upper() + lab[1:]} & "
                         f"{r['auc']:.3f} ({r['auc_lo']:.3f}--{r['auc_hi']:.3f}) & {r['ap']:.3f} & {r['recall_top5']:.3f} \\\\")
        lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\par\\vspace{3pt}\\parbox{0.98\\columnwidth}{\\footnotesize \\emph{Note.} Attributes, graph and training labels use only the "
              "first 70\\% of the transactions in time. Scored accounts are those not flagged in that period; the target is being flagged in the "
              "remaining 30\\%. Intervals are bootstrap percentiles over scored accounts (500 resamples).}", "\\end{table}"]
    tex["tab:temporal"] = fit_width("\n".join(lines), "\\columnwidth")

# ------------------------------------------------------------------ strict labels
if "hi" in present and extra["hi"]["strict"] is not None:
    t = extra["hi"]["strict"]
    meta = json.load(open(src / "results_rev" / "hi" / "strict_meta.json"))
    agg = t.group_by("method").agg(auc=pl.col("auc").mean(), sd=pl.col("auc").std(), ap=pl.col("ap").mean(), r5=pl.col("recall_top5").mean())
    summary["strict"] = {"meta": meta, "rows": agg.to_dicts()}
    order_ = ["gradient boosting", "logistic regression", "ZRL-U/profile_logit", "ZRL-U/class_risk", "ZRL-U/deviation"]
    labs = ["Gradient boosting, default", "Logistic regression, default", "ZRL-U profiles", "ZRL-U class risk", "ZRL-U deviation (no labels)"]
    lines = ["\\begin{table}[t]", "\\caption{Stricter label: accounts with at least two flagged transactions (HI-Small, five folds)}",
             "\\label{tab:strict}", "\\centering", "\\footnotesize", "\\begin{tabular}{@{}lrrr@{}}", "\\toprule", "Method & AUC (SD) & AP & R@5\\% \\\\", "\\midrule"]
    for mth, lab in zip(order_, labs):
        r = agg.filter(pl.col("method") == mth).to_dicts()
        if r:
            lines.append(f"{lab} & {r[0]['auc']:.3f} ({r[0]['sd']:.3f}) & {r[0]['ap']:.3f} & {r[0]['r5']:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              f"\\par\\vspace{{3pt}}\\parbox{{0.98\\columnwidth}}{{\\footnotesize \\emph{{Note.}} {meta['positives']:,} accounts "
              f"({meta['share'] * 100:.2f}\\%) meet the stricter label.}}", "\\end{table}"]
    tex["tab:strict"] = "\n".join(lines)

# ------------------------------------------------------------------ weight laws under noise
if "hi" in present and extra["hi"]["weights"] is not None:
    wl = extra["hi"]["weights"]
    agg = wl.group_by("noise_attributes", "law").agg(auc=pl.col("auc").mean(), sd=pl.col("auc").std(), ap=pl.col("ap").mean(),
                                                    wn=pl.col("weight_on_noise").mean(), neff=pl.col("effective_attributes").mean(),
                                                    noise_rank=pl.col("mean_rank_of_noise").mean()).sort("noise_attributes", "law")
    summary["weights"] = agg.to_dicts()
    levels = sorted(agg["noise_attributes"].unique().to_list())
    LAWS = [("equal", "Equal ($s=0$)"), ("zeta", "Zeta ($s=1$)"), ("geometric", "Geometric ($0.85^{k}$)"), ("linear", "Linear in rank")]
    lines = ["\\begin{table}[t]", "\\caption{Weight laws when pure-noise attributes are added (HI-Small sample of 100,000 accounts, label-free ranking, five folds)}",
             "\\label{tab:noise}", "\\centering", "\\footnotesize", "\\begin{tabular}{@{}l" + "rr" * len(levels) + "@{}}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{lv} noise attr.}}" for lv in levels) + " \\\\",
             " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(levels))),
             "Weight law & " + " & ".join("AUC (SD) & On noise" for _ in levels) + " \\\\", "\\midrule"]
    for law, lab in LAWS:
        cells = []
        for lv in levels:
            r = agg.filter((pl.col("noise_attributes") == lv) & (pl.col("law") == law)).to_dicts()[0]
            cells += [f"{r['auc']:.3f} ({r['sd']:.3f})", f"{r['wn'] * 100:.0f}\\%"]
        lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\par\\vspace{3pt}\\parbox{0.98\\columnwidth}{\\footnotesize \\emph{Note.} Uniform random attributes are appended to the 30 real "
              "ones and all attributes are ranked by entropy, without labels. On noise = share of the total weight that falls on the random "
              "attributes.}", "\\end{table}"]
    tex["tab:noise"] = fit_width("\n".join(lines), "\\columnwidth")
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for (law, lab), color in zip(LAWS, SERIES):
        d = agg.filter(pl.col("law") == law).sort("noise_attributes")
        ax.errorbar(d["noise_attributes"], d["auc"], yerr=d["sd"], color=color, lw=2, marker="o", ms=5, capsize=3,
                    label=lab.replace("$", "").replace("^{k}", "^k"))
    ax.set_xlabel("Pure-noise attributes added to the 30 real ones")
    ax.set_ylabel("Held-out ROC AUC")
    ax.set_xticks(levels)
    ax.legend(frameon=False, loc="lower left", fontsize=8.5)
    fig.savefig(out / "fig_noise.png")
    plt.close(fig)

# ------------------------------------------------------------------ tuning details
for k in present:
    f = src / "results_rev" / k / "tuned_meta.json"
    if f.exists():
        tm = json.load(open(f))
        summary.setdefault("tuning", {})[k] = {"best": tm["gradient_boosting_best"], "logistic_best_C": tm["logistic_best_C"],
                                              "trials": len(tm["gradient_boosting_trials"])}
    f = src / "results_rev" / k / "gnn_meta.json"
    if f.exists():
        summary.setdefault("gnn", {})[k] = json.load(open(f))

# -------------------------------------------- sensitivity and ablation (label-ranked variant, first run)
if "hi" in present and old["hi"].filter(pl.col("sweep") == "s").height:
    s0 = old["hi"].filter(pl.col("rep") == 0)
    st = read(src / "results_hi" / "structure.csv", schema_overrides={"value": pl.Utf8}).filter(pl.col("rep") == 0)
    m = 30
    lines = ["\\begin{table}[t]", "\\caption{Sensitivity to the parameters (HI-Small, five folds, label-ranked weights)}", "\\label{tab:sens}",
             "\\centering", "\\footnotesize", "\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule",
             "Parameter & Value & $N_{\\mathrm{eff}}$ & RMSE & AUC & AP & R@5\\% \\\\", "\\midrule"]
    summary["sensitivity"] = []

    def sens_row(param, value, sweep, sval, s_exp, q):
        mth = "ZRL (equal weights)" if sweep == "weights" else "ZRL"
        d = s0.filter((pl.col("method") == mth + "/profile_logit") & (pl.col("sweep") == sweep) & (pl.col("value") == sval))
        e = st.filter((pl.col("method") == mth) & (pl.col("sweep") == sweep) & (pl.col("value") == sval))
        if not d.height:
            return
        w = zrl.zeta_weights(m, s_exp, q)
        row = {"param": param, "value": value, "neff": 1.0 / float((w**2).sum()), "rmse": float(np.sqrt(e["mse"].mean())),
               "auc": float(d["auc"].mean()), "sd": float(d["auc"].std()), "ap": float(d["ap"].mean()), "recall_top5": float(d["recall_top5"].mean())}
        summary["sensitivity"].append(row)
        lines.append(f"{param} & {value} & {row['neff']:.1f} & {row['rmse']:.4f} & {row['auc']:.3f} ({row['sd']:.3f}) & {row['ap']:.3f} & "
                     f"{row['recall_top5']:.3f} \\\\")

    sens_row("$s$", "0", "weights", "equal", 0.0, 1.0)
    sens_row("$s$", "0.5", "s", "0.5", 0.5, 1.0)
    sens_row("$s$", "1", "main", "main", 1.0, 1.0)
    for v in ("1.5", "2.0", "3.0"):
        sens_row("$s$", v.rstrip("0").rstrip("."), "s", v, float(v), 1.0)
    lines.append("\\addlinespace")
    for v in ("2.0", "4.0"):
        sens_row("$q$", v[0], "q", v, 1.0, float(v))
    lines.append("\\addlinespace")
    for v in ("16", "32"):
        sens_row("$K$", v, "k", v, 1.0, 1.0)
    sens_row("$K$", "64", "main", "main", 1.0, 1.0)
    for v in ("128", "256"):
        sens_row("$K$", v, "k", v, 1.0, 1.0)
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\par\\vspace{3pt}\\parbox{0.98\\columnwidth}{\\footnotesize \\emph{Note.} One parameter is varied at a time from $s=1$, $q=1$, "
              "$K=64$. $N_{\\mathrm{eff}}$ is the effective number of attributes. These sweeps were run with the label-ranked weights (ZRL-MI). "
              "RMSE values with different weights refer to different graphs.}", "\\end{table}"]
    tex["tab:sens"] = "\n".join(lines)

    main_o = s0.filter((pl.col("method") == "ZRL/profile_logit") & (pl.col("sweep") == "main")).sort("fold")
    lines = ["\\begin{table}[t]", "\\caption{Attribute ablations (HI-Small, five folds, label-ranked weights)}", "\\label{tab:ablation}", "\\centering",
             "\\footnotesize", "\\begin{tabular}{@{}lrrrr@{}}", "\\toprule", "Variant & AUC & AP & R@5\\% & Diff.\\ ($p$) \\\\", "\\midrule"]
    summary["ablation"] = {}
    for label, method, sweep in (("All 30 attributes", "ZRL/profile_logit", "main"),
                                 ("Without non-backtracking centrality", "ZRL (without non-backtracking centrality)/profile_logit", "attributes"),
                                 ("Behavioural attributes only", "ZRL (behavioural attributes only)/profile_logit", "attributes")):
        d = s0.filter((pl.col("method") == method) & (pl.col("sweep") == sweep)).sort("fold")
        if not d.height:
            continue
        diff, p, wins, n = corrected_t(d["auc"].to_numpy(), main_o["auc"].to_numpy())
        summary["ablation"][label] = {"auc": float(d["auc"].mean()), "ap": float(d["ap"].mean()), "recall_top5": float(d["recall_top5"].mean()),
                                      "diff": diff, "p": p}
        dtxt = "--" if sweep == "main" else f"{diff:+.3f} ({ptxt(p).strip('$')})"
        lines.append(f"{label} & {d['auc'].mean():.3f} & {d['ap'].mean():.3f} & {d['recall_top5'].mean():.3f} & {dtxt} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    tex["tab:ablation"] = fit_width("\n".join(lines), "\\columnwidth")

    pts = [r for r in summary["sensitivity"] if r["param"] == "$s$"]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    xs = [float(r["value"]) for r in pts]
    ax.errorbar(xs, [r["auc"] for r in pts], yerr=[r["sd"] for r in pts], color=SERIES[0], lw=2, marker="o", ms=6, capsize=3)
    for x, r in zip(xs, pts):
        ax.annotate(f"{r['auc']:.3f}", (x, r["auc"]), textcoords="offset points", xytext=(6, 8), fontsize=8.5, color=INK2)
    ax.set_xlabel("Zeta exponent s")
    ax.set_ylabel("Held-out ROC AUC")
    ax.set_ylim(min(r["auc"] - r["sd"] for r in pts) - 0.01, max(r["auc"] + r["sd"] for r in pts) + 0.012)
    fig.savefig(out / "fig_auc_vs_s.png")
    plt.close(fig)

# ------------------------------------------------------------------ scalability
raw = read(src / "results_extra" / "scalability.csv")
if raw is not None:
    sc = raw.group_by("n").agg(edges=pl.col("edges").first(), fit_seconds=pl.col("fit_seconds").mean(), fit_sd=pl.col("fit_seconds").std().fill_null(0.0),
                               refine_seconds=pl.col("refine_seconds").mean(), summary_seconds=pl.col("summary_seconds").mean(),
                               matvec_seconds=pl.col("matvec_seconds").mean(), peak_rss_gb=pl.col("peak_rss_gb").max(),
                               dense_float32_gb=pl.col("dense_float32_gb").first(), runs=pl.len()).sort("n")
    summary["scalability"] = sc.to_dicts()
    slope = float(np.polyfit(np.log(sc["n"].to_numpy()), np.log(sc["fit_seconds"].to_numpy()), 1)[0])
    summary["scalability_slope"] = slope
    lines = ["\\begin{table}[t]", "\\caption{Cost of one complete fit as the number of accounts grows}", "\\label{tab:scale}", "\\centering",
             "\\footnotesize", "\\begin{tabular}{@{}rrrrrrr@{}}", "\\toprule",
             " & & \\multicolumn{3}{c}{Time (s)} & \\multicolumn{2}{c}{Memory (GB)} \\\\", "\\cmidrule(lr){3-5} \\cmidrule(lr){6-7}",
             "Accounts & Edges & Total (sd) & Refine & Summary & Peak & Dense $W$ \\\\", "\\midrule"]
    for r in sc.to_dicts():
        mant, expo = f"{r['edges']:.2e}".split("e")
        lines.append(f"{r['n']:,} & ${mant}\\times10^{{{int(expo)}}}$ & {r['fit_seconds']:.0f} ({r['fit_sd']:.1f}) & {r['refine_seconds']:.0f} & "
                     f"{r['summary_seconds']:.0f} & {r['peak_rss_gb']:.1f} & {r['dense_float32_gb']:,.0f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              f"\\par\\vspace{{3pt}}\\parbox{{0.98\\columnwidth}}{{\\footnotesize \\emph{{Note.}} 30 attributes, $K=64$, 512 landmarks, 8 threads. "
              f"Each size is run {int(sc['runs'].max())} times, each in a fresh process on an otherwise idle machine, so peak memory belongs to that "
              f"run alone; behaviour under concurrent load was not measured. Dense $W$ is the memory the graph would need in single precision. "
              f"The fitted log--log slope of total time against accounts is {slope:.2f}.}}", "\\end{table}"]
    tex["tab:scale"] = fit_width("\n".join(lines), "\\columnwidth")
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.loglog(sc["n"], sc["fit_seconds"], color=SERIES[0], lw=2, marker="o", ms=6)
    ax.set_xlabel("Number of accounts n")
    ax.set_ylabel("Time for one complete fit (s)")
    ax.annotate(f"slope {slope:.2f}", (sc["n"][len(sc) // 2], sc["fit_seconds"][len(sc) // 2]), textcoords="offset points", xytext=(10, -14),
                fontsize=9, color=INK2)
    fig.savefig(out / "fig_scalability.png")
    plt.close(fig)

# ------------------------------------------------------- reduced graph and classes (ZRL-U)
fit_file = src / "results_hi" / "full_fit_rev.npz"
if fit_file.exists():
    f = np.load(fit_file)
    mt = json.load(open(src / "results_hi" / "meta_rev.json"))
    feats = mt["features"]
    summary["meta_rev"] = {k: {x: v for x, v in json.load(open(src / f"results_{k}" / "meta_rev.json")).items() if x != "features"} for k in present}
    R, sizes, pos, cm = f["R"], f["sizes"], f["positives"], f["class_mean"]
    rate = pos / sizes
    base = pos.sum() / sizes.sum()
    order = np.argsort(-rate)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5), gridspec_kw={"width_ratios": [1.15, 1]})
    im = axes[0].imshow(R[np.ix_(order, order)], cmap=BLUES, interpolation="nearest")
    axes[0].grid(False)
    axes[0].set_xlabel("Class (ordered by laundering rate)")
    axes[0].set_ylabel("Class")
    cb = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    cb.set_label("Class-pair density", color=INK2)
    cb.outline.set_visible(False)
    axes[1].bar(range(len(R)), rate[order] * 100, color=SERIES[0], width=0.8)
    axes[1].axhline(base * 100, color=MUTED, lw=1.2, ls="--")
    axes[1].annotate(f"overall rate {base * 100:.2f}%", (len(R) - 1, base * 100), textcoords="offset points", xytext=(0, 5), ha="right",
                     fontsize=8.5, color=INK2)
    axes[1].set_xlabel("Class (ordered by laundering rate)")
    axes[1].set_ylabel("Laundering accounts in class (%)")
    axes[1].grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out / "fig_reduced_graph.png")
    plt.close(fig)
    cum = np.cumsum(pos[order]) / pos.sum()
    summary["classes"] = {"k": int(len(R)), "base_rate": float(base), "classes_ge_2x": int((rate >= 2 * base).sum()),
                          "classes_zero": int((pos == 0).sum()), "share_in_top8": float(cum[7]), "share_in_top16": float(cum[15]),
                          "max_rate": float(rate.max())}
    names = {"nbt_centrality": "non-backtracking centrality", "degree": "partners", "n_currencies": "currencies used",
             "cross_cur_share": "cross-currency share", "pagerank_in": "PageRank (receiver)", "pagerank_out": "PageRank (sender)",
             "in_partners": "paying partners", "out_partners": "receiving partners", "n_out": "payments sent", "n_in": "payments received",
             "partner_degree": "partners' degree", "reciprocity": "reciprocity", "active_days": "active days",
             "cross_bank_share": "cross-bank share", "out_ratio": "share sent", "repeat_out": "payments per partner",
             "share_ach": "ACH share", "share_bitcoin": "Bitcoin share", "share_cheque": "cheque share", "share_wire": "wire share",
             "share_cash": "cash share", "share_credit_card": "credit-card share", "share_reinvestment": "reinvestment share",
             "self_share": "self-transfer share", "night_share": "night share", "out_amt_mean": "mean amount sent",
             "out_amt_std": "spread of amount sent", "out_amt_max": "largest amount sent", "in_amt_mean": "mean amount received",
             "in_amt_std": "spread of amount received"}
    summary["weights_top"] = [[names.get(a, a), w] for a, w in mt["top_weights"]]
    lines = ["\\begin{table*}[t]", "\\caption{The five classes with the highest laundering rates (HI-Small, ZRL-U, all accounts, descriptive)}",
             "\\label{tab:classes}", "\\centering", "\\footnotesize", "\\begin{tabular}{@{}crrrrl@{}}", "\\toprule",
             "Class & Accounts & Laundering & Rate & Lift & Most distinctive attributes (class mean of the scaled rank; overall mean 0.50) \\\\", "\\midrule"]
    summary["top_classes"] = []
    for i, c in enumerate(order[:5]):
        top = np.argsort(-np.abs(cm[c] - 0.5))[:3]
        traits = "; ".join(f"{names.get(feats[j], feats[j])} ({cm[c][j]:.2f})" for j in top)
        summary["top_classes"].append({"size": int(sizes[c]), "positives": int(pos[c]), "rate": float(rate[c]), "lift": float(rate[c] / base),
                                       "traits": traits})
        lines.append(f"{'ABCDE'[i]} & {int(sizes[c]):,} & {int(pos[c])} & {rate[c] * 100:.1f}\\% & {rate[c] / base:.1f} & {traits} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    tex["tab:classes"] = "\n".join(lines)

# ------------------------------------------------------------------ near-tied relevance
ties = read(src / "results_rev" / "ties.csv")
if ties is not None:
    meta = json.load(open(src / "results_rev" / "ties_meta.json"))
    summary["ties"] = {"meta": meta, "rows": ties.to_dicts()}
    short = {"nbt_centrality": "non-backtracking centr.", "n_out": "payments sent", "out_partners": "receiving partners",
             "pagerank_out": "PageRank (sender)", "degree": "partners"}
    lines = ["\\begin{table}[t]", f"\\caption{{Order of the {meta['near_tied']} attributes whose label-free relevance is nearly tied (HI-Small, ZRL-U)}}",
             "\\label{tab:ties}", "\\centering", "\\footnotesize", "\\begin{tabular}{@{}llrrr@{}}", "\\toprule",
             "Order & Largest weight on & RMSE & AUC & Top 25\\% \\\\", "\\midrule"]
    for r in ties.to_dicts():
        first = "shared" if r["order"] == "shared weight" else short.get(r["first_attribute"], r["first_attribute"].replace("_", " "))
        lines.append(f"{r['order'].capitalize()} & {first} & {r['rmse']:.4f} & {r['profile_auc']:.4f} & {r['share_top25'] * 100:.1f}\\% \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\par\\vspace{3pt}\\parbox{0.98\\columnwidth}{\\footnotesize \\emph{Note.} Near-tied = relevance within 0.001 of the highest. "
              "Shared weight = the near-tied attributes all receive the mean of the weights of the ranks they span. AUC of the profile "
              "classifier, five folds.}", "\\end{table}"]
    tex["tab:ties"] = fit_width("\n".join(lines), "\\columnwidth")

ORDER = ["tab:structure", "tab:seeds", "tab:main", "tab:sig", "tab:temporal", "tab:strict", "tab:noise", "tab:ties", "tab:sens", "tab:ablation", "tab:scale",
         "tab:classes"]
(out / "tables.tex").write_text("\n\n".join(tex[k] for k in ORDER if k in tex) + "\n")
json.dump(summary, open(out / "summary.json", "w"), indent=1)
print("tables", [k for k in ORDER if k in tex], "datasets", present, "folds", summary["folds"])
