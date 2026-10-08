"""Turn the result files of the scale experiments into LaTeX tables, figures and a summary JSON.

Usage: python report.py <cloud_results_dir> <out_dir>
Expects results_hi, results_li, results_hm and results_extra inside the first directory;
datasets that are missing are skipped. Light: reads small CSV files only.
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

scores, structure, meta, fits = {}, {}, {}, {}
for key, _ in DATASETS:
    d = src / f"results_{key}"
    if (d / "scores.csv").exists():
        scores[key] = pl.read_csv(d / "scores.csv", schema_overrides={"value": pl.Utf8}, infer_schema_length=None)
        structure[key] = pl.read_csv(d / "structure.csv", schema_overrides={"value": pl.Utf8}, infer_schema_length=None)
    if (d / "meta.json").exists():
        meta[key] = json.load(open(d / "meta.json"))
        fits[key] = np.load(d / "full_fit.npz")
summary = {"datasets": {}, "folds": {k: int(v.select(pl.struct("rep", "fold").n_unique()).item()) for k, v in scores.items()}}
tex = []

METHODS = [  # (label, method id, sweep filter or None, uses labels in scoring)
    ("Gradient boosting, all attributes", "gradient boosting (all attributes)", None),
    ("Gradient boosting, behavioural attributes", "gradient boosting (behavioural attributes)", None),
    ("Logistic regression, all attributes", "logistic regression (all attributes)", None),
    ("ZRL profiles (zeta weights, regularity partition)", "ZRL/profile_logit", "main"),
    ("ZRL profiles, equal weights", "ZRL (equal weights)/profile_logit", None),
    ("Profiles on a $k$-means partition", "k-means partition/profile_logit", None),
    ("Profiles on a random partition", "random equal-sized partition/profile_logit", None),
    ("ZRL class risk", "ZRL/class_risk", "main"),
    ("$k$-means class risk", "k-means partition/class_risk", None),
    ("ZRL deviation (no labels)", "ZRL/deviation", "main"),
    ("Isolation forest (no labels)", "isolation forest (no labels)", None),
]
METRICS = [("auc", "AUC"), ("ap", "AP"), ("recall_top1", "R@1\\%"), ("recall_top5", "R@5\\%")]


def rows_of(df, method, sweep=None):
    d = df.filter(pl.col("method") == method)
    if sweep:
        d = d.filter(pl.col("sweep") == sweep)
    return d.sort(["rep", "fold"])


def fmt(x, digits=3):
    return f"{x:.{digits}f}"


def corrected_t(a, b):
    """Nadeau-Bengio corrected resampled t-test for repeated k-fold results (test/train = 1/4)."""
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    if n < 2 or d.var(ddof=1) == 0:
        return float(d.mean()), float("nan"), float("nan"), int((d > 0).sum()), n
    t = d.mean() / np.sqrt((1 / n + 0.25) * d.var(ddof=1))
    return float(d.mean()), float(t), float(2 * stats.t.sf(abs(t), n - 1)), int((d > 0).sum()), n


# ---------------------------------------------------------------- main table
lines = ["\\begin{table*}[t]", "\\caption{Detection of laundering accounts on held-out accounts (mean over folds, standard deviation in parentheses)}",
         "\\label{tab:main}", "\\centering", "\\footnotesize",
         "\\begin{tabular}{@{}l" + "cccc" * len(scores) + "@{}}", "\\toprule",
         " & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{name}}}" for key, name in DATASETS if key in scores) + " \\\\",
         " ".join(f"\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(scores))),
         "Method & " + " & ".join(" & ".join(m for _, m in METRICS) for key, _ in DATASETS if key in scores) + " \\\\", "\\midrule"]
for i, (label, method, sweep) in enumerate(METHODS):
    cells = []
    for key, _ in DATASETS:
        if key not in scores:
            continue
        d = rows_of(scores[key], method, sweep)
        summary["datasets"].setdefault(key, {})[label] = {m: [float(d[m].mean()), float(d[m].std() or 0.0)] for m, _ in METRICS} if d.height else None
        for m, _ in METRICS:
            cells.append(f"{fmt(d[m].mean())} ({fmt(d[m].std() or 0.0)})" if m == "auc" and d.height else (fmt(d[m].mean()) if d.height else "--"))
    lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    if i in (2, 6, 8):
        lines.append("\\addlinespace")
lines += ["\\bottomrule", "\\end{tabular}",
          "\\par\\vspace{3pt}\\parbox{0.98\\textwidth}{\\footnotesize \\emph{Note.} AUC = area under the ROC curve; AP = average precision; "
          "R@1\\% and R@5\\% = share of the held-out laundering accounts found among the 1\\% and 5\\% highest-scored held-out accounts. "
          "Profiles are scored by a logistic regression on the 64-class profile and the deviation; class risk scores an account by the "
          "laundering rate of its class among training accounts.}", "\\end{table*}"]
tex.append("\n".join(lines))

# ------------------------------------------------------- paired significance
if "hi" in scores:
    lines = ["\\begin{table}[t]", "\\caption{Paired comparisons with ZRL profiles (AUC), corrected resampled $t$-test}", "\\label{tab:sig}",
             "\\centering", "\\footnotesize", "\\begin{tabular}{@{}llrrr@{}}", "\\toprule",
             "Data & Compared method & Diff. & Wins & $p$ \\\\", "\\midrule"]
    summary["paired"] = {}
    for key, name in DATASETS:
        if key not in scores:
            continue
        main = rows_of(scores[key], "ZRL/profile_logit", "main")["auc"].to_numpy()
        for label, method, sweep in METHODS:
            if method in ("ZRL/profile_logit",) or "class_risk" in method or "deviation" in method or "isolation" in method:
                continue
            other = rows_of(scores[key], method, sweep)["auc"].to_numpy()
            if len(other) != len(main):
                continue
            diff, t, p, wins, n = corrected_t(main, other)
            summary["paired"].setdefault(key, {})[label] = {"diff": diff, "t": t, "p": p, "wins": wins, "n": n}
            ptxt = "--" if np.isnan(p) else ("$<.001$" if p < 0.001 else f"{p:.3f}".lstrip("0"))
            lines.append(f"{name} & {label.replace(', all attributes', '').replace(', behavioural attributes', ' (behav.)')} & "
                         f"{diff:+.3f} & {wins}/{n} & {ptxt} \\\\")
        lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\par\\vspace{3pt}\\parbox{0.98\\columnwidth}{\\footnotesize \\emph{Note.} Diff.\\ is the AUC of ZRL profiles minus that of the "
              "compared method; Wins counts the folds in which ZRL profiles are ahead.}", "\\end{table}"]
    tex.append("\n".join(lines))

# ---------------------------------------------------------- partition quality
lines = ["\\begin{table}[t]", "\\caption{Quality of 64-class summaries of the zeta-weighted graph}", "\\label{tab:structure}", "\\centering",
         "\\footnotesize", "\\begin{tabular}{@{}llrrr@{}}", "\\toprule", "Data & Partition & RMSE & Irreg. & Index \\\\", "\\midrule"]
summary["structure"] = {}
for key, name in DATASETS:
    if key not in structure:
        continue
    st = structure[key]
    for label, method in (("Regularity (ZRL)", "ZRL"), ("$k$-means", "k-means partition"), ("Random", "random equal-sized partition")):
        d = st.filter((pl.col("method") == method) & pl.col("sweep").is_in(["main", "partition"]))
        if not d.height:
            continue
        rmse, irr, idx, sec = float(np.sqrt(d["mse"].mean())), float(d["irregular_fraction"].mean()), float(d["index"].mean()), float(d["seconds"].mean())
        summary["structure"].setdefault(key, {})[method] = {"rmse": rmse, "irregular": irr, "index": idx, "seconds": sec}
        lines.append(f"{name} & {label} & {rmse:.4f} & {irr:.3f} & {idx:.4f} \\\\")
    lines.append("\\addlinespace")
lines += ["\\bottomrule", "\\end{tabular}",
          "\\par\\vspace{3pt}\\parbox{0.98\\columnwidth}{\\footnotesize \\emph{Note.} RMSE = root-mean-square error of replacing each edge weight by "
          "its class-pair density, estimated on two million random pairs; the bound of Proposition~\\ref{prop:existence} for $r=6$ is "
          f"{np.pi / np.sqrt(6):.3f}. Irreg.\\ = share of class pairs found irregular at $\\varepsilon=0.05$ on a stratified sample. "
          "Index = size-weighted mean of squared class-pair densities.}", "\\end{table}"]
tex.append("\n".join(lines))

# --------------------------------------------------------------- sensitivity
if "hi" in scores and scores["hi"].filter(pl.col("sweep") == "s").height:
    s, st = scores["hi"], structure["hi"]
    folds0 = s.filter((pl.col("rep") == 0))
    lines = ["\\begin{table}[t]", "\\caption{Sensitivity of ZRL to its parameters (HI-Small, five folds)}", "\\label{tab:sens}", "\\centering",
             "\\footnotesize", "\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule",
             "Parameter & Value & $N_{\\mathrm{eff}}$ & RMSE & AUC & AP & R@5\\% \\\\", "\\midrule"]
    m = meta["hi"]["m"] if "hi" in meta else 30
    summary["sensitivity"] = []

    def sens_row(param, value, sweep, sval, s_exp, q, k):
        d = folds0.filter((pl.col("method") == ("ZRL (equal weights)/profile_logit" if sweep == "weights" else "ZRL/profile_logit"))
                          & (pl.col("sweep") == sweep) & ((pl.col("value") == sval) if sval else pl.lit(True)))
        e = st.filter((pl.col("rep") == 0) & (pl.col("method") == ("ZRL (equal weights)" if sweep == "weights" else "ZRL"))
                      & (pl.col("sweep") == sweep) & ((pl.col("value") == sval) if sval else pl.lit(True)))
        if not d.height:
            return
        w = zrl.zeta_weights(m, s_exp, q)
        neff = 1.0 / float((w**2).sum())
        summary["sensitivity"].append({"param": param, "value": value, "neff": neff, "rmse": float(np.sqrt(e["mse"].mean())),
                                       "auc": float(d["auc"].mean()), "sd": float(d["auc"].std()), "ap": float(d["ap"].mean()),
                                       "recall_top5": float(d["recall_top5"].mean())})
        lines.append(f"{param} & {value} & {neff:.1f} & {np.sqrt(e['mse'].mean()):.4f} & {d['auc'].mean():.3f} ({d['auc'].std():.3f}) & "
                     f"{d['ap'].mean():.3f} & {d['recall_top5'].mean():.3f} \\\\")

    sens_row("$s$", "0", "weights", "equal", 0.0, 1.0, 64)
    sens_row("$s$", "0.5", "s", "0.5", 0.5, 1.0, 64)
    sens_row("$s$", "1", "main", "main", 1.0, 1.0, 64)
    for v in ("1.5", "2.0", "3.0"):
        sens_row("$s$", v.rstrip("0").rstrip("."), "s", v, float(v), 1.0, 64)
    lines.append("\\addlinespace")
    for v in ("2.0", "4.0"):
        sens_row("$q$", v[0], "q", v, 1.0, float(v), 64)
    lines.append("\\addlinespace")
    for v in ("16", "32"):
        sens_row("$K$", v, "k", v, 1.0, 1.0, int(v))
    sens_row("$K$", "64", "main", "main", 1.0, 1.0, 64)
    for v in ("128", "256"):
        sens_row("$K$", v, "k", v, 1.0, 1.0, int(v))
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\par\\vspace{3pt}\\parbox{0.98\\columnwidth}{\\footnotesize \\emph{Note.} One parameter is varied at a time from the main "
              "configuration ($s=1$, $q=1$, $K=64$). $N_{\\mathrm{eff}}$ is the effective number of attributes. RMSE values with different "
              "weights refer to different graphs and are not comparable across rows of $s$ and $q$.}", "\\end{table}"]
    tex.append("\n".join(lines))

    # ablations
    lines = ["\\begin{table}[t]", "\\caption{Ablations of ZRL (HI-Small, five folds)}", "\\label{tab:ablation}", "\\centering", "\\footnotesize",
             "\\begin{tabular}{@{}lrrrr@{}}", "\\toprule", "Variant & AUC & AP & R@5\\% & Diff.\\ ($p$) \\\\", "\\midrule"]
    main = folds0.filter((pl.col("method") == "ZRL/profile_logit") & (pl.col("sweep") == "main")).sort("fold")
    summary["ablation"] = {}
    variants = [("ZRL, main configuration", "ZRL/profile_logit", "main"),
                ("Equal weights ($s=0$)", "ZRL (equal weights)/profile_logit", "weights"),
                ("Entropy ranking (no labels in weights)", "ZRL (entropy ranking)/profile_logit", "ranking"),
                ("Without non-backtracking centrality", "ZRL (without non-backtracking centrality)/profile_logit", "attributes"),
                ("Behavioural attributes only", "ZRL (behavioural attributes only)/profile_logit", "attributes"),
                ("$k$-means partition", "k-means partition/profile_logit", "partition"),
                ("Random partition", "random equal-sized partition/profile_logit", "partition")]
    for label, method, sweep in variants:
        d = folds0.filter((pl.col("method") == method) & (pl.col("sweep") == sweep)).sort("fold")
        if not d.height:
            continue
        diff, t, p, wins, n = corrected_t(d["auc"].to_numpy(), main["auc"].to_numpy())
        summary["ablation"][label] = {"auc": float(d["auc"].mean()), "ap": float(d["ap"].mean()), "recall_top5": float(d["recall_top5"].mean()),
                                      "diff": diff, "p": p, "wins_for_variant": wins}
        dtxt = "--" if method == "ZRL/profile_logit" else f"{diff:+.3f} ({'<.001' if p < 0.001 else f'{p:.3f}'.lstrip('0')})"
        lines.append(f"{label} & {d['auc'].mean():.3f} & {d['ap'].mean():.3f} & {d['recall_top5'].mean():.3f} & {dtxt} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\par\\vspace{3pt}\\parbox{0.98\\columnwidth}{\\footnotesize \\emph{Note.} Diff.\\ is the AUC of the variant minus that of the "
              "main configuration, with the $p$-value of the corrected resampled $t$-test over five folds.}", "\\end{table}"]
    tex.append("\n".join(lines))

    # figure: AUC against s
    pts = [r for r in summary["sensitivity"] if r["param"] == "$s$"]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    xs = [float(r["value"]) for r in pts]
    ax.errorbar(xs, [r["auc"] for r in pts], yerr=[r["sd"] for r in pts], color=SERIES[0], lw=2, marker="o", ms=6, capsize=3)
    for x, r in zip(xs, pts):
        ax.annotate(f"{r['auc']:.3f}", (x, r["auc"]), textcoords="offset points", xytext=(6, 8), fontsize=8.5, color=INK2)
    ax.set_xlabel("Zeta exponent s")
    ax.set_ylabel("Held-out ROC AUC")
    lo = min(r["auc"] - r["sd"] for r in pts)
    ax.set_ylim(lo - 0.01, max(r["auc"] + r["sd"] for r in pts) + 0.012)
    fig.savefig(out / "fig_auc_vs_s.png")
    plt.close(fig)

# --------------------------------------------------------------------- noise
noise_file = src / "results_extra" / "noise.csv"
if noise_file.exists():
    nz = pl.read_csv(noise_file)
    agg = nz.group_by("noise_attributes", "method").agg(auc=pl.col("auc").mean(), sd=pl.col("auc").std(), ap=pl.col("ap").mean(),
                                                         wn=pl.col("weight_on_noise").mean()).sort("noise_attributes", "method")
    summary["noise"] = agg.to_dicts()
    levels = sorted(agg["noise_attributes"].unique().to_list())
    lines = ["\\begin{table}[t]", "\\caption{Robustness to pure-noise attributes (HI-Small sample of 100,000 accounts, five folds)}",
             "\\label{tab:noise}", "\\centering", "\\footnotesize", "\\begin{tabular}{@{}rrrrrr@{}}", "\\toprule",
             " & \\multicolumn{2}{c}{Weight on noise} & \\multicolumn{3}{c}{AUC (standard deviation)} \\\\",
             "\\cmidrule(lr){2-3} \\cmidrule(lr){4-6}",
             "Noise attributes & $s=0$ & $s=1$ & ZRL, $s=0$ & ZRL, $s=1$ & Boosting \\\\", "\\midrule"]
    for lv in levels:
        g = {r["method"]: r for r in agg.filter(pl.col("noise_attributes") == lv).to_dicts()}
        lines.append(f"{lv} & {g['ZRL s=0']['wn']:.2f} & {g['ZRL s=1']['wn']:.2f} & {g['ZRL s=0']['auc']:.3f} ({g['ZRL s=0']['sd']:.3f}) & "
                     f"{g['ZRL s=1']['auc']:.3f} ({g['ZRL s=1']['sd']:.3f}) & "
                     f"{g['gradient boosting']['auc']:.3f} ({g['gradient boosting']['sd']:.3f}) \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\par\\vspace{3pt}\\parbox{0.98\\columnwidth}{\\footnotesize \\emph{Note.} Uniform random attributes are appended to the 30 real ones. "
              "Weight on noise is the total zeta weight that falls on the random attributes.}", "\\end{table}"]
    tex.append("\n".join(lines))
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for method, color, label in (("ZRL s=1", SERIES[0], "ZRL, zeta weights (s = 1)"), ("ZRL s=0", SERIES[1], "ZRL, equal weights (s = 0)")):
        d = agg.filter(pl.col("method") == method).sort("noise_attributes")
        ax.errorbar(d["noise_attributes"], d["auc"], yerr=d["sd"], color=color, lw=2, marker="o", ms=6, capsize=3, label=label)
    ax.set_xlabel("Pure-noise attributes added to the 30 real ones")
    ax.set_ylabel("Held-out ROC AUC")
    ax.set_xticks(levels)
    ax.legend(frameon=False, loc="lower left")
    fig.savefig(out / "fig_noise.png")
    plt.close(fig)

# --------------------------------------------------------------- scalability
scale_file = src / "results_extra" / "scalability.csv"
if scale_file.exists():
    raw = pl.read_csv(scale_file)
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
              f"run alone. Total includes sorting, refinement, the exact summary and the sampled checks. Dense $W$ is the memory the graph would "
              f"need in single precision. The fitted log--log slope of total time against accounts is {slope:.2f}.}}", "\\end{table}"]
    tex.append("\n".join(lines))
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.loglog(sc["n"], sc["fit_seconds"], color=SERIES[0], lw=2, marker="o", ms=6)
    ax.set_xlabel("Number of accounts n")
    ax.set_ylabel("Time for one complete fit (s)")
    ax.annotate(f"slope {slope:.2f}", (sc["n"][len(sc) // 2], sc["fit_seconds"][len(sc) // 2]), textcoords="offset points", xytext=(10, -14),
                fontsize=9, color=INK2)
    fig.savefig(out / "fig_scalability.png")
    plt.close(fig)

# ------------------------------------------------------- reduced graph, classes
if "hi" in fits:
    f, mt = fits["hi"], meta["hi"]
    feats = mt["features"]
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

    w = f["weights"]
    top_w = np.argsort(-w)[:8]
    summary["weights"] = [{"attribute": feats[j], "weight": float(w[j]), "mi": float(f["relevance"][j])} for j in top_w]
    cum = np.cumsum(pos[order]) / pos.sum()
    summary["classes"] = {"k": int(len(R)), "base_rate": float(base), "classes_ge_2x": int((rate >= 2 * base).sum()),
                          "classes_zero": int((pos == 0).sum()), "share_in_top8": float(cum[7]), "share_in_top16": float(cum[15]),
                          "max_rate": float(rate.max())}
    names = {"nbt_centrality": "non-backtracking centrality", "degree": "partners", "n_currencies": "currencies used",
             "cross_cur_share": "cross-currency share", "pagerank_in": "PageRank (receiver)", "pagerank_out": "PageRank (sender)",
             "in_partners": "sending partners", "out_partners": "receiving partners", "n_out": "payments sent", "n_in": "payments received",
             "partner_degree": "partners' degree", "reciprocity": "reciprocity", "active_days": "active days",
             "cross_bank_share": "cross-bank share", "out_ratio": "share sent", "repeat_out": "payments per partner",
             "share_ach": "ACH share", "share_bitcoin": "Bitcoin share", "share_cheque": "cheque share", "share_wire": "wire share",
             "share_cash": "cash share", "share_credit_card": "credit-card share", "share_reinvestment": "reinvestment share",
             "self_share": "self-transfer share", "night_share": "night share", "out_amt_mean": "mean amount sent",
             "out_amt_std": "spread of amount sent", "out_amt_max": "largest amount sent", "in_amt_mean": "mean amount received",
             "in_amt_std": "spread of amount received"}
    lines = ["\\begin{table*}[t]", "\\caption{The five classes with the highest laundering rates (HI-Small, all accounts, descriptive)}",
             "\\label{tab:classes}", "\\centering", "\\footnotesize", "\\begin{tabular}{@{}crrrrl@{}}", "\\toprule",
             "Class & Accounts & Laundering & Rate & Lift & Most distinctive attributes (class mean of the scaled rank; overall mean 0.50) \\\\", "\\midrule"]
    summary["top_classes"] = []
    for i, c in enumerate(order[:5]):
        diff = cm[c] - 0.5
        top = np.argsort(-np.abs(diff))[:3]
        traits = "; ".join(f"{names.get(feats[j], feats[j])} ({cm[c][j]:.2f})" for j in top)
        summary["top_classes"].append({"size": int(sizes[c]), "positives": int(pos[c]), "rate": float(rate[c]), "lift": float(rate[c] / base),
                                       "traits": traits})
        lines.append(f"{'ABCDE'[i]} & {int(sizes[c]):,} & {int(pos[c])} & {rate[c] * 100:.1f}\\% & {rate[c] / base:.1f} & {traits} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    tex.append("\n".join(lines))
for key in meta:
    summary.setdefault("meta", {})[key] = {k: v for k, v in meta[key].items() if k != "features"}

ORDER = ["tab:structure", "tab:main", "tab:sig", "tab:sens", "tab:noise", "tab:ablation", "tab:scale", "tab:classes"]
def fit_width(block, label, width):
    """Scale a table that is wider than the text block down to the available width."""
    if "\\label{" + label + "}" not in block:
        return block
    return block.replace("\\begin{tabular}", "\\resizebox{" + width + "}{!}{\\begin{tabular}").replace("\\end{tabular}", "\\end{tabular}}")


tex = [fit_width(fit_width(b, "tab:main", "\\textwidth"), "tab:ablation", "\\columnwidth") for b in tex]
tex.sort(key=lambda block: next((i for i, lab in enumerate(ORDER) if "\\label{" + lab + "}" in block), len(ORDER)))
(out / "tables.tex").write_text("\n\n".join(tex) + "\n")
json.dump(summary, open(out / "summary.json", "w"), indent=1)
print("tables", len(tex), "datasets", list(scores), "folds", summary["folds"])
