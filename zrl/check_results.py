"""Compare a fresh run with the published one.

Usage: python check_results.py <results_dir> [write]

With "write", stores the mean AUC and AP of every method and configuration in
<results_dir>/EXPECTED_MEANS.csv. Without it, compares the run in <results_dir> with that
file and fails if any mean differs by more than the tolerance. Runs on different machines
are not bit-identical (see REPRODUCIBILITY.md), so means are compared, not file hashes.
"""
import sys
from pathlib import Path

import polars as pl

res = Path(sys.argv[1])
TOLERANCE = 0.01
KEYS = ["dataset", "method", "sweep", "value"]
frames = []
for d in sorted(res.glob("results_*")):
    f = d / "scores.csv"
    if f.exists() and d.name != "results_extra":
        s = pl.read_csv(f, schema_overrides={"value": pl.Utf8}, infer_schema_length=None)
        frames.append(s.group_by(["method", "sweep", "value"]).agg(auc=pl.col("auc").mean(), ap=pl.col("ap").mean(), folds=pl.len())
                      .with_columns(dataset=pl.lit(d.name)))
means = pl.concat(frames).select(KEYS + ["auc", "ap", "folds"]).sort(KEYS)
expected = res / "EXPECTED_MEANS.csv"
if len(sys.argv) > 2 and sys.argv[2] == "write":
    means.write_csv(expected)
    print(f"wrote {expected} ({means.height} rows)")
    sys.exit(0)
ref = pl.read_csv(expected, schema_overrides={"value": pl.Utf8})
j = means.join(ref, on=KEYS, suffix="_expected")
worst = j.with_columns(d=(pl.col("auc") - pl.col("auc_expected")).abs()).sort("d", descending=True)
print(f"compared {j.height} of {ref.height} configurations; largest difference in mean AUC: {worst['d'][0]:.4f} "
      f"({worst['dataset'][0]}, {worst['method'][0]})")
if j.height < ref.height or worst["d"][0] > TOLERANCE:
    sys.exit(f"results differ from the published run by more than {TOLERANCE}")
print(f"all means agree with the published run within {TOLERANCE}")
