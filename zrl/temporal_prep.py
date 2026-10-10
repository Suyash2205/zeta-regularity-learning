"""Choose the cut-off time for the time-based test and write the later labels.

Usage: python temporal_prep.py <transactions.csv> <later_labels.parquet> [quantile]

Prints the cut-off (the time before which the given share of transactions lies, default 0.7) in the
format ZRL_T_MAX expects, and writes one row per account that takes part in a flagged transaction
at or after the cut-off.
"""
import sys

import polars as pl

src, out = sys.argv[1], sys.argv[2]
quantile = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
COLS = ["ts", "from_bank", "from_acct", "to_bank", "to_acct", "amt_recv", "cur_recv", "amt_paid", "cur_paid", "fmt", "label"]
tx = (pl.scan_csv(src, has_header=True, new_columns=COLS, schema_overrides={"from_bank": pl.Utf8, "to_bank": pl.Utf8})
      .select("ts", "label", src=pl.col("from_bank") + "_" + pl.col("from_acct"), dst=pl.col("to_bank") + "_" + pl.col("to_acct")).collect())
cut = tx["ts"].sort()[int(quantile * (tx.height - 1))]   # timestamps are text in YYYY/MM/DD HH:MM, which sorts by time
later = tx.filter((pl.col("ts") >= cut) & (pl.col("label") == 1))
accounts = pl.concat([later.select(account="src"), later.select(account="dst")]).unique().with_columns(flagged_later=pl.lit(1, dtype=pl.Int8))
accounts.write_parquet(out)
before = tx.filter(pl.col("ts") < cut)
print(f"transactions before the cut-off {before.height} ({before['label'].sum()} flagged), after {tx.height - before.height} "
      f"({int(tx['label'].sum() - before['label'].sum())} flagged); last transaction {tx['ts'].max()}", file=sys.stderr)
print(cut)
