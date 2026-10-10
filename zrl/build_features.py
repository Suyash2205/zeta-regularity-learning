"""Build one row of behavioural attributes per account from the IBM AML transactions file.

Usage: python build_features.py <HI-Small_Trans.csv> <out.parquet>

Set ZRL_T_MAX (format YYYY/MM/DD HH:MM) to use only transactions strictly before that time.
The column n_flagged (number of flagged transactions of the account) is a label, not an attribute.
"""
import os
import sys

import polars as pl

src, out = sys.argv[1], sys.argv[2]

COLS = ["ts", "from_bank", "from_acct", "to_bank", "to_acct", "amt_recv", "cur_recv",
        "amt_paid", "cur_paid", "fmt", "label"]
tx = (
    pl.scan_csv(src, has_header=True, new_columns=COLS, schema_overrides={"from_bank": pl.Utf8, "to_bank": pl.Utf8})
    .with_columns(
        src=pl.col("from_bank") + "_" + pl.col("from_acct"),
        dst=pl.col("to_bank") + "_" + pl.col("to_acct"),
        ts=pl.col("ts").str.to_datetime("%Y/%m/%d %H:%M"),
        log_amt=(pl.col("amt_paid") + 1.0).log(),
    )
    # amounts are in different currencies, so standardise the log amount inside each currency
    .with_columns(
        z_amt=((pl.col("log_amt") - pl.col("log_amt").mean().over("cur_paid"))
               / pl.col("log_amt").std().over("cur_paid")),
        cross_bank=(pl.col("from_bank") != pl.col("to_bank")).cast(pl.Float64),
        cross_cur=(pl.col("cur_recv") != pl.col("cur_paid")).cast(pl.Float64),
        self_tx=(pl.col("src") == pl.col("dst")).cast(pl.Float64),
        night=((pl.col("ts").dt.hour() < 6) | (pl.col("ts").dt.hour() >= 22)).cast(pl.Float64),
    )
    .collect()
)
if os.environ.get("ZRL_T_MAX"):
    from datetime import datetime
    tx = tx.filter(pl.col("ts") < datetime.strptime(os.environ["ZRL_T_MAX"], "%Y/%m/%d %H:%M"))
print("transactions", tx.height, "laundering", int(tx["label"].sum()))

FORMATS = ["ACH", "Cheque", "Credit Card", "Wire", "Cash", "Bitcoin", "Reinvestment"]
sent = tx.group_by("src").agg(
    n_out=pl.len(),
    out_partners=pl.col("dst").n_unique(),
    out_amt_mean=pl.col("z_amt").mean(),
    out_amt_std=pl.col("z_amt").std().fill_null(0.0),
    out_amt_max=pl.col("z_amt").max(),
    cross_bank_share=pl.col("cross_bank").mean(),
    cross_cur_share=pl.col("cross_cur").mean(),
    self_share=pl.col("self_tx").mean(),
    night_share=pl.col("night").mean(),
    n_currencies=pl.col("cur_paid").n_unique(),
    active_days=pl.col("ts").dt.date().n_unique(),
    *[(pl.col("fmt") == f).mean().alias("share_" + f.lower().replace(" ", "_")) for f in FORMATS],
    launder_out=pl.col("label").sum(),
).rename({"src": "account"})
recv = tx.group_by("dst").agg(
    n_in=pl.len(),
    in_partners=pl.col("src").n_unique(),
    in_amt_mean=pl.col("z_amt").mean(),
    in_amt_std=pl.col("z_amt").std().fill_null(0.0),
    launder_in=pl.col("label").sum(),
).rename({"dst": "account"})

acc = sent.join(recv, on="account", how="full", coalesce=True).fill_null(0.0).with_columns(
    out_ratio=pl.col("n_out") / (pl.col("n_out") + pl.col("n_in")),
    repeat_out=pl.col("n_out") / pl.col("out_partners").clip(lower_bound=1),
    n_tx=pl.col("n_out") + pl.col("n_in"),
    is_laundering=((pl.col("launder_out") + pl.col("launder_in")) > 0).cast(pl.Int8),
    n_flagged=(pl.col("launder_out") + pl.col("launder_in")).cast(pl.Int32),
).drop("launder_out", "launder_in")
acc.write_parquet(out)
print("accounts", acc.height, "laundering accounts", int(acc["is_laundering"].sum()))
print(acc.select(pl.col("n_tx").quantile(q).alias(f"q{q}") for q in (0.5, 0.9, 0.99, 0.999)))
for n in (2000, 3000, 4000, 6000):
    top = acc.sort("n_tx", descending=True).head(n)
    print(f"top {n}: min n_tx {top['n_tx'].min()}, laundering accounts {int(top['is_laundering'].sum())}")
