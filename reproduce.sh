#!/usr/bin/env bash
# One command to reproduce every table and figure of the paper.
#
#   ./reproduce.sh          full run: three data sets, all experiments (about 2 hours on 16 CPUs, 128 GB RAM)
#   ./reproduce.sh revision the experiments added in revision (about 90 minutes on 16 CPUs)
#   ./reproduce.sh smoke    quick check on a 30,000-account sample of HI-Small (about 15 minutes, 16 GB RAM)
#
# Environment: ZRL_THREADS (default 4), ZRL_MAX_GB (memory the run may use; it aborts above this),
# ZRL_SKIP_INSTALL=1 to use an existing .venv as it is.
set -euo pipefail
MODE="${1:-full}"
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
[ -d .venv ] || "$PY" -m venv .venv
[ "${ZRL_SKIP_INSTALL:-0}" = 1 ] || .venv/bin/python -m pip -q install -r requirements.txt
RUN=".venv/bin/python"
mkdir -p data results logs
BASE="https://www.kaggle.com/api/v1/datasets/download/ealtman2019/ibm-transactions-for-anti-money-laundering-aml?file_name="
sha() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

fetch() {  # file name; verifies against checksums.sha256 when the file is listed there
  local f="data/$1"
  [ -s "$f" ] || curl -sL -o "$f" "${BASE}$1"
  local want; want=$(grep " $f\$" checksums.sha256 | cut -d' ' -f1 || true)
  local got; got=$(sha "$f")
  echo "$got  $f" >> logs/data_checksums.txt
  if [ -n "$want" ] && [ "$want" != "$got" ]; then echo "checksum mismatch for $f" >&2; exit 1; fi
}

prepare() {  # transactions file, short name
  fetch "$1"
  [ -s "data/$2_accounts.parquet" ] || $RUN zrl/build_features.py "data/$1" "data/$2_accounts_raw.parquet" > "logs/$2_features.log"
  [ -s "data/$2_accounts.parquet" ] || $RUN zrl/structural.py "data/$1" "data/$2_accounts_raw.parquet" "data/$2_accounts.parquet" > "logs/$2_structural.log"
}

: > logs/data_checksums.txt
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-3}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
T="${ZRL_THREADS:-4}"

if [ "$MODE" = "smoke" ]; then
  export ZRL_MAX_GB="${ZRL_MAX_GB:-12}"
  prepare HI-Small_Trans.csv hi
  $RUN - <<'PY'
import polars as pl
a = pl.read_parquet("data/hi_accounts.parquet")
pos = a.filter(pl.col("is_laundering") == 1).sort("account").head(1200)
neg = a.filter(pl.col("is_laundering") == 0).sort("account").sample(28800, seed=0)
pl.concat([pos, neg]).write_parquet("data/smoke_accounts.parquet")
PY
  $RUN zrl/experiment_scale.py data/smoke_accounts.parquet results/results_hi full 1 "$T" | tee logs/smoke.log
  ZRL_SCALE_REPEATS=1 ZRL_THREADS="$T" $RUN zrl/extras.py data/smoke_accounts.parquet results/results_extra scale | tee -a logs/smoke.log
  $RUN zrl/report.py results results/paper
  echo "smoke run finished: see results/paper"
  exit 0
fi

export ZRL_MAX_GB="${ZRL_MAX_GB:-60}" ZRL_PARALLEL_BLOCK="${ZRL_PARALLEL_BLOCK:-1}"
prepare HI-Small_Trans.csv hi
prepare LI-Small_Trans.csv li
prepare HI-Medium_Trans.csv hm

if [ "$MODE" = "revision" ]; then
  # Experiments added in revision (label-free main configuration and its robustness). About 90 minutes on 16 CPUs.
  REV="$RUN zrl/revision.py"
  ZRL_THREADS=5 $REV data/hi_accounts.parquet results/results_hi unsup 3 > logs/rev_hi_unsup.log 2>&1 &
  ZRL_THREADS=4 $REV data/li_accounts.parquet results/results_li unsup 3 > logs/rev_li_unsup.log 2>&1 &
  ZRL_THREADS=4 $REV data/hm_accounts.parquet results/results_hm unsup 1 > logs/rev_hm_unsup.log 2>&1 &
  wait
  ( ZRL_THREADS=5 $REV data/hi_accounts.parquet results/results_rev seeds > logs/rev_seeds.log 2>&1
    ZRL_THREADS=5 $REV data/hi_accounts.parquet results/results_rev ties > logs/rev_ties.log 2>&1 ) &
  ( ZRL_THREADS=3 $REV data/hi_accounts.parquet results/results_rev/hi strict > logs/rev_strict.log 2>&1
    ZRL_THREADS=3 $REV data/hi_accounts.parquet results/results_rev/hi weights > logs/rev_weights.log 2>&1 ) &
  ( for d in hi li; do ZRL_THREADS=4 $REV data/${d}_accounts.parquet results/results_rev/$d tuned > logs/rev_tuned_$d.log 2>&1; done ) &
  wait
  # Time-based test: attributes from the first 70% of the transactions, labels from the rest.
  for pair in "hi HI-Small_Trans.csv" "li LI-Small_Trans.csv"; do
    set -- $pair
    CUT=$($RUN zrl/temporal_prep.py "data/$2" "data/$1_later.parquet" 2> "logs/rev_temporal_prep_$1.log")
    ZRL_T_MAX="$CUT" $RUN zrl/build_features.py "data/$2" "data/$1_past_raw.parquet" > "logs/rev_temporal_features_$1.log" 2>&1
    ZRL_T_MAX="$CUT" $RUN zrl/structural.py "data/$2" "data/$1_past_raw.parquet" "data/$1_past.parquet" >> "logs/rev_temporal_features_$1.log" 2>&1
    ZRL_THREADS=6 $REV "data/$1_past.parquet" "results/results_rev/$1" temporal "data/$1_later.parquet" > "logs/rev_temporal_$1.log" 2>&1 &
  done
  wait
  # Graph neural network baseline (needs PyTorch).
  .venv/bin/python -m pip -q install -r requirements-gnn.txt
  for pair in "hi HI-Small_Trans.csv" "li LI-Small_Trans.csv"; do
    set -- $pair
    ZRL_THREADS=7 $REV "data/$1_accounts.parquet" "results/results_rev/$1" gnn "data/$2" > "logs/rev_gnn_$1.log" 2>&1 &
  done
  wait
  .venv/bin/python -m pip freeze | grep -i -E "^(torch|numpy|scipy|polars|scikit-learn)==" > results/results_rev/versions.txt
  rm -rf results/logs_revision && cp -r logs results/logs_revision
  echo "revision experiments finished: see results/results_rev and results/results_*/scores_rev.csv"
  exit 0
fi

# The three cross-validation experiments and the noise test run side by side.
$RUN zrl/experiment_scale.py data/hi_accounts.parquet results/results_hi full 3 5 > logs/hi.log 2>&1 &
$RUN zrl/experiment_scale.py data/li_accounts.parquet results/results_li main 3 4 > logs/li.log 2>&1 &
$RUN zrl/experiment_scale.py data/hm_accounts.parquet results/results_hm main 1 4 > logs/hm.log 2>&1 &
ZRL_THREADS=3 $RUN zrl/extras.py data/hi_accounts.parquet results/results_extra noise > logs/noise.log 2>&1 &
wait
# Scalability timings run last, alone on the machine, each size twice in a fresh process.
ZRL_THREADS=8 ZRL_SCALE_REPEATS="${ZRL_SCALE_REPEATS:-2}" $RUN zrl/extras.py data/hm_accounts.parquet results/results_extra scale > logs/scale.log 2>&1
$RUN zrl/report.py results results/paper > logs/report.log 2>&1

# Checksums of every result table
( cd results && for f in results_*/scores.csv results_*/structure.csv results_extra/noise.csv; do echo "$(sha "$f")  $f"; done ) > results/HASHES.txt
# Runs on different machines are not bit-identical, so the check compares means within a tolerance;
# identical hashes are reported when they occur.
if [ -f results/EXPECTED_HASHES.txt ] && diff -q results/EXPECTED_HASHES.txt results/HASHES.txt >/dev/null; then echo "result tables are bit-identical to the published run"; fi
if [ -f results/EXPECTED_MEANS.csv ]; then $RUN zrl/check_results.py results; fi
rm -rf results/logs && cp -r logs results/logs
echo "done: tables and figures are in results/paper"
