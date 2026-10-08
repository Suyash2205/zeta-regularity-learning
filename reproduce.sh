#!/usr/bin/env bash
# One command to reproduce every table and figure of the paper.
#
#   ./reproduce.sh          full run: three data sets, all experiments (about 2 hours on 16 CPUs, 128 GB RAM)
#   ./reproduce.sh smoke    quick check on a 30,000-account sample of HI-Small (about 10 minutes, 8 GB RAM)
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
  export ZRL_MAX_GB="${ZRL_MAX_GB:-6}"
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

# The three cross-validation experiments and the noise test run side by side.
$RUN zrl/experiment_scale.py data/hi_accounts.parquet results/results_hi full 3 5 > logs/hi.log 2>&1 &
$RUN zrl/experiment_scale.py data/li_accounts.parquet results/results_li main 3 4 > logs/li.log 2>&1 &
$RUN zrl/experiment_scale.py data/hm_accounts.parquet results/results_hm main 1 4 > logs/hm.log 2>&1 &
ZRL_THREADS=3 $RUN zrl/extras.py data/hi_accounts.parquet results/results_extra noise > logs/noise.log 2>&1 &
wait
# Scalability timings run last, alone on the machine, each size twice in a fresh process.
ZRL_THREADS=8 ZRL_SCALE_REPEATS="${ZRL_SCALE_REPEATS:-2}" $RUN zrl/extras.py data/hm_accounts.parquet results/results_extra scale > logs/scale.log 2>&1
$RUN zrl/report.py results results/paper > logs/report.log 2>&1

# Checksums of every result table, for comparison with results/EXPECTED_HASHES.txt
( cd results && for f in results_*/scores.csv results_*/structure.csv results_extra/noise.csv; do echo "$(sha "$f")  $f"; done ) > results/HASHES.txt
if [ -f results/EXPECTED_HASHES.txt ]; then diff results/EXPECTED_HASHES.txt results/HASHES.txt && echo "all result tables match the expected hashes"; fi
rm -rf results/logs && cp -r logs results/logs
echo "done: tables and figures are in results/paper"
