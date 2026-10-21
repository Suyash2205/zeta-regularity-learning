#!/usr/bin/env bash
# Check the published results without rerunning the experiments. See VERIFY.md.
#
#   ./verify.sh quick                       checksums, tables and every number quoted in the paper (seconds, no data)
#   ./verify.sh dumps hi|li|hm [repeats]    recompute all partition-based results from the saved partitions (minutes)
#   ./verify.sh spot hi|li <piece> [args]   rerun one piece from scratch: partition | seed N | kmeans | balanced | fold REP FOLD
#
# The account attributes are downloaded from the GitHub release of this repository the first time they are
# needed and checked against results/DATA.sha256. Environment: ZRL_THREADS (default 4), ZRL_SKIP_INSTALL=1.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
[ -d .venv ] || "$PY" -m venv .venv
[ "${ZRL_SKIP_INSTALL:-0}" = 1 ] || .venv/bin/python -m pip -q install -r requirements.txt
RELEASE="https://github.com/Suyash2205/zeta-regularity-learning/releases/download/verification-data-v1"
sha() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

MODE="${1:-quick}"
if [ "$MODE" = dumps ] || [ "$MODE" = spot ]; then
  f="data/${2:?name a data set: hi, li or hm}_accounts.parquet"
  mkdir -p data
  if [ ! -s "$f" ]; then
    echo "downloading $f from the release"
    curl -fsSL -o "$f" "$RELEASE/$(basename "$f")"
    want=$(grep " $f\$" results/DATA.sha256 | cut -d' ' -f1)
    if [ "$(sha "$f")" != "$want" ]; then rm -f "$f"; echo "checksum mismatch for $f" >&2; exit 1; fi
  fi
fi
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-3}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
exec .venv/bin/python -W ignore zrl/verify.py "$@"
