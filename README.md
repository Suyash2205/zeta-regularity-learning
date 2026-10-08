# ZRL: Zeta-Weighted Regularity Learning

Code and result files for the paper *Zeta-Weighted Regularity Learning: Exact Matrix-Free Profiling of Accounts in Dense Payment Graphs for Trusted Circular-Economy Platforms* (Suyash Humne and Sagar Korde, K J Somaiya School of Engineering, Somaiya Vidyavihar University).

ZRL summarises a dense similarity graph over bank accounts without ever storing it:

1. account attributes are ranked and weighted with a truncated Hurwitz zeta series;
2. the weighted similarity graph is partitioned into equal-sized classes in the spirit of Szemerédi's regularity lemma;
3. every account is described by its mean similarity to each class (its *profile*).

The reduced graph and all profiles are computed exactly from all n² edge weights, in time linear in the number of accounts, using sorted prefix sums.

## Reproduce everything with one command

```bash
git clone https://github.com/Suyash2205/zeta-regularity-learning.git
cd zeta-regularity-learning
./reproduce.sh            # full run
```

The script creates a virtual environment, installs the pinned packages, downloads the three public data files (verifying checksums), builds the account attributes, runs every experiment, and writes every table and figure of the paper to `results/paper/`. It finishes by comparing the mean results with those published in `results/EXPECTED_MEANS.csv` (runs on different machines agree closely but are not bit-identical; see `REPRODUCIBILITY.md`, Section 8).

| Mode | Command | Needs | Time |
|---|---|---|---|
| Full | `./reproduce.sh` | 16 CPUs, 64 GB RAM or more, 10 GB disk | about 2 hours |
| Quick check | `./reproduce.sh smoke` | 4 CPUs, 16 GB RAM | about 15 minutes |

The quick check runs the whole pipeline on a 30,000-account sample of HI-Small. Its numbers are not those of the paper.

Useful environment variables: `ZRL_THREADS` (worker threads), `ZRL_MAX_GB` (the run aborts if it uses more memory than this), `ZRL_SKIP_INSTALL=1` (use an existing `.venv`).

## What is in the repository

| Path | Contents |
|---|---|
| `zrl/zrl.py` | Zeta weights, attribute relevance, and the dense reference implementation used for small graphs and for the regularity check |
| `zrl/zrl_scale.py` | The matrix-free model: exact products with the implicit graph, refinement, exact summary, sampled checks |
| `zrl/build_features.py` | The 24 behavioural attributes of every account, from the transaction file |
| `zrl/structural.py` | The 6 structural attributes: PageRank (both directions), partners, partners' degree, reciprocity, non-backtracking centrality |
| `zrl/experiment_scale.py` | Cross-validated evaluation, sweeps, ablations and comparison methods |
| `zrl/extras.py` | Noise-attribute test and scalability timings |
| `zrl/report.py` | Turns result files into the paper's tables and figures |
| `zrl/check_results.py` | Compares a fresh run with the published means |
| `results/` | Fold-level scores, partition statistics, fold assignments, held-out predictions, logs and hashes from our runs |
| `REPRODUCIBILITY.md` | Seeds, software versions, hardware, exact attribute definitions, every hyperparameter |
| `checksums.sha256` | SHA-256 of the data files |

## Data

IBM Transactions for Anti-Money Laundering (AML), files `HI-Small_Trans.csv`, `LI-Small_Trans.csv` and `HI-Medium_Trans.csv`, from <https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml>, released by their authors under the Community Data License Agreement. The data are not redistributed here; `reproduce.sh` downloads them.

## Licence

MIT, see `LICENSE`.
