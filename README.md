# ZRL: zeta-weighted, regularity-inspired summarisation of account similarity graphs

Code and result files for the paper *Zeta-Weighted, Regularity-Inspired Summarisation of Account Similarity Graphs for Risk Profiling in Digital Payments* (Suyash Humne and Sagar Korde, K J Somaiya School of Engineering, Somaiya Vidyavihar University).

ZRL summarises a dense *similarity* graph over bank accounts (who behaves like whom, not who pays whom) without ever storing it:

1. account attributes are ranked and weighted with a truncated Hurwitz zeta series;
2. the weighted similarity graph is partitioned into equal-sized classes by a heuristic inspired by Szemerédi's regularity lemma;
3. every account is described by its mean similarity to each class (its *profile*).

Given the partition, the reduced graph and all profiles are computed exactly from all n² edge weights, in time linear in the number of accounts, using sorted prefix sums. The partition itself is heuristic and is not certified to be regular.

What the results show, in short: 64 classes reproduce the edge weights with an error of about 0.03; a classifier on the profiles is slightly ahead of logistic regression on the attributes and clearly behind tuned gradient boosting and a graph neural network; a k-means partition summarised in the same exact way is slightly more accurate than the regularity-inspired one; and under a time-based test every method is weak. ZRL is a summary for human review, not a detector.

## Check the results in minutes

A full rerun takes hours on a large machine. The intermediate results of the published run (the class of every
account in every partition, the folds, the fitted reduced graphs) are saved in `results/`, and the account
attributes are attached to the release `verification-data-v1`, so the results can be checked without it:

```bash
./verify.sh quick              # 10 s: result files unchanged; tables and every number quoted in the paper follow from them
./verify.sh dumps hi           # 6 min, 2 GB: recompute every partition-based result of HI-Small from the saved partitions
./verify.sh spot hi partition  # 3 min: rebuild the ZRL-U partition from scratch and compare (other pieces: see VERIFY.md)
```

`VERIFY.md` explains what each level shows, how closely it agrees on other machines, and what it does not cover.

## Reproduce everything with one command

```bash
git clone https://github.com/Suyash2205/zeta-regularity-learning.git
cd zeta-regularity-learning
./reproduce.sh            # first-round experiments
./reproduce.sh revision   # experiments added in revision; writes the tables of the paper
```

The script creates a virtual environment, installs the pinned packages, downloads the three public data files (verifying checksums), builds the account attributes, runs the experiments, and writes the tables and figures to `results/paper/`. The first command finishes by comparing the mean results with those published in `results/EXPECTED_MEANS.csv` (runs on different machines agree closely but are not bit-identical; see `REPRODUCIBILITY.md`, Section 8).

| Mode | Command | Needs | Time |
|---|---|---|---|
| Full | `./reproduce.sh` | 16 CPUs, 64 GB RAM or more, 10 GB disk | about 2 hours |
| Revision | `./reproduce.sh revision` (after the full run) | 16 CPUs, 64 GB RAM | about 2.5 hours, most of it the graph neural network on CPUs |
| Quick check | `./reproduce.sh smoke` | 4 CPUs, 16 GB RAM | about 15 minutes |

The quick check runs the whole pipeline on a 30,000-account sample of HI-Small. Its numbers are not those of the paper.

With Docker: `docker build -t zrl .`, then `docker run --rm -v "$PWD/results:/zrl/results" zrl smoke` (or no argument for the full run, or `revision`).

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
| `zrl/revision.py` | Label-free main configuration (ZRL-U), landmark seeds, near-tied relevance, stricter label, tuned baselines, GraphSAGE, weight laws under noise, time-based test |
| `zrl/temporal_prep.py` | Cut-off time and later labels for the time-based test |
| `Dockerfile` | Container that runs `reproduce.sh` |
| `zrl/report.py` | Turns result files into the paper's tables and figures |
| `zrl/check_results.py` | Compares a fresh run with the published means |
| `zrl/verify.py`, `verify.sh`, `VERIFY.md` | Checks of the published results from the saved intermediate results, without a full rerun |
| `results/` | Fold-level scores, partition statistics, fold assignments, held-out predictions, logs and hashes from our runs |
| `REPRODUCIBILITY.md` | Seeds, software versions, hardware, exact attribute definitions, every hyperparameter |
| `checksums.sha256` | SHA-256 of the data files |
| `results/MANIFEST.sha256`, `results/DATA.sha256` | SHA-256 of every result file and of the account attribute files |

## Data

IBM Transactions for Anti-Money Laundering (AML), files `HI-Small_Trans.csv`, `LI-Small_Trans.csv` and `HI-Medium_Trans.csv`, from <https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml>, released by their authors under the Community Data License Agreement – Sharing 1.0. The transaction files are not redistributed here; `reproduce.sh` downloads them. The account attributes built from them are attached to the release `verification-data-v1` under the same licence (see `DATA_LICENSE.md`).

## Licence

MIT, see `LICENSE`.
