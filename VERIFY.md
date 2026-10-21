# Checking the published results without rerunning everything

Rerunning every experiment takes several hours on a 16-CPU machine with 64 GB or more of memory
(`./reproduce.sh`). To make the results checkable in minutes, this repository keeps the intermediate results
of the published run, and `./verify.sh` checks them at three levels. Each level needs less trust than the one
before it.

| Level | Command | What it shows | Time and memory | Needs |
|---|---|---|---|---|
| 1 | `./verify.sh quick` | The tables of the paper and every number quoted in its text follow from the result files, and no result file has changed since publication | about 10 seconds | nothing |
| 2 | `./verify.sh dumps hi` (or `li`, `hm`) | The result files follow from the data: from the saved class assignments the script recomputes the reduced graph, the reconstruction error, the class concentration, the irregularity sample and the cross-validated AUC and AP of every fold | HI-Small about 6 minutes, 2 GB; LI-Small about 7 minutes, 2.2 GB; HI-Medium about 36 minutes, 11 GB | the account attributes (downloaded automatically) |
| 3 | `./verify.sh spot hi <piece>` | A piece chosen by the reviewer, rerun from scratch, gives the published value | 1 to 5 minutes, 2 to 3 GB | the account attributes |

With Docker: `docker build -t zrl .`, then `docker run --rm --entrypoint ./verify.sh zrl quick` (or `dumps hi`, and
so on).

The timings are from an 8 GB Apple-silicon laptop (HI-Small, LI-Small) and an `e2-highmem-4` cloud machine
with 4 CPUs and 32 GB (HI-Medium).

## Level 1: from the result files to the paper

`./verify.sh quick` does three things.

1. It checks the SHA-256 checksum of each of the result files listed in `results/MANIFEST.sha256`.
2. It rebuilds the tables of the paper and `summary.json` from the result files with `zrl/report.py` and
   compares them with the published ones. `tables.tex` must be byte-identical; it is the file the paper
   includes. `summary.json` must be identical apart from the order of rows inside a list.
3. It looks up each number quoted in the text of the paper (abstract and results; listed in
   `results/paper/paper_numbers.csv` with the place it comes from) in the rebuilt summary, rounds it as
   printed, and compares.

A changed score in any result file fails step 1. If the checksum list is changed too, steps 2 and 3 still
fail, because the rebuilt tables and quoted numbers no longer match the paper.

## Level 2: from the data to the result files

The expensive part of ZRL is finding the partition, which needs many products with the implicit graph. Once the
partition is known, everything else (the reduced graph, every profile, the error, the classifiers) takes
seconds to minutes. The class of every account in every published partition is saved:

| File | Contents |
|---|---|
| `results/results_{hi,li,hm}/partitions_rev.npz` | ZRL-U, ZRL-MI, k-means and random partitions, one class number per account (accounts in sorted order of their identifier) |
| `results/results_rev/hi/fair_partition.npz` | balanced k-means partition of HI-Small |
| `results/results_{hi,li,hm}/folds.npz` | the test fold of every account in every repeat, and the labels |
| `results/results_{hi,li,hm}/full_fit_rev.npz` | the reduced graph, weights, class sizes and class means of ZRL-U |
| `results/results_{hi,li}/predictions_rep0.npz` | held-out scores of logistic regression and gradient boosting, first repeat |

`./verify.sh dumps hi` loads the account attributes with the same code as the published run (it imports
`zrl/revision.py`), checks the labels, the attribute orders, the zeta weights and the folds, and then for every
saved partition recomputes and compares:

- the reduced graph and class sizes (ZRL-U);
- the reconstruction error and the index;
- the share of irregular class pairs, using the same random sample as the published run;
- the eight class-concentration figures;
- the AUC and average precision of the profile classifier, class risk and deviation in each of the five folds of
  the first repeat (`./verify.sh dumps hi 3` does all fifteen).

The account attributes are not in the repository itself, because the largest file exceeds GitHub's limit for
a single file. They are attached to the release
[`verification-data-v1`](https://github.com/Suyash2205/zeta-regularity-learning/releases/tag/verification-data-v1),
and `verify.sh` downloads them and checks them against `results/DATA.sha256`. To build them yourself from the
original IBM files, run the `prepare` step of `reproduce.sh` (see the README).

**How closely Level 2 agrees.** On both machines, everything that follows from the partition by arithmetic
agrees closely: the reconstruction error, the index and the irregularity sample to six or more decimals, the
reduced graph to 0.0001, the class concentration exactly, and the AUC of class risk and deviation to 0.00003. On HI-Small the
balanced k-means scores, computed on the laptop in the first place, are identical.

One quantity does not reproduce exactly: the profile classifier, a logistic regression on 65 strongly
correlated columns (64 profile entries and the deviation). In most folds its Newton solver finds the Hessian
ill-conditioned, falls back to L-BFGS, and L-BFGS stops at its iteration limit of 200 before it has converged.
Where it stops depends on the linear-algebra library and on the number of threads, so it differs between
machines, including between the published run and a Linux machine with the same package versions. Measured
differences of the profile classifier from the published values:

| Data set | Machine | Largest difference in one fold (AUC) | Largest difference of the five-fold mean (AUC) |
|---|---|---|---|
| HI-Small | laptop | 0.004 | 0.001 |
| LI-Small | laptop | 0.013 | 0.004 |
| HI-Medium | Linux, 4 CPUs | 0.012 | 0.003 |

The largest differences are on the random and k-means partitions. Whether the solver converges in a given fold
also differs between machines, so the script compares the profile classifier in every fold with the tolerances
at the top of `zrl/verify.py`: 0.015 in AUC for a single fold and 0.005 for the mean over the folds. Class risk
and deviation must agree within 0.002, and in practice agree within 0.00003. A reader should therefore treat
differences in the profile classifier's AUC of less than about 0.005 as numerical noise, the same conclusion the paper draws from the near-tied attribute orders.

## Level 3: rerun a piece from scratch

Level 2 trusts the saved partitions. Level 3 removes that trust for a piece the reviewer chooses:

```bash
./verify.sh spot hi partition     # the ZRL-U partition, rebuilt from the attributes
./verify.sh spot hi seed 3        # the partition with landmark seed 3 (seeds table)
./verify.sh spot hi kmeans        # the k-means partition
./verify.sh spot hi balanced      # the balanced k-means partition
./verify.sh spot li fold 2 4      # repeat 2, fold 4: logistic regression, gradient boosting (default and,
                                  # in repeat 0, tuned) and, on HI-Small, spline logistic regression, trained from scratch
```

The partition is not bit-identical across machines, because floating-point sums in the landmark products are
rounded differently by different BLAS libraries and a few accounts near a median split change sides. On the
laptop, the rebuilt ZRL-U partition of HI-Small put 99.96% of the accounts in the same class as the published
one (adjusted Rand index 0.9993), and its error was 0.032768 against the published 0.032766. For LI-Small the figures were
99.93% and 0.030653 against 0.030651; the profile classifier on the rebuilt partition had a five-fold mean AUC
of 0.8516 against the published 0.8556, the solver effect described above. Level 3 therefore reports the share
of identical accounts and compares the error, the AUC and the concentration with tolerances.

Measured results of every check on the machines named above are in `results/verification/`.

## What these checks do not cover

- The GraphSAGE baseline (needs PyTorch; the published run took 37 minutes on HI-Small and 50 on LI-Small with 7 CPU threads), the weight-law test under
  noise, the stricter label and the time-based test are not part of Levels 2 and 3. Their result files are
  covered by Level 1, and `./reproduce.sh revision` reruns them.
- The scalability timings depend on the machine and cannot be checked elsewhere; their logs are in
  `results/logs/`.
- Level 3 on HI-Medium needs about 17 GB of memory and is not offered by default; `./verify.sh dumps hm` runs on
  a machine with 16 GB or more.
