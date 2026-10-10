# Reproducibility details

Everything here describes the code as it is in this repository. No hyperparameter of ZRL was tuned on the data; every value below was set once. Sections 1–8 describe the first round of experiments (`./reproduce.sh`), in which the attribute order came from labels (ZRL-MI). Section 9 describes the experiments added in revision (`./reproduce.sh revision`), including the label-free configuration ZRL-U that the paper now uses as its main one.

## 1. Software and hardware

| | Full runs | Laptop timing quoted in the paper |
|---|---|---|
| Machine | Google Cloud `e2-highmem-16`: 16 virtual CPUs, 128 GB RAM, no GPU | Apple-silicon MacBook, 8 GB RAM |
| OS | Debian GNU/Linux 12 | macOS |
| Python | 3.11.2 | 3.11.6 |
| Packages | numpy 2.4.6, scipy 1.17.1, polars 2.0.0, scikit-learn 1.9.1, matplotlib 3.11.2 | same |
| BLAS | OpenBLAS bundled with the numpy wheel, `OPENBLAS_NUM_THREADS=3` per experiment | Accelerate/OpenBLAS default |

## 2. Random seeds

| Source of randomness | Where | Seed |
|---|---|---|
| Ranking set (stratified 20% of accounts) | `train_test_split(test_size=0.2, stratify=y, random_state=0)` | 0 |
| Cross-validation folds | `StratifiedKFold(5, shuffle=True, random_state=rep)`, `rep = 0, 1, 2` | 0, 1, 2 |
| Landmark accounts (512), stratified regularity sample, random pairs for the RMSE | one `numpy.random.default_rng(seed)` per fit, used in that order | 0 |
| Power iteration in the regularity check | same generator, standard-normal start vector, 25 iterations | 0 |
| k-means partition | `MiniBatchKMeans(64, n_init=3, batch_size=8192, random_state=0)` | 0 |
| Random equal-sized partition | `default_rng(0).permutation(n) % 64` | 0 |
| Gradient boosting | `HistGradientBoostingClassifier(max_iter=200, random_state=fold)` | fold index 0–4 |
| Isolation forest | `IsolationForest(n_estimators=200, random_state=fold)`, fitted on 100,000 training accounts drawn with `default_rng(fold)` | fold index 0–4 |
| Noise attributes | `default_rng(1)`: uniform random columns and the 100,000-account sample | 1 |
| Scalability subsamples | `default_rng(rep)`, `rep = 0, 1, 2` | 0, 1, 2 |

Fold assignments of every account are saved in `results/results_*/folds.npz` (`fold_of[rep, account]`; −1 marks the ranking set; accounts are in sorted order of the account identifier).

## 3. Evaluation protocol

1. A stratified 20% of the accounts forms the **ranking set**. Its labels fix the order of the attributes (mutual information).
2. The other 80% is split into five stratified folds.
3. The similarity graph, the partition, the reduced graph and the profiles are computed on **all** accounts and use **no** labels.
4. Every supervised model (the profile classifier, class risk, logistic regression, gradient boosting) is trained on **four folds plus the ranking set** and scored on the fifth fold. The ranking-set labels are therefore used twice, for the attribute order and as training data, and are never part of a test fold.
5. Steps 2–4 are repeated with three different fold seeds on HI-Small and LI-Small (15 test folds) and once on HI-Medium (5 test folds). Sweeps and ablations use the five folds of the first repeat.

Paired differences use the corrected resampled t-test: `t = mean(d) / sqrt((1/J + 0.25) * var(d))` with `J` test folds and `J − 1` degrees of freedom. The factor 0.25 is the test/train ratio of plain five-fold cross-validation; with the ranking set added to training the true ratio is 0.19, so the test as run is slightly conservative.

## 4. Attributes (30 per account)

Account identifier: bank code and account number joined by an underscore. `S` is the set of transactions the account sends, `R` the set it receives. Sent-side attributes of an account with no sent transactions are 0, and likewise for the received side.

**Behavioural (24), from `zrl/build_features.py`**

| Attribute | Definition |
|---|---|
| `n_out`, `n_in` | number of transactions in `S`, in `R` |
| `out_ratio` | `n_out / (n_out + n_in)` |
| `out_partners`, `in_partners` | distinct receivers in `S`; distinct senders in `R` |
| `repeat_out` | `n_out / max(out_partners, 1)` |
| `out_amt_mean`, `out_amt_std`, `out_amt_max` | mean, sample standard deviation (0 for one transaction) and maximum over `S` of `z`, where `z = (log(1 + amount paid) − mean) / sd` and mean and sd are taken over all transactions in the same payment currency |
| `in_amt_mean`, `in_amt_std` | mean and sample standard deviation of `z` over `R` |
| `cross_bank_share` | share of `S` whose sending and receiving banks differ |
| `cross_cur_share` | share of `S` whose receiving currency differs from the payment currency |
| `self_share` | share of `S` in which sender and receiver are the same account |
| `night_share` | share of `S` with hour before 06:00 or from 22:00 |
| `n_currencies` | distinct payment currencies in `S` |
| `active_days` | distinct calendar dates in `S` |
| `share_ach`, `share_cheque`, `share_credit_card`, `share_wire`, `share_cash`, `share_bitcoin`, `share_reinvestment` | share of `S` in each payment format |

**Structural (6), from `zrl/structural.py`**

The payment graph has one directed, unweighted edge for every distinct (sender, receiver) pair with sender ≠ receiver; repeated payments and amounts are ignored. Its adjacency matrix is `A`. The undirected graph `U` has an edge wherever `A` has one in either direction; `D` is its degree matrix.

| Attribute | Definition |
|---|---|
| `degree` | degree in `U` (number of distinct partners) |
| `partner_degree` | sum of the degrees of the account's neighbours in `U`, divided by `max(degree, 1)` |
| `reciprocity` | number of accounts `v` with both `u → v` and `v → u`, divided by `max(out-degree, 1)` |
| `pagerank_in` | PageRank on `A`: damping 0.85, uniform start, 60 power iterations, mass of accounts without out-edges spread uniformly |
| `pagerank_out` | the same on the reversed graph |
| `nbt_centrality` | solution `x` of `(I − tU + t²(D − I)) x = (1 − t²) 1`. The spectral radius `ρ` of the non-backtracking matrix is the largest-magnitude eigenvalue of the companion matrix `[[U, I − D], [I, 0]]`, computed with ARPACK (`scipy.sparse.linalg.eigs`, tolerance 1e-4, at most 500 iterations); `t = 0.5 / ρ`; the system is solved by conjugate gradients with relative tolerance 1e-10 |

**Scaling.** Every attribute is replaced by `(r − 1) / (n − 1)`, where `r` is its average rank (`scipy.stats.rankdata(method="average")`), so tied values share one rank.

## 5. The model

| Step | Exact procedure |
|---|---|
| Relevance, with labels | Mutual information between the label and the scaled rank cut into 20 equal-width bins; plug-in estimate, natural logarithm, no smoothing, empty cells skipped; a constant attribute scores 0. Computed on the ranking set only |
| Relevance, without labels | Raw attribute min–max scaled, 20-bin histogram, `1 + Σ p ln p / ln 20` with empty bins skipped |
| Order | Stable sort by decreasing relevance |
| Weights | `w_k = (k − 1 + q)^(−s) / Σ_j (j − 1 + q)^(−s)`; main configuration `s = 1`, `q = 1` |
| Graph | `W_uv = Σ_k w_k (1 − |x_uk − x_vk|)` for `u ≠ v`, `W_uu = 0` |
| Landmarks | 512 accounts drawn once, uniformly without replacement, reused in every round |
| Refinement round | For each class `V_a`: form the block of `W` with rows `V_a` and landmark columns; subtract, for each class `c` of the landmarks, the mean of the sub-block (rows `V_a`, landmarks in class `c`); take the leading left singular vector from the eigen-decomposition of the Gram matrix (`numpy.linalg.eigh`, no iteration); put accounts strictly above the median in one half. If a half is empty, accounts are split alternately. To bound memory, a class with more than `6e8 / (4 L)` accounts uses only the first `⌊6e8 / (4 |V_a|)⌋` landmarks (this affects the first one or two rounds on large data) |
| Rounds | 6, giving `K = 64` classes |
| Reduced graph | `R_ab = (Σ_{u∈V_a, v∈V_b} W_uv) / (|V_a| |V_b|)` for `a ≠ b`, and divided by `|V_a| (|V_a| − 1)` for `a = b`; exact |
| Profile | `p_ub = (Σ_{v∈V_b} W_uv) / (|V_b| − [u ∈ V_b])`; exact |
| Deviation | `δ_u = sqrt(Σ_b (|V_b| / n) (p_ub − R_{c(u) b})²)` |
| Regularity check | 100 accounts per class drawn without replacement; for every pair of classes the sample block is centred on its mean, its leading singular vectors are found by 25 power iterations, and the four pairs of subsets given by the signs of the two vectors are tested, each subset needing at least an `ε` share of its class in the sample; the pair is irregular if any tested pair of subsets has a density more than `ε = 0.05` from the block density |
| RMSE | 2,000,000 ordered pairs `u ≠ v` drawn uniformly; root mean square of `W_uv − R_{c(u) c(v)}` |

Memory: the regularity check forms a dense block on `100 K` sampled accounts, about 0.2 GB for `K = 64` and 2.6 GB (with temporaries, about 8 GB) for `K = 256`, whatever the number of accounts. This is why the quick check needs 16 GB although it uses only 30,000 accounts; a single fit with `K = 64` on all 515,088 accounts of HI-Small needs about 2 GB.

What is exact and what is estimated: given a partition, the reduced graph, the profiles, the deviations and the index are exact. The partition itself is heuristic (split directions come from landmark columns), and the share of irregular pairs and the RMSE are estimated from samples.

## 6. Comparison methods

| Method | Settings |
|---|---|
| Profile classifier | `StandardScaler` then `LogisticRegression(C=1.0, class_weight="balanced", solver="newton-cholesky", max_iter=200)` on the 64 profile values and the deviation |
| Class risk | `(laundering training accounts in class + overall training rate) / (training accounts in class + 1)` |
| Logistic regression | the same pipeline on the 30 scaled attributes |
| Gradient boosting | `HistGradientBoostingClassifier(max_iter=200, random_state=fold)`, all other settings at scikit-learn defaults (learning rate 0.1, 31 leaves, early stopping on a 10% validation split because the training set exceeds 10,000 rows) |
| Isolation forest | see Section 2; score is the negated `score_samples` |
| k-means partition | see Section 2, on the attributes multiplied by `sqrt(w)` |

## 7. Result files

`results/results_hi`, `results_li`, `results_hm` each hold `scores.csv` (one row per method, repeat and fold, with AUC, AP and recall at 1%, 5% and 10%), `structure.csv` (partition statistics per fit), `folds.npz`, `meta.json` and `full_fit.npz` (reduced graph, weights, class sizes and class means). `predictions_rep0.npz` holds the held-out scores of the three main methods for the first repeat (HI-Small and LI-Small). `results/results_extra` holds `noise.csv` and `scalability.csv`. `results/logs` holds the run logs, and `results/HASHES.txt` the SHA-256 of every result table.

`results/paper` holds the generated tables (`tables.tex`), figures and `summary.json`.

## 8. What was verified, and how closely results reproduce

The results in `results/` come from `./reproduce.sh` run on a fresh machine from a clean clone of this repository (commit `2abb2ab`; later commits change only documentation, the quick-check memory default, the final results check and the width of three generated tables). The quick check `./reproduce.sh smoke` was also run to completion on the same fresh machine.

Before that, the same experiments had been run once on another machine of the same type, with the attribute tables prepared on a laptop. Comparing the two independent runs over all 87 method-and-configuration combinations of the three data sets:

| | Largest difference in mean AUC |
|---|---|
| All 87 configurations | 0.0052 (k-means class risk, HI-Small) |
| ZRL profile classifier, main configuration | 0.0001 (HI-Small), 0.0003 (LI-Small), 0.0001 (HI-Medium) |
| Attribute-only models (logistic regression, gradient boosting) | 0.0002 |

The runs are **not bit-identical**. Single fold-level AUC values differ by up to 0.015. Known causes: (i) the attribute tables are built with multithreaded aggregations whose floating-point sums can differ in the last digits, which changes ranks among near-ties; (ii) the start vector of the Arnoldi iteration for the spectral radius is not seeded (the radius itself agreed to four digits: 17.07, 18.56 and 26.78 for HI-Small, LI-Small and HI-Medium); (iii) median splits amplify last-digit differences, because an account at the median can fall on either side; (iv) mini-batch k-means is sensitive to the same perturbations. For this reason `reproduce.sh` ends by comparing means with `results/EXPECTED_MEANS.csv` within a tolerance of 0.01 (`zrl/check_results.py`), and reports bit-identical hashes when they occur.

Scalability timings (`results/results_extra/scalability.csv`) were measured after all experiments had finished, each run in a fresh process, twice per size.

## 9. Experiments added in revision

`./reproduce.sh revision` runs `zrl/revision.py`. The splits are those of Section 3. All of these analyses were added after the first results were known and are exploratory.

| Task | What it does | Output |
|---|---|---|
| `unsup` | ZRL-U: attributes ranked by `1 − normalised entropy` of the 20-bin histogram of the min–max scaled raw attribute (no labels), `s = 1`, `q = 1`, 64 classes. Scored on every fold next to ZRL-MI, a k-means partition and a random partition of the same graph; concentration of laundering accounts per partition | `results/results_*/scores_rev.csv`, `structure_rev.csv`, `meta_rev.json` |
| `seeds` | ZRL-U on HI-Small with landmark seeds 0–4 | `results/results_rev/seeds.csv` |
| `ties` | The attributes whose entropy relevance is within 0.001 of the highest are reordered (reversed, three random orders) or given a shared weight | `results/results_rev/ties.csv` |
| `strict` | Label = at least two flagged transactions (HI-Small) | `results/results_rev/hi/strict.csv` |
| `tuned` | Gradient boosting: 16 random settings (learning rate, leaves, minimum leaf size, L2, class weighting; up to 500 iterations with early stopping); logistic regression: four values of `C`. Chosen once by average precision on a 20% validation split of the training part of the first fold, then used in all five folds | `results/results_rev/{hi,li}/tuned.csv`, `tuned_meta.json` |
| `gnn` | GraphSAGE on the directed transaction graph: two layers, separate mean aggregation over payers and payees, hidden size 64, dropout 0.2, Adam (learning rate 0.01, weight decay 5e-4), 200 full-batch epochs, class-weighted loss, state chosen by average precision on 10% of the training accounts; `torch.manual_seed(fold)`. Five folds, CPU only | `results/results_rev/{hi,li}/gnn.csv` |
| `weights` | Equal, zeta (`s = 1`), geometric (ratio 0.85) and linear weight laws with 0, 30 and 60 uniform noise attributes, entropy ranking, 100,000-account sample of HI-Small | `results/results_rev/hi/weights.csv` |
| `temporal` | Transactions are cut at the time before which 70% of them lie (`zrl/temporal_prep.py`: 2022/09/07 14:55 for HI-Small, 2022/09/07 14:48 for LI-Small). Attributes, graph and training labels come from before the cut; accounts not flagged before it are scored for being flagged after it. Bootstrap intervals from 500 resamples | `results/results_rev/{hi,li}/temporal.csv`, `temporal_meta.json` |

Software for these runs: the packages of Section 1 plus `torch` (CPU wheel), pinned in `requirements-gnn.txt`; the installed versions are recorded in `results/results_rev/versions.txt`. Machine: Google Cloud `e2-standard-16` (16 virtual CPUs, 64 GB RAM, Debian 12). Logs are in `results/logs_revision`.

Container: the image defined by `Dockerfile` was built on that machine and `docker run zrl smoke` ran to completion inside it (logs in `results/logs_revision/docker`). The full and revision runs were made outside the container, with the same pinned packages.

Near-tied relevance: on HI-Small the five attributes with the highest entropy relevance differ by less than 2e-6, so their order is arbitrary. Across six orderings the profile AUC ranges from 0.865 to 0.871 (`results/results_rev/ties.csv`); differences smaller than this between ZRL-U and other methods should not be over-read.

The sweeps over `s`, `q` and `K` and the attribute ablations were not rerun with the label-free ranking; they are reported with the label-ranked weights of the first round.

