# Licence of the account attribute files

The release `verification-data-v1` of this repository contains three files of account attributes,
`hi_accounts.parquet`, `li_accounts.parquet` and `hm_accounts.parquet`. They are computed by
`zrl/build_features.py` and `zrl/structural.py` from the files `HI-Small_Trans.csv`, `LI-Small_Trans.csv` and
`HI-Medium_Trans.csv` of the IBM Transactions for Anti-Money Laundering (AML) data set
(<https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml>), which its
authors (E. Altman and others, IBM) released under the Community Data License Agreement – Sharing, Version 1.0.

As Enhanced Data under that agreement, the attribute files are published under the same terms:
Community Data License Agreement – Sharing, Version 1.0, <https://cdla.dev/sharing-1-0/>. The data are
synthetic; no real person or account is described. The code of this repository is under the MIT licence
(`LICENSE`); this notice covers only the attribute files.

Original data: E. Altman, J. Blanuša, L. von Niederhäusern, B. Egressy, A. Anghel and K. Atasu,
"Realistic Synthetic Financial Transactions for Anti-Money Laundering Models", NeurIPS 2023 Datasets and
Benchmarks Track.
