# Logs of the verification runs

Output of `./verify.sh` on the published results, as described in `VERIFY.md`.

| Log | Machine |
|---|---|
| `dumps_hi.log`, `dumps_li.log`, `spot_*.log` | Apple-silicon laptop, 8 GB, macOS, Python 3.11.6, packages of `requirements.txt` |
| `dumps_hm.log` | Google Cloud `e2-highmem-4` (4 virtual CPUs, 32 GB), Debian 12, Python 3.11.2, packages of `requirements.txt` |

The HI-Medium attributes in the release were built on the cloud machine from `HI-Medium_Trans.csv`, whose
checksum matched `checksums.sha256`; the HI-Small and LI-Small attributes were built on the laptop.
