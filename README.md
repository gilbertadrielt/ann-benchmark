# ANN benchmark: LSH, IVF-PQ, HNSW and IVF-RaBitQ on four embedding datasets

All methods come from Faiss 1.15.1. Every number in the report is produced by this code.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_all.py --smoke              # 5-10 minutes, synthetic data: checks every stage
python run_all.py                      # the real run: ~5 GB download, then several hours
```

The real run is resumable: if it stops, run `python run_all.py --from <stage>`
(the failing stage is printed), and finished work is skipped.
Run the final timed run on one machine, plugged in, with other apps closed.

## Stages

| Stage | Script | Output |
| --- | --- | --- |
| download | scripts/download.py | data/raw/*.hdf5 |
| prepare | scripts/prepare.py | data/<ds>/*.npy, meta.json |
| ground_truth | scripts/ground_truth.py | exact top-100, results/dataset_stats.csv, sanity_checks.csv |
| bench | scripts/run_bench.py | results/<ds>.csv (E1, E2) |
| ablations | scripts/ablations.py | A2, A4 rows in results/<ds>.csv |
| ood | scripts/ood_study.py | results/ood/, results/e6_routing.csv (E4, E6) |
| failure | scripts/failure_cases.py | results/failure/ (E5) |
| lecture | scripts/lecture_check.py | results/e7_lecture.csv (E7) |
| scaling | scripts/scaling.py | results/e8_scaling.csv (E8) |
| plots | scripts/make_plots.py | results/figures/*.pdf|png, results/tables/*.csv|md |

One stage on its own: `python run_all.py --only plots`.
One method on one dataset: `python -m scripts.run_bench --dataset sift --method hnsw`.

## Notes

- Settings and grids live only in `config.py`.
- Below 12 GB of RAM, AG News is cut to its first 400,000 vectors (shown in table T1).
- Downloads resume; if one keeps failing, download the URL from `config.py` in a browser
  and save it under `data/raw/` with the same file name.
- Do not commit `data/` or `indexes/`; commit `results/`.
