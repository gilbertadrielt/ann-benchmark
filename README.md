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

On Intel chips with performance and efficiency cores (12th gen and later), run
`python -m scripts.cpu_check` once and add the `--cpus` option it prints to every
`run_all.py` command, so timed runs stay on the fast cores (Windows and Linux only).

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

## Run one method at a time

`run_all.py` can take many hours. Everything is resumable and results are appended to
`results/<ds>.csv`, so you can run the pieces separately, in any order, and stop with Ctrl+C
at any time. Datasets: `sift`, `yahoo`, `coco`, `agnews`. Methods: `flat`, `lsh`, `ivfpq`,
`hnsw`, `ivfrabitq`.

**Step 1. Prepare one dataset (once per dataset).** Each of these accepts dataset names,
so you can start with the smallest, SIFT (~500 MB), instead of downloading all ~5 GB:

```bash
python -m scripts.download sift
python -m scripts.prepare sift
python -m scripts.ground_truth sift
```

Without a name, each command processes every dataset. If you see
`data/<ds>/meta.json missing`, this step has not been run for that dataset.

**Step 2. Run one method on that dataset:**

```bash
python -m scripts.run_bench --dataset sift --method flat
python -m scripts.run_bench --dataset sift --method lsh
python -m scripts.run_bench --dataset sift --method ivfrabitq
python -m scripts.run_bench --dataset sift --method ivfpq
python -m scripts.run_bench --dataset sift --method hnsw       # slowest
```

For COCO, which has two query sets, add `--queries i2i` or `--queries t2i` to run just one
(by default both run):

```bash
python -m scripts.run_bench --dataset coco --queries t2i --method ivfpq
```

**Step 3. Repeat for the other datasets** (`yahoo`, `coco`, `agnews`): run step 1 with the
new name, then step 2.

**Step 4. Extra experiments**, each of which can also be run on its own once the
datasets they need are prepared (ablations default to `sift` and `yahoo`):

```bash
python -m scripts.ablations sift
python -m scripts.ood_study
python -m scripts.failure_cases
python -m scripts.lecture_check
python -m scripts.scaling
python -m scripts.make_plots       # works on whatever is in results/
```

Tips: run timed benchmarks one at a time with other apps closed, because latency is measured.
`python -m scripts.run_bench --all` runs every dataset, query set and method in one process.

## Notes

- Settings and grids live only in `config.py`.
- Below 12 GB of RAM, AG News is cut to its first 400,000 vectors (shown in table T1).
- Downloads resume; if one keeps failing, download the URL from `config.py` in a browser
  and save it under `data/raw/` with the same file name.
- Do not commit `data/` or `indexes/`; commit `results/`.
