"""Steps 3-4 (E1, E2): build every index in a method's grid, sweep its search grid, log to results/<ds>.csv.

    python -m scripts.run_bench --dataset sift --method hnsw
    python -m scripts.run_bench --dataset coco --queries t2i --method ivfpq
    python -m scripts.run_bench --all            # every dataset x query set x method

Indexes are cached under indexes/<ds>/ and shared by both COCO query sets.
Rows already in the CSV are skipped, so an interrupted run resumes where it stopped.
"""
import argparse

import config
from ann import bench, data, io


def run_one(ds, qs, method):
    meta = data.meta(ds)
    if qs not in meta["query_sets"]:
        raise SystemExit(f"{ds} has query sets {meta['query_sets']}, not {qs}")
    xb = data.load_db(ds)
    xt = data.load_train(ds)
    xq = data.load_queries(ds, qs)
    gt_I, gt_D = data.load_gt(ds, qs)
    print(f"[run_bench] {ds}/{qs} {method}: n_db={len(xb):,} d={xb.shape[1]} queries={len(xq):,}", flush=True)
    bench.run("main", ds, qs, method, config.build_grid(method, ds), config.search_sweeps,
              xb, xt, xq, gt_I, gt_D, io.results_csv(ds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(config.DATASETS))
    ap.add_argument("--queries", default=None, help="query set: main, or i2i / t2i for coco (default: all)")
    ap.add_argument("--method", choices=config.METHODS)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.all:
        for ds, qs in sorted(config.TARGETS, key=lambda t: config.BENCH_ORDER.index(t[0])):
            for m in config.METHODS:
                run_one(ds, qs, m)
        return
    if not (a.dataset and a.method):
        ap.error("give --dataset and --method, or --all")
    for qs in ([a.queries] if a.queries else data.meta(a.dataset)["query_sets"]):
        run_one(a.dataset, qs, a.method)


if __name__ == "__main__":
    main()
