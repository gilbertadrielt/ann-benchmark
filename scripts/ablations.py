"""Step 5a: the two ablations that need new runs. A1, A3 and A5 reuse the main rows.

A2  PQ with asymmetric (ADC) vs symmetric (SDC) distances, M in {16, 32, 64},
    exhaustive search over the first A2_QUERIES queries.
A4  IVF-RaBitQ with 1-bit vs 4-bit codes and query bits qb in {0, 1, 4, 8}
    (qb = 0 keeps the query in float32), at nlist = SMALL_NLIST, on A4_QUERIES queries.
    qb = 0 is about 100x slower in Faiss 1.15.1, so it runs with k_factor = 1 only.

Rows go to results/<ds>.csv with experiment = "A2" or "A4".

    python -m scripts.ablations [datasets...]      # default: sift yahoo
"""
import sys

import config
from ann import bench, data, io


def a2_sweeps(method, bp):
    return [[{"sdc": False}], [{"sdc": True}]]


def a4_sweeps(method, bp):
    probes = [p for p in config.NPROBES if p <= min(bp["nlist"], config.A4_MAX_NPROBE)]
    return [[{"nprobe": p, "k_factor": kf, "qb": qb} for p in probes]
            for qb in config.A4_QBS for kf in ([1] if qb == 0 else config.A4_KFACTORS)]


def run(ds):
    xb, xt = data.load_db(ds), data.load_train(ds)
    log = io.results_csv(ds)
    n = config.A2_QUERIES
    xq = data.load_queries(ds, "main", n)
    gt_I, gt_D = data.load_gt(ds, "main", n)
    print(f"[ablations] {ds} A2: PQ ADC vs SDC on {len(xq)} queries", flush=True)
    bench.run("A2", ds, "main", "pq", [{"M": m} for m in config.PQ_MS], a2_sweeps,
              xb, xt, xq, gt_I, gt_D, log, early_stop=False)

    n = config.A4_QUERIES
    xq = data.load_queries(ds, "main", n)
    gt_I, gt_D = data.load_gt(ds, "main", n)
    print(f"[ablations] {ds} A4: RaBitQ bits x query bits on {len(xq)} queries", flush=True)
    bench.run("A4", ds, "main", "ivfrabitq", [{"nlist": config.SMALL_NLIST, "bits": b} for b in (1, 4)],
              a4_sweeps, xb, xt, xq, gt_I, gt_D, log)


def main(argv):
    for ds in (argv or config.ABLATION_DATASETS):
        run(ds)


if __name__ == "__main__":
    main(sys.argv[1:])
