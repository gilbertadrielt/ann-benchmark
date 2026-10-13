"""Step 5f (E8, second half): how query time grows with database size on SIFT.

Prefixes of the SIFT database (100k, 250k, 500k, 1M) share the same queries and the same
250k training sample; exact ground truth is recomputed for each size, because neighbour
ids change with the subset. Three indexes, all with exact re-ranking:
  PQ32 (exhaustive scan), IVF1024,PQ32 and RR128,IVF1024,RaBitQ (1 bit).
Every (nprobe, k_factor) is logged to results/e8_scaling.csv; make_plots picks, per size
and method, the fastest configuration with recall@10 >= 0.90.
The first half of E8 (IVF-PQ vs IVF-RaBitQ at equal bytes on all four datasets) needs
no new runs: those builds are in the main grid.

    python -m scripts.scaling
"""
import numpy as np

import config
from ann import bench, data
from scripts.ground_truth import exact_search

DS = "sift"


def sweeps(method, bp):
    if method == "pq_refine":
        return [[{"k_factor": kf} for kf in (1, 2, 4, 8, 16, 32, 64)]]
    probes = [p for p in config.NPROBES if p <= bp["nlist"]]
    extra = {"qb": config.QB_DEFAULT} if method == "ivfrabitq" else {}
    return [[{"nprobe": p, "k_factor": kf, **extra} for p in probes] for kf in config.KFACTORS]


def main():
    xb_full, xt = data.load_db(DS), data.load_train(DS)
    xq = data.load_queries(DS, "main", config.E8_QUERIES)
    log = config.RESULTS / "e8_scaling.csv"
    for n in config.E8_SIZES:
        if n > len(xb_full):
            print(f"[scaling] skipping n={n:,}: database has only {len(xb_full):,} vectors")
            continue
        xb = np.ascontiguousarray(xb_full[:n])
        gt_path = data.ds_dir(DS) / f"gt_D_e8_{n}.npy"
        if gt_path.exists():
            gt_D, gt_I = np.load(gt_path), np.load(data.ds_dir(DS) / f"gt_I_e8_{n}.npy")
        else:
            gt_D, gt_I = exact_search(xb, xq)
            np.save(gt_path, gt_D)
            np.save(data.ds_dir(DS) / f"gt_I_e8_{n}.npy", gt_I)
        print(f"[scaling] n={n:,}", flush=True)
        for method, spec in [("pq_refine", config.E8_SPECS["pq"]), ("ivfpq", config.E8_SPECS["ivfpq"]),
                             ("ivfrabitq", config.E8_SPECS["ivfrabitq"])]:
            bp = {**spec, "n_db": n}
            bench.run("E8", DS, "main", method, [bp], sweeps, xb, xt, xq, gt_I, gt_D, log, cache=False)
        del xb


if __name__ == "__main__":
    main()
