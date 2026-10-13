"""Steps 5b-5c: distribution-shift diagnostics (E4) and gap-corrected routing (E6) on COCO.

The I2I vs T2I benchmark rows themselves come from run_bench (same cached indexes,
two query sets). This script adds what explains them:
  results/ood/diagnostics.csv      per query: set, cosine to 1st neighbour, RC10, distance to nearest IVF centroid
  results/ood/pca.npz              2-D PCA of 5,000 database vectors, 1,000 image and 1,000 caption queries
  results/e6_routing.csv           recall@10 vs nprobe for routing shifts alpha in {0, 0.5, 1}

E6: a caption query is shifted by the mean image-minus-caption offset ONLY when choosing
which IVF cells to scan; distances are still computed with the original query. The offset is
estimated on calibration captions that are never used for evaluation.

    python -m scripts.ood_study
"""
import faiss
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import config
from ann import bench, data
from ann.metrics import all_threads, recall_hits

DS = "coco"


def main():
    out = config.RESULTS / "ood"
    out.mkdir(parents=True, exist_ok=True)
    xb, xt = data.load_db(DS), data.load_train(DS)
    xq = {qs: data.load_queries(DS, qs) for qs in ("i2i", "t2i")}
    gt = {qs: data.load_gt(DS, qs) for qs in ("i2i", "t2i")}
    calib = np.load(data.ds_dir(DS) / "xq_t2i_calib.npy")

    all_threads()
    print(f"[ood_study] building IVF{config.SMALL_NLIST},Flat on {DS}", flush=True)
    owner, _ = bench.get_index(DS, "ivfflat", {"nlist": config.SMALL_NLIST}, xb, xt)
    ivf = faiss.downcast_index(owner)        # keep `owner` alive: it owns the C++ index that `ivf` points to
    centroids = ivf.quantizer.reconstruct_n(0, ivf.nlist)

    # ---- E4 diagnostics
    rows = []
    for qs in ("i2i", "t2i"):
        cD, _ = ivf.quantizer.search(xq[qs], 1)
        gt_D = gt[qs][1]
        cos1 = 1.0 - gt_D[:, 0] / 2.0                 # unit vectors: cos = 1 - d^2 / 2
        rc = data.load_rc10(DS, qs)
        rows.append(pd.DataFrame({"query_set": qs, "query": np.arange(len(xq[qs])), "cos_1nn": cos1,
                                  "rc10": rc, "centroid_dist": np.sqrt(np.clip(cD[:, 0], 0, None))}))
    diag = pd.concat(rows, ignore_index=True)
    diag.to_csv(out / "diagnostics.csv", index=False)
    print(diag.groupby("query_set")[["cos_1nn", "rc10", "centroid_dist"]].median().round(3).to_string())

    rng = np.random.default_rng(config.SEED)
    sample = np.asarray(xb[np.sort(rng.choice(len(xb), min(5000, len(xb)), replace=False))])
    pca = PCA(2, random_state=config.SEED).fit(sample)
    n_show = min(1000, len(xq["i2i"]))
    np.savez(out / "pca.npz", db=pca.transform(sample), i2i=pca.transform(xq["i2i"][:n_show]),
             t2i=pca.transform(xq["t2i"][:n_show]), explained=pca.explained_variance_ratio_)

    # ---- E6 gap-corrected routing
    gap = np.asarray(xb).mean(0) - calib.mean(0)       # image mean minus caption mean (calibration rows only)
    q = xq["t2i"]
    gt_D = gt["t2i"][1]
    res = []
    for alpha in config.E6_ALPHAS:
        q_route = np.ascontiguousarray(q + alpha * gap, dtype=np.float32)
        faiss.normalize_L2(q_route)
        for nprobe in config.E6_NPROBES:
            if nprobe > ivf.nlist:
                continue
            ivf.nprobe = nprobe                         # search_preassigned checks Iq has nprobe columns
            _, Iq = ivf.quantizer.search(q_route, nprobe)          # cells chosen with the shifted query
            Dq = ((q[:, None, :] - centroids[Iq]) ** 2).sum(-1).astype(np.float32)
            _, I = ivf.search_preassigned(q, config.K, Iq, Dq)     # scored with the original query
            r = float(recall_hits(I, q, xb, gt_D).mean())
            res.append({"alpha": alpha, "nprobe": nprobe, "recall10": r})
            print(f"  E6 alpha={alpha:.1f} nprobe={nprobe:3d} recall={r:.3f}")
    pd.DataFrame(res).to_csv(config.RESULTS / "e6_routing.csv", index=False)


if __name__ == "__main__":
    main()
