"""Step 5e (E7): reproduce the lecture's SIFT1M numbers with our own runs.

Slide 63 (from Pinecone's PQ article): IndexPQ m = 8, 8 bits, and IndexIVFPQ nlist = 2048,
m = 8, nprobe = 48, with recall measured at k = 100. We use the same settings, but average
over our 2,000 queries instead of one.
Slide 36: IndexLSH(128, nbits, rotate=True, train_thresholds=False), top 100 by Hamming
distance, mean cosine similarity between each query and its 100 results, against nbits.

Writes results/e7_lecture.csv.

    python -m scripts.lecture_check
"""
import faiss
import numpy as np
import pandas as pd

import config
from ann import bench, data
from ann.metrics import time_search

DS = "sift"


def mean_cosine(xq, xb, I):
    q = xq / np.linalg.norm(xq, axis=1, keepdims=True)
    out = []
    for s in range(0, len(xq), 200):
        ids = I[s:s + 200]
        v = np.asarray(xb[np.where(ids >= 0, ids, 0).ravel()], dtype=np.float32).reshape(ids.shape + (xb.shape[1],))
        v = v / np.maximum(np.linalg.norm(v, axis=2, keepdims=True), 1e-12)
        out.append((v * q[s:s + 200, None, :]).sum(-1).mean(1))
    return float(np.concatenate(out).mean())


def id_recall_at(I, gt_I, k):
    return float(np.mean([len(set(a[:k]) & set(b[:k])) / k for a, b in zip(I, gt_I)]))


def main():
    xb, xt = data.load_db(DS), data.load_train(DS)
    xq = data.load_queries(DS, "main")
    gt_I, _ = data.load_gt(DS, "main")
    k = 100
    rows = [{"item": "exact top-100", "setting": "IndexFlatL2", "recall_at_100": 1.0,
             "mean_cosine_top100": mean_cosine(xq, xb, gt_I), "qps": None, "bytes_per_vec": 4 * xb.shape[1]}]

    for method, bp, sp, label in [
        ("pq", {"M": 8}, {}, "slide 63: PQ m=8, 8 bits"),
        ("ivfpq", {"nlist": config.E7_IVF_NLIST, "M": 8}, {"nprobe": config.E7_NPROBE},
         f"slide 63: IVF{config.E7_IVF_NLIST},PQ8, nprobe={config.E7_NPROBE}"),
    ]:
        index, info = bench.get_index(DS, method, bp, xb, xt, refine=False)
        if "nprobe" in sp:
            faiss.extract_index_ivf(index).nprobe = sp["nprobe"]
        _, I, best, _ = time_search(index, xq, k)
        rows.append({"item": label, "setting": str(bp | sp), "recall_at_100": id_recall_at(I, gt_I, k),
                     "mean_cosine_top100": mean_cosine(xq, xb, I), "qps": len(xq) / best,
                     "bytes_per_vec": info["base_bytes"] / len(xb)})
        print(f"[lecture_check] {label}: recall@100 {rows[-1]['recall_at_100']:.3f}, QPS {rows[-1]['qps']:.0f}")
        del index

    for nbits in config.E7_LSH_BITS:
        lsh = faiss.IndexLSH(xb.shape[1], nbits, True, False)
        lsh.train(xt)
        lsh.add(xb)
        _, I, best, _ = time_search(lsh, xq, k)
        rows.append({"item": "slide 36: LSH bits curve", "setting": f"nbits={nbits}", "nbits": nbits,
                     "recall_at_100": id_recall_at(I, gt_I, k), "mean_cosine_top100": mean_cosine(xq, xb, I),
                     "qps": len(xq) / best, "bytes_per_vec": lsh.code_size})
        print(f"[lecture_check] LSH nbits={nbits}: mean cosine of top-100 {rows[-1]['mean_cosine_top100']:.3f}")
        del lsh
    pd.DataFrame(rows).to_csv(config.RESULTS / "e7_lecture.csv", index=False)


if __name__ == "__main__":
    main()
