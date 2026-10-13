"""Timing, recall, memory and query-difficulty measurements."""
import os
import time

import faiss
import numpy as np
from threadpoolctl import threadpool_limits

import config


def all_threads():
    faiss.omp_set_num_threads(os.cpu_count() or 1)


def time_search(index, xq, k, reps=config.TIMING_REPS, budget_s=config.TIMING_BUDGET_S):
    """Single-threaded batch search: one warm-up, then up to `reps` timed runs; best time kept.

    threadpoolctl also pins the BLAS library, which exact search and IVF cell
    assignment use; without it, QPS for those methods is inflated.
    """
    faiss.omp_set_num_threads(1)
    try:
        with threadpool_limits(limits=1):
            index.search(xq[: min(100, len(xq))], k)            # warm-up
            times, D, I = [], None, None
            for _ in range(reps):
                t0 = time.perf_counter()
                D, I = index.search(xq, k)
                times.append(time.perf_counter() - t0)
                if sum(times) > budget_s:
                    break
    finally:
        all_threads()
    return D, I, min(times), len(times)


def squared_dist_to_ids(xq, xb, ids):
    """Exact squared L2 distance from each query to each listed database id; -1 ids give inf."""
    ok = ids >= 0
    cand = np.asarray(xb[np.where(ok, ids, 0).ravel()], dtype=np.float32).reshape(ids.shape + (xb.shape[1],))
    d2 = ((cand - xq[:, None, :]) ** 2).sum(-1)
    d2[~ok] = np.inf
    return d2


def recall_hits(I, xq, xb, gt_D, k=config.K, chunk=500):
    """Per-query tie-aware recall@k.

    A returned id is a hit if its exact squared distance is within the k-th true
    squared distance (relative tolerance 1e-4 absorbs float32 rounding). Missing
    results (id -1) are misses. Returns an array of length n_queries in [0, 1].
    """
    out = np.empty(len(xq), dtype=np.float32)
    thr = gt_D[:, k - 1].astype(np.float64)
    for s in range(0, len(xq), chunk):
        e = min(s + chunk, len(xq))
        d2 = squared_dist_to_ids(xq[s:e], xb, I[s:e, :k])
        t = thr[s:e, None]
        out[s:e] = (d2 <= t * (1 + 1e-4) + 1e-6).sum(1) / k
    return out


def recall_at_k(I, xq, xb, gt_D, k=config.K):
    return float(recall_hits(I, xq, xb, gt_D, k).mean())


def id_recall(I, gt_I, k=config.K):
    """Plain set overlap of ids, kept as a cross-check of the tie-aware recall."""
    return float(np.mean([len(set(a[:k]) & set(b[:k])) / k for a, b in zip(I, gt_I)]))


def relative_contrast(xq, xb_sample, gt_D, k=config.K, chunk=500):
    """RC_k = mean distance from the query to the database / distance to its k-th neighbour. Low = hard."""
    xs = np.asarray(xb_sample, dtype=np.float32)
    xs_n = (xs ** 2).sum(1)
    mean_d = np.empty(len(xq), dtype=np.float64)
    for s in range(0, len(xq), chunk):
        q = xq[s:s + chunk]
        d2 = (q ** 2).sum(1)[:, None] + xs_n[None, :] - 2.0 * q @ xs.T
        mean_d[s:s + chunk] = np.sqrt(np.clip(d2, 0, None)).mean(1)
    kth = np.sqrt(np.clip(gt_D[:, k - 1].astype(np.float64), 1e-12, None))
    return (mean_d / kth).astype(np.float32)
