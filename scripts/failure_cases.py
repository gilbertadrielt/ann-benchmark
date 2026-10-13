"""Step 5d (E5): which queries fail, and is it the query or the method?

For each dataset/query set and each ANN method, take the main-grid configuration whose
mean recall@10 is closest to 0.90, re-run it (all threads, cached index) and keep per-query recall.
Writes:
  results/failure/<ds>_<set>_<method>.npy   per-query recall@10
  results/failure/summary.csv               chosen config, mean recall, Spearman rho(recall, RC10)
  results/failure/worst_queries.csv         10 worst queries per dataset with RC10 and every method's recall
  results/failure/overlap.csv               Jaccard overlap of the 50 worst queries between methods

    python -m scripts.failure_cases
"""
import itertools
import json

import faiss
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config
from ann import bench, data, io, methods
from ann.metrics import all_threads, recall_hits

ANN = ["lsh", "ivfpq", "hnsw", "ivfrabitq"]


def main():
    out = config.RESULTS / "failure"
    out.mkdir(parents=True, exist_ok=True)
    df = io.load_results()
    df = df[df.experiment == "main"]
    summary, worst, overlap = [], [], []
    for ds, qs in config.TARGETS:
        if not data.has_gt(ds, qs):
            continue
        sub = df[(df.dataset == ds) & (df.query_set == qs)]
        if sub.empty:
            continue
        xb, xt = data.load_db(ds), data.load_train(ds)
        xq = data.load_queries(ds, qs)
        gt_I, gt_D = data.load_gt(ds, qs)
        rc = data.load_rc10(ds, qs)
        per = {}
        for m in ANN:
            s = sub[sub.method == m]
            if s.empty:
                continue
            row = s.iloc[(s.recall10 - 0.90).abs().argsort().iloc[0]]
            bp, sp = json.loads(row.build_params), json.loads(row.search_params)
            index, _ = bench.get_index(ds, m, bp, xb, xt)
            methods.set_search_params(index, sp)
            all_threads()
            _, I = index.search(xq, config.K)
            hits = recall_hits(I, xq, xb, gt_D)
            np.save(out / f"{ds}_{qs}_{m}.npy", hits)
            per[m] = hits
            rho = spearmanr(hits, rc).statistic if hits.std() > 0 else float("nan")
            summary.append({"dataset": ds, "query_set": qs, "method": m, "build_params": row.build_params,
                            "search_params": row.search_params, "mean_recall10": float(hits.mean()),
                            "share_below_0.5": float((hits < 0.5).mean()), "spearman_rho_rc10": float(rho)})
            print(f"[failure_cases] {ds}/{qs} {m}: recall {hits.mean():.3f}, rho(recall, RC10) = {rho:.2f}")
            del index
        if not per:
            continue
        mean_hits = np.mean(list(per.values()), axis=0)
        for qi in np.argsort(mean_hits)[:10]:
            worst.append({"dataset": ds, "query_set": qs, "query": int(qi), "rc10": float(rc[qi]),
                          **{f"recall_{m}": float(per[m][qi]) for m in per}})
        n_worst = min(50, len(xq))
        sets = {m: set(np.argsort(h, kind="stable")[:n_worst]) for m, h in per.items()}
        for a, b in itertools.combinations(sets, 2):
            overlap.append({"dataset": ds, "query_set": qs, "method_a": a, "method_b": b,
                            "jaccard_worst50": len(sets[a] & sets[b]) / len(sets[a] | sets[b])})
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(worst).to_csv(out / "worst_queries.csv", index=False)
    pd.DataFrame(overlap).to_csv(out / "overlap.csv", index=False)


if __name__ == "__main__":
    faiss.omp_set_num_threads(1)
    main()
