"""Step 2: exact top-100 ground truth, sanity checks and dataset statistics.

Writes data/<ds>/gt_I_<set>.npy, gt_D_<set>.npy (squared L2), rc10_<set>.npy,
results/dataset_stats.csv (table T1) and results/sanity_checks.csv.
A failed sanity check stops the pipeline with a clear message.

    python -m scripts.ground_truth [datasets...]
"""
import sys

import faiss
import numpy as np
import pandas as pd

import config
from ann import data
from ann.metrics import all_threads, relative_contrast


def exact_search(xb, xq, k=config.GT_K, chunk=1000):
    all_threads()
    index = faiss.IndexFlatL2(xb.shape[1])
    index.add(xb)
    D = np.empty((len(xq), k), dtype=np.float32)
    I = np.empty((len(xq), k), dtype=np.int64)
    for s in range(0, len(xq), chunk):
        D[s:s + chunk], I[s:s + chunk] = index.search(xq[s:s + chunk], k)
    return D, I


def _upsert(path, rows, keys):
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        merged = old.merge(new[keys], on=keys, how="left", indicator=True)
        old = old[(merged["_merge"] == "left_only").values]
        new = pd.concat([old, new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)


def run(ds):
    meta = data.meta(ds)
    d_dir = data.ds_dir(ds)
    xb = data.load_db(ds)
    rc_idx = np.load(d_dir / "rc_idx.npy")
    xs = np.asarray(xb[rc_idx])
    stats, checks = [], []
    for qs in meta["query_sets"]:
        xq = data.load_queries(ds, qs)
        if data.has_gt(ds, qs):
            gt_I, gt_D = data.load_gt(ds, qs)
            print(f"[ground_truth] {ds}/{qs}: already computed")
        else:
            print(f"[ground_truth] {ds}/{qs}: exact top-{config.GT_K} for {len(xq):,} queries over {len(xb):,} vectors", flush=True)
            gt_D, gt_I = exact_search(xb, xq)
            np.save(d_dir / f"gt_I_{qs}.npy", gt_I)
            np.save(d_dir / f"gt_D_{qs}.npy", gt_D)

        # 1. agreement with the file's own neighbours (skipped if the database was subsampled)
        prov_path = d_dir / f"gt_provided_{qs}.npy"
        if prov_path.exists():
            prov = np.load(prov_path)
            agree = float(np.mean([len(set(a[:10]) & set(b[:10])) / 10 for a, b in zip(gt_I, prov)]))
            ok = agree >= 0.95
            checks.append({"dataset": ds, "query_set": qs, "check": "agreement with provided top-10",
                           "value": round(agree, 4), "pass": ok,
                           "note": "" if agree >= 0.99 else "small gaps come from float16 storage or ties"})
            if not ok:
                raise SystemExit(f"{ds}/{qs}: only {agree:.3f} agreement with the provided neighbours. "
                                 "Usually a missed normalisation or misaligned query rows; check prepare.py output.")
        # 2. self-match: database vectors find themselves first at distance 0
        if qs == meta["query_sets"][0]:
            rng = np.random.default_rng(config.SEED)
            ids = rng.choice(len(xb), 5, replace=False)
            sD, sI = exact_search(xb, np.ascontiguousarray(xb[ids]), k=1)
            ok = bool(np.all(sD[:, 0] <= 1e-3 * max(1.0, float(np.abs(xb[ids]).max()) ** 2)))
            checks.append({"dataset": ds, "query_set": qs, "check": "self-match at distance 0",
                           "value": float(sD.max()), "pass": ok, "note": ""})
            if not ok:
                raise SystemExit(f"{ds}: self-match failed; the database file may be corrupt")
        # 3. ties at the 10th/11th neighbour (why recall is tie-aware)
        ties = int(np.sum(np.isclose(gt_D[:, 9], gt_D[:, 10], rtol=1e-6, atol=1e-9)))
        checks.append({"dataset": ds, "query_set": qs, "check": "queries with a tie at rank 10/11",
                       "value": ties, "pass": True, "note": ""})

        rc_path = d_dir / f"rc10_{qs}.npy"
        rc = relative_contrast(xq, xs, gt_D)
        np.save(rc_path, rc)
        stats.append({"dataset": ds, "label": config.DATASETS[ds]["label"], "query_set": qs,
                      "n_db": meta["n_db"], "dim": meta["dim"], "n_queries": len(xq), "metric": meta["metric"],
                      "raw_size_mb": round(meta["n_db"] * meta["dim"] * 4 / 1e6), "subsampled": meta["subsampled"],
                      "rc10_median": float(np.median(rc)), "rc10_q25": float(np.percentile(rc, 25)),
                      "rc10_q75": float(np.percentile(rc, 75))})
        print(f"    RC10 median {np.median(rc):.3f} (IQR {np.percentile(rc, 25):.3f}-{np.percentile(rc, 75):.3f}), ties {ties}")

    # 4. COCO pairing: caption row i belongs to image row i
    if ds == "coco":
        qi, qt = data.load_queries(ds, "i2i"), data.load_queries(ds, "t2i")
        paired = float((qi * qt).sum(1).mean())
        shuffled = float((qi * qt[np.random.default_rng(config.SEED).permutation(len(qt))]).sum(1).mean())
        ok = paired > shuffled + 0.02
        checks.append({"dataset": ds, "query_set": "i2i/t2i", "check": "caption/image pairing (cosine paired vs shuffled)",
                       "value": round(paired - shuffled, 4), "pass": ok,
                       "note": f"paired {paired:.3f}, shuffled {shuffled:.3f}"})
        if not ok:
            print("    WARNING: caption rows do not pair with image rows; E4 still runs (it compares query sets as a whole)")
    _upsert(config.RESULTS / "dataset_stats.csv", stats, ["dataset", "query_set"])
    _upsert(config.RESULTS / "sanity_checks.csv", checks, ["dataset", "query_set", "check"])
    for c in checks:
        print(f"    check: {c['check']}: {c['value']} -> {'PASS' if c['pass'] else 'WARN'}")


def main(argv):
    for ds in (argv or config.BENCH_ORDER):
        run(ds)


if __name__ == "__main__":
    main(sys.argv[1:])
