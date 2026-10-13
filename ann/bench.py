"""Shared benchmark loop: build or load an index, sweep search parameters, log one row per config."""
import gc
import json
import os
import time

import faiss

import config
from ann import io, methods
from ann.metrics import all_threads, id_recall, recall_hits, time_search


def get_index(ds, method, bp, xb, xt, cache=True, refine=None):
    """Return (index ready to search, build info). Compressed indexes get the re-ranking layer
    unless refine=False (the cache always holds the base index only)."""
    n, d = xb.shape
    refine = method in methods.REFINED if refine is None else refine
    if method == "flat":
        all_threads()
        t0 = time.perf_counter()
        index = faiss.IndexFlatL2(d)
        index.add(xb)
        info = {"train_s": 0.0, "add_s": time.perf_counter() - t0, "build_threads": os.cpu_count(),
                "base_bytes": 4 * d * n}
        return index, info

    path = config.INDEXES / ds / f"{methods.tag(method, bp)}.faiss"
    info_path = path.with_suffix(".json")
    base, info = None, None
    if cache and path.exists() and info_path.exists():
        info = json.loads(info_path.read_text())
        if info.get("n_db") == n and info.get("dim") == d:
            base = faiss.read_index(str(path))
    if base is None:
        all_threads()
        base = methods.build_base(method, d, bp)
        t0 = time.perf_counter()
        base.train(xt)
        train_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        base.add(xb)
        add_s = time.perf_counter() - t0
        info = {"train_s": train_s, "add_s": add_s, "build_threads": os.cpu_count(), "n_db": n, "dim": d}
        if cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            faiss.write_index(base, str(tmp))
            os.replace(tmp, path)
            info["base_bytes"] = path.stat().st_size
            info_path.write_text(json.dumps(info))
        else:
            info["base_bytes"] = int(faiss.serialize_index(base).nbytes)
    if refine:
        return methods.wrap_refine(base, xb), info
    return base, info


def memory_fields(method, info, n, d):
    codes = info["base_bytes"]
    total = codes + (4 * d * n if method in methods.REFINED else 0)
    return {"total_bytes": total, "bytes_per_vec": total / n, "bytes_per_vec_codes": codes / n}


def evaluate(index, xq, xb, gt_I, gt_D, sp):
    """Time one search configuration and measure its recall. Returns (metrics, per-query hits)."""
    methods.set_search_params(index, sp)
    D, I, best_s, reps = time_search(index, xq, config.K)
    hits = recall_hits(I, xq, xb, gt_D)
    m = {"recall10": float(hits.mean()), "recall10_id": id_recall(I, gt_I),
         "qps": len(xq) / best_s, "latency_ms": 1000.0 * best_s / len(xq), "n_reps": reps}
    return m, hits


def _stops(r, prev, flat_steps):
    """Early-stop rule shared by the sweep and the resume check. Returns (stop, flat_steps)."""
    flat_steps = flat_steps + 1 if prev is not None and r - prev < config.PLATEAU_GAIN else 0
    return r >= config.RECALL_STOP or flat_steps >= config.PLATEAU_STEPS, flat_steps


def _finished(sweeps, done, key, early_stop):
    """True if every sweep of this build is already fully logged (so the index need not be loaded)."""
    for sweep in sweeps:
        prev, flat = None, 0
        for sp in sweep:
            if key(sp) not in done:
                return False
            r = done[key(sp)]
            stop, flat = _stops(r, prev, flat)
            prev = r
            if early_stop and stop:
                break
    return True


def run(experiment, ds, qs, method, build_grid, sweeps_fn, xb, xt, xq, gt_I, gt_D, log_path,
        cache=True, ds_cache=None, early_stop=True, verbose=True):
    """Build every config in build_grid, sweep its search params, append one CSV row per config.

    A sweep of increasing search budget stops once recall >= RECALL_STOP or recall has stopped
    improving (larger budgets are only slower). Resumable: configs already in log_path are skipped.
    """
    done = io.done_rows(log_path)
    n, d = xb.shape
    for bp in build_grid:
        sweeps = sweeps_fn(method, bp)

        def key(sp):
            return (experiment, qs, method, io.pjson(bp), io.pjson(sp))
        if _finished(sweeps, done, key, early_stop):
            continue
        index, info = get_index(ds_cache or ds, method, bp, xb, xt, cache=cache)
        mem = memory_fields(method, info, n, d)
        for sweep in sweeps:
            prev, flat_steps = None, 0
            for sp in sweep:
                if key(sp) in done:
                    r = done[key(sp)]
                else:
                    m, _ = evaluate(index, xq, xb, gt_I, gt_D, sp)
                    row = io.base_row(experiment, ds, qs, method, bp, sp, n, d, len(xq))
                    row.update(m)
                    row.update(mem)
                    row.update({k: info[k] for k in ("train_s", "add_s", "build_threads")})
                    row["recipe"] = methods.describe(method, d, bp)
                    io.append_row(log_path, row)
                    r = m["recall10"]
                    if verbose:
                        print(f"  {experiment:5s} {ds}/{qs} {method:9s} {io.pjson(bp):38s} {io.pjson(sp):40s} "
                              f"recall={r:.3f} qps={m['qps']:.0f}", flush=True)
                stop, flat_steps = _stops(r, prev, flat_steps)
                prev = r
                if early_stop and stop:
                    break
        del index
        gc.collect()
