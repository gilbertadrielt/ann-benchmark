"""Step 1b: turn the HDF5 files into float32 NumPy arrays under data/<dataset>/.

For each dataset this writes:
  xb.npy                database vectors (normalised for cosine datasets)
  xq_<set>.npy          the first N_QUERIES queries of each query set
  xt.npy                fixed-seed training sample (up to 250,000 vectors)
  rc_idx.npy            database sample for relative contrast
  gt_provided_<set>.npy the file's own top-100 ids, used only as a sanity check
  meta.json             sizes, metric and provenance
COCO gets one shared image database with query sets i2i and t2i, plus
xq_t2i_calib.npy: caption queries never used for evaluation (E6 only).

    python -m scripts.prepare [datasets...] [--force]
"""
import json
import sys

import faiss
import h5py
import numpy as np

import config
from ann.data import ds_dir


def _read(path, key):
    with h5py.File(path, "r") as f:
        return f[key][:].astype(np.float32, copy=False)


def _read_opt(path, key, n):
    with h5py.File(path, "r") as f:
        return f[key][:n] if key in f else None


def _norm(x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    faiss.normalize_L2(x)
    return x


def _stats(name, x):
    norms = np.linalg.norm(x[: min(len(x), 100_000)], axis=1)
    print(f"    {name}: shape {x.shape}, values [{x.min():.3g}, {x.max():.3g}], "
          f"norm min/mean/max {norms.min():.3f}/{norms.mean():.3f}/{norms.max():.3f}")


def prepare(ds, force=False):
    out = ds_dir(ds)
    if (out / "meta.json").exists() and not force:
        print(f"[prepare] {ds}: already prepared")
        return
    spec = config.DATASETS[ds]
    cos = spec["metric"] == "cos"
    files = {qs: config.RAW / fn for qs, (fn, _) in spec["files"].items()}
    first = next(iter(files.values()))
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.SEED)
    print(f"[prepare] {ds}")

    xb = _read(first, "train")
    n_full = len(xb)
    if ds == "coco":
        other = _read(files["t2i"], "train")
        if other.shape != xb.shape or not np.array_equal(other, xb):
            print("    WARNING: i2i and t2i databases differ; using the i2i database for both query sets")
        del other
    subsampled = bool(spec["max_db"] and n_full > spec["max_db"])
    if subsampled:
        print(f"    machine has {config.RAM_GB:.0f} GB RAM: keeping the first {spec['max_db']:,} of {n_full:,} vectors")
        xb = np.ascontiguousarray(xb[: spec["max_db"]])
    xb = _norm(xb) if cos else np.ascontiguousarray(xb)
    np.save(out / "xb.npy", xb)
    _stats("xb", xb)

    n_queries = {}
    for qs, path in files.items():
        xq_all = _read(path, "test")
        xq_all = _norm(xq_all) if cos else np.ascontiguousarray(xq_all)
        xq = np.ascontiguousarray(xq_all[: config.N_QUERIES])
        np.save(out / f"xq_{qs}.npy", xq)
        n_queries[qs] = len(xq)
        _stats(f"xq_{qs}", xq)
        if ds == "coco" and qs == "t2i":
            start = max(len(xq), len(xq_all) // 2)       # rows 5,000-9,999: disjoint from evaluation rows
            calib = np.ascontiguousarray(xq_all[start: start + config.N_CALIB])
            np.save(out / "xq_t2i_calib.npy", calib)
            print(f"    xq_t2i_calib: rows {start}-{start + len(calib) - 1} ({len(calib)} captions, E6 only)")
        prov = _read_opt(path, "neighbors", len(xq))
        if prov is not None and not subsampled:
            np.save(out / f"gt_provided_{qs}.npy", prov.astype(np.int64))

    n_train = min(config.N_TRAIN, len(xb))
    xt = np.ascontiguousarray(xb[np.sort(rng.choice(len(xb), n_train, replace=False))])
    np.save(out / "xt.npy", xt)
    np.save(out / "rc_idx.npy", np.sort(rng.choice(len(xb), min(config.N_RC, len(xb)), replace=False)))
    meta = {"dataset": ds, "n_db": len(xb), "n_db_file": n_full, "subsampled": subsampled,
            "dim": xb.shape[1], "metric": spec["metric"], "query_sets": list(files),
            "n_queries": n_queries, "n_train": n_train, "normalised": cos,
            "source_files": {qs: p.name for qs, p in files.items()}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"    n_db={len(xb):,} dim={xb.shape[1]} train sample={n_train:,} queries={n_queries}")


def main(argv):
    force = "--force" in argv
    names = [a for a in argv if not a.startswith("--")] or config.BENCH_ORDER
    for ds in names:
        prepare(ds, force)


if __name__ == "__main__":
    main(sys.argv[1:])
