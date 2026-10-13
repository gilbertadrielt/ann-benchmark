"""Loading prepared arrays, and generating synthetic HDF5 files for --smoke runs."""
import json

import h5py
import numpy as np

import config


def ds_dir(ds):
    return config.DATA / ds


def meta(ds):
    path = ds_dir(ds) / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python -m scripts.prepare` first")
    return json.loads(path.read_text())


def load_db(ds):
    """Database vectors as a read-only memory map (float32, C-contiguous)."""
    return np.load(ds_dir(ds) / "xb.npy", mmap_mode="r")


def load_train(ds):
    return np.load(ds_dir(ds) / "xt.npy")


def load_queries(ds, qs, n=None):
    xq = np.load(ds_dir(ds) / f"xq_{qs}.npy")
    return np.ascontiguousarray(xq[:n]) if n else xq


def load_gt(ds, qs, n=None):
    d = ds_dir(ds)
    gt_I, gt_D = np.load(d / f"gt_I_{qs}.npy"), np.load(d / f"gt_D_{qs}.npy")
    return (gt_I[:n], gt_D[:n]) if n else (gt_I, gt_D)


def load_rc10(ds, qs):
    return np.load(ds_dir(ds) / f"rc10_{qs}.npy")


def has_gt(ds, qs):
    return (ds_dir(ds) / f"gt_D_{qs}.npy").exists()


# ---------------------------------------------------------------- synthetic data (--smoke only)
def _exact_neighbors(train, test, k=100):
    import faiss
    index = faiss.IndexFlatL2(train.shape[1])
    index.add(train)
    D, I = index.search(test, k)
    return np.sqrt(np.clip(D, 0, None)), I


def _write_h5(path, train, test, distance, store_dtype="float32"):
    t32 = np.ascontiguousarray(train.astype(np.float32))
    q32 = np.ascontiguousarray(test.astype(np.float32))
    if distance in ("angular", "normalized"):
        t32 = t32 / np.linalg.norm(t32, axis=1, keepdims=True)
        q32 = q32 / np.linalg.norm(q32, axis=1, keepdims=True)
    D, I = _exact_neighbors(np.ascontiguousarray(t32), np.ascontiguousarray(q32))
    with h5py.File(path, "w") as f:
        f.attrs["distance"] = distance
        f.attrs["dimension"] = train.shape[1]
        f.attrs["point_type"] = "float"
        f.create_dataset("train", data=train.astype(store_dtype))
        f.create_dataset("test", data=test.astype(store_dtype))
        f.create_dataset("neighbors", data=I.astype(np.int64))
        f.create_dataset("distances", data=D.astype(np.float64))


def _clustered(rng, n, d, n_clusters, spread):
    centers = rng.standard_normal((n_clusters, d))
    labels = rng.integers(0, n_clusters, n)
    return centers[labels] + spread * rng.standard_normal((n, d)), centers


def make_synthetic(ds):
    """Write small files with the same names, keys, dtypes and quirks as the real datasets."""
    rng = np.random.default_rng(config.SEED)
    n, nq = config.SMOKE_SIZES[ds]
    d = config.DATASETS[ds]["dim"]
    config.RAW.mkdir(parents=True, exist_ok=True)
    if ds == "sift":                                   # non-negative integers 0..255, like SIFT
        x, _ = _clustered(rng, n + nq, d, 50, 0.6)
        x = np.clip(np.round(40 * (x + 2.5)), 0, 255)
        _write_h5(config.RAW / config.DATASETS[ds]["files"]["main"][0], x[:n], x[n:], "euclidean")
    elif ds == "yahoo":                                # unit-normalised sentence embeddings
        x, _ = _clustered(rng, n + nq, d, 80, 0.8)
        _write_h5(config.RAW / config.DATASETS[ds]["files"]["main"][0], x[:n], x[n:], "normalized")
    elif ds == "agnews":                               # unnormalised, high-dimensional
        x, _ = _clustered(rng, n + nq, d, 60, 0.7)
        x = x * rng.uniform(0.5, 2.0, (n + nq, 1))
        _write_h5(config.RAW / config.DATASETS[ds]["files"]["main"][0], x[:n], x[n:], "euclidean")
    elif ds == "coco":                                 # float16, shared image database, caption queries shifted
        imgs, _ = _clustered(rng, n + nq, d, 70, 0.7)
        imgs = 10 * imgs / np.linalg.norm(imgs, axis=1, keepdims=True)
        gap = rng.standard_normal(d)
        gap = 6 * gap / np.linalg.norm(gap)
        caps = imgs[n:] + gap + 4 * rng.standard_normal((nq, d)) / np.sqrt(d)
        files = config.DATASETS[ds]["files"]
        _write_h5(config.RAW / files["i2i"][0], imgs[:n], imgs[n:], "angular", "float16")
        _write_h5(config.RAW / files["t2i"][0], imgs[:n], caps, "angular", "float16")
    else:
        raise ValueError(ds)
