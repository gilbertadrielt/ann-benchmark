"""Every setting of the benchmark: paths, datasets, seeds and parameter grids.

Nothing else in the repository hard-codes a path, a grid or a seed.
Set ANN_SMOKE=1 (or run `python run_all.py --smoke`) to run the whole
pipeline on small synthetic data in a separate `smoke/` folder.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SMOKE = os.environ.get("ANN_SMOKE", "0") == "1"
WORK = ROOT / "smoke" if SMOKE else ROOT
DATA = WORK / "data"
RAW = DATA / "raw"
INDEXES = WORK / "indexes"
RESULTS = WORK / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

SEED = 2026
K = 10                                   # recall@10 everywhere
GT_K = 100                               # depth of our exact ground truth
N_QUERIES = 200 if SMOKE else 2000       # evaluation queries per query set
N_TRAIN = 250_000                        # training sample for IVF / PQ / LSH thresholds
N_RC = 2_000 if SMOKE else 10_000        # database sample for relative contrast
N_CALIB = 300 if SMOKE else 5_000        # COCO caption queries used only by E6
TIMING_REPS = 3                          # timed repetitions, best one kept
TIMING_BUDGET_S = 30.0                   # stop repeating once a config has used this much time
RECALL_STOP = 0.999                      # larger search budgets after this are dominated, skipped
PLATEAU_GAIN = 0.001                     # ...and so are budgets once recall gains < 0.001
PLATEAU_STEPS = 2                        #    for 2 steps in a row (the recall cap of a compressed code)


def _ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().total / 1e9
    except Exception:
        return 16.0


RAM_GB = _ram_gb()
LOW_MEMORY = RAM_GB < 12                 # below 12 GB, AG News is cut to 400,000 vectors

_ANNB = "http://ann-benchmarks.com"
_VIBE = "https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main"
_COCO = "https://github.com/fabiocarrara/str-encoders/releases/download/v0.1.3"

# files: query_set -> (file name, url). All files share the ANN-Benchmarks HDF5 layout:
# datasets train (database), test (queries), neighbors, distances (true top-100).
DATASETS = {
    "sift": {
        "files": {"main": ("sift-128-euclidean.hdf5", f"{_ANNB}/sift-128-euclidean.hdf5")},
        "metric": "l2", "dim": 128, "max_db": None,
        "label": "SIFT1M", "approx_mb": 501,
    },
    "yahoo": {
        "files": {"main": ("yahoo-minilm-384-normalized.hdf5", f"{_VIBE}/yahoo-minilm-384-normalized.hdf5")},
        "metric": "cos", "dim": 384, "max_db": None,
        "label": "Yahoo-MiniLM-384", "approx_mb": 1040,
    },
    "coco": {
        "files": {"i2i": ("coco-i2i-512-angular.hdf5", f"{_COCO}/coco-i2i-512-angular.hdf5"),
                  "t2i": ("coco-t2i-512-angular.hdf5", f"{_COCO}/coco-t2i-512-angular.hdf5")},
        "metric": "cos", "dim": 512, "max_db": None,
        "label": "COCO-CLIP", "approx_mb": 272,
    },
    "agnews": {
        "files": {"main": ("agnews-mxbai-1024-euclidean.hdf5", f"{_VIBE}/agnews-mxbai-1024-euclidean.hdf5")},
        "metric": "l2", "dim": 1024, "max_db": 400_000 if LOW_MEMORY else None,
        "label": "AG News-mxbai-1024", "approx_mb": 3160,
    },
}
DATASET_ORDER = ["agnews", "sift", "yahoo", "coco"]      # download order: largest first
BENCH_ORDER = ["sift", "yahoo", "coco", "agnews"]        # run order: cheapest first
TARGETS = [("sift", "main"), ("yahoo", "main"), ("coco", "i2i"), ("coco", "t2i"), ("agnews", "main")]

METHODS = ["flat", "lsh", "ivfpq", "hnsw", "ivfrabitq"]
METHOD_LABEL = {"flat": "Exact (Flat)", "lsh": "LSH", "ivfpq": "IVF-PQ", "hnsw": "HNSW",
                "ivfrabitq": "IVF-RaBitQ", "pq": "PQ (exhaustive)", "ivfflat": "IVF-Flat"}
# Categorical colours in fixed order (validated for colour-blind separation);
# Flat is a neutral grey baseline.
METHOD_COLOR = {"flat": "#6b6a65", "lsh": "#2a78d6", "ivfpq": "#eb6834", "hnsw": "#1baf7a",
                "ivfrabitq": "#eda100", "pq": "#e87ba4"}
METHOD_MARKER = {"flat": "*", "lsh": "o", "ivfpq": "s", "hnsw": "^", "ivfrabitq": "D", "pq": "v"}

# ---------------------------------------------------------------- grids
if SMOKE:
    NLISTS = {"coco": [64]}
    DEFAULT_NLISTS = [64]
    NPROBES = [1, 4, 16, 64]
    KFACTORS = [1, 4]
    EFSEARCH = [10, 32, 128]
    LSH_KFACTORS = [1, 16]
    HNSW_BUILD = [{"M": 8, "efC": 40}, {"M": 16, "efC": 40}]
    LSH_BITS = [64, 256]
else:
    NLISTS = {"coco": [256, 1024]}
    DEFAULT_NLISTS = [1024, 4096]
    NPROBES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    KFACTORS = [1, 4, 16]
    EFSEARCH = [10, 16, 32, 64, 128, 256, 512]
    LSH_KFACTORS = [1, 4, 16, 64]
    HNSW_BUILD = [{"M": m, "efC": 200} for m in (8, 16, 32, 48)]
    LSH_BITS = [64, 128, 256, 512, 1024, 2048]

PQ_MS = [16, 32, 64]
QB_DEFAULT = 4                           # RaBitQ query bits in the main grid
SMALL_NLIST = 64 if SMOKE else 1024      # nlist used by A3, A4, E6, E8
# E8: IVF-PQ code size that matches 1-bit RaBitQ's d/8 bytes, at a shared nlist
EQUAL_BYTES = {"sift": (SMALL_NLIST, 16), "yahoo": (SMALL_NLIST, 48),
               "coco": (SMALL_NLIST, 64), "agnews": (64 if SMOKE else 4096, 128)}


def nlists(ds):
    return NLISTS.get(ds, DEFAULT_NLISTS)


def build_grid(method, ds):
    """Build-time parameters for one method on one dataset (main experiment, E1)."""
    reduced = ds == "agnews"
    if method == "flat":
        return [{}]
    if method == "lsh":
        if reduced:
            bits = [64, 256] if SMOKE else [256, 1024, 2048]
            return [{"nbits": b, "train_thresholds": True} for b in bits]
        return [{"nbits": b, "train_thresholds": t} for b in LSH_BITS for t in (False, True)]
    if method == "ivfpq":
        if reduced:
            nl = 64 if SMOKE else 4096
            grid = [{"nlist": nl, "M": m} for m in ([32, 128] if SMOKE else [32, 64, 128])]
        else:
            grid = [{"nlist": nl, "M": m} for nl in nlists(ds) for m in (PQ_MS[:1] if SMOKE else PQ_MS)]
        nl, m = EQUAL_BYTES[ds]
        if {"nlist": nl, "M": m} not in grid:
            grid.append({"nlist": nl, "M": m})       # the equal-bytes build for E8
        return grid
    if method == "hnsw":
        if reduced:
            return [{"M": 16, "efC": 40 if SMOKE else 100}]
        return list(HNSW_BUILD)
    if method == "ivfrabitq":
        nls = [64 if SMOKE else 4096] if reduced else nlists(ds)
        return [{"nlist": nl, "bits": b} for nl in nls for b in (1, 4)]
    raise ValueError(method)


def search_sweeps(method, bp):
    """Search parameters as sweeps of increasing budget; a sweep stops once recall >= RECALL_STOP
    or recall stops improving (see PLATEAU_GAIN)."""
    if method == "flat":
        return [[{}]]
    if method == "lsh":
        return [[{"k_factor": kf} for kf in LSH_KFACTORS]]
    if method == "hnsw":
        return [[{"efSearch": ef} for ef in EFSEARCH]]
    if method in ("ivfpq", "ivfrabitq"):
        probes = [p for p in NPROBES if p <= bp["nlist"]]
        extra = {"qb": QB_DEFAULT} if method == "ivfrabitq" else {}
        return [[{"nprobe": p, "k_factor": kf, **extra} for p in probes] for kf in KFACTORS]
    raise ValueError(method)


# ---------------------------------------------------------------- extra experiments
ABLATION_DATASETS = ["sift", "yahoo"]                      # new runs for A2 and A4
A2_QUERIES = 100 if SMOKE else 1000                        # exhaustive PQ is slow on 1M vectors
A4_QBS = [0, 1, 4, 8]                                     # qb = 0: full-precision query
A4_KFACTORS = [1, 4]
A4_QUERIES = 100 if SMOKE else 500
A4_MAX_NPROBE = 64
# In Faiss 1.15.1 the full-precision query path (qb = 0) scans about 100x slower than
# qb >= 1, so A4 runs qb = 0 on 500 queries, nprobe <= 64 and k_factor = 1 only.
E6_ALPHAS = [0.0, 0.5, 1.0]
E6_NPROBES = [1, 2, 4, 8, 16, 32, 64]
E7_IVF_NLIST = 64 if SMOKE else 2048
E7_NPROBE = 16 if SMOKE else 48
E7_LSH_BITS = [4, 8, 16, 32, 64, 96, 128]
E8_SIZES = [2_500, 5_000, 10_000, 20_000] if SMOKE else [100_000, 250_000, 500_000, 1_000_000]
E8_QUERIES = 100 if SMOKE else 1000
E8_SPECS = {"pq": {"M": 32}, "ivfpq": {"nlist": SMALL_NLIST, "M": 32},
            "ivfrabitq": {"nlist": SMALL_NLIST, "bits": 1}}

# ---------------------------------------------------------------- synthetic data for --smoke
SMOKE_SIZES = {"sift": (20_000, 600), "yahoo": (15_000, 600), "coco": (8_000, 900), "agnews": (12_000, 600)}
