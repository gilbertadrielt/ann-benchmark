"""CSV logging (append-only, resumable) and machine description."""
import csv
import json
import os
import platform
from datetime import datetime
from functools import lru_cache

import faiss
import pandas as pd

import config

COLUMNS = [
    "experiment", "dataset", "query_set", "method", "build_params", "search_params", "recipe",
    "recall10", "recall10_id", "qps", "latency_ms", "n_reps",
    "train_s", "add_s", "build_threads",
    "total_bytes", "bytes_per_vec", "bytes_per_vec_codes",
    "n_db", "dim", "n_queries", "k", "faiss_version", "machine", "timestamp",
]


def pjson(p):
    return json.dumps(p, sort_keys=True)


@lru_cache(maxsize=1)
def machine():
    cpu = platform.processor() or platform.machine()
    try:
        if platform.system() == "Linux":
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
        elif platform.system() == "Darwin":
            import subprocess
            cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    except Exception:
        pass
    return f"{cpu} | {os.cpu_count()} threads | {config.RAM_GB:.0f} GB | {platform.system()}"


def results_csv(ds):
    return config.RESULTS / f"{ds}.csv"


def append_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in COLUMNS})


def done_rows(path):
    """{(experiment, query_set, method, build_params, search_params): recall10} already logged."""
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype={"build_params": str, "search_params": str})
    return {(r.experiment, r.query_set, r.method, r.build_params, r.search_params): float(r.recall10)
            for r in df.itertuples()}


def base_row(experiment, ds, qs, method, bp, sp, n_db, dim, n_queries):
    return {
        "experiment": experiment, "dataset": ds, "query_set": qs, "method": method,
        "build_params": pjson(bp), "search_params": pjson(sp),
        "n_db": n_db, "dim": dim, "n_queries": n_queries, "k": config.K,
        "faiss_version": faiss.__version__, "machine": machine(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def load_results(ds=None):
    """All logged rows (every dataset, or one) as a DataFrame with parsed parameter dicts."""
    paths = [results_csv(ds)] if ds else sorted(config.RESULTS.glob("*.csv"))
    frames = []
    for p in paths:
        if p.exists() and p.stem in config.DATASETS:
            frames.append(pd.read_csv(p, dtype={"build_params": str, "search_params": str}))
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    df["bp"] = df["build_params"].map(json.loads)
    df["sp"] = df["search_params"].map(json.loads)
    return df
