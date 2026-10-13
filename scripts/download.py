"""Step 1a: download the five HDF5 files (about 5 GB) into data/raw/.

Resumes partial downloads, retries on network errors, and checks that each file
opens as HDF5 with `train` and `test` datasets. In --smoke mode it writes small
synthetic files with the same names and layout instead.

    python -m scripts.download            # all datasets
    python -m scripts.download sift coco  # some of them
"""
import sys
import time
import urllib.error
import urllib.request

import h5py

import config
from ann.data import make_synthetic


def _check_h5(path):
    with h5py.File(path, "r") as f:
        keys = set(f.keys())
        if not {"train", "test"} <= keys:
            raise RuntimeError(f"{path.name}: expected datasets train and test, found {sorted(keys)}")
        return f["train"].shape, f["test"].shape


def _fetch(url, dest, retries=5):
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "ann-benchmark/1.0"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:          # server ignored the range: start over
                    have = 0
                total = int(r.headers.get("Content-Length", 0)) + have
                mode = "ab" if have else "wb"
                done, last = have, time.time()
                with open(part, mode) as f:
                    while True:
                        chunk = r.read(1 << 22)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if time.time() - last > 10:
                            pct = f"{100 * done / total:.0f}%" if total else ""
                            print(f"    {dest.name}: {done / 1e6:,.0f} MB {pct}", flush=True)
                            last = time.time()
            if total and part.stat().st_size < total:
                raise IOError(f"incomplete: {part.stat().st_size} of {total} bytes")
            part.replace(dest)
            return
        except (urllib.error.URLError, IOError, TimeoutError, ConnectionError) as e:
            print(f"    attempt {attempt}/{retries} failed: {e}", flush=True)
            if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403, 404):
                break
            time.sleep(min(60, 5 * attempt))
    raise RuntimeError(f"could not download {url}. Download it in a browser and save it as {dest}")


def main(names=None):
    names = names or config.DATASET_ORDER
    config.RAW.mkdir(parents=True, exist_ok=True)
    for ds in names:
        spec = config.DATASETS[ds]
        if config.SMOKE:
            if not all((config.RAW / fn).exists() for fn, _ in spec["files"].values()):
                print(f"[download] {ds}: writing synthetic smoke-test files")
                make_synthetic(ds)
        for qs, (fname, url) in spec["files"].items():
            dest = config.RAW / fname
            if not dest.exists():
                print(f"[download] {ds}/{qs}: {url} (about {spec['approx_mb']:,} MB for the dataset)", flush=True)
                _fetch(url, dest)
            tr, te = _check_h5(dest)
            print(f"[download] {ds}/{qs}: ok, train {tr}, test {te}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
