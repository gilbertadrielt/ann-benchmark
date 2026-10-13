"""Run the whole benchmark, start to finish, in the right order.

    python run_all.py --smoke        # 5-10 minutes on synthetic data: checks the whole pipeline
    python run_all.py                # the real run: downloads ~5 GB, then several hours of compute
    python run_all.py --from bench   # resume from a stage (finished configs are skipped anyway)
    python run_all.py --only plots   # one stage

Every stage is resumable: rerunning skips downloads, prepared arrays, ground truth,
cached indexes and logged configurations that already exist.
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = ["download", "prepare", "ground_truth", "bench", "ablations", "ood", "failure", "lecture", "scaling", "plots"]


def commands(stage, cfg):
    py = [sys.executable, "-m"]
    if stage == "download":
        return [py + ["scripts.download"]]
    if stage == "prepare":
        return [py + ["scripts.prepare"]]
    if stage == "ground_truth":
        return [py + ["scripts.ground_truth"]]
    if stage == "bench":
        out = []
        for ds in cfg.BENCH_ORDER:
            for qs in cfg.DATASETS[ds]["files"]:
                for m in cfg.METHODS:
                    out.append(py + ["scripts.run_bench", "--dataset", ds, "--queries", qs, "--method", m])
        return out
    if stage == "ablations":
        return [py + ["scripts.ablations"]]
    if stage == "ood":
        return [py + ["scripts.ood_study"]]
    if stage == "failure":
        return [py + ["scripts.failure_cases"]]
    if stage == "lecture":
        return [py + ["scripts.lecture_check"]]
    if stage == "scaling":
        return [py + ["scripts.scaling"]]
    if stage == "plots":
        return [py + ["scripts.make_plots"]]
    raise ValueError(stage)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="synthetic data in ./smoke, tiny grids")
    ap.add_argument("--from", dest="start", choices=STAGES, help="start at this stage")
    ap.add_argument("--only", nargs="+", choices=STAGES, help="run only these stages")
    a = ap.parse_args()
    if a.smoke:
        os.environ["ANN_SMOKE"] = "1"
    sys.path.insert(0, str(ROOT))
    import config as cfg

    if a.only:
        stages = [s for s in STAGES if s in a.only]
    elif a.start:
        stages = STAGES[STAGES.index(a.start):]
    else:
        stages = STAGES
    print(f"Mode: {'SMOKE TEST (synthetic data)' if cfg.SMOKE else 'FULL RUN'} | output: {cfg.WORK}")
    print(f"Machine: {os.cpu_count()} threads, {cfg.RAM_GB:.0f} GB RAM"
          + (" -> AG News will be cut to 400,000 vectors" if cfg.LOW_MEMORY and not cfg.SMOKE else ""))
    cfg.RESULTS.mkdir(parents=True, exist_ok=True)
    log = cfg.RESULTS / "run_log.csv"
    t_all = time.time()
    for stage in stages:
        for cmd in commands(stage, cfg):
            label = " ".join(cmd[2:])
            print(f"\n=== [{stage}] {label}", flush=True)
            t0 = time.time()
            rc = subprocess.call(cmd, cwd=ROOT, env=os.environ.copy())
            dt = time.time() - t0
            with open(log, "a") as f:
                f.write(f"{datetime.now().isoformat(timespec='seconds')},{stage},{label},{rc},{dt:.1f}\n")
            if rc != 0:
                print(f"\nStage '{stage}' failed (exit code {rc}) after {dt / 60:.1f} min.")
                print(f"Fix the error above, then resume with:  python run_all.py{' --smoke' if cfg.SMOKE else ''} --from {stage}")
                sys.exit(rc)
            print(f"=== done in {dt / 60:.1f} min", flush=True)
    print(f"\nAll stages finished in {(time.time() - t_all) / 3600:.2f} h. Figures: {cfg.FIGURES}  Tables: {cfg.TABLES}")


if __name__ == "__main__":
    main()
