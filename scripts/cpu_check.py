"""Find the fast CPU threads (P-cores) of a hybrid Intel chip, for `run_all.py --cpus`.

Times the same single-threaded matrix work on each logical CPU in turn. On chips that mix
performance and efficiency cores, the efficiency cores come out clearly slower.

    python -m scripts.cpu_check
"""
import time

import numpy as np
import psutil
from threadpoolctl import threadpool_limits


def to_spec(cpus):
    """[0, 1, 2, 5, 6] -> '0-2,5-6'."""
    parts, start = [], cpus[0]
    for prev, cur in zip(cpus, cpus[1:] + [None]):
        if cur != prev + 1:
            parts.append(f"{start}-{prev}" if prev > start else f"{start}")
            start = cur
    return ",".join(parts)


def main():
    proc = psutil.Process()
    if not hasattr(proc, "cpu_affinity"):
        print("This OS (macOS) does not support CPU pinning; run without --cpus.")
        return
    original = proc.cpu_affinity()
    a = np.random.default_rng(0).random((400, 400), dtype=np.float32)
    times = {}
    try:
        with threadpool_limits(limits=1):
            for cpu in original:
                proc.cpu_affinity([cpu])
                a @ a                                   # warm-up on this CPU
                t0 = time.perf_counter()
                for _ in range(200):
                    a @ a
                times[cpu] = time.perf_counter() - t0
    finally:
        proc.cpu_affinity(original)
    fastest = min(times.values())
    fast = sorted(c for c, t in times.items() if t < 1.3 * fastest)
    for c, t in times.items():
        print(f"CPU {c:2d}: {t:.3f} s  {'fast' if c in fast else 'slow'}")
    if len(fast) == len(times):
        print("\nAll CPUs run at about the same speed: no need for --cpus.")
    else:
        print(f"\nFast CPUs: {fast}\nAdd this to run_all.py:  --cpus {to_spec(fast)}")


if __name__ == "__main__":
    main()
