"""Figures for report section 3.4 (AG News), built from results/agnews.csv only.

Colours and markers follow config.py of the benchmark repo so they match the other sections.
    python agnews_figures.py <agnews.csv> <out_dir>
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

CSV, OUT = sys.argv[1], Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

COLOR = {"flat": "#6b6a65", "lsh": "#2a78d6", "ivfpq": "#eb6834", "hnsw": "#1baf7a", "ivfrabitq": "#eda100"}
MARK = {"flat": "*", "lsh": "o", "ivfpq": "s", "hnsw": "^", "ivfrabitq": "D"}
LABEL = {"flat": "Exact (Flat)", "lsh": "LSH", "ivfpq": "IVF-PQ", "hnsw": "HNSW", "ivfrabitq": "IVF-RaBitQ"}
ORDER = ["flat", "lsh", "ivfpq", "hnsw", "ivfrabitq"]
ANN = ["lsh", "ivfpq", "hnsw", "ivfrabitq"]
PQ_RAMP = {32: "#ef8a5e", 64: "#d4561f", 128: "#9c3a12"}      # validated ordinal ramps (one hue each)
RQ_RAMP = {1: "#e0a100", 4: "#9a6a00"}
INK, MUTED, GRID, RAW = "#0b0b0b", "#52514e", "#e6e5e0", "#cfcdc6"

plt.rcParams.update({
    "font.family": "Inter", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "legend.fontsize": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.6, "lines.markersize": 4.5,
    "legend.frameon": False, "figure.dpi": 110, "savefig.bbox": "tight", "axes.titlelocation": "left",
})
thousands = FuncFormatter(lambda v, _: f"{v:,.0f}")

df = pd.read_csv(CSV)
df["bp"] = df.build_params.map(json.loads)
df["sp"] = df.search_params.map(json.loads)
df["build_s"] = df.train_s + df.add_s
flat = df[df.method == "flat"].iloc[0]
FLAT_QPS, FLAT_MS = flat.qps, flat.latency_ms


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", name)


def pareto(sub):
    sub = sub.sort_values(["recall10", "qps"], ascending=[False, False])
    keep, best = [], -1.0
    for _, r in sub.iterrows():
        if r.qps > best:
            keep.append(r)
            best = r.qps
    return pd.DataFrame(keep).sort_values("recall10")


def best_at(m, thr):
    sub = df[(df.method == m) & (df.recall10 >= thr)]
    return None if sub.empty else sub.loc[sub.qps.idxmax()]


# ---------------------------------------------------------------- Figure 1: recall vs QPS frontier
fig, ax = plt.subplots(figsize=(6.5, 4.0))
for t in (0.90, 0.95, 0.99):
    ax.axvline(t, color="#d6d4cc", lw=1.0, zorder=0)
for m in ANN:
    sub = df[df.method == m]
    ax.scatter(sub.recall10, sub.qps, s=10, color=COLOR[m], alpha=0.18, lw=0, zorder=1)
    f = pareto(sub)
    ax.plot(f.recall10, f.qps, color=COLOR[m], marker=MARK[m], ms=5, mec="white", mew=0.6,
            label=LABEL[m], zorder=3)
ax.axhline(FLAT_QPS, color=MUTED, lw=0.9, ls=":", zorder=2)
ax.text(0.405, FLAT_QPS * 1.08, f"Exact search speed ({FLAT_QPS:,.0f} QPS)", fontsize=7.5, color=MUTED)
ax.scatter([1.0], [FLAT_QPS], marker="*", s=110, color=COLOR["flat"], zorder=4, label=LABEL["flat"],
           edgecolor="white", lw=0.6)
ax.set_yscale("log")
ax.set_xlim(0.40, 1.005)
ax.set_ylim(200, 3.2e4)
ax.yaxis.set_major_formatter(thousands)
ax.set_xlabel("Recall@10")
ax.set_ylabel("Queries per second (1 thread, log scale)")
ax.set_title("AG News-mxbai-1024 (400,000 vectors): best speed reachable at each recall")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=5, handletextpad=0.4, columnspacing=1.4)
save(fig, "agnews_fig1_recall_vs_qps")

# ---------------------------------------------------------------- Figure 2: QPS at recall targets
fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.5), sharey=True)
rows = list(reversed(ORDER))
for ax, thr in zip(axes, (0.90, 0.95, 0.99)):
    for i, m in enumerate(rows):
        r = best_at(m, thr)
        if r is None:
            continue
        ax.barh(i, r.qps, height=0.62, color=COLOR[m], edgecolor="white", lw=0)
        ax.text(max(r.qps, FLAT_QPS) * 1.15, i, f"{r.qps:,.0f}  ({r.qps / FLAT_QPS:.1f}×)", va="center",
                fontsize=6.8, color=INK)
    ax.axvline(FLAT_QPS, color=MUTED, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(150, 1.3e5)
    ax.set_xticks([1e3, 1e4])
    ax.xaxis.set_major_formatter(thousands)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Recall@10 ≥ {thr:.2f}")
    ax.set_xlabel("QPS (log scale)")
axes[0].set_yticks(range(len(rows)))
axes[0].set_yticklabels([LABEL[m] for m in rows])
fig.tight_layout(w_pad=0.6)
save(fig, "agnews_fig2_qps_at_targets")

# ---------------------------------------------------------------- Figure 3: memory and build time per build
builds = df.groupby(["method", "build_params"], sort=False).first().reset_index()


def blabel(r):
    p = json.loads(r.build_params)
    return {"flat": "Flat", "lsh": f"LSH {p.get('nbits')} bits", "ivfpq": f"IVF-PQ M={p.get('M')}",
            "hnsw": f"HNSW M={p.get('M')}, efC={p.get('efC')}",
            "ivfrabitq": f"IVF-RaBitQ {p.get('bits')}-bit"}[r.method]


def bkey(r):
    p = json.loads(r.build_params)
    return (ORDER.index(r.method), p.get("nbits", 0), p.get("M", 0), p.get("bits", 0))


builds["key"] = builds.apply(bkey, axis=1)
builds = builds.sort_values("key", ascending=False).reset_index(drop=True)
raw = 4 * int(flat.dim)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3.6), sharey=True, gridspec_kw={"width_ratios": [1.15, 1]})
for i, r in builds.iterrows():
    extra = r.bytes_per_vec - raw
    a1.barh(i, raw, height=0.66, color=RAW, lw=0)
    if extra > 0:
        a1.barh(i, extra, left=raw + 25, height=0.66, color=COLOR[r.method], lw=0)
    lab = f"{r.bytes_per_vec:,.0f}" + (f"  (+{extra:,.0f})" if extra > 0 else "")
    a1.text(r.bytes_per_vec + 90, i, lab, va="center", fontsize=6.8, color=INK)
    a2.barh(i, r.build_s, height=0.66, color=COLOR[r.method], lw=0)
    a2.text(r.build_s * 1.1, i, f"{r.build_s:.1f} s", va="center", fontsize=6.8, color=INK)
a1.set_yticks(range(len(builds)))
a1.set_yticklabels([blabel(r) for _, r in builds.iterrows()])
a1.set_xlim(0, 6200)
a1.xaxis.set_major_formatter(thousands)
a1.set_xlabel("Bytes per vector, total index\n(grey = raw float32 vectors, colour = codes or graph)")
a1.set_title("Memory per vector")
a2.set_xscale("log")
a2.set_xlim(5, 400)
a2.set_xlabel("Build time, train + add\n(seconds, 8 threads, log scale)")
a2.set_title("Index build time")
a2.set_xticks([10, 30, 100, 300])
a2.xaxis.set_major_formatter(thousands)
for a in (a1, a2):
    a.grid(axis="y", visible=False)
fig.tight_layout(w_pad=1.0)
save(fig, "agnews_fig3_memory_build")

# ---------------------------------------------------------------- Figure 4: IVF probing vs re-ranking
from matplotlib.lines import Line2D  # noqa: E402

ivf = df[df.method.isin(["ivfpq", "ivfrabitq"])].copy()
ivf["nprobe"] = ivf.sp.map(lambda p: p["nprobe"])
ivf["kf"] = ivf.sp.map(lambda p: p["k_factor"])
ceiling = ivf[(ivf.kf == 16)].groupby("nprobe").recall10.max()
TICKS = [1, 4, 16, 64, 256]

fig, axes = plt.subplots(1, 3, figsize=(6.5, 3.05))
panels = [("ivfpq", "M", PQ_RAMP, "IVF-PQ (nlist = 4,096)", lambda v: f"M = {v}"),
          ("ivfrabitq", "bits", RQ_RAMP, "IVF-RaBitQ (nlist = 4,096)", lambda v: f"{v}-bit codes")]
for ax, (m, key, ramp, title, fmt) in zip(axes[:2], panels):
    ax.plot(ceiling.index, ceiling.values, color=RAW, lw=5, solid_capstyle="round", zorder=1)
    handles = []
    for v, c in ramp.items():
        sub = ivf[(ivf.method == m) & (ivf.bp.map(lambda p: p.get(key)) == v)]
        rr = sub[sub.kf == 16].sort_values("nprobe")
        nr = sub[sub.kf == 1].sort_values("nprobe")
        ax.plot(rr.nprobe, rr.recall10, color=c, marker=MARK[m], ms=3.6, lw=1.5, zorder=3)
        ax.plot(nr.nprobe, nr.recall10, color=c, ls="--", lw=1.2, zorder=2)
        handles.append(Line2D([], [], color=c, marker=MARK[m], ms=3.6, lw=1.5, label=fmt(v)))
    ax.legend(handles=handles, loc="lower right", fontsize=6.6, labelspacing=0.3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(TICKS)
    ax.set_xticklabels([str(t) for t in TICKS])
    ax.set_ylim(0.3, 1.01)
    ax.set_xlabel("nprobe (clusters scanned)")
    ax.set_title(title)
axes[0].set_ylabel("Recall@10")

ax = axes[2]
sel = [("ivfrabitq", {"bits": 1, "nlist": 4096}, 4, "IVF-RaBitQ 1-bit, k_factor = 4"),
       ("ivfpq", {"M": 128, "nlist": 4096}, 16, "IVF-PQ M = 128, k_factor = 16")]
for m, bp, kf, lab in sel:
    sub = ivf[(ivf.method == m) & (ivf.build_params == json.dumps(bp, sort_keys=True)) & (ivf.kf == kf)]
    sub = sub.sort_values("nprobe")
    ax.plot(sub.nprobe, sub.latency_ms, color=COLOR[m], marker=MARK[m], ms=3.6, label=lab, zorder=3)
    if m == "ivfrabitq":
        for npb in (16, 64, 256):
            r = sub[sub.nprobe == npb].iloc[0]
            off = (-40, 3) if npb == 256 else (4, -9)
            ax.annotate(f"R = {r.recall10:.2f}", (npb, r.latency_ms), xytext=off, textcoords="offset points",
                        fontsize=6.4, color=INK)
ax.axhline(FLAT_MS, color=MUTED, lw=0.9, ls=":")
ax.text(1.0, FLAT_MS * 1.17, f"Exact search ({FLAT_MS:.2f} ms)", fontsize=6.6, color=MUTED)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(TICKS)
ax.set_xticklabels([str(t) for t in TICKS])
ax.set_ylim(0.03, 40)
ax.set_xlim(0.8, 420)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_xlabel("nprobe (clusters scanned)")
ax.set_ylabel("ms per query (log scale)")
ax.set_title("Search time")
ax.legend(loc="upper left", fontsize=6.2, labelspacing=0.3)

style = [Line2D([], [], color=INK, lw=1.5, marker="s", ms=3.4, label="re-ranked (k_factor = 16)"),
         Line2D([], [], color=INK, lw=1.2, ls="--", label="no re-ranking (k_factor = 1)"),
         Line2D([], [], color=RAW, lw=5, label="best re-ranked recall at each nprobe")]
fig.legend(handles=style, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=7)
fig.tight_layout(w_pad=0.8, rect=(0, 0.07, 1, 1))
save(fig, "agnews_fig4_ivf_probing")
