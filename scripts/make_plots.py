"""Step 6: every figure (F1-F11) and table (T1-T6) of the report, from results/ only.

Figures go to results/figures/ as PDF (vector, for the report) and 300-dpi PNG.
Tables go to results/tables/ as CSV and Markdown. A figure whose inputs are missing is
skipped with a message instead of stopping the script, so it can be re-run at any time.

    python -m scripts.make_plots
"""
import json
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from ann import io  # noqa: E402

C, MK, LBL = config.METHOD_COLOR, config.METHOD_MARKER, config.METHOD_LABEL
ANN = ["lsh", "ivfpq", "hnsw", "ivfrabitq"]
SEQ = ["#86b6ef", "#3987e5", "#256abf", "#104281"]        # one-hue ordinal ramp for parameter values
PAIR = ["#2a78d6", "#eb6834"]                              # two-way comparisons (off / on)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
PANELS = [("sift", "main"), ("yahoo", "main"), ("coco", "i2i"), ("agnews", "main")]

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.6, "lines.markersize": 4.5,
    "legend.frameon": False, "figure.dpi": 110, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


# ---------------------------------------------------------------- helpers
def save(fig, name):
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES / f"{name}.pdf")
    fig.savefig(config.FIGURES / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"[make_plots] wrote {name}")


def table(df, name, floatfmt=3):
    config.TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.TABLES / f"{name}.csv", index=False)
    cols = list(df.columns)

    def fmt(v):
        if isinstance(v, float):
            if np.isnan(v):
                return ""
            return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.{floatfmt}f}"
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    (config.TABLES / f"{name}.md").write_text("\n".join(lines) + "\n")
    print(f"[make_plots] wrote table {name}")


def step(fn):
    def wrapped(*a, **k):
        try:
            fn(*a, **k)
        except Exception as e:                            # keep going: one missing input must not block the rest
            print(f"[make_plots] skipped {fn.__name__}: {type(e).__name__}: {e}")
            if config.SMOKE:
                traceback.print_exc()
    return wrapped


def pareto(df):
    df = df.sort_values(["recall10", "qps"], ascending=[False, False])
    keep, best = [], -1.0
    for _, r in df.iterrows():
        if r.qps > best:
            keep.append(r)
            best = r.qps
    return pd.DataFrame(keep).sort_values("recall10")


def main_rows(df, ds, qs):
    return df[(df.experiment == "main") & (df.dataset == ds) & (df.query_set == qs)]


def build_s(r):
    return float(r.train_s) + float(r.add_s)


def best_at(sub, thr):
    ok = sub[sub.recall10 >= thr]
    return None if ok.empty else ok.loc[ok.qps.idxmax()]


def panel_nlist(ds):
    return config.EQUAL_BYTES[ds][0]


def bp_is(df, **kw):
    return df[df.bp.map(lambda p: all(p.get(k) == v for k, v in kw.items()))]


def sp_is(df, **kw):
    return df[df.sp.map(lambda p: all(p.get(k) == v for k, v in kw.items()))]


def frontier_axes(ax, df, title, linestyle="-", label_suffix="", legend=True):
    for m in ANN:
        sub = df[df.method == m]
        if sub.empty:
            continue
        f = pareto(sub)
        ax.plot(f.recall10, f.qps, linestyle=linestyle, color=C[m], marker=MK[m],
                label=LBL[m] + label_suffix, markeredgecolor="white", markeredgewidth=0.6)
    flat = df[df.method == "flat"]
    if not flat.empty and linestyle == "-":
        ax.plot([1.0], [flat.qps.max()], linestyle="none", marker=MK["flat"], markersize=9, color=C["flat"],
                label=LBL["flat"], markeredgecolor="white", markeredgewidth=0.6)
    ax.set_yscale("log")
    ax.set_xlim(0.5, 1.005)
    ax.set_title(title, loc="left")
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("Queries per second (1 thread, log)")
    if legend:
        ax.legend(loc="lower left")


# ---------------------------------------------------------------- tables
@step
def t1():
    p = config.RESULTS / "dataset_stats.csv"
    df = pd.read_csv(p)
    df = df[["label", "query_set", "n_db", "dim", "n_queries", "metric", "raw_size_mb", "rc10_median",
             "rc10_q25", "rc10_q75", "subsampled"]]
    table(df.rename(columns={"label": "dataset", "raw_size_mb": "raw MB"}), "T1_datasets")


@step
def t2(df):
    rows = []
    for ds, qs in config.TARGETS:
        sub = main_rows(df, ds, qs)
        for m in config.METHODS:
            s = sub[sub.method == m]
            if s.empty:
                continue
            row = {"dataset": ds, "query_set": qs, "method": LBL[m]}
            for thr in (0.90, 0.95):
                b = best_at(s, thr)
                key = f"{int(thr * 100)}"
                if b is None:
                    row.update({f"QPS@{key}": "not reached", f"bytes/vec@{key}": "", f"build s@{key}": "",
                                f"config@{key}": f"max recall {s.recall10.max():.3f}"})
                else:
                    row.update({f"QPS@{key}": round(float(b.qps)), f"bytes/vec@{key}": round(float(b.bytes_per_vec), 1),
                                f"build s@{key}": round(build_s(b), 1),
                                f"config@{key}": f"{b.build_params} {b.search_params}"})
            rows.append(row)
    table(pd.DataFrame(rows), "T2_best_qps")


@step
def t3(df):
    rows = []
    for m in ANN:
        r = {"method": LBL[m]}
        for qs in ("i2i", "t2i"):
            s = main_rows(df, "coco", qs)
            s = s[s.method == m]
            b = best_at(s, 0.90) if not s.empty else None
            r[f"{qs} config@0.90"] = "not reached" if b is None else f"{b.build_params} {b.search_params}"
            r[f"{qs} QPS@0.90"] = np.nan if b is None else float(b.qps)
        r["QPS ratio t2i/i2i"] = r["t2i QPS@0.90"] / r["i2i QPS@0.90"] if r["i2i QPS@0.90"] == r["i2i QPS@0.90"] else np.nan
        rows.append(r)
    table(pd.DataFrame(rows), "T3_shift_budget")
    # re-ranking under shift: recall at k_factor 1 vs 16 (or the largest available), fixed nprobe
    rows = []
    for m in ("ivfpq", "ivfrabitq"):
        for qs in ("i2i", "t2i"):
            s = main_rows(df, "coco", qs)
            s = bp_is(s[s.method == m], nlist=panel_nlist("coco"))
            if s.empty:
                continue
            first_bp = s.build_params.iloc[0]
            s = s[s.build_params == first_bp]
            nprobe = min([p for p in config.NPROBES if p <= panel_nlist("coco")], key=lambda p: abs(p - 16))
            s = sp_is(s, nprobe=nprobe)
            kfs = sorted(s.sp.map(lambda p: p["k_factor"]).unique())
            if not kfs:
                continue
            r1 = sp_is(s, k_factor=kfs[0]).recall10.max()
            rk = sp_is(s, k_factor=kfs[-1]).recall10.max()
            rows.append({"method": LBL[m], "query_set": qs, "build": first_bp, "nprobe": nprobe,
                         f"recall k_factor={kfs[0]}": r1, f"recall k_factor={kfs[-1]}": rk, "gain": rk - r1})
    table(pd.DataFrame(rows), "T3b_rerank_under_shift")


@step
def t4(df):
    out = []
    for ds in config.ABLATION_DATASETS:
        main = main_rows(df, ds, "main")
        lsh = sp_is(main[main.method == "lsh"], k_factor=1)
        for (nb, tt), g in lsh.groupby([lsh.bp.map(lambda p: p["nbits"]), lsh.bp.map(lambda p: p["train_thresholds"])]):
            out.append({"ablation": "A1 LSH thresholds", "dataset": ds, "setting": f"nbits={nb}, train_thresholds={tt}",
                        "recall10": g.recall10.max(), "qps": g.qps.max()})
        a2 = df[(df.experiment == "A2") & (df.dataset == ds)]
        for _, r in a2.iterrows():
            out.append({"ablation": "A2 PQ ADC vs SDC", "dataset": ds,
                        "setting": f"M={r.bp['M']}, {'SDC' if r.sp['sdc'] else 'ADC'}", "recall10": r.recall10, "qps": r.qps})
        for m in ("ivfpq", "ivfrabitq"):
            s = bp_is(main[main.method == m], nlist=panel_nlist(ds))
            for kf, g in s.groupby(s.sp.map(lambda p: p["k_factor"])):
                out.append({"ablation": "A3 re-ranking", "dataset": ds, "setting": f"{LBL[m]}, k_factor={kf}",
                            "recall10": g.recall10.max(), "qps": g[g.recall10 == g.recall10.max()].qps.max(),
                            "bytes_per_vec": g.bytes_per_vec.iloc[0] if kf > 1 else g.bytes_per_vec_codes.iloc[0]})
        a4 = sp_is(df[(df.experiment == "A4") & (df.dataset == ds)], k_factor=1)
        for (bits, qb), g in a4.groupby([a4.bp.map(lambda p: p["bits"]), a4.sp.map(lambda p: p["qb"])]):
            r16 = g[g.sp.map(lambda p: p["nprobe"]) == min(g.sp.map(lambda p: p["nprobe"]), key=lambda v: abs(v - 16))]
            out.append({"ablation": "A4 RaBitQ bits", "dataset": ds, "setting": f"bits={bits}, qb={qb}, nprobe~16",
                        "recall10": r16.recall10.max(), "qps": r16.qps.max(), "bytes_per_vec": g.bytes_per_vec_codes.iloc[0]})
        h = main[main.method == "hnsw"]
        for M, g in h.groupby(h.bp.map(lambda p: p["M"])):
            out.append({"ablation": "A5 HNSW degree", "dataset": ds, "setting": f"M={M}",
                        "recall10": g.recall10.max(), "qps": g[g.recall10 == g.recall10.max()].qps.max(),
                        "bytes_per_vec": g.bytes_per_vec.iloc[0], "build_s": build_s(g.iloc[0])})
    table(pd.DataFrame(out), "T4_ablations")


@step
def t5_t6(df):
    e7 = config.RESULTS / "e7_lecture.csv"
    if e7.exists():
        table(pd.read_csv(e7), "T5_lecture_check")
    rows = []
    for ds, qs in PANELS:
        nl, M = config.EQUAL_BYTES[ds]
        sub = main_rows(df, ds, qs)
        pq = bp_is(sub[sub.method == "ivfpq"], nlist=nl, M=M)
        rq = bp_is(sub[sub.method == "ivfrabitq"], nlist=nl, bits=1)
        bpq, brq = (best_at(pq, 0.90) if not pq.empty else None), (best_at(rq, 0.90) if not rq.empty else None)
        rows.append({"dataset": ds, "dim": config.DATASETS[ds]["dim"], "nlist": nl, "IVF-PQ M": M,
                     "PQ code bytes": float(pq.bytes_per_vec_codes.iloc[0]) if not pq.empty else np.nan,
                     "RaBitQ code bytes": float(rq.bytes_per_vec_codes.iloc[0]) if not rq.empty else np.nan,
                     "PQ QPS@0.90": float(bpq.qps) if bpq is not None else np.nan,
                     "RaBitQ QPS@0.90": float(brq.qps) if brq is not None else np.nan})
    t = pd.DataFrame(rows)
    t["RaBitQ/PQ QPS ratio"] = t["RaBitQ QPS@0.90"] / t["PQ QPS@0.90"]
    table(t, "T6_equal_bytes")


# ---------------------------------------------------------------- figures
@step
def f1(df):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6), constrained_layout=True)
    for ax, (ds, qs) in zip(axes.ravel(), PANELS):
        sub = main_rows(df, ds, qs)
        if sub.empty:
            ax.set_visible(False)
            continue
        title = config.DATASETS[ds]["label"] + (" (image queries)" if ds == "coco" else "")
        frontier_axes(ax, sub, title, legend=(ds == "sift"))
    save(fig, "F1_recall_vs_qps")


@step
def f2(df):
    fig, ax = plt.subplots(figsize=(4.6, 3.6), constrained_layout=True)
    frontier_axes(ax, main_rows(df, "coco", "i2i"), "COCO-CLIP: image queries (solid) vs caption queries (dashed)",
                  legend=False)
    frontier_axes(ax, main_rows(df, "coco", "t2i"), ax.get_title(), linestyle="--", label_suffix=" (captions)",
                  legend=False)
    handles = [plt.Line2D([], [], color=C[m], marker=MK[m], label=LBL[m]) for m in ANN]
    handles += [plt.Line2D([], [], color=MUTED, linestyle="-", label="image queries (I2I)"),
                plt.Line2D([], [], color=MUTED, linestyle="--", label="caption queries (T2I)")]
    ax.legend(handles=handles, loc="lower left")
    save(fig, "F2_shift_i2i_vs_t2i")


@step
def f3(df):
    for ds, qs in PANELS:
        sub = main_rows(df, ds, qs)
        if sub.empty:
            continue
        nl = panel_nlist(ds)
        fig, axes = plt.subplots(1, 4, figsize=(9.0, 2.6), constrained_layout=True)
        lsh = sub[sub.method == "lsh"]
        lsh = lsh[lsh.bp.map(lambda p: p["train_thresholds"])]
        for i, kf in enumerate(sorted(lsh.sp.map(lambda p: p["k_factor"]).unique())):
            g = sp_is(lsh, k_factor=kf)
            g = g.assign(nb=g.bp.map(lambda p: p["nbits"])).sort_values("nb")
            axes[0].plot(g.nb, g.recall10, marker="o", color=SEQ[i % 4], label=f"k_factor={kf}")
        axes[0].set_xscale("log", base=2)
        axes[0].set(title="LSH: bits", xlabel="nbits", ylabel="Recall@10")
        for ax, m, key, name in [(axes[1], "ivfpq", "M", "IVF-PQ: M"), (axes[3], "ivfrabitq", "bits", "IVF-RaBitQ: bits")]:
            s = sp_is(bp_is(sub[sub.method == m], nlist=nl), k_factor=1)
            for i, v in enumerate(sorted(s.bp.map(lambda p: p[key]).unique())):
                g = s[s.bp.map(lambda p: p[key]) == v]
                g = g.assign(np_=g.sp.map(lambda p: p["nprobe"])).sort_values("np_")
                ax.plot(g.np_, g.recall10, marker="s", color=SEQ[i % 4], label=f"{key}={v}")
            ax.set_xscale("log", base=2)
            ax.set(title=f"{name} (nlist={nl}, no re-rank)", xlabel="nprobe")
        h = sub[sub.method == "hnsw"]
        for i, M in enumerate(sorted(h.bp.map(lambda p: p["M"]).unique())):
            g = h[h.bp.map(lambda p: p["M"]) == M]
            g = g.assign(ef=g.sp.map(lambda p: p["efSearch"])).sort_values("ef")
            axes[2].plot(g.ef, g.recall10, marker="^", color=SEQ[i % 4], label=f"M={M}")
        axes[2].set_xscale("log", base=2)
        axes[2].set(title="HNSW: M", xlabel="efSearch")
        for ax in axes:
            ax.set_ylim(0, 1.02)
            if ax.lines:
                ax.legend(loc="lower right")
        fig.suptitle(config.DATASETS[ds]["label"], x=0.01, ha="left", fontsize=9)
        save(fig, f"F3_parameters_{ds}")


@step
def f4(df):
    for ds in config.ABLATION_DATASETS:
        main = main_rows(df, ds, "main")
        if main.empty:
            continue
        nl = panel_nlist(ds)
        fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.0), constrained_layout=True)
        ax = axes[0, 0]
        lsh = sp_is(main[main.method == "lsh"], k_factor=1)
        for i, tt in enumerate((False, True)):
            g = lsh[lsh.bp.map(lambda p: p["train_thresholds"]) == tt]
            g = g.assign(nb=g.bp.map(lambda p: p["nbits"])).sort_values("nb")
            ax.plot(g.nb, g.recall10, marker="o", color=PAIR[i], linestyle="-" if tt else "--",
                    label="learned thresholds" if tt else "threshold 0")
        ax.set_xscale("log", base=2)
        ax.set(title="A1: LSH thresholds (no re-rank)", xlabel="nbits", ylabel="Recall@10", ylim=(0, 1.02))
        ax.legend(loc="upper left")

        ax = axes[0, 1]
        a2 = df[(df.experiment == "A2") & (df.dataset == ds)]
        if not a2.empty:
            Ms = sorted(a2.bp.map(lambda p: p["M"]).unique())
            x = np.arange(len(Ms))
            for i, sdc in enumerate((False, True)):
                vals = [a2[(a2.bp.map(lambda p: p["M"]) == M) & (a2.sp.map(lambda p: p["sdc"]) == sdc)].recall10.max() for M in Ms]
                ax.bar(x + (i - 0.5) * 0.38, vals, width=0.36, color=PAIR[i], label="SDC" if sdc else "ADC")
            ax.set_xticks(x, [f"M={M}" for M in Ms])
            ax.set(title="A2: PQ asymmetric vs symmetric", ylabel="Recall@10", ylim=(0, 1.02))
            ax.legend(loc="upper left")
            ax.grid(axis="x", visible=False)

        ax = axes[0, 2]
        for m, ls in (("ivfpq", "-"), ("ivfrabitq", "--")):
            s = bp_is(main[main.method == m], nlist=nl)
            if s.empty:
                continue
            first = s.build_params.iloc[0]
            s = s[s.build_params == first]
            for i, kf in enumerate(sorted(s.sp.map(lambda p: p["k_factor"]).unique())):
                g = sp_is(s, k_factor=kf)
                g = g.assign(np_=g.sp.map(lambda p: p["nprobe"])).sort_values("np_")
                ax.plot(g.np_, g.recall10, color=SEQ[(i + 1) % 4], linestyle=ls,
                        label=f"{LBL[m]}, k_factor={kf}")
        ax.set_xscale("log", base=2)
        ax.set(title=f"A3: exact re-ranking (nlist={nl})", xlabel="nprobe", ylabel="Recall@10", ylim=(0, 1.02))
        ax.legend(loc="best", fontsize=6.5)

        ax = axes[1, 0]
        a4 = sp_is(df[(df.experiment == "A4") & (df.dataset == ds)], k_factor=1)
        qbs = sorted(a4.sp.map(lambda p: p["qb"]).unique()) if not a4.empty else []
        for bits, ls in ((1, "-"), (4, "--")):
            for i, qb in enumerate(qbs):
                g = a4[(a4.bp.map(lambda p: p["bits"]) == bits) & (a4.sp.map(lambda p: p["qb"]) == qb)]
                g = g.assign(np_=g.sp.map(lambda p: p["nprobe"])).sort_values("np_")
                ax.plot(g.np_, g.recall10, color=SEQ[i % 4], linestyle=ls, label=f"{bits}-bit, qb={qb}")
        ax.set_xscale("log", base=2)
        ax.set(title="A4: RaBitQ code and query bits", xlabel="nprobe", ylabel="Recall@10 (no re-rank)", ylim=(0, 1.02))
        ax.legend(loc="best", fontsize=6, ncol=2)

        ax = axes[1, 1]
        h = main[main.method == "hnsw"]
        for i, M in enumerate(sorted(h.bp.map(lambda p: p["M"]).unique())):
            g = h[h.bp.map(lambda p: p["M"]) == M]
            g = g.assign(ef=g.sp.map(lambda p: p["efSearch"])).sort_values("ef")
            ax.plot(g.ef, g.qps, marker="^", color=SEQ[i % 4], label=f"M={M}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set(title="A5: HNSW degree, speed side", xlabel="efSearch", ylabel="QPS (log)")
        ax.legend(loc="upper right")
        axes[1, 2].set_visible(False)
        fig.suptitle(f"Ablations on {config.DATASETS[ds]['label']}", x=0.01, ha="left", fontsize=9)
        save(fig, f"F4_ablations_{ds}")


def _t2_pick(sub):
    b = best_at(sub, 0.90)
    return b if b is not None else sub.loc[sub.recall10.idxmax()]


@step
def f5_f6(df):
    for name, parts, xlabel in [
        ("F5_memory", [("bytes_per_vec_codes", "index (codes, ids, links)"), ("raw", "raw vectors kept for re-ranking")],
         "Bytes per vector"),
        ("F6_build_time", [("train_s", "train"), ("add_s", "add")], "Build time (s, all cores)"),
    ]:
        fig, axes = plt.subplots(1, 4, figsize=(9.0, 2.6), constrained_layout=True)
        for ax, (ds, qs) in zip(axes, PANELS):
            sub = main_rows(df, ds, qs)
            if sub.empty:
                ax.set_visible(False)
                continue
            ms = [m for m in config.METHODS if not sub[sub.method == m].empty]
            picks = {m: _t2_pick(sub[sub.method == m]) for m in ms}
            left = np.zeros(len(ms))
            for j, (col, label) in enumerate(parts):
                vals = np.array([float(picks[m].bytes_per_vec - picks[m].bytes_per_vec_codes) if col == "raw"
                                 else float(picks[m][col]) for m in ms])
                ax.barh(range(len(ms)), vals, left=left, color=PAIR[j], height=0.6, label=label,
                        edgecolor="white", linewidth=1)
                left += vals
            ax.set_yticks(range(len(ms)), [LBL[m] for m in ms])
            ax.invert_yaxis()
            ax.set(title=config.DATASETS[ds]["label"], xlabel=xlabel)
            ax.grid(axis="y", visible=False)
        axes[0].legend(loc="lower right", fontsize=6.5)
        fig.suptitle("Configuration with the highest QPS at recall@10 >= 0.90 (highest recall if never reached)",
                     x=0.01, ha="left", fontsize=8, color=MUTED)
        save(fig, name)


@step
def f7():
    diag = pd.read_csv(config.RESULTS / "ood" / "diagnostics.csv")
    pca = np.load(config.RESULTS / "ood" / "pca.npz")
    fig, axes = plt.subplots(1, 4, figsize=(9.0, 2.5), constrained_layout=True)
    ax = axes[0]
    ax.scatter(*pca["db"].T, s=2, color="#c3c2b7", label="database (images)", rasterized=True)
    ax.scatter(*pca["i2i"].T, s=3, color=PAIR[0], label="image queries", rasterized=True)
    ax.scatter(*pca["t2i"].T, s=3, color=PAIR[1], label="caption queries", rasterized=True)
    ax.set(title="PCA of CLIP vectors", xlabel="PC 1", ylabel="PC 2")
    ax.legend(loc="best", markerscale=3, fontsize=6.5)
    for ax, col, label in [(axes[1], "cos_1nn", "Cosine to 1st true neighbour"),
                           (axes[2], "rc10", "Relative contrast RC10 (low = hard)"),
                           (axes[3], "centroid_dist", "Distance to nearest IVF centroid")]:
        lo, hi = diag[col].quantile(0.005), diag[col].quantile(0.995)
        bins = np.linspace(lo, hi, 40)
        for i, qs in enumerate(("i2i", "t2i")):
            ax.hist(diag[diag.query_set == qs][col], bins=bins, color=PAIR[i], alpha=0.65,
                    label="image queries" if qs == "i2i" else "caption queries")
        ax.set(xlabel=label, ylabel="Queries")
        ax.grid(axis="x", visible=False)
    axes[1].legend(loc="upper left", fontsize=6.5)
    save(fig, "F7_shift_diagnostics")


@step
def f8():
    for ds, qs in config.TARGETS:
        files = {m: config.RESULTS / "failure" / f"{ds}_{qs}_{m}.npy" for m in ANN}
        files = {m: p for m, p in files.items() if p.exists()}
        if not files:
            continue
        rc = np.load(config.DATA / ds / f"rc10_{qs}.npy")
        summ = pd.read_csv(config.RESULTS / "failure" / "summary.csv")
        fig, axes = plt.subplots(1, len(files), figsize=(2.3 * len(files), 2.4), constrained_layout=True, sharey=True)
        axes = np.atleast_1d(axes)
        rng = np.random.default_rng(0)
        for ax, (m, p) in zip(axes, files.items()):
            hits = np.load(p)
            ax.scatter(rc, hits + rng.uniform(-0.02, 0.02, len(hits)), s=3, color=C[m], alpha=0.5, rasterized=True)
            ax.set_xscale("log")
            s = summ[(summ.dataset == ds) & (summ.query_set == qs) & (summ.method == m)]
            rho = float(s.spearman_rho_rc10.iloc[0]) if not s.empty else float("nan")
            ax.set(title=f"{LBL[m]}  (Spearman rho = {rho:.2f})", xlabel="RC10 (log)", ylim=(-0.05, 1.05))
        axes[0].set_ylabel("Per-query recall@10")
        fig.suptitle(f"{config.DATASETS[ds]['label']} {qs}: configs with mean recall near 0.90",
                     x=0.01, ha="left", fontsize=8.5)
        save(fig, f"F8_failure_{ds}_{qs}")


@step
def f9():
    e6 = pd.read_csv(config.RESULTS / "e6_routing.csv")
    fig, ax = plt.subplots(figsize=(4.0, 3.0), constrained_layout=True)
    for i, (alpha, g) in enumerate(e6.groupby("alpha")):
        ax.plot(g.nprobe, g.recall10, marker="o", color=SEQ[(i + 1) % 4],
                label="normal routing" if alpha == 0 else f"shifted routing, alpha={alpha}")
    ax.set_xscale("log", base=2)
    ax.set(title="E6: gap-corrected routing, caption queries", xlabel="nprobe", ylabel="Recall@10", ylim=(0, 1.02))
    ax.legend(loc="lower right")
    save(fig, "F9_gap_corrected_routing")


@step
def f10():
    e7 = pd.read_csv(config.RESULTS / "e7_lecture.csv")
    lsh = e7[e7["item"].str.contains("LSH")].sort_values("nbits")
    exact = e7[e7["item"] == "exact top-100"].mean_cosine_top100.iloc[0]
    fig, ax = plt.subplots(figsize=(4.0, 3.0), constrained_layout=True)
    ax.plot(lsh.nbits, lsh.mean_cosine_top100, marker="o", color=C["lsh"], label="LSH top-100 by Hamming distance")
    ax.axhline(exact, color=C["flat"], linestyle="--", linewidth=1.2, label="exact top-100")
    ax.set(title="E7: lecture slide 36, SIFT1M", xlabel="nbits", ylabel="Mean cosine similarity of the top 100")
    ax.legend(loc="lower right")
    save(fig, "F10_lecture_lsh_bits")


@step
def f11():
    eq = pd.read_csv(config.TABLES / "T6_equal_bytes.csv")
    sc = pd.read_csv(config.RESULTS / "e8_scaling.csv", dtype={"build_params": str, "search_params": str})
    sc["n"] = sc.build_params.map(lambda s: json.loads(s)["n_db"])
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), constrained_layout=True)
    ax = axes[0]
    e = eq.dropna(subset=["RaBitQ/PQ QPS ratio"])
    ax.plot(e.dim, e["RaBitQ/PQ QPS ratio"], marker="D", color=C["ivfrabitq"])
    for _, r in e.iterrows():
        ax.annotate(r.dataset, (r.dim, r["RaBitQ/PQ QPS ratio"]), textcoords="offset points", xytext=(4, 4),
                    fontsize=7, color=MUTED)
    ax.axhline(1.0, color=MUTED, linewidth=0.8)
    ax.set_xscale("log", base=2)
    ax.set(title="Equal bytes per vector: IVF-RaBitQ / IVF-PQ", xlabel="Dimension", ylabel="QPS ratio at recall@10 = 0.90")
    ax = axes[1]
    for m, lab in (("pq_refine", "PQ32, exhaustive"), ("ivfpq", "IVF-PQ"), ("ivfrabitq", "IVF-RaBitQ")):
        g = sc[(sc.method == m) & (sc.recall10 >= 0.90)]
        if g.empty:
            continue
        best = g.loc[g.groupby("n").qps.idxmax()].sort_values("n")
        color = C["pq"] if m == "pq_refine" else C[m]
        ax.plot(best.n, 1000.0 / best.qps, marker=MK["pq" if m == "pq_refine" else m], color=color, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(title="SIFT prefixes: query time at recall@10 = 0.90", xlabel="Database size", ylabel="ms per query (log)")
    ax.legend(loc="upper left")
    save(fig, "F11_dimension_and_scale")


def main():
    df = io.load_results()
    if df.empty:
        print("[make_plots] no results yet: run the benchmark first")
    t1()
    t2(df)
    t3(df)
    t4(df)
    t5_t6(df)
    f1(df)
    f2(df)
    f3(df)
    f4(df)
    f5_f6(df)
    f7()
    f8()
    f9()
    f10()
    f11()


if __name__ == "__main__":
    main()
