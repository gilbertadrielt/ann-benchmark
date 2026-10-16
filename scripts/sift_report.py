"""Reproduce the SIFT1M comparison from logged measurements (no new timed runs)."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import config
from ann.bench import _finished
from ann import io


def markdown(df):
    def cell(v):
        return f'{v:.4f}' if isinstance(v, float) else str(v)
    return '\n'.join(['| ' + ' | '.join(df.columns) + ' |',
                      '| ' + ' | '.join(['---'] * len(df.columns)) + ' |'] +
                     ['| ' + ' | '.join(cell(v) for v in row) + ' |'
                      for row in df.itertuples(index=False, name=None)])


def main():
    out = config.RESULTS / 'sift_report'
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(io.results_csv('sift'))
    df = df[(df.experiment == 'main') & (df.dataset == 'sift') & (df.query_set == 'main')].copy()
    if df.empty:
        raise ValueError('No SIFT main measurements')
    for col, expected in [('n_db', 1_000_000), ('dim', 128), ('k', 10)]:
        if not df[col].eq(expected).all():
            raise ValueError(f'Expected {col}={expected}')
    for col in ['machine', 'faiss_version', 'n_queries', 'build_threads']:
        if df[col].nunique() != 1:
            raise ValueError(f'Mixed {col}: separate benchmark runs before comparing')
    done = io.done_rows(io.results_csv('sift'))
    coverage = []
    for m in config.METHODS:
        for bp in config.build_grid(m, 'sift'):
            def key(sp):
                return ('main', 'main', m, io.pjson(bp), io.pjson(sp))
            coverage.append({'method': m, 'build_params': io.pjson(bp),
                             'complete_under_stop_rules': _finished(config.search_sweeps(m, bp), done, key, True)})
    pd.DataFrame(coverage).to_csv(out / 'coverage.csv', index=False)
    df['build_s'] = df.train_s + df.add_s
    df['base_MB'] = df.bytes_per_vec_codes * df.n_db / 1e6
    df['total_MB'] = df.total_bytes / 1e6
    builds = df.drop_duplicates(['method', 'build_params'])
    builds[['method', 'build_params', 'train_s', 'add_s', 'build_s', 'base_MB', 'total_MB']].to_csv(out / 'all_builds.csv', index=False)
    picks, tables = [], []
    baseline = df[df.method == 'flat'].iloc[0]
    for target in [.9, .95, .99]:
        records = []
        for m in config.METHODS:
            eligible = df[(df.method == m) & (df.recall10 >= target)]
            if eligible.empty:
                records.append({'Algorithm': config.METHOD_LABEL[m], 'Recall@10': 'not reached'})
                continue
            r = eligible.sort_values('latency_ms').iloc[0]
            picks.append(dict(r, target_recall=target))
            records.append({'Algorithm': config.METHOD_LABEL[m], 'Recall@10': r.recall10,
                            'Latency ms': r.latency_ms, 'QPS': r.qps,
                            'Speedup vs Flat': baseline.latency_ms / r.latency_ms,
                            'Base MB': r.base_MB, 'Total MB': r.total_MB, 'Build s': r.build_s,
                            'Build parameters': r.build_params, 'Search parameters': r.search_params})
        tab = pd.DataFrame(records)
        tab.to_csv(out / f'target_{int(target*100)}.csv', index=False)
        tables.append(f'## Fastest measured configuration at recall ≥ {target:.0%}\n\n' + markdown(tab))
    pd.DataFrame(picks).to_csv(out / 'selected_configs.csv', index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    frontiers = []
    for m in config.METHODS:
        g = df[df.method == m].sort_values(['latency_ms', 'recall10'], ascending=[True, False])
        if g.empty:
            continue
        frontier = g[g.recall10.gt(g.recall10.cummax().shift(fill_value=-1))]
        frontiers.append(frontier)
        ax.scatter(g.latency_ms, g.recall10, s=10, alpha=.14, color=config.METHOD_COLOR[m])
        ax.plot(frontier.latency_ms, frontier.recall10, marker=config.METHOD_MARKER[m],
                color=config.METHOD_COLOR[m], label=config.METHOD_LABEL[m])
    pd.concat(frontiers).to_csv(out / 'latency_recall_frontiers.csv', index=False)
    ax.set(xscale='log', xlabel='Batch time per query (ms, log scale; lower is better)',
           ylabel='Recall@10', ylim=(0, 1.02), title='SIFT1M: recall versus search latency')
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(out / f'recall_latency.{ext}', dpi=200)
    plt.close(fig)
    selected = pd.DataFrame(picks)
    selected = selected[selected.target_recall == .95].set_index('method').reindex(config.METHODS).dropna(subset=['latency_ms'])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = [config.METHOD_LABEL[m] for m in selected.index]
    axes[0].barh(labels, selected.base_MB, label='Base index')
    axes[0].barh(labels, selected.total_MB - selected.base_MB, left=selected.base_MB, label='Raw vectors for reranking')
    axes[0].set(xlabel='Estimated resident index size (decimal MB)', title='Memory at recall ≥ 95%')
    axes[1].barh(labels, selected.train_s, label='Train')
    axes[1].barh(labels, selected.add_s, left=selected.train_s, label='Add', color='#ed9750')
    axes[1].set(xlabel='Recorded base index build time (s)', title='Build time at recall ≥ 95%')
    for ax in axes:
        ax.invert_yaxis()
        ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(out / f'memory_build.{ext}', dpi=200)
    plt.close(fig)
    ranges = builds.groupby('method').agg(build_min_s=('build_s', 'min'), build_max_s=('build_s', 'max'),
                                         memory_min_MB=('total_MB', 'min'), memory_max_MB=('total_MB', 'max')).reset_index()
    ranges.to_csv(out / 'build_memory_ranges.csv', index=False)
    complete = all(r['complete_under_stop_rules'] for r in coverage)
    intro = f'''# SIFT1M benchmark report

Generated from results/sift.csv; rerun with `venv/bin/python -m scripts.sift_report`.

{len(df)} measured search configurations, {len(builds)} distinct builds, all five configured algorithms.
Grid coverage under the repository's recall/plateau stopping rules: {'complete' if complete else 'INCOMPLETE; see coverage.csv'}.
Dataset: 1,000,000 unnormalised 128-dimensional vectors, squared L2 distance,
{int(df.n_queries.iloc[0]):,} evaluation queries, top-10 results. Machine: {df.machine.iloc[0]}.
Faiss {df.faiss_version.iloc[0]}; recorded measurement dates: {df.timestamp.min()} to {df.timestamp.max()}.

## Measurement interpretation

Recall is the repository's tie-aware recall@10, with a relative distance tolerance of 1e-4.
Search uses one Faiss/BLAS thread, a warm-up, and up to three timed batches; the best batch time is retained.
Latency is batch duration divided by query count, not individually timed request latency or p95/p99.
Build uses {int(df.build_threads.iloc[0])} configured threads. Build time is recorded training plus insertion of the base index;
it excludes serialization, loading and construction of the refinement wrapper. Cached build times are reused across searches.
Memory uses serialized base-index size plus float32 raw vectors for refinement (512 MB here);
it excludes process/runtime overhead, temporary buffers and training peak RAM. Flat uses raw-vector size.
LSH, IVF-PQ and IVF-RaBitQ all include exact reranking, including its search cost and retained vectors.
Consequently small compressed codes do not imply small total resident memory in this experiment.
Threshold comparisons choose the fastest measured configuration meeting each target independently;
actual recall may overshoot, and selections can use different builds. No interpolation or extrapolation is used.
Stopping rules truncate sweeps at recall ≥ 0.999 or two gains below 0.001; these are measured-grid results,
not proofs of global optimality. Close timing differences have no confidence intervals.

![Recall versus latency](recall_latency.png)

![Memory and build comparison](memory_build.png)
'''
    findings = '''## Findings and algorithm trade-offs

- HNSW has the lowest measured latency at all three recall targets (90%, 95%, 99%). At 95%, HNSW M16 / efSearch=64 delivers 0.96375 recall in 0.11162 ms, about 4.96× faster than Flat, using 656.26 MB and a 55.12 s base build.
- IVF-PQ is the strongest measured compromise at 90–95% recall when memory and build cost matter. At 95%, M16 / nlist=4096 / nprobe=64 / k_factor=16 achieves 0.96465 recall in 0.17323 ms, with 538.26 MB and a 14.88 s build. HNSW is about 1.55× faster but uses 118 MB more and takes about 3.71× longer to build at these selected operating points.
- IVF-RaBitQ at 95% reaches 0.96055 recall in 0.18497 ms with 546.20 MB and a 13.02 s build. IVF-PQ is slightly faster with less memory here, while RaBitQ builds faster. At 99%, RaBitQ's best measured latency (0.40650 ms) beats IVF-PQ (0.45830 ms), but uses more memory (606.20 versus 554.26 MB).
- Exact Flat guarantees full recall, has the lowest estimated resident index size (512 MB), and the quickest recorded build (0.107 s). Its search costs 0.55364 ms/query. It is attractive for exactness or frequent rebuilds; ANN amortizes a larger build cost over searches.
- LSH is slower than Flat at every selected recall target: 1.27353, 1.67396 and 3.40851 ms at 90%, 95% and 99%. It builds quickly, but exhaustive Hamming scans and reranking make it uncompetitive on this SIFT1M run.
- Increasing HNSW efSearch or IVF nprobe generally buys recall at additional search cost. More graph links increase HNSW memory/build cost; more code bits/subquantizers increase compressed-index memory. Larger reranking candidate pools improve recall while increasing exact-distance work.

There is no single winner across all metrics: HNSW wins measured query speed; IVF-PQ balances speed, memory and build time around 95%; Flat wins exactness, build time and total index size in this reranking setup. These conclusions apply to this machine, query set and grid. PQ-only and IVF-Flat are extra experiment implementations without SIFT measurements in this CSV and are not ranked.

## Build and memory ranges across every measured build

'''
    (out / 'report.md').write_text(intro + '\n\n'.join(tables) + '\n\n' + findings + markdown(ranges) + '\n')
    print(f'Wrote {out / "report.md"}; coverage complete={complete}')


if __name__ == '__main__':
    main()
