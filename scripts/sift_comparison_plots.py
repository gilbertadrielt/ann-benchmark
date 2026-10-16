"""Generate comparison plots from the SIFT report's recorded measurements.

Run after scripts.sift_report: python -m scripts.sift_comparison_plots
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

import config


def main():
    root = config.RESULTS / 'sift_report'
    out = root / 'plots'
    out.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(root / 'selected_configs.csv')
    builds = pd.read_csv(root / 'all_builds.csv')
    raw = pd.read_csv(config.RESULTS / 'sift.csv')
    raw = raw[(raw.experiment == 'main') & (raw.query_set == 'main')]
    methods = config.METHODS
    colors = [config.METHOD_COLOR[m] for m in methods]
    labels = [config.METHOD_LABEL[m] for m in methods]
    targets = [.9, .95, .99]
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    entries = []
    with PdfPages(out / 'all_comparisons.pdf') as pdf:
        def save(fig, name, caption):
            fig.tight_layout(rect=(0, .07, 1, .94))
            fig.text(.02, .015, caption, fontsize=8, wrap=True)
            fig.savefig(out / f'{name}.png', dpi=220)
            fig.savefig(out / f'{name}.pdf')
            pdf.savefig(fig)
            plt.close(fig)
            entries.append((name, caption))

        # Curves use measured per-method frontiers; faint points show all configurations.
        for zoom in [False, True]:
            fig, ax = plt.subplots(figsize=(9, 5.5))
            for m, label, color in zip(methods, labels, colors):
                g = raw[raw.method == m].sort_values(['latency_ms', 'recall10'], ascending=[True, False])
                frontier = g[g.recall10.gt(g.recall10.cummax().shift(fill_value=-1))]
                ax.scatter(g.latency_ms, g.recall10, color=color, s=12, alpha=.18)
                ax.plot(frontier.latency_ms, frontier.recall10, marker=config.METHOD_MARKER[m], color=color, label=label)
            ax.set(xscale='log', xlabel='Batch time per query (ms; log scale)', ylabel='Recall@10',
                   ylim=(.9 if zoom else 0, 1.005), title='SIFT1M: recall versus latency' + (' — high recall' if zoom else ''))
            ax.grid(alpha=.2)
            ax.legend(loc='lower right')
            save(fig, 'recall_latency_high' if zoom else 'recall_latency_all',
                 'Faint points: all measured configurations. Lines: per-algorithm measured Pareto frontier; connecting segments are visual guides.')

        metrics = [('latency_ms', 'Latency (ms/query; log scale)', 'latency_targets', True),
                   ('qps', 'Queries per second (log scale)', 'throughput_targets', True),
                   ('total_MB', 'Estimated resident index size (MB)', 'memory_targets', False),
                   ('build_s', 'Base index build time (s; log scale)', 'build_targets', True),
                   ('recall10', 'Achieved recall@10', 'achieved_recall_targets', False)]
        for metric, ylabel, name, log in metrics:
            fig, axes = plt.subplots(1, 3, figsize=(13, 5))
            for ax, target in zip(axes, targets):
                g = selected[selected.target_recall == target].set_index('method').reindex(methods)
                bars = ax.bar(np.arange(len(methods)), g[metric], color=colors)
                if log:
                    ax.set_yscale('log')
                ax.set_xticks(np.arange(len(methods)), labels, rotation=35, ha='right')
                ax.set(title=f'Recall target ≥ {target:.0%}', ylabel=ylabel)
                if metric == 'recall10':
                    ax.set_ylim(.85, 1.025)
                    ax.axhline(target, color='black', linestyle='--', linewidth=1)
                else:
                    ax.margins(y=.25)
                ax.bar_label(bars, labels=[f'{v:.4f}' if metric in ['latency_ms', 'recall10'] else f'{v:,.2f}' for v in g[metric]], fontsize=8, padding=3)
                ax.grid(axis='y', alpha=.2)
                ax.set_axisbelow(True)
            fig.suptitle('SIFT1M: ' + ylabel.split(' (')[0] + ' at selected operating points')
            save(fig, name, 'Each bar uses the fastest measured configuration meeting its recall target. Builds may differ between targets; see selected_configs.csv.')

        fig, ax = plt.subplots(figsize=(10, 5))
        baseline = raw[raw.method == 'flat'].latency_ms.iloc[0]
        x = np.arange(len(methods))
        for i, target in enumerate(targets):
            g = selected[selected.target_recall == target].set_index('method').reindex(methods)
            vals = baseline / g.latency_ms
            bars = ax.bar(x + (i - 1) * .25, vals, width=.25, label=f'Recall ≥ {target:.0%}')
            ax.bar_label(bars, fmt='%.2f×', fontsize=8, padding=3)
        ax.axhline(1, color='black', linestyle='--', linewidth=1)
        ax.set_xticks(x, labels)
        ax.set(ylabel='Search speedup versus Exact Flat', title='SIFT1M: speedup at each recall target')
        ax.margins(y=.2)
        ax.legend()
        save(fig, 'speedup_targets', 'Speedup = Flat latency / selected latency. Above 1× is faster than Flat; below 1× is slower.')

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        g = selected[selected.target_recall == .95].set_index('method').reindex(methods)
        for ax, metric, xlabel in [(axes[0], 'total_MB', 'Estimated resident index size (MB)'),
                                  (axes[1], 'build_s', 'Base index build time (s; log scale)')]:
            for m, label, color in zip(methods, labels, colors):
                r = g.loc[m]
                ax.scatter(r[metric], r.latency_ms, s=90, color=color, label=label)
            ax.set(xlabel=xlabel, ylabel='Latency (ms/query; log scale)', yscale='log')
            if metric == 'build_s':
                ax.set_xscale('log')
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        fig.suptitle('SIFT1M: memory, build time and query speed at recall ≥ 95%')
        save(fig, 'resource_latency_tradeoffs', 'Lower and further left is better. Each point uses that algorithm’s fastest measured configuration reaching 95% recall.')

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, col, xlabel in [(axes[0], 'total_MB', 'Estimated resident index size (MB)'),
                                (axes[1], 'build_s', 'Base index build time (s; log scale)')]:
            for j, (m, color) in enumerate(zip(methods, colors)):
                values = builds[builds.method == m][col].sort_values().to_numpy()
                offsets = np.linspace(-.14, .14, len(values)) if len(values) > 1 else [0]
                ax.hlines(j, values.min(), values.max(), color=color, linewidth=2, alpha=.5)
                ax.scatter(values, j + np.asarray(offsets), color=color, s=32)
            ax.set_yticks(range(len(methods)), labels)
            ax.invert_yaxis()
            ax.set_xlabel(xlabel)
            if col == 'build_s':
                ax.set_xscale('log')
            ax.grid(axis='x', alpha=.2)
        fig.suptitle('SIFT1M: resources across all 27 measured builds')
        save(fig, 'all_build_resources', 'Each dot is one distinct build; horizontal lines show observed ranges. Vertical offsets only separate overlapping dots.')

    gallery = ['# SIFT1M comparison plots', '',
               'Generated from existing benchmark measurements; no new benchmark runs.', '',
               'Reproduce: `venv/bin/python -m scripts.sift_report` then `venv/bin/python -m scripts.sift_comparison_plots`.', '',
               'Search latency is single-threaded batch time/query, not request percentiles. Memory is serialized base size plus retained reranking vectors, not process RSS.',
               'Build time is recorded base-index training + insertion, excluding refinement setup, loading and serialization. MB uses decimal units.', '',
               '[Download the complete plot collection](all_comparisons.pdf)', '']
    for name, caption in entries:
        gallery.extend([f'## {name.replace("_", " ").title()}', '', f'![{name}]({name}.png)', '', caption, ''])
    (out / 'gallery.md').write_text('\n'.join(gallery))
    print(f'Created {len(entries)} plots as PNG/PDF, combined PDF and gallery in {out}')


if __name__ == '__main__':
    main()
