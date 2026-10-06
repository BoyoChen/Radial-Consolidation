"""Paired validation accuracy and evidence gap, manuscript Figure 4."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

from .paths import ROOT, OUT
BASE = ROOT
SOURCE = BASE / 'data/evaluation.json'
data = json.loads(SOURCE.read_text(encoding='utf-8'))
assert data['epoch'] == 20
batches = [(name, name, ['Unrestricted','Hard Freeze','Radial-only'], n) for name,n in [('MNIST main',5),('MNIST replication',5),('N-MNIST',7)]]
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#777777', 'axes.linewidth': .7,
    'pdf.fonttype': 42, 'svg.fonttype': 'none'})
colors = ['#717987', '#2F7DE1', '#F36B3B']
# Keep canonical data/statistics order; change only the plotted order.
display_order = [0, 2, 1]
display_labels = ['Unrestricted', 'Radial-only', 'Hard Freeze']
fig = plt.figure(figsize=(10.3, 5.5))
grid = fig.add_gridspec(2, 3, left=.082, right=.99, top=.915, bottom=.10,
                      hspace=.36, wspace=.27)
audit = []
for col, (name, rid, names, n) in enumerate(batches):
    arms = [sorted([r for r in data['rows'] if r['batch'] == rid and r['condition'] == arm],
                   key=lambda r: r['seed']) for arm in names]
    seeds = [[r['seed'] for r in a] for a in arms]
    assert seeds[0] == seeds[1] == seeds[2] and len(set(seeds[0])) == n
    if col == 2:
        assert all(r['metrics']['no_decision'] == 1 for r in arms[0])
    for row, metric in enumerate(['valid', 'gap']):
        values = np.array([[r['metrics'][metric] * (100 if row == 0 else 1)
                            for r in a] for a in arms])
        sub = grid[row, col].subgridspec(2, 1, height_ratios=[7, 1], hspace=.10)
        ax = fig.add_subplot(sub[0])
        low = fig.add_subplot(sub[1], sharex=ax)
        ylim = (94.4, 98.4) if row == 0 else ((.19, .33) if col < 2 else (.30, .44))
        ticks = [95, 96, 97, 98] if row == 0 else ([.20, .24, .28, .32] if col < 2 else [.30, .34, .38, .42])
        ax.set(xlim=(-.42, 2.42), ylim=ylim, yticks=ticks)
        ax.tick_params(axis='x', bottom=False, labelbottom=False)
        ax.tick_params(axis='y', labelsize=10, length=3)
        low.set(xticks=[0, 1, 2], xticklabels=display_labels)
        low.tick_params(axis='x', labelsize=10, length=3, pad=5)
        if col < 2:
            low.set(ylim=(-1, 1), yticks=[])
            for spine in low.spines.values():
                spine.set_visible(False)
            low.tick_params(axis='x', length=0)
        else:
            low.set(ylim=(-.12, .12) if row == 0 else (-.004, .004), yticks=[0])
            low.tick_params(axis='y', labelsize=10, length=3)
            ax.spines['bottom'].set_visible(False)
            for target, y in [(ax, 0), (low, 1)]:
                target.plot([-.018, .018], [y-.025, y+.025], transform=target.transAxes,
                            color='#777777', lw=.8, clip_on=False)
        for v in values[display_order].T:
            start = 1 if col == 2 else 0
            ax.plot(range(start, 3), v[start:], color='#b5b5b5', lw=1.1, zorder=1)
        for x, arm_index in enumerate(display_order):
            color = colors[arm_index]
            target = low if col == 2 and x == 0 else ax
            if target is ax:
                assert values[arm_index].min() > ylim[0] and values[arm_index].max() < ylim[1]
            target.scatter(np.full(n, x), values[arm_index], s=19, color=color,
                           marker='x' if target is low else 'o', zorder=3)
            target.hlines(values[arm_index].mean(), x-.13, x+.13, color=color, lw=2, zorder=4)
        contrasts = {}
        for k, label in [(0, 'RO - U'), (1, 'RO - HF')]:
            delta = values[2] - values[k]
            mean = float(delta.mean())
            half = float(t.ppf(.975, n-1) * delta.std(ddof=1) / np.sqrt(n))
            contrasts[label] = {'mean': mean, 'ci95': [mean-half, mean+half]}
        if row == 0:
            ax.set_title(name, fontsize=13, pad=12)
        if col == 0:
            ax.set_ylabel('Validation accuracy (%)' if row == 0 else 'Evidence gap', labelpad=9)
        audit.append({'batch': name, 'metric': metric, 'seeds': seeds[0],
                      'values': values.tolist(), 'contrasts': contrasts,
                      'active_y_limits': ylim, 'broken_axis': col == 2})

stem = OUT / 'figure_04_accuracy_gap'
for ext in ['pdf', 'png', 'jpg', 'svg']:
    fig.savefig(stem.with_suffix('.'+ext), dpi=220,
                **({'pil_kwargs': {'quality': 95}} if ext == 'jpg' else {}))
plt.close(fig)
(OUT / 'accuracy_gap_statistics.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
