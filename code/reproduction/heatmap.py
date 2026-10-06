"""Two matched 2x2 score maps, using the four-seed final evidence only."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from scipy.stats import t

from .paths import ROOT, OUT
BASE = ROOT
SOURCE = BASE / 'data/evaluation.json'
data = json.loads(SOURCE.read_text(encoding='utf-8'))
assert data['epoch'] == 20
metrics = ['valid', 'readout_a1_test']
means = np.zeros((2, 2, 2))
audit = []
seed_reference = None
for i, epoch in enumerate([2, 10]):
    for j, homeostasis in enumerate(['continue', 'freeze']):
        rows = sorted([r for r in data['rows'] if r['batch'] == 'Kernel-homeostasis' and r['boundary'] == epoch and r['homeostasis'] == homeostasis], key=lambda r: r['seed'])
        seeds = [r['seed'] for r in rows]
        assert len(rows) == len(set(seeds)) == 4
        if seed_reference is None:
            seed_reference = seeds
        assert seeds == seed_reference
        for k, metric in enumerate(metrics):
            values = 100 * np.array([r['metrics'][metric] for r in rows])
            means[k, i, j] = values.mean()
            half = t.ppf(.975, 3)*values.std(ddof=1)/2
            audit.append(dict(epoch=epoch, homeostasis=homeostasis, metric=metric,
                              seeds=seeds, values=values.tolist(), mean=values.mean(),
                              ci95=[values.mean()-half, values.mean()+half]))
# Assert the rendered values agree with the complete original manuscript table.
expected = np.array([[[97.575, 97.700], [96.246, 96.321]],
                     [[98.635, 98.755], [98.582, 98.513]]])
np.testing.assert_allclose(np.round(means, 3), expected, atol=1e-10)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none'})
fig = plt.figure(figsize=(7.16, 3.0), facecolor='white')
norm = Normalize(96, 99)
cmap = plt.get_cmap('Blues')
for k, label in enumerate(['(a) Valid', '(b) Readout (Test)']):
    ax = fig.add_axes([.09 + k*.435, .13, .31, .65])
    im = ax.imshow(means[k], norm=norm, cmap=cmap, aspect='equal', interpolation='nearest')
    ax.set_xticks([0, 1], ['Continue', 'Freeze'])
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Homeostasis', labelpad=8)
    ax.set_yticks([0, 1], ['2', '10'])
    ax.set_ylabel('Epoch', labelpad=6)
    ax.tick_params(length=0, pad=7)
    ax.set_xticks([.5], minor=True)
    ax.set_yticks([.5], minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    ax.tick_params(which='minor', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(2):
        for j in range(2):
            value = means[k, i, j]
            rgb = cmap(norm(value))[:3]
            luminance = np.dot(rgb, [.2126, .7152, .0722])
            ax.text(j, i, f'{value:.3f}', ha='center', va='center', fontsize=13,
                    weight='bold', color='white' if luminance < .5 else '#172f40')
    fig.text(.245 + k*.435, .04, label, ha='center', fontsize=11)
cax = fig.add_axes([.88, .165, .018, .57])
cb = fig.colorbar(im, cax=cax, ticks=[96, 97, 98, 99])
cb.ax.tick_params(labelsize=8)
cb.set_label('Accuracy (%)', labelpad=4, fontsize=9)
stem = OUT / 'figure_03_kernel_homeostasis'
for ext in ['pdf', 'svg', 'png', 'jpg']:
    kwargs = {'pil_kwargs': {'quality': 95}} if ext == 'jpg' else {}
    fig.savefig(stem.with_suffix('.'+ext), dpi=240, **kwargs)
plt.close(fig)
manifest = dict(source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                evaluation_epoch=20, rows='kernel freeze boundary before epoch 2 or 10',
                columns=['Homeostasis Continue', 'Homeostasis Freeze'],
                panels=['Native validation accuracy', 'Hidden linear Readout test accuracy'],
                shared_color_limits=[96, 99], cells=audit,
                native_test_in_evaluation_data=True)
(OUT / 'heatmap_statistics.json').write_text(
    json.dumps(manifest, indent=2), encoding='utf-8')
print('PASS: four matched seeds per cell; values match original table; shared scale 96-99%.')
print(means.tolist())
