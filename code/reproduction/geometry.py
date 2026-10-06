"""Single table: saved kernel snapshots and equal-duration paired changes."""
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
SOURCE = BASE / 'data/kernels.npz'
digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
reference = json.loads((BASE / 'data/geometry_reference.json').read_text(encoding='utf-8'))
assert digest == reference['kernel_sha256']
seeds = [34100, 34101, 34102, 34103]
epochs = [10, 30, 50, 70, 90]
with np.load(SOURCE) as z:
    W = {e: np.stack([z[f's{s}_e{e:03d}'] for s in seeds]).astype(float) for e in epochs}
assert all(w.shape == (4, 300, 600) and np.isfinite(w).all() for w in W.values())

def stats(x):
    x = np.asarray(x, dtype=float)
    mean = float(x.mean())
    half = float(t.ppf(.975, len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)))
    return dict(mean=mean, ci95=[mean-half, mean+half], seeds=seeds, values=x.tolist())

labels = ['Participation ratio', 'Global DC energy fraction', 'Mean receiver DC fraction', 'Signed mass']
snapshots = {}
for e, batch in W.items():
    values = []
    for w in batch:
        power = np.linalg.svd(w, compute_uv=False)**2
        dc = 100*(w.reshape(300, 6, 10, 10).mean((2, 3))**2).sum(1)
        energy = (w*w).sum(1)
        assert np.all(energy > 0)
        values.append([power.sum()**2/(power**2).sum(), dc.sum()/energy.sum(),
                       np.mean(dc/energy), w.sum()/np.abs(w).sum()])
    snapshots[e] = np.array(values)
previous = json.loads((BASE / 'data/geometry_reference.json').read_text(encoding='utf-8'))
for e in [10, 50, 90]:
    for i, key in enumerate(labels):
        np.testing.assert_allclose(snapshots[e][:,i], previous['snapshots'][str(e)][key]['values'], rtol=1e-10, atol=1e-12)

intervals = []
for start, end in zip(epochs[:-1], epochs[1:]):
    assert end-start == 20
    a, b = W[start], W[end]
    delta = b-a
    norms = np.linalg.norm(delta, axis=-1)
    assert np.all(norms > 0)
    cosine = (a*b).sum(-1)/(np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1))
    orth = delta-(((delta*a).sum(-1)/(a*a).sum(-1))[..., None])*a
    np.testing.assert_allclose((orth*a).sum(-1), 0, atol=1e-10)
    assert np.all(snapshots[start] > 0), 'Relative changes require nonzero, positive reference values here.'
    intervals.append(dict(start=start, end=end, duration=end-start,
        cosine=stats(cosine.mean(1)),
        angle_degrees=stats(np.degrees(np.arccos(np.clip(cosine, -1, 1))).mean(1)),
        snapshot_deltas={key:stats(snapshots[end][:,i]-snapshots[start][:,i]) for i,key in enumerate(labels)},
        snapshot_relative_percent={key:stats(100*(snapshots[end][:,i]-snapshots[start][:,i])/snapshots[start][:,i]) for i,key in enumerate(labels)},
        relative_l2=stats(np.linalg.norm(delta, axis=(1, 2))/np.linalg.norm(a, axis=(1, 2))),
        orthogonal_fraction=stats((np.linalg.norm(orth, axis=-1)/norms).mean(1))))

plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9, 'pdf.fonttype':42, 'svg.fonttype':'none'})
fig = plt.figure(figsize=(10.6, 4.0), facecolor='white')
ax = fig.add_axes([.015, .025, .97, .95])
ax.axis('off')
rows = []
for i, label in enumerate(labels):
    cells = [label]+[f'{snapshots[e][:,i].mean():.5f}' for e in epochs]
    rows.append(cells)
snapshot_rows = rows
rows = []
for key, label in [('cosine','Mean direction cosine'), ('angle_degrees','Mean direction angle (0-180°)'),
                   ('orthogonal_fraction','Orthogonal fraction (0-100%)')]:
    cells = [label, '-']
    for row in intervals:
        v = row[key]
        if key == 'angle_degrees':
            cells.append(f"{v['mean']:.1f}°")
        elif key == 'orthogonal_fraction':
            cells.append(f"{100*v['mean']:.1f}%")
        else:
            cells.append(f"{v['mean']:.3f}")
    rows.append(cells)
for title, title_y, data, headers, bounds in [
    ('(A) Aggregate kernel statistics at each checkpoint', .97, snapshot_rows,
     ['Statistic']+[f'Epoch {e}' for e in epochs], [0, .50, 1, .42]),
    ('(B) Directional change over consecutive 20-epoch intervals', .40, rows,
     ['Statistic', 'Baseline']+[f'{r["start"]} → {r["end"]}' for r in intervals],
     [0, 0, 1, .35]),
]:
    ax.text(0, title_y, title, transform=ax.transAxes, fontsize=11,
            fontweight='bold', va='center', color='#19374b')
    tab = ax.table(cellText=data, colLabels=headers,
                   colWidths=[.30]+[.14]*5, cellLoc='center', bbox=bounds)
    tab.auto_set_font_size(False)
    tab.set_fontsize(9.2)
    for (i,j), cell in tab.get_celld().items():
        cell.set_edgecolor('#c9d3dc')
        cell.set_linewidth(.5)
        if i == 0:
            cell.set_facecolor('#e7f0f6')
            cell.set_text_props(weight='bold')
        if i > 0 and j == 0:
            cell.set_text_props(ha='left')
stem = OUT / 'table_02_geometry'
for ext in ['pdf','svg','png','jpg']:
    kwargs = {'pil_kwargs':{'quality':95}} if ext == 'jpg' else {}
    fig.savefig(stem.with_suffix('.'+ext), dpi=240, **kwargs)
plt.close(fig)
audit = dict(kernel_sha256=digest, epochs=epochs, seeds=seeds, neuron_count=300,
    snapshots={str(e):{key:stats(snapshots[e][:,i]) for i,key in enumerate(labels)} for e in epochs},
    intervals=intervals, interpolation=False, interval_duration_epochs=20,
    comparison_reference='previous displayed epoch', first_comparison=None,
    overlapping_snapshot_seedwise_match=True, display_ci=False,
    display_snapshot_relative_change=False, snapshot_relative_change_retained_in_audit=True,
    display_units={'snapshots':'original units', 'snapshot_relative_change':'percent, not percentage points',
                   'orthogonal_fraction':'percent', 'angle_degrees':'degrees'},
    relative_change_aggregation='100*(current-previous)/previous within each seed, then mean across four seeds',
    angle_aggregation='mean of per-receiver arccos, then mean across seeds; not arccos of mean cosine',
    relative_l2_retained_in_audit=True)
(OUT / 'geometry_statistics.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
print('PASS: 20 saved kernels; four matched seeds; equal 20-epoch intervals; overlapping snapshots reproduced.')
from .common import assert_numeric_match
assert_numeric_match(audit, reference)
