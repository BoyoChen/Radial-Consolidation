from pathlib import Path
import hashlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import t

def save_split_panels(final_dir, coords, selected, shown, indices, lim, colors, reference, epochs):
    """Publication-size panels; methodological prose belongs in the LaTeX caption."""
    final_dir.mkdir(exist_ok=True)
    outputs = {}
    with plt.rc_context({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
                         'ytick.labelsize': 7, 'legend.fontsize': 7}):
        top = plt.figure(figsize=(7.16, 1.62), facecolor='white')
        grid = top.add_gridspec(1, 7, left=.065, right=.993, bottom=.30, top=.85, wspace=.17)
        for panel, (epoch, index) in enumerate(zip(shown, indices)):
            ax = top.add_subplot(grid[0, panel])
            xy = coords[index, :, :2]
            ax.scatter(xy[:, 0], xy[:, 1], c='#bfc7ce', s=1.2, alpha=.5, edgecolors='none')
            ax.axhline(0, color='#d9dfe5', lw=.4)
            ax.axvline(0, color='#d9dfe5', lw=.4)
            for color, neuron in zip(colors, selected):
                x, y = xy[neuron]
                ax.plot([0, x], [0, y], color=color, lw=.8, alpha=.85)
                ax.scatter(x, y, color=color, s=8, edgecolors='white', linewidth=.25, zorder=3)
            ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xticks=[-.5, 0, .5], yticks=[-.5, 0, .5])
            ax.set_aspect('equal')
            ax.set_title(f'Epoch {epoch}', fontsize=8, pad=5)
            ax.tick_params(length=2, pad=2)
            if panel == 0:
                ax.set_ylabel('Fixed PC2', labelpad=2)
            else:
                ax.tick_params(labelleft=False)
        top.text(.53, .14, 'Fixed PC1', ha='center', fontsize=8)
        handles = [Line2D([0], [0], marker='o', color=c, lw=.8, markersize=3, label=f'N{n:03d}')
                   for n, c in zip(selected, colors)]
        top.legend(handles=handles, loc='lower center', bbox_to_anchor=(.53, -.005), ncol=5,
                   frameon=False, handlelength=1.1, columnspacing=1.5)
        bottom = plt.figure(figsize=(7.16, 1.85), facecolor='white')
        ax = bottom.add_axes([.085, .235, .90, .715])
        for key, xs, color, label in [
            ('reference_angles', epochs, '#236993', 'Angle from epoch 2'),
            ('four_epoch_angles', epochs[1:-1], '#b44940', 'Change over preceding 4 epochs')]:
            values = np.array(reference[key])
            mean = values.mean(0)
            half = t.ppf(.975, 3) * values.std(0, ddof=1) / 2
            ax.plot(xs, mean, color=color, lw=1.1, label=label)
            ax.fill_between(xs, mean-half, mean+half, color=color, alpha=.15, linewidth=0)
        for e in shown:
            ax.axvline(e, color='#dae0e6', lw=.5, zorder=0)
        ax.axvline(50, color='#737d87', ls='--', lw=.7)
        ax.set(xlabel='Training epoch', ylabel='Angle (degrees)', xlim=(2, 100), ylim=(0, 90), xticks=shown)
        ax.legend(loc='upper right', frameon=False, ncol=2)
        ax.grid(axis='y', alpha=.16)
        for suffix, fig in [('snapshots', top), ('angles', bottom)]:
            stem = final_dir / f'kernel_direction_drift_{suffix}'
            for ext in ['pdf', 'svg', 'png', 'jpg']:
                kwargs = {'pil_kwargs': {'quality': 95, 'optimize': True}} if ext == 'jpg' else {}
                if ext == 'pdf':
                    kwargs['metadata'] = {'Title': '', 'Author': ''}
                target = stem.with_suffix('.' + ext)
                fig.savefig(target, dpi=260, **kwargs)
                outputs[target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
            plt.close(fig)
    return outputs
