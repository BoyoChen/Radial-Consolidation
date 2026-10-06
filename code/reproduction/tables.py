"""Generate numerical tables, CSV and LaTeX from per-seed evidence."""
import csv
import matplotlib.pyplot as plt
import numpy as np
from . import paths
from .common import read_data, save_json, mean_ci, save_figure


PROTOCOL = [
 ['Input channels / time steps','6 / 40','2 / 20'],
 ['Hidden channels','300','330'], ['Hidden kernel / stride','10 x 10 / 2','9 x 9 / 2'],
 ['Output neurons per class','20','45'], ['Trace decay gamma','0.66','0.97'],
 ['Lateral inhibition factor','3','30'], ['Temporal margin','0.30','0.35'],
 ['Positive / negative teaching gains','40 / 120','32 / 100'],
 ['Group softmax temperature','0.25','0.25'], ['Threshold bounds','[0.1, 10]','[1, 100]'],
 ['Threshold log-step','0.00125','0.00125'], ['Hidden amplifier log-step','6.25e-6','0'],
 ['Output amplifier log-step','1.25e-6','0'], ['Per-receiver normalization','L1','L1']]


def draw_table(ax, rows, headers, widths=None, size=9):
    ax.axis('off')
    table=ax.table(cellText=rows,colLabels=headers,colWidths=widths,cellLoc='center',bbox=[0,0,1,1])
    table.auto_set_font_size(False);table.set_fontsize(size)
    for (i,j),cell in table.get_celld().items():
        cell.set_edgecolor('#c9d3dc');cell.set_linewidth(.5)
        if i==0:
            cell.set_facecolor('#e7f0f6');cell.set_text_props(weight='bold')
        if i>0 and j==0:cell.set_text_props(ha='left')


def export_table(name,headers,rows):
    with (paths.OUT/f'{name}.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(headers);writer.writerows(rows)
    def escape(s):
        return str(s).replace('%',r'\%').replace('_',r'\_').replace('&',r'\&')
    lines=[r'\begin{tabular}{'+'l'*len(headers)+'}',r'\hline']
    for row in [headers]+rows:lines.append(' & '.join(map(escape,row))+r' \\')
    lines += [r'\hline',r'\end{tabular}']
    (paths.OUT/f'{name}.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def build():
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(8,5.0));fig.subplots_adjust(left=.02,right=.98,bottom=.03,top=.97)
    draw_table(ax,PROTOCOL,['Setting','MNIST','N-MNIST'],[.52,.24,.24])
    save_figure(fig,'table_01_protocol');plt.close(fig)
    export_table('table_01_protocol',['Setting','MNIST','N-MNIST'],PROTOCOL)
    evidence=read_data('evaluation.json')['rows']
    rows=[];contrasts=[];audit=[]
    for batch,n in [('MNIST main',5),('MNIST replication',5),('N-MNIST',7)]:
        arms={}
        for condition in ['Unrestricted','Hard Freeze','Radial-only']:
            arm=sorted([r for r in evidence if r['batch']==batch and r['condition']==condition],key=lambda r:r['seed'])
            assert len(arm)==n and len({r['seed'] for r in arm})==n
            arms[condition]=arm
            display=condition+(' @2' if condition!='Unrestricted' else '')
            if batch=='N-MNIST' and condition=='Unrestricted':display+=' (100% no-decision)'
            rows.append([batch,display,
                         str(n),f"{100*np.mean([r['metrics']['valid'] for r in arm]):.3f}",
                         f"{100*np.mean([r['metrics']['test'] for r in arm]):.3f}"])
        ro,hf=arms['Radial-only'],arms['Hard Freeze']
        assert [r['seed'] for r in ro]==[r['seed'] for r in hf]
        cells=[batch]
        for metric in ['valid','test','gap']:
            factor=1 if metric=='gap' else 100
            stats=mean_ci([factor*(r['metrics'][metric]-h['metrics'][metric]) for r,h in zip(ro,hf)])
            decimals=5 if metric=='gap' else 3
            low,high=stats['ci95']
            cells.append(f"{stats['mean']:+.{decimals}f} [{low:.{decimals}f}, {high:.{decimals}f}]")
            audit.append(dict(batch=batch,metric=metric,**stats))
        contrasts.append(cells)
    fig=plt.figure(figsize=(11,5.1))
    top=fig.add_axes([.02,.38,.96,.59]);bottom=fig.add_axes([.02,.035,.96,.245])
    headers=['Dataset / batch','Condition','Seeds','Validation (%)','Test (%)']
    draw_table(top,rows,headers,[.22,.34,.08,.18,.18],8.5)
    diff_headers=['Radial minus hard freeze','Validation diff. (pp)','Test diff. (pp)','Timing-gap difference']
    draw_table(bottom,contrasts,diff_headers,[.24,.25,.25,.26],8.2)
    save_figure(fig,'table_03_results');plt.close(fig)
    export_table('table_03_accuracy',headers,rows)
    export_table('table_03_paired',diff_headers,contrasts)
    save_json('paired_statistics.json',audit)
    import json
    geo=json.loads((paths.OUT/'geometry_statistics.json').read_text(encoding='utf-8'))
    metrics=list(geo['snapshots']['10'])
    export_table('table_02_snapshots',['Statistic']+[f'Epoch {e}' for e in geo['epochs']],
        [[m]+[f"{geo['snapshots'][str(e)][m]['mean']:.5f}" for e in geo['epochs']] for m in metrics])
    export_table('table_02_intervals',['Statistic']+[f"{r['start']} to {r['end']}" for r in geo['intervals']],
        [[key]+[r[key]['mean']*(100 if key=='orthogonal_fraction' else 1) for r in geo['intervals']]
         for key in ['cosine','angle_degrees','orthogonal_fraction']])
