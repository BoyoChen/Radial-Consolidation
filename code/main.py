"""Rebuild manuscript figures from bundled evidence: python code/main.py --analyze"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import runpy
import time

# Bound linear-algebra threading before importing NumPy, including on large servers.
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(key,'2')
os.environ.setdefault('MPLBACKEND','Agg')


def verify_inputs(root):
    manifest=json.loads((root/'data/manifest.json').read_text(encoding='utf-8'))
    for name,expected in manifest.items():
        path=root/name
        if not path.is_file():raise RuntimeError(f'Missing bundled data: {name}')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected['sha256']:
            raise RuntimeError(f'Bundled data checksum mismatch: {name}')
    print(f'PASS: {len(manifest)} bundled data checksums')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analyze',action='store_true',help='Rebuild all figures and tables (the default; no training)')
    parser.add_argument('--output',type=Path,help='Output directory (default: repository/result)')
    parser.add_argument('--verify-only',action='store_true',help='Verify bundled source data without rendering')
    args=parser.parse_args()
    from reproduction import paths
    verify_inputs(paths.ROOT)
    if args.verify_only:return
    if args.output:paths.OUT=args.output.resolve()
    protected=[paths.ROOT/folder for folder in ['data','code','tests','docs','paper','.git','.github']]
    if paths.OUT==paths.ROOT or any(paths.OUT==p or p in paths.OUT.parents for p in protected):
        raise ValueError('Output must not replace source, paper, metadata or bundled data')
    paths.OUT.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    from reproduction import trajectory,tables,booklet
    from reproduction.common import save_json,assert_numeric_match,read_data
    trajectory.build()
    for module in ['geometry','heatmap','accuracy_gap']:
        runpy.run_module('reproduction.'+module,run_name='__main__')
    tables.build()
    # Cross-check final paired statistics against an independently exported baseline.
    actual=json.loads((paths.OUT/'paired_statistics.json').read_text())
    assert_numeric_match(actual,read_data('paired_reference.json'))
    booklet.build()
    required=['figure_01_method','figure_02_drift','figure_03_kernel_homeostasis','figure_04_accuracy_gap',
              'table_01_protocol','table_02_geometry','table_03_results','algorithm_01_update','figures_and_tables']
    import pymupdf as fitz
    for name in required:
        with fitz.open(paths.OUT/f'{name}.pdf') as doc:
            assert len(doc)==(8 if name=='figures_and_tables' else 1)
    # Git checkouts must preserve the bytes covered by public artifact hashes.
    for path in paths.OUT.iterdir():
        if path.is_file() and path.suffix in {'.json','.csv','.tex','.svg','.txt'}:
            path.write_bytes(path.read_bytes().replace(b'\r\n',b'\n'))
    previews={f'{name}.png' for name in required if name.startswith('figure_')}
    release_files=[p for p in sorted(paths.OUT.iterdir()) if p.is_file()
                   and (p.suffix in {'.pdf','.csv','.tex','.json'} or p.name in previews)
                   and p.name!='reproduction_manifest.json']
    save_json('reproduction_manifest.json',dict(status='PASS',elapsed_seconds=time.perf_counter()-started,
        python=platform.python_version(),platform=platform.system(),
        packages={p:importlib.metadata.version(p) for p in ['numpy','scipy','matplotlib','pillow','reportlab','pymupdf']},
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in release_files}))
    print(f'PASS: 4 figures, 3 tables, Algorithm 1 and 8-page overview in {time.perf_counter()-started:.1f}s')
    print(paths.OUT/'figures_and_tables.pdf')


if __name__=='__main__':
    main()
