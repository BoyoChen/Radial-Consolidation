"""Validate regenerated artifacts; never train or contact external services."""
import argparse
import csv
import json
import re
from pathlib import Path
import sys
import hashlib
import numpy as np
import pymupdf as fitz

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from reproduction.common import assert_numeric_match


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'result')
    args=parser.parse_args()
    result=args.output.resolve()
    expected=['figure_01_method','figure_02_drift','figure_03_kernel_homeostasis','figure_04_accuracy_gap',
              'table_01_protocol','table_02_geometry','table_03_results','algorithm_01_update',
              'kernel_direction_drift_snapshots','kernel_direction_drift_angles','figures_and_tables']
    for stem in expected:
        with fitz.open(result/f'{stem}.pdf') as pdf:
            assert len(pdf)==(8 if stem=='figures_and_tables' else 1),stem
            assert all(p.rect.width>0 and p.rect.height>0 for p in pdf),stem
    def read(name):return json.loads((result/name).read_text(encoding='utf-8'))
    paired=json.loads((ROOT/'data/paired_reference.json').read_text())
    assert_numeric_match(read('paired_statistics.json'),paired)
    geometry=json.loads((ROOT/'data/geometry_reference.json').read_text())
    assert_numeric_match(read('geometry_statistics.json'),geometry)
    for name,rows in [('table_01_protocol',15),('table_02_snapshots',5),('table_02_intervals',4),
                      ('table_03_accuracy',10),('table_03_paired',4)]:
        with (result/f'{name}.csv').open(newline='',encoding='utf-8') as file:
            assert len(list(csv.reader(file)))==rows,name
    manifest=read('reproduction_manifest.json')
    assert manifest['status']=='PASS'
    for name,digest in manifest['files'].items():
        assert hashlib.sha256((result/name).read_bytes()).hexdigest()==digest,name
    # Public source should not retain local paths, orchestration IDs or credentials.
    patterns=[r'\bex\d{3,}\b',r'\b(?:Prefect|TensorBoard|mslab)\b',r'(?i)(?<![a-z0-9])[A-Z]:[/\\]',
              r'hooks[.]slack',r'github_pat_',r'ghp_[a-zA-Z0-9]{20,}']
    targets=[*ROOT.glob('*.md'),ROOT/'CITATION.cff',* (ROOT/'data').glob('*.json')]
    for path in targets:
        for pattern in patterns:
            assert not re.search(pattern,path.read_text(encoding='utf-8')),(path.name,pattern)
    print('PASS: 11 PDFs, 5 CSVs, paired/geometry statistics, artifact checksums and public metadata')


if __name__=='__main__':main()
