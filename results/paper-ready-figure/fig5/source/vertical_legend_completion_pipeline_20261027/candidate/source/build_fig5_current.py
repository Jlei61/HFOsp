#!/usr/bin/env python3
"""Validate/recover the registered new Fig5; Z/M figures now belong to Fig6."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path)
    args=ap.parse_args()
    current=json.loads((ROOT/'results/paper-ready-figure/fig5/current_version.json').read_text())
    assert current['asset_id']=='spatial_snn_propagation_and_core_count'
    record=json.loads((ROOT/current['manifest']).read_text())
    for name,digest in record['outputs_sha256'].items():
        p=ROOT/name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
        if args.output_dir:
            args.output_dir.mkdir(parents=True,exist_ok=True);shutil.copy2(p,args.output_dir/p.name)
    print(json.dumps(dict(figure='Fig5',validated=len(record['outputs_sha256']),status=current['status'],
        human_visual_acceptance=current['human_visual_acceptance'],cohort_comparison_complete=record['cohort_comparison_complete'])))

if __name__=='__main__':main()
