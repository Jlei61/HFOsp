#!/usr/bin/env python3
"""Validate or copy the frozen former Fig5 exports under their current Fig6 identity."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path)
    args=p.parse_args()
    manifest=json.loads((ROOT/'results/paper-ready-figure/fig6/current_version.json').read_text())
    for name,digest in manifest['outputs_sha256'].items():
        source=ROOT/name
        assert hashlib.sha256(source.read_bytes()).hexdigest()==digest,name
        if args.output_dir:
            args.output_dir.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,args.output_dir/source.name)
    print(json.dumps(dict(figure='Fig6',validated=len(manifest['outputs_sha256']),scientific_content_changed=False)))

if __name__=='__main__':main()
