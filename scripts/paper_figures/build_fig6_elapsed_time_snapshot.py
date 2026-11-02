#!/usr/bin/env python3
"""Verify the published Fig6 time-colored snapshot and optionally copy its exports."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / 'results/paper-ready-figure/fig6/candidates/selected_r08_r09_elapsed_time_20261102'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        help='Copy the verified frozen PNG/PDF/SVG exports; does not rerun the simulation.')
    args = parser.parse_args()
    manifest = json.loads((PACKAGE / 'publication_manifest.json').read_text())
    for name, digest in manifest['files_sha256'].items():
        path = PACKAGE / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Snapshot mismatch: {name}')
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name in manifest['figure_exports']:
            source = PACKAGE / name
            shutil.copy2(source, args.output_dir / source.name)
    print(json.dumps(dict(figure='Fig6', version=manifest['version'],
                         validated_files=len(manifest['files_sha256']),
                         figure_exports=len(manifest['figure_exports']),
                         simulation_rerun=False, formal_pointer_changed=False), indent=2))


if __name__ == '__main__':
    main()
