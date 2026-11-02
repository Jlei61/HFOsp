#!/usr/bin/env python3
"""Verify or restore the author-accepted Figure 4 exports.

The default verifies the frozen release. --output-dir restores its original
PNG/PDF/SVG files to a separate empty directory, byte for byte. This is an
artifact rebuild, not a new rendering from model parameters or an SNN run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / 'results/paper-ready-figure/fig4'
VERSION = 'snn_network_optimization_final_20261102'
LAYOUT = 'snn_layout_ei_labels_20261031'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify(package):
    pointer = read(package / 'current_version.json')
    registry = read(package / 'figure4_panel_registry.json')
    manifest = read(package / 'release_manifest.json')
    for record in (pointer, registry, manifest):
        if (record.get('status') != 'AUTHOR_ACCEPTED_FINAL'
                or record.get('version') != VERSION
                or record.get('layout_version') != LAYOUT):
            raise ValueError('Figure 4 pointer, registry and release manifest disagree.')
    if pointer['panel_letters'] != list('ABCDEFGHI') or 'pending_panel_a_revision' in pointer:
        raise ValueError('The accepted Figure 4 must be A-I without a pending revision.')
    for name, expected in manifest['files'].items():
        path = package / name
        if not path.is_file() or digest(path) != expected['sha256']:
            raise ValueError(f'Missing or changed Figure 4 release file: {name}')
    if pointer['panels'] != registry['panels']:
        raise ValueError('Figure 4 panel/data contracts disagree.')
    if set(pointer['panels']) != set('ABCDEFGHI'):
        raise ValueError('The accepted Figure 4 must contain A-I.')
    if pointer['panels']['D']['version'] != 'native_no_core_background':
        raise ValueError('Legacy Figure 4 D cannot replace the native no-core panel.')
    if pointer['panels']['E']['version'] != 'native_single_core_bursts':
        raise ValueError('Legacy Figure 4 E cannot replace the native core panel.')
    for name, expected in manifest.get('repo_files', {}).items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Figure 4 producer changed: {name}')
    actual = {str(p.relative_to(package)) for p in (package / 'figures').iterdir()
              if p.suffix in ('.png', '.pdf', '.svg')}
    if actual != set(registry['outputs']):
        raise ValueError('Unexpected or missing current Figure 4 exports.')
    for name, expected in registry['outputs'].items():
        if digest(package / name) != expected:
            raise ValueError(f'Current output hash changed: {name}')
    return dict(status='PASS', version=VERSION, layout_version=LAYOUT,
                author_acceptance='CONFIRMED', files_verified=len(manifest['files']),
                export_files_verified=len(registry['outputs']))


def restore(package, output):
    output = output.resolve()
    if output == package.resolve() or package.resolve() in output.parents:
        raise ValueError('Use an output directory outside the accepted Figure 4 package.')
    if output.exists() and any(output.iterdir()):
        raise ValueError('The output directory must be empty.')
    (output / 'figures').mkdir(parents=True, exist_ok=True)
    registry = read(package / 'figure4_panel_registry.json')
    for name, expected in registry['outputs'].items():
        shutil.copy2(package / name, output / name)
        if digest(output / name) != expected:
            raise ValueError(f'Restored output differs: {name}')
    shutil.copy2(package / 'figures/README.md', output / 'figures/README.md')
    report = dict(status='PASS', version=VERSION, export_files=len(registry['outputs']),
                  all_exports_byte_identical=True, method='Frozen native export restoration; no analysis rerun.')
    (output / 'restore_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=PACKAGE)
    parser.add_argument('--output-dir', type=Path, help='Separate empty directory for exact export restoration.')
    args = parser.parse_args()
    report = verify(args.package)
    if args.output_dir:
        report['restore'] = restore(args.package, args.output_dir)
        report['output'] = str(args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
