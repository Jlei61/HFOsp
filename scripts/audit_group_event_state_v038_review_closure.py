#!/usr/bin/env python3
"""Read-only integrity checks for the v0.3.8 review-repair evidence package."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): value.update(chunk)
    return value.hexdigest()


def audit(root, reports=None):
    start = time.time(); expected = {}; counts = {}; queue_results = {}
    reports = reports or root/'final_reports'
    summary_path = reports/'summary_main.json'; summary = json.loads(summary_path.read_text())
    def bind(path, digest):
        path = str(path)
        if path in expected and expected[path] != digest: raise ValueError(f'conflicting provenance: {path}')
        expected[path] = digest
    actual_queues = {path.parent.name for path in root.glob('*_batch/manifest.json')}
    if actual_queues != set(summary['repair_queues']): raise ValueError('summary omits or invents a repair queue')
    for name in sorted(actual_queues):
        directory = root/name; manifest = directory/'manifest.json'; status = json.loads((directory/'queue_status.json').read_text())
        plan = json.loads(manifest.read_text()); bind(manifest,status['manifest_sha256'])
        if status['status'] != 'COMPLETE' or status['failed'] or status['pending'] or status['running']:
            raise ValueError(f'incomplete queue: {name}')
        if set(status['jobs']) != {job['id'] for job in plan['jobs']}: raise ValueError('queue denominator mismatch')
        for relative,digest in plan['source_hashes'].items(): bind(Path(plan['source_root'])/relative,digest)
        expected_flags = plan.get('expected_output_flags',dict(development_targets_read=False,sealed_partition_opened=False,seizure_targets_read=False))
        for job in plan['jobs']:
            result = json.loads(Path(job['output']).read_text()); state = status['jobs'][job['id']]
            if result['status'] != state['status']: raise ValueError('worker/supervisor status mismatch')
            if any(result.get(k) is not value for k,value in expected_flags.items()): raise ValueError('partition contract mismatch')
            bind(job['output'],state['output_sha256'])
            for path,digest in job['input_hashes'].items(): bind(path,digest)
        queue_results[name] = {k:status[k] for k in ('complete','not_estimable','failed','pending','running')}
    for name in ('checkpoint_replay','checkpoint_replay_verified'):
        cards = list((root/name).glob('**/card.json')); counts[name] = len(cards)
        if len(cards) != 165: raise ValueError('checkpoint denominator mismatch')
        for path in cards:
            card = json.loads(path.read_text()); bind(card['replay_bundle'],card['replay_bundle_sha256'])
    for card in summary['source_card_manifest']:
        bind(card['_loaded_source_card'],card['_loaded_source_sha256'])
        if card.get('_loaded_overlay_card'): bind(card['_loaded_overlay_card'],card['_loaded_overlay_sha256'])
    lineage = json.loads(Path(summary['state_lineage']['path']).read_text())
    bind(summary['state_lineage']['path'],summary['state_lineage']['sha256'])
    bind(summary['model_inventory']['path'],summary['model_inventory']['sha256'])
    inventory = json.loads(Path(summary['model_inventory']['path']).read_text())
    for row in inventory['h2a_adapters']:
        bind(row['card'],row['card_sha256']); bind(row['checkpoint'],row['checkpoint_sha256'])
    for path,row in inventory['frozen_decoders'].items(): bind(path,row['checkpoint_sha256'])
    for row in lineage['h1_cards']:
        for prefix in ('source_card','checkpoint','trajectory'): bind(row[prefix],row[prefix+'_sha256'])
    for path in (root/'dual_branch_controls').glob('**/card.json'):
        card = json.loads(path.read_text()); bind(card['replay_card'],card['replay_card_sha256'])
    failures = []
    def check(item):
        path,digest = item
        return path if sha(path) != digest else None
    with ThreadPoolExecutor(max_workers=4) as pool:
        failures = [path for path in pool.map(check,expected.items()) if path is not None]
    if failures: raise ValueError(f'changed evidence: {failures}')
    figures = reports/'figures'; pngs = sorted(figures.glob('*.png')); readme = (figures/'README.md').read_text()
    for png in pngs:
        metadata = json.loads(png.with_suffix('.metadata.json').read_text())
        if metadata['summary_sha256'] != sha(summary_path): raise ValueError('figure/summary state mismatch')
        if metadata['producer_sha256'] != sha(metadata['producer_path']): raise ValueError('figure producer changed')
        if not png.with_suffix('.pdf').is_file() or f'### {png.name}' not in readme: raise ValueError('figure delivery incomplete')
    if len(pngs) != 6: raise ValueError('expected six reviewed scientific figures')
    return {'status':'PASS','summary_status':summary['status'],'summary_sha256':sha(summary_path),
            'checked_file_count':len(expected),'checked_hashes':expected,'replay_counts':counts,'queues':queue_results,
            'figures':[{str(p):sha(p) for p in (png,png.with_suffix('.pdf'),png.with_suffix('.metadata.json'))} for png in pngs],
            'elapsed_seconds':time.time()-start,
            'interpretation':'artifact integrity and registered execution only; not evidence of biological establishment'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reports',type=Path)
    args = parser.parse_args(); result = audit(args.root, args.reports); args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','summary_status','checked_file_count','elapsed_seconds')}))
