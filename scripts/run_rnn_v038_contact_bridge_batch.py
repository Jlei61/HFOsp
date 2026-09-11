#!/usr/bin/env python3
"""Bounded CPU subprocess pool with logs, source hashes and resume checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    root = args.root
    manifest = json.loads((root / 'execution_manifest.json').read_text())
    code = root / 'execution_code'
    code.mkdir(exist_ok=True)
    for name in ('audit_rnn_v038_contact_bridge.py', 'contact_bridge_metrics.py', Path(__file__).name):
        source = Path(__file__).with_name(name)
        dest = code / name
        if dest.exists() and sha(dest) != sha(source):
            raise ValueError('immutable execution code differs: ' + name)
        if not dest.exists():
            dest.write_bytes(source.read_bytes())
    (root / 'logs').mkdir(exist_ok=True)
    env = dict(os.environ, LD_LIBRARY_PATH='/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')

    def worker(job):
        if sha(job['source_card']) != job['source_sha256']:
            raise ValueError('source card changed: ' + job['id'])
        output = root / 'per_subject' / job['id']
        card = output / 'card.json'
        if card.exists():
            old = json.loads(card.read_text())
            if sha(output / 'predictions_and_scores.npz') != old['predictions_sha256']:
                raise ValueError('saved predictions changed')
            # Smoke may use the workspace path with exactly the same code bytes.
            for path, digest in old['code_hashes'].items():
                if sha(code / Path(path).name) != digest:
                    raise ValueError('cannot mix scoring code versions')
            return {'id': job['id'], 'returncode': 0, 'status': old['status'], 'resumed': True}
        cmd = [sys.executable, str(code / 'audit_rnn_v038_contact_bridge.py'), '--snapshot', str(root / 'source_snapshot'),
               '--source-card', job['source_card'], '--replay', job['replay'], '--output', str(output)]
        with (root / 'logs' / (job['id'] + '.log')).open('w') as log:
            result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        return {'id': job['id'], 'returncode': result.returncode,
                'status': json.loads(card.read_text())['status'] if card.exists() else 'FAILED'}

    results = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=manifest['workers']) as pool:
        for future in as_completed([pool.submit(worker, job) for job in manifest['jobs']]):
            result = future.result()
            results.append(result)
            status = {'total': len(manifest['jobs']), 'finished': len(results),
                      'failed': sum(x['returncode'] != 0 for x in results), 'elapsed_seconds': time.time() - started, 'jobs': results}
            tmp = root / 'batch_status.tmp'
            tmp.write_text(json.dumps(status, indent=2) + '\n')
            tmp.replace(root / 'batch_status.json')
            print(json.dumps(result), flush=True)
    if any(x['returncode'] != 0 for x in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
