#!/usr/bin/env python3
"""One fresh, fingerprinted data build replays all three original H1 families."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import gc
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.audit_group_event_state_v038_checkpoint_replay import replay
from scripts.audit_group_event_state_v038_dual_credit import file_hash
from src.topic5_group_event_state.v037.h1_data import build_h1_subject_data
from src.topic5_group_event_state.v037.contracts import atomic_json


def main():
    p = argparse.ArgumentParser(); p.add_argument('--h1-root', type=Path, required=True)
    p.add_argument('--subject', required=True); p.add_argument('--seed', type=int, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--replay-root', type=Path, required=True)
    p.add_argument('--device', default='cuda:0'); args = p.parse_args(); start = time.time()
    if args.output.exists(): raise FileExistsError(args.output)
    source = args.h1_root / 'event' / args.subject / f'seed{args.seed}' / 'card.json'
    original = json.loads(source.read_text())
    data = build_h1_subject_data(args.subject, seed=args.seed, horizons_seconds=original['horizons_seconds'])
    results = {}
    for family in ('event', 'grid', 'dual'):
        src = args.h1_root / family / args.subject / f'seed{args.seed}' / 'card.json'
        out = args.replay_root / args.h1_root.name / family / args.subject / f'seed{args.seed}' / 'card.json'
        if out.exists(): raise FileExistsError(out)
        card = replay(src, out, torch.device(args.device), data_override=data, verify_preprocessing=True)
        results[family] = {'path': str(out), 'sha256': file_hash(out), 'full_endpoint_replay_qualified': card['full_endpoint_replay_qualified']}
        del card; gc.collect()
        if str(args.device).startswith('cuda'): torch.cuda.empty_cache()
    result = {'status': 'COMPLETE', 'subject': args.subject, 'seed': args.seed,
              'families': results, 'selected_input_and_dictionary_sha256': data.representation_provenance['rich_grammar']['selected_input_and_dictionary_sha256'],
              'model_weights_updated': False, 'seizure_targets_read': False,
              'development_targets_read': False, 'sealed_partition_opened': False,
              'elapsed_seconds': time.time() - start}
    atomic_json(args.output, result); print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
