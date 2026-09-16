#!/usr/bin/env python3
"""Fixed-recipe H2a budget sensitivity with frozen upstream checkpoints."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_group_event_state_v038_dual_credit import file_hash
from src.topic5_group_event_state.v037.contracts import atomic_json
from src.topic5_group_event_state.v037.h2a import H2ATrainConfig, train_h2a_subject


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-card', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--maximum-static-epochs', type=int, choices=(80, 240), default=80)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    old = json.loads(args.source_card.read_text())
    hp = Path(old['state_provenance']['checkpoint'])
    dp = Path(old['decoder_provenance']['checkpoint'])
    paths = [args.source_card, hp, hp.with_name('card.json'), hp.with_name('trajectory_and_targets.npz'), dp]
    hashes = {str(path): file_hash(path) for path in paths}
    cfg = H2ATrainConfig(**old['config'])
    if args.preflight_only:
        cfg = replace(cfg, max_epochs_static=2, max_epochs_state=2, max_epochs_oracle=2)
    else:
        cfg = replace(cfg, max_epochs_state=360, max_epochs_static=args.maximum_static_epochs)
    # Reproduce the original cold start, not a resumed checkpoint with lost
    # Adam moments. Decoder construction resets its registered RNG seed.
    result = train_h2a_subject(old['subject'], old['state_seed'], device=torch.device(args.device),
                               out_dir=args.output.parent, config=cfg,
                               h1_root=hp.parents[2], decoder_root=dp.parents[4],
                               state_family=old['state_provenance']['family'])
    prefix_parity = {}
    if not args.preflight_only:
        for name in ('static', 'state', 'B_mark'):
            reference = {int(row['epoch']): row for row in old['stages'][name]['history']}
            differences = []
            for row in result['stages'][name]['history']:
                if int(row['epoch']) in reference:
                    differences.append(abs(float(row['inner_grammar']) - float(reference[int(row['epoch'])]['inner_grammar'])))
            prefix_parity[name] = max(differences, default=0.0)
        # A different optimization trajectory is reported as a sensitivity,
        # never silently described as an exact continuation of the old fit.
    for path, digest in hashes.items():
        if file_hash(Path(path)) != digest:
            raise ValueError(f'frozen source changed: {path}')
    result.update(status='COMPLETE', repair_source_card=str(args.source_card),
                  repair_source_sha256=hashes[str(args.source_card)], repair_input_hashes=hashes,
                  repair_type='fixed_recipe_cold_start_budget_sensitivity',
                  repair_original_epoch_budget=old['config']['max_epochs_state'],
                  repair_epoch_budget=cfg.max_epochs_state, repair_preflight_only=args.preflight_only,
                  repair_static_epoch_budget=cfg.max_epochs_static,
                  repair_original_inner_history_max_abs_difference=prefix_parity,
                  repair_upstream_checkpoint_updated=False,
                  evaluation_replay_bundle_sha256=file_hash(Path(result['evaluation_replay_bundle'])))
    atomic_json(args.output, result)
    print(json.dumps({k: result[k] for k in ('status', 'subject', 'state_seed', 'repair_epoch_budget')}), flush=True)


if __name__ == '__main__':
    main()
