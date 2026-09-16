#!/usr/bin/env python3
"""Attribute dual temporal gains using frozen branch-specific controls."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.audit_group_event_state_v038_dual_credit import file_hash
from scripts.audit_group_event_state_v038_dual_random_background import _parent_parity_audit
from src.topic5_group_event_state.v037.contracts import atomic_json
from src.topic5_group_event_state.v037.h1_dual_train import NestedDualReadout
from src.topic5_group_event_state.v037.h1_train import _selection_score, _block_shift


def main():
    p = argparse.ArgumentParser(); p.add_argument('--replay-card', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--device', default='cuda:0'); args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    started = time.time(); replay = json.loads(args.replay_card.read_text())
    if replay['family'] != 'dual': raise ValueError('dual only')
    bundle_path = Path(replay['replay_bundle'])
    if file_hash(bundle_path) != replay['replay_bundle_sha256']: raise ValueError('replay bundle changed')
    bundle = torch.load(bundle_path, map_location='cpu', weights_only=False)
    source_card = Path(replay['source_card']); original = json.loads(source_card.read_text())
    for path, digest in replay['input_hashes'].items():
        if file_hash(Path(path)) != digest: raise ValueError('original source changed')
    data = bundle['data']; saved = bundle['source_checkpoint']; cfg = saved['config']; tr = bundle['trajectory']
    device = torch.device(args.device); tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    q = bundle['q'].to(device); bm = tensor(tr['fixed_mark_state']); bc = tensor(tr['background_current'])
    es = tensor(tr['event_state']); bs = tensor(tr['background_state']); exposure = tensor(tr['target_exposure_seconds'])
    target = {k: v.to(device) for k, v in bundle['target'].items()}; valid = {k: v.to(device) for k, v in bundle['valid'].items()}
    fit = torch.as_tensor(np.flatnonzero(data.rate.phase == 'FIT'), device=device)
    sel_np = np.flatnonzero(data.rate.phase == 'SELECTION'); selection = torch.as_tensor(sel_np, device=device)
    horizons = tuple(original['horizons_seconds']); bd = len(cfg['taus_seconds']) * data.burden_mark.shape[1]
    ed = len(cfg['taus_seconds']) * cfg['burden_channels_per_tau']
    model = NestedDualReadout(q.shape[1], bd, bm.shape[1]-bd, bc.shape[1], bs.shape[1], ed, es.shape[1]-ed, saved['widths'], len(horizons)).to(device)
    model.load_state_dict(saved['readout']); model.eval()
    def predict(event, background):
        return model.predict(q, bmark=bm, background_current=bc, event_state=event, background_state=background)
    def score(prediction, mask=valid):
        return _selection_score(prediction, target, mask, exposure, model.log_dispersion, selection, sel_np, horizons)
    with torch.no_grad():
        correct = predict(es, bs); full = score(correct)
        try: parity = _parent_parity_audit(full, original['selection_scores']['S_dual'], horizons)
        except ValueError as error: parity = {'all_endpoint_total_auditable': False, 'reason': str(error)}
        fixed = {'event': score(predict(es[fit].mean(0,keepdim=True).expand_as(es),bs)),
                 'background': score(predict(es,bs[fit].mean(0,keepdim=True).expand_as(bs)))}
        no_event = score(predict(None,bs)); by_horizon = {}
        for h, seconds in enumerate(horizons):
            key = str(int(seconds)); cell = {}
            for branch, values in [('event', es), ('background',bs)]:
                shifted, eligible = _block_shift(values,data,sel_np,float(seconds))
                mask = {name: value & eligible[:,None] & (torch.arange(value.shape[1],device=device)[None] == h) for name,value in valid.items()}
                paired_correct = score(correct,mask)['by_horizon'][key]
                paired_shift = score(predict(shifted,bs) if branch=='event' else predict(es,shifted),mask)['by_horizon'][key]
                full_h = full['by_horizon'][key]; constant_h = fixed[branch]['by_horizon'][key]
                cell[branch] = {
                    'dynamic_over_branch_constant': None if full_h is None or constant_h is None else constant_h['total']-full_h['total'],
                    'correct_time_over_branch_shift': None if paired_correct is None or paired_shift is None else paired_shift['total']-paired_correct['total'],
                    'n_shift_eligible_anchors': int(eligible[selection].sum()),
                    'correct': full_h, 'branch_constant': constant_h,
                    'paired_correct': paired_correct, 'branch_shifted': paired_shift,
                }
            cell['event']['increment_over_persistent_background'] = None if full['by_horizon'][key] is None else no_event['by_horizon'][key]['total']-full['by_horizon'][key]['total']
            by_horizon[key] = cell
    result = {'status': 'COMPLETE', 'subject': original['subject'], 'seed': original['seed'],
              'family': 'dual', 'source_card': str(source_card), 'source_checkpoint_sha256': replay['input_hashes'][str(source_card.with_name('checkpoint.pt'))],
              'replay_card': str(args.replay_card), 'replay_card_sha256': file_hash(args.replay_card),
              'replay_bundle_sha256': replay['replay_bundle_sha256'], 'score_parity': parity,
              'by_horizon': by_horizon, 'model_weights_updated': False,
              'interpretation': 'one branch shifted or replaced by FIT mean; other branch and baseline inputs fixed; post-review frozen diagnostic',
              'development_targets_read': False, 'sealed_partition_opened': False, 'seizure_targets_read': False,
              'elapsed_seconds': time.time()-started}
    atomic_json(args.output,result); print(json.dumps({k:result[k] for k in ['status','subject','seed','elapsed_seconds']}),flush=True)


if __name__=='__main__': main()
