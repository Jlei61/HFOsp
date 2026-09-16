#!/usr/bin/env python3
"""Corrected publication check for the synthetic old-summary window.

``v040/synthetic.py:old_summary_oracle`` reports
``old_summary_published_at_query_fraction`` from packets in ``[q-360, q)``
rather than from the registered old-summary window ``[t-360, t-240)``.  Because
a minute packet publishes with its closed one-hour block, the packets nearest
the query are always unpublished, so that field is systematically wrong and is
not usable.  This script recomputes it on the registered window.

It is deliberately named outside the ``*group_event_state_v040*.py`` glob that
feeds the frozen source digest, so it can be added while fits are in flight.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.topic5_group_event_state.v040 import data as D
from src.topic5_group_event_state.v040.synthetic import OLD_WINDOW_MINUTES

ROOT = Path('/data/hfosp_group_event_state_epilepsy_state_v040')


def check(payload, split, role, horizons=(1, 5, 30, 120)):
    pk = payload['packets']
    table = D.target_table(payload, split, role, 30)
    rows = []
    for h in horizons:
        sel = table[table[:, 1] == h]
        if not len(sel):
            continue
        published, defined = [], []
        for target, _, q in sel:
            lo = pk['end'][target] - OLD_WINDOW_MINUTES[0] * 60.
            hi = pk['end'][target] - OLD_WINDOW_MINUTES[1] * 60.
            win = np.flatnonzero((pk['start'] >= lo - 1e-6) & (pk['end'] <= hi + 1e-6))
            defined.append(bool(len(win)))
            published.append(bool(len(win) and pk['release'][win].max() <= pk['end'][q] + 1e-6))
        rows.append(dict(horizon=int(h), n_targets=int(len(sel)),
                         window_exists_fraction=float(np.mean(defined)),
                         fully_published_at_query_fraction=float(np.mean(published))))
    return rows


if __name__ == '__main__':
    out = {}
    for pkt in sorted((ROOT / 'synthetic' / 'packets').glob('synthetic_*.pt')):
        if pkt.name.endswith('.witness.pt'):
            continue
        payload = torch.load(pkt, map_location='cpu', weights_only=False)
        split = D.build_split(payload, payload['subject'], 20260906, 'inner0', 'S-E')
        cut = split['support_start'] + .5 * (split['support_end'] - split['support_start'])
        pk = payload['packets']
        outer = dict(split)
        outer['outer_packet'] = ((pk['start'] >= cut) & (pk['end'] <= split['support_end'])
                                 & split['valid_packet'])
        out[payload['subject']] = dict(
            registered_window_minutes_before_target=list(OLD_WINDOW_MINUTES),
            inner_validation=check(payload, split, 'inner'),
            untouched_future_test=check(payload, outer, 'outer'))
    result = dict(
        status='COMPLETE', scope='corrected secondary diagnostic only; no score is recomputed',
        supersedes=('old_summary_oracle.rows[].old_summary_published_at_query_fraction, which reads '
                    '[q-360, q) instead of the registered [t-360, t-240) window and is therefore invalid'),
        worlds=out)
    path = ROOT / 'contracts' / 'synthetic_publication_check.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, indent=2, ensure_ascii=False)[:2500])
