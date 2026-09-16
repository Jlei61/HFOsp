#!/usr/bin/env python3
"""Why did every conditional fine-identity head select step 0?

Distinguishes "the state adds nothing" from "this readout family cannot use any
input".  The positive control is the explicit recent history itself: the
per-contact participation logit over the readable window is, by construction, a
direct estimate of who participates next.  If that cannot beat the intercept on
the same conditional set score, the readout is the limiting factor, not the state.

Named outside the ``*group_event_state_v040*.py`` digest glob so it can run while
fits are in flight. Read-only: it loads a frozen producer and fits nothing that
is written back.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.topic5_group_event_state.v040 import consumers as C
from src.topic5_group_event_state.v0312.frozen import conditional_set_lp, identity_units

ROOT = Path('/data/hfosp_group_event_state_epilepsy_state_v040')


def score(logits, members, community):
    u = identity_units(members, community)
    return float((-conditional_set_lp(logits, members, community)[u]).sum() / u.sum()), int(u.sum())


def main(selected, device='cpu'):
    model, prep, cfg, record = C.load_selected(selected, device)
    community = torch.as_tensor(prep.payload['shaft_index'], device=prep.device)
    role = 'outer'
    train = C.adapter_data(model, prep, cfg, 'fit', 1)
    test = C.adapter_data(model, prep, cfg, role, 1)
    unique = np.unique(train['packet'])
    cut = unique[max(0, int(.8 * len(unique)) - 1)]
    tr = C.subset_adapter(train, train['packet'] <= cut)
    va = C.subset_adapter(train, train['packet'] > cut)
    out = dict(selected=str(selected), split=dict(
        train_events=int(len(tr['identity'])), validation_events=int(len(va['identity'])),
        test_events=int(len(test['identity'])),
        train_units=int(identity_units(tr['identity'], community).sum()),
        validation_units=int(identity_units(va['identity'], community).sum()),
        test_units=int(identity_units(test['identity'], community).sum())))

    # Reference points on the TEST split, with no fitting at all.
    pr = (tr['identity'].sum(0) + .5) / (len(tr['identity']) + 1)
    intercept = torch.logit(pr).expand(len(test['identity']), -1)
    s_int, n = score(intercept, test['identity'], community)
    s_hist, _ = score(test['history'], test['identity'], community)
    s_hist_c, _ = score(test['history'] + torch.logit(pr), test['identity'], community)
    out['no_fit_reference'] = dict(
        n_units=n, intercept_only=s_int,
        readable_recent_history_as_logits=s_hist,
        history_plus_intercept=s_hist_c,
        note=('lower is better; the recent-history logit is a direct empirical estimate of who '
              'participates, so it is the positive control for this readout family'))

    # Validation trajectory of the registered head, C+H+S.
    features = C.FEATURES['C+H+S']
    def get(d):
        return torch.cat([d[k] for k in features] + [d['cond']], -1)
    x, xv, xt = get(tr), get(va), get(test)
    center = x.mean(0)
    scale = x.std(0, unbiased=False).clamp(min=.1)
    xs, xvs, xts = (x - center) / scale, (xv - center) / scale, (xt - center) / scale
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.seed)
        net = nn.Linear(x.shape[1], tr['identity'].shape[1]).to(x.device)
    with torch.no_grad():
        net.weight.zero_()
        net.bias.copy_(torch.logit(pr))

    def loss(d, xx):
        u = identity_units(d['identity'], community)
        return -conditional_set_lp(net(xx), d['identity'], community)[u].sum() / u.sum().clamp(min=1)

    opt = torch.optim.AdamW(net.parameters(), lr=.01, weight_decay=1e-3)
    trace = []
    with torch.no_grad():
        trace.append(dict(step=0, train=float(loss(tr, xs)), validation=float(loss(va, xvs)),
                          test=float(loss(test, xts)), weight_norm=0.))
    for step in range(1, 401):
        opt.zero_grad()
        v = loss(tr, xs)
        v.backward()
        opt.step()
        if step % 20 == 0:
            with torch.no_grad():
                trace.append(dict(step=step, train=float(loss(tr, xs)),
                                  validation=float(loss(va, xvs)), test=float(loss(test, xts)),
                                  weight_norm=float(net.weight.norm())))
    out['C+H+S_trajectory'] = trace
    best = min(trace, key=lambda r: (r['validation'], r['step']))
    out['interpretation'] = dict(
        best_validation_step=best['step'],
        train_improved=trace[0]['train'] - min(r['train'] for r in trace),
        validation_improved=trace[0]['validation'] - min(r['validation'] for r in trace),
        test_at_best_validation=best['test'],
        test_at_step_zero=trace[0]['test'],
        test_at_step_400=trace[-1]['test'])
    path = ROOT / 'contracts' / 'identity_head_diagnosis.json'
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'C+H+S_trajectory'}, indent=2))
    print('trajectory (step, train, validation, test, |W|):')
    for r in out['C+H+S_trajectory'][:12]:
        print('  %4d  %.5f  %.5f  %.5f  %.4f' % (r['step'], r['train'], r['validation'], r['test'], r['weight_norm']))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else
         str(ROOT / 'runs/epilepsiae_1125__S_marks__S-E__outer__persistent__seed20260906/selected.pt'))
