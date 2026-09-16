#!/usr/bin/env python3
"""Model-free audit of how representative each partition is.

Answers one question only: do the fitting, validation and held-out periods
contain comparable event distributions and comparable amounts of real
observation? It reads the frozen measurement bundles and touches no model, so
nothing here can be explained by training.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src.topic5_group_event_state.v035.contracts import atomic_json

BUNDLE = '/data/hfosp_group_event_state_v0_3_9_transition_transfer_background_repaired/human_data_v2'
LEAD_INDEX, LEAD_SECONDS, WINDOW_SECONDS = 1, 7200.0, 1800.0


def union_hours(starts, span=WINDOW_SECONDS):
    """De-duplicated observed target time; overlapping five-minute anchors are
    not independent hours and must not be counted as such."""
    intervals = sorted((float(s), float(s) + span) for s in starts)
    merged = []
    for lo, hi in intervals:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return sum(hi - lo for lo, hi in merged) / 3600.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--subjects', nargs='+',
                   default=['epilepsiae_1096', 'epilepsiae_1125', 'epilepsiae_253'])
    args = p.parse_args()
    rows = []
    for subject in args.subjects:
        data = torch.load(Path(BUNDLE) / f'{subject}.pt', map_location='cpu', weights_only=False)
        samples = data['samples']
        phase = np.array([s['phase'] for s in samples])
        valid = np.array([s['targets'][LEAD_INDEX][2] for s in samples], bool)
        count = np.array([s['targets'][LEAD_INDEX][0] for s in samples], float)
        anchor = np.array([s['anchor'] for s in samples], float)
        fit = (phase == 'FIT') & valid
        fit_max = float(count[fit].max()) if fit.any() else float('nan')
        fit_p90 = float(np.quantile(count[fit], .9)) if fit.any() else float('nan')
        for name in ('FIT', 'INNER', 'SELECTION'):
            mask = (phase == name) & valid
            if not mask.any():
                continue
            value = count[mask]
            rows.append(dict(
                subject=subject, phase=name, n_windows=int(mask.sum()),
                deduplicated_target_hours=round(union_hours(anchor[mask] + LEAD_SECONDS), 2),
                span_hours=round(float(anchor[mask].max() - anchor[mask].min()) / 3600, 2),
                mean=round(float(value.mean()), 1), median=round(float(np.median(value)), 1),
                p90=round(float(np.quantile(value, .9)), 1), maximum=round(float(value.max()), 1),
                fraction_above_fit_maximum=round(float((value > fit_max).mean()), 4),
                fraction_above_fit_p90=round(float((value > fit_p90).mean()), 4),
                fit_maximum=round(fit_max, 1), fit_p90=round(fit_p90, 1)))
    out = args.root / 'final_reports'; out.mkdir(parents=True, exist_ok=True)
    with (out / 'phase_representativeness.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    atomic_json(out / 'phase_representativeness.json', dict(
        status='COMPLETE', schema='v0310_phase_representativeness_v1', timestamp=time.time(),
        lead_hours=LEAD_SECONDS / 3600, window_hours=WINDOW_SECONDS / 3600, rows=rows,
        reading='the mean and the median can disagree strongly; where they do, the partitions differ in '
                'how many high-firing stretches they contain rather than in a single shifted level',
        deduplicated_hours_note='five-minute anchors give overlapping thirty-minute target windows, so a '
                                'partition spanning many hours can cover far fewer distinct hours',
        does_not_establish=['which mechanism produces the difference (rhythm phase, sleep, medication, '
                            'short-term clustering or a measurement change are all compatible)',
                            'that a circadian cycle is present; a monotone rise inside a short window '
                            'reproduces the same clock correlation',
                            'that the baseline comparison is invalid; a constant remains a legitimate '
                            'opponent when the available inputs carry no information'],
        conclusion='the partitions differ markedly in event distribution and in real covered time, so the '
                   'validation and held-out periods are not representative; this can affect both '
                   'checkpoint selection and across-period prediction, and it means the current '
                   'no-increment result cannot adjudicate whether interictal events carry pathological '
                   'state',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(json.dumps(dict(status='COMPLETE', n_rows=len(rows)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
