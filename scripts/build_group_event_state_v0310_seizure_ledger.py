#!/usr/bin/env python3
"""CPU-only rare-seizure support ledger. No state and no outcome score here."""
from __future__ import annotations
import argparse, csv, hashlib, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from src.topic5_group_event_state.v0310 import seizure as S
from src.topic5_group_event_state.v035.contracts import atomic_json

BUNDLE = '/data/hfosp_group_event_state_v0_3_9_transition_transfer_background_repaired/human_data_v2'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--subjects', nargs='+', default=['epilepsiae_1096', 'epilepsiae_1125', 'epilepsiae_253'])
    p.add_argument('--history-hours', type=float, default=2.0)
    args = p.parse_args()
    out = args.root / 'seizure'; out.mkdir(parents=True, exist_ok=True)
    rows, summaries = [], {}
    for subject in args.subjects:
        subject_rows, summary, grid = S.build_ledger(subject, BUNDLE, args.history_hours)
        rows += subject_rows; summaries[subject] = summary
        np.savez_compressed(out / f'{subject}_query_grid.npz',
                            times=grid['times'], eligible=grid['eligible'], coverage=grid['coverage'],
                            phase=grid['phase'], inside_support=grid['inside_support'],
                            seconds_since_previous_onset=grid['seconds_since_previous_onset'],
                            excluded_blocks=grid['excluded_blocks'],
                            exclusion_intervals=grid['exclusion_intervals'], support=grid['support'])
        print(json.dumps({k: summary[k] for k in ('subject', 'n_raw_seizures', 'n_design_seizures_before_80pct',
                                                  'n_clusters', 'n_eligible_first_onsets', 'eligible_by_phase',
                                                  's_a_status', 's_b_status', 's_c_status',
                                                  'n_eligible_queries', 'n_excluded_blocks')},
                         ensure_ascii=False), flush=True)
    header = list(rows[0]) if rows else []
    with (out / 'seizure_episode_ledger.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=header); writer.writeheader(); writer.writerows(rows)
    atomic_json(out / 'seizure_ledger_summary.json',
                dict(status='COMPLETE', schema='v0310_seizure_ledger_v1', timestamp=time.time(),
                     history_hours=args.history_hours, subjects=summaries, n_rows=len(rows),
                     ledger_csv=str(out / 'seizure_episode_ledger.csv'),
                     source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     module_sha256=hashlib.sha256((ROOT / 'src/topic5_group_event_state/v0310/seizure.py').read_bytes()).hexdigest(),
                     state_outcome_scores_computed=False,
                     upstream_selected_with_seizure_outcomes=False,
                     note='support inventory only; S-A/S-B/S-C scores are computed after the upstream '
                          'completion/closure manifest is frozen'))
    print(json.dumps(dict(status='COMPLETE', rows=len(rows))), flush=True)


if __name__ == '__main__':
    main()
