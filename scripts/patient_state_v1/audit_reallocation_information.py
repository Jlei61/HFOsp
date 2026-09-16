"""Quantify how much pre15 information is fixed by each reallocation condition."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scripts.patient_state_v1.common import RUN, write_json

OUT = RUN/'preseizure_mark_reallocation_v1_38'

def main():
    windows = pd.read_csv(OUT/'frozen_pre15_windows.csv'); rows = []
    for condition in ['global_counts', 'interval_counts', 'local_counts']:
        strata = json.loads((OUT/f'{condition}_strata.json').read_text())
        for i, window in enumerate(windows.itertuples()):
            contributing = [s for s in strata if s['window_counts'][i] > 0]
            total = sum(s['n_events'] for s in contributing); expected = 0.; variance = 0.
            for s in contributing:
                n, k, draw = s['n_events'], s['n_tb'], s['window_counts'][i]
                p = k/n; expected += draw*p
                if n > 1: variance += draw*p*(1-p)*(n-draw)/(n-1)
            rows.append(dict(condition=condition, sz=window.sz, label=window.label, n_pre15=window.n_events, n_events_in_contributing_strata=total, pre15_fraction_of_contributing_events=window.n_events/total if total else None, expected_tb_count=expected, conditional_tb_count_sd=float(np.sqrt(variance)), count_is_fixed_under_null=variance == 0))
    table = pd.DataFrame(rows); table.to_csv(OUT/'conditional_information.csv', index=False)
    nonempty = table[(table.condition == 'interval_counts') & (table.n_pre15 > 0)]
    dominated = nonempty[nonempty.pre15_fraction_of_contributing_events >= .9]
    write_json(OUT/'conditioning_scope_audit.json', dict(status='COMPLETE', interval_windows_at_least90percent_pre15=dominated.to_dict('records'), warning='Keeping interval counts includes the tested pre15 labels. When pre15 is most/all of an interval, the conditional null largely or completely fixes its mark composition. Therefore a large replication fraction cannot itself establish an earlier slow background or reject a genuine pre15 drift.', earlier_only_control='../preseizure_past_baseline_v1_39/earlier_only_baselines.csv', interpretation='This audit limits the causal/temporal interpretation of the reallocation result; it does not invalidate the exact conditional count calculation.'))
    print(nonempty[['sz', 'label', 'n_pre15', 'n_events_in_contributing_strata', 'pre15_fraction_of_contributing_events', 'conditional_tb_count_sd']].to_string(index=False))

if __name__ == '__main__':
    main()
