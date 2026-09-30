"""Collect audited finite-state observations without assigning bifurcations."""
from common import np,read,write
from onset_state_continuation import DEST
from datetime import datetime
import csv


def collect():
    rows=[]
    for label,c in read(DEST/'conditions.json').items():
        p=DEST/label/'independent_audit.json'
        if not p.exists():continue
        a=read(p);assert a['status']=='AUDIT_PASS'
        tail=a['windows'][-1]
        ret=DEST/label/'period_return/result.json'
        period=read(ret) if ret.exists() else None
        rows.append(dict(label=label,field=c['field'],D=a['D'],mean_Z=1-a['D'],
            dt_ms=c.get('dt_ms',.05),observed_ms=a['observed_ms'],
            final_window_ms=str(tail['window_ms']),
            final_global_mean_hz=tail['mean_rates_global_A_B_surround'][0],
            final_core_A_mean_hz=tail['mean_rates_global_A_B_surround'][1],
            final_core_B_mean_hz=tail['mean_rates_global_A_B_surround'][2],
            final_surround_mean_hz=tail['mean_rates_global_A_B_surround'][3],
            final_complete_events=tail['complete_events']['n'],
            final_quiet_fraction=tail['quiet_fraction'],
            final_spatial_persistent_fraction=tail['persistent_spatial_fraction'],
            final_group_rate_relative_variation=tail['group_rate_relative_variation'],
            candidate_period_ms=period['interpolated_period_ms'] if period else None,
            candidate_full_state_return_error=period['interpolated_return']['combined_relative_rms'] if period else None,
            fixed_Z=True,dynamic_M=True,bifurcation_type='NOT_ESTABLISHED'))
    path=DEST/'audited_state_evidence.csv'
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(DEST/'audited_state_evidence.json',dict(updated_local=datetime.now().astimezone().isoformat(),rows=rows,
        scope='Finite-window observations of explicit conditional drift and separately labeled periodic seeds. No automatic stable/unstable branch, separatrix, chaos or bifurcation labels.',
        evidence_rule='Only completed independent_audit.json rows are collected; running prefixes are omitted.',
        model_promoted=False))
    print(path,len(rows))


if __name__=='__main__':collect()
