"""Reuse the frozen label-free three-summary evaluation on actual SNN events."""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import csv,json,sys,hashlib,warnings
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/topic4_sef_hfo/core_spatial_readout_v10_20260916'
SOURCE=ROOT/'.worktrees/topic4-continuous-core-state-r1'
sys.path[:0]=[str(SOURCE),str(SOURCE/'src/snn_engine')]
from scripts.report_topic4_label_free_dense import describe,metrics
from scripts.analyze_topic4_three_observable_bo import patient,DISPLAY
sys.path.insert(0,str(ROOT/'scripts/topic4_burst_regime'))
import metrics_v2
from scipy.ndimage import gaussian_filter1d
read=lambda p:json.loads(Path(p).read_text())
def safe(x):
    if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [safe(v) for v in x]
    if isinstance(x,np.ndarray):return safe(x.tolist())
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x
def write(name,x):(OUT/name).write_text(json.dumps(safe(x),indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def main():
    ev,names,parent=patient();target=describe(ev.fit,names);states=read(OUT/'reduced_states.json')
    rows=[];details={};events=[];coverage=[]
    for key in ('a','b','cd'):
        folder=OUT/'native'/key;result=read(folder/'result.json');ob=read(folder/'observation.json')
        z=np.load(folder/'trajectory.npz');assert names==z['contact_names'].tolist()
        assert np.array_equal(z['sheet_activity_counts'].sum((1,2)),z['six_group_counts_2ms'][:,:3].sum(1))
        ids=np.array([i for i in z['primary_event_indices'] if ob['events'][i]['window_ms'][0]>=2000 and ob['events'][i]['window_ms'][1]<=12000],int)
        allids=[i for i,e in enumerate(ob['events']) if e['window_ms'][0]>=2000 and e['window_ms'][1]<=12000]
        times=z['centroid_ms'][ids];q=describe(times,names)
        v=metrics(q,target,names);lab=z['event_mode'][ids]
        mode_counts={label:int((lab==value).sum()) for label,value in [('TA',1),('TB',0),('unassigned',-1)]}
        counts=z['six_group_counts_2ms'];sizes=np.bincount(z['region'],minlength=6);r=counts/np.asarray(sizes)/.002
        mean=r[1000:].mean(0);smooth=gaussian_filter1d(r,1.5,axis=0)
        dynamics=metrics_v2.run_metrics(folder/'trajectory.npz',result)
        for i,e in enumerate(ob['events']):
            selected=i in ids
            events.append(dict(native_key=key,event=i,window_start_ms=e['window_ms'][0],window_stop_ms=e['window_ms'][1],
                primary_eligible=e['primary_eligible'],analysis_selected=selected,prolonged=e['prolonged'],
                exclusion_reasons=';'.join(e['primary_exclusion_reasons']),centroid_support=int(np.isfinite(z['centroid_ms'][i]).sum()),
                label=int(z['event_mode'][i])))
        contacts=[n for i,n in enumerate(names) if q['mean_rank'][i] is not None and np.isfinite(q['mean_rank'][i])]
        pairs={s:[k for k,vv in q['pairs'].items() if vv['shaft']==s and vv['order_probability'] is not None] for s in ('SCL','ICL')}
        row=dict(native_key=key,J=result['g'],reduced_rows='c/d' if key=='cd' else key,native_state=' / '.join(dynamics[k]['label'] for k in ('coreAE','coreBE')),
            N_valid=len(ids),N_detected_in_window=len(allids),N_excluded_in_window=len(allids)-len(ids),
            rank_error=v[0],within_shaft_order_error=v[1],participation_error=v[3],both_shafts_fraction=v[4],
            TA_count=mode_counts['TA'],TB_count=mode_counts['TB'],unassigned_count=mode_counts['unassigned'],
            mean_A_hz=float(mean[0]),mean_B_hz=float(mean[1]),mean_surround_E_hz=float(mean[2]),
            A_CV=dynamics['coreAE']['cv'],B_CV=dynamics['coreBE']['cv'],
            median_A_IEI_ms=1000*dynamics['coreAE']['iei_median_s'],median_B_IEI_ms=1000*dynamics['coreBE']['iei_median_s'],
            native_window_s=[2,12],state_correspondence='QUALITATIVE_TWO_CORE_BURSTS_ONLY' if key=='a' else 'NOT_REPRODUCED_IN_THIS_COLD_START',
            source=str(folder/'trajectory.npz'))
        rows.append(row);coverage.append(dict(native_key=key,contacts_estimable=contacts,
            pairs_estimable={s:len(pairs[s]) for s in pairs},pair_joint_event_counts={k:v['n'] for k,v in q['pairs'].items()}))
        details[key]=dict(summary=q,valid_event_indices=ids.tolist(),all_event_indices_in_window=allids,
            fixed_snapshot_event_indices=ids[:2].tolist(),mode_counts=mode_counts,dynamics=dynamics,
            mean_six_rates_hz=mean.tolist(),q05_six_rates_3ms_smoothed_hz=np.quantile(smooth[1000:],.05,axis=0).tolist(),
            q95_six_rates_3ms_smoothed_hz=np.quantile(smooth[1000:],.95,axis=0).tolist())
    state_table=[]
    for s in states:
        proxy=next(r for r in rows if r['native_key']==s['native_key'])
        state_table.append(dict(state=s['label'],state_name=s['title'],J=s['J_exact'],
            closure_mean_A_hz=s['mean'][0],closure_mean_B_hz=s['mean'][1],
            spatial_state_validated=False,branch_conditioned_rank_error=None,branch_conditioned_order_error=None,
            branch_conditioned_participation_error=None,
            parameter_matched_native_key=s['native_key'],parameter_matched_native_N=proxy['N_valid'],
            parameter_matched_rank_error=proxy['rank_error'],parameter_matched_order_error=proxy['within_shaft_order_error'],
            parameter_matched_participation_error=proxy['participation_error'],
            mapping_status=proxy['state_correspondence']))
    write('native_summary.json',rows);write('native_details.json',details);write('state_correspondence.json',state_table)
    write('metric_provenance.json',dict(patient_fit_N=len(ev.fit),contact_names=names,display_order=DISPLAY,
        patient_fit_centroid_sha256=hashlib.sha256(np.ascontiguousarray(ev.fit).tobytes()).hexdigest(),
        evaluator=parent['sources']['evaluator'],patient_target=target,
        metric_source=str(SOURCE/'scripts/report_topic4_label_free_dense.py'),
        metric_source_sha256=hashlib.sha256((SOURCE/'scripts/report_topic4_label_free_dense.py').read_bytes()).hexdigest(),
        definitions=dict(rank='Absolute difference of each contact mean normalized event rank; mean within each shaft, then equal shaft weight',
            order='Absolute difference of P(contact j later than i | both participate), with exact ties weighted 0.5; all within-shaft pairs, then equal shaft weight',
            participation='Absolute difference of each contact participation probability; mean within each shaft, then equal shaft weight'),
        pooling='All eligible events pooled before summary, never average TA/TB summaries',
        selection='Frozen repaired observer; isolated 250-ms event windows fully contained in 2-12 s; overlap and prolonged exclusions unchanged',
        threshold_refit=False,statistical_unit='One topology 2511 and one noise 848101 per J, not independent events/networks',
        mode_labels='Frozen nearest-template diagnostic; observing both labels does not validate two recovered propagation distributions',coverage=coverage))
    for name,data in [('native_summary.csv',rows),('event_inventory.csv',events),('state_correspondence.csv',state_table)]:
        with (OUT/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(safe(data))
    print(json.dumps(safe(rows),indent=2),flush=True)

if __name__=='__main__':main()
