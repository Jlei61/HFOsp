"""Existing-output diagnostic review, with all detections explicitly separate from primary."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
from scripts import analyze_topic4_core_connectivity_search as an

IDS=['endpoint__baseline','up4p5__baseline','near_upper__baseline',
     'up4p5__EE_core_to_out_scale_1.25','up4p5__EE_same_core_scale_1.25',
     'up4p5__EI_same_core_scale_0.75','up4p5__EE_kernel_perp_scale_1.5',
     'near_upper__EE_kernel_perp_scale_1.5']
OUT=Path('/data/hfosp/topic4_sef_hfo/core_connectivity_search_20260910/rapid_audit_20260911')

def main():
    plan=an.rt.read(an.run.OUT/'plan.json');lookup={c['id']:c for c in plan['candidates']}
    cases=[lookup[cid] for cid in IDS];units={};overlap=[]
    for c in cases:
        for seed in plan['seeds']:
            r,a,primary=an.load_unit(an.run.result_path('screen',c['id'],2511,seed),1500)
            ids=an.all_detected_ids(r,1500)
            for i in ids:
                e=r['events'][i];lo,hi=e['window_ms'];dt=float(a['contact_envelope_dt_ms'])
                time=(np.arange(round(lo/dt),round(hi/dt))+.5)*dt
                mass=np.maximum(a['contact_envelope'][round(lo/dt):round(hi/dt)],0)
                shared=np.zeros(len(time),bool)
                for j,other in enumerate(r['events']):
                    if j!=i:shared|=(time>=other['window_ms'][0])&(time<other['window_ms'][1])
                overlap.append(dict(candidate=c['id'],seed=seed,event=int(i),primary=int(i) in primary,
                    shared_window_mass_fraction=float(mass[shared].sum()/mass.sum()) if mass.sum() else None))
            units[(c['id'],seed)]=(r,a,ids)
    result=an.render(units,cases,plan['seeds'],an.patient_examples(),OUT,
        lambda c:c['id'],event_population='全部检测事件的开发诊断，非原primary评分集合')
    an.rt.write(OUT/'selected_review_provenance.json',dict(status='DEVELOPMENT_ONLY',candidates=IDS,
        selection='Three baselines plus five conditions showing both labels and improved rod participation in the paired all-detection screen; not independent validation or primary-score nomination.',
        event_selection='Nearest own run-mode feature mean among all post-burnin complete-window detections.',
        shared_window_mass=overlap,output=result,patient_population='Frozen patient FIT and Fig2C examples unchanged; model all-detection population differs from primary.'))
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
