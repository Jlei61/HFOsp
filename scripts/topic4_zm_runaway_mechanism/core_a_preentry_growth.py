"""Bounded full-state growth comparison before prolonged Core-A activity.

Both conditions still have only short events in the audited 20 s record.
Expansion here would separate irregularity from the later long-event entry;
it would not by itself establish chaos or a bifurcation.
"""
from common import OUT,np,read,write,log
import core_a_type_analysis as growth
import onset_state_continuation as flow
import core_a_resource_branch as local
import argparse

BASE=OUT/'core_a_bifurcation_type_20260924/reference_stability_gap'
DEST=BASE/'preentry_growth_comparison'
growth.DEST=DEST;flow.DEST=DEST;local.DEST=DEST


def register():
    DEST.mkdir(exist_ok=True);assert not (DEST/'contract.json').exists()
    fields={};conditions={};coordinates={}
    for label,parent in [('DA0270','actual_D0270'),('native_midpoint','actual_native_entry_midpoint')]:
        source=BASE/parent/'from_interictal_history'
        audit=read(source/'whole_record_audit.json');assert audit['status']=='AUDIT_PASS'
        row=next(r for r in audit['thresholds'] if r['threshold_hz']==5.)
        complete=[r['duration_ms'] for r in row['activities'] if not r['left_censored'] and not r['right_censored']]
        assert max(complete)<200
        initial=source/'checkpoint10000.npz';z=np.load(initial)['syn'][5].copy()
        fields[label]=z;coordinates[label]=audit['coordinates']
        conditions[label]=dict(label=label,field=label,initial=str(initial),source_dt_ms=.05,dt_ms=.05,
            duration_ms=10000,previous_elapsed_ms=0,prior_same_field_elapsed_ms=10000,tangent=True)
    np.savez_compressed(DEST/'fields.npz',**fields);write(DEST/'conditions.json',conditions)
    write(DEST/'contract.json',dict(
        question='Is full-state expansion already present in the still-short-event state at native9507.45ms, compared with the near-nine-burst D_A=.270 record? This distinguishes loss of simple periodicity from the subsequent onset of seconds-long Core-A activity.',
        coordinates=coordinates,conditions=conditions,
        equations='Unchanged full3479-group spatial drift, graph, physical delays, private variance and locked response. Only original Core-A Z fields differ; outsideA native9s, eachZ held, everyE M dynamic. Original constant external mean, no future innovations.',
        controls='Resume the exact existing full10s checkpoint at each field; integrate the same10--20s exposure with a passive tangent. These are deterministic conditions, not independent replicates. Exact nominal/passive replay and centered full-state derivative checks precede each run.',
        readout='One complete delayed-state direction, fixed cell-weighted norm,100ms renormalization,first2s discarded,all1s blocks. Compare with the already audited nominal event record; rates/M histories are saved by the same producer.',
        budget='Two10s coarse-step growth runs only. A material positive difference warrants a separate time-step control before interpretation.',
        interpretation='Finite-time growth cannot alone certify asymptotic chaos, stable periodicity, or a bifurcation. If expansion precedes prolonged events, do not equate a first periodic instability with the requested onset threshold.',model_promoted=False))
    log('PREENTRY GROWTH REGISTERED',coordinates)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','check','run','audit'])
    p.add_argument('--label',choices=['DA0270','native_midpoint']);p.add_argument('--device',type=int,default=1)
    a=p.parse_args();{'register':register,'check':lambda:growth.check(a.label,a.device),
        'run':lambda:growth.run(a.label,a.device),'audit':lambda:local.audit(a.label)}[a.command]()
