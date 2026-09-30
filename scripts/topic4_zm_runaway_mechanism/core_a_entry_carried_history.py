"""Distinguish a field contrast from dependence on the transplanted history."""
from common import OUT,np,read,write,log
import onset_state_continuation as flow
import core_a_resource_branch as local
from datetime import datetime
import argparse

BASE=OUT/'core_a_bifurcation_type_20260924/reference_stability_gap'
LOWER=BASE/'actual_native_entry_midpoint/from_interictal_history'
UPPER=BASE/'actual_D0300/from_interictal_history'
DEST=BASE/'actual_entry_carried_history'
flow.DEST=DEST;local.DEST=DEST


def register():
    DEST.mkdir(exist_ok=True);assert not (DEST/'contract.json').exists()
    for p in [LOWER,UPPER]:assert read(p/'whole_record_audit.json')['status']=='AUDIT_PASS'
    source=LOWER/'final_state.npz';a=np.load(source);z=np.load(UPPER/'final_state.npz')['syn'][5]
    from common import model
    s=model(40);A=s.E&(s.geo['group_region']==0)
    assert np.array_equal(a['syn'][5,~A],z[~A])
    np.savez_compressed(DEST/'fields.npz',entry=z)
    conditions={'from_nearby_short_history':dict(label='from_nearby_short_history',field='entry',initial=str(source),
        source_dt_ms=.05,dt_ms=.05,duration_ms=10000,previous_elapsed_ms=0)}
    write(DEST/'conditions.json',conditions)
    write(DEST/'contract.json',dict(created_local=datetime.now().astimezone().isoformat(),
        question='Does the first long-activity field Z_A=.700 also recruit prolonged activity when it inherits a nearby short-event history atZ_A=.712351, instead of the common original9s interictal history?',
        conditions=conditions,coordinates={'entry':read(UPPER/'whole_record_audit.json')['coordinates']},
        intervention='Only native within-Core-A Z changes from the original9507.453ms field to9594.907ms field. Preserve the entire20s short-field fast/M/delay state. OutsideCoreA Zbitwise unchanged. AllZheld thereafter; allMdynamic. Same originalconstantmeaninput, no futurecountinnovations.',
        baseline=str(UPPER),
        readout='Same CoreA5Hz/20ms quiet and local/global/spatial measures. The baseline startsfromcommonoriginalhistory, not the same fast state; this explicitly tests history sensitivity at the SAME final field.',
        decision='Prolonged activity from both histories weakens a claim that only the initial large resource transplant explains the contrast. Continuedshortactivity fromnearbyhistory motivates coexistence/basin testing, but finite10s alone cannot prove bistability or branch survival. Neither outcome identifies the bifurcation.',
        budget='One10s originalstep continuation; source/10ms replay and independent readout. No additionalparameterpoint, refit or automaticrampcampaign.',model_promoted=False))
    log('CARRIED ENTRY HISTORY REGISTERED')


def audit():
    import core_a_local_return_neighborhood as whole
    whole.DEST=DEST;whole.LABEL='from_nearby_short_history'
    whole.flow.DEST=DEST;whole.local.DEST=DEST
    whole.audit()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','check','run','audit'])
    p.add_argument('--device',type=int,default=0);a=p.parse_args()
    {'register':register,'check':lambda:flow.check(a.device),
     'run':lambda:flow.run('from_nearby_short_history',a.device),
     'audit':audit}[a.command]()
