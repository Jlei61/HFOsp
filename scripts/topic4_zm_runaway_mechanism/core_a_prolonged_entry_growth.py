"""Attractor-class diagnostic at the first verified prolonged-event field."""
from common import OUT,np,read,write,log
import core_a_type_analysis as growth
import onset_state_continuation as flow
import core_a_resource_branch as local
import argparse

BASE=OUT/'core_a_bifurcation_type_20260924/reference_stability_gap'
SOURCE=BASE/'actual_D0300/from_interictal_history'
DEST=BASE/'prolonged_entry_growth'
growth.DEST=DEST;flow.DEST=DEST;local.DEST=DEST


def register():
    DEST.mkdir(exist_ok=True);assert not (DEST/'contract.json').exists()
    audit=read(SOURCE/'whole_record_audit.json');assert audit['status']=='AUDIT_PASS'
    assert read(SOURCE/'jobs.json')['status']=='COMPLETE'
    z=np.load(SOURCE/'final_state.npz')['syn'][5];np.savez_compressed(DEST/'fields.npz',entry=z)
    conditions={label:dict(label=label,field='entry',initial=str(SOURCE/'final_state.npz'),
        source_dt_ms=.05,dt_ms=dt,duration_ms=duration,previous_elapsed_ms=0,
        prior_same_field_elapsed_ms=20000,tangent=True)
        for label,dt,duration in [('coarse',.05,10000),('fine',.025,5000)]}
    write(DEST/'conditions.json',conditions)
    write(DEST/'contract.json',dict(
        question='At the first directly verified seconds-long Core-A activity field D_A=.300, is the full delayed-state dynamics expanding, or is it compatible with a stable long-period attractor? This determines which invariant-set mechanism must be pursued at entry.',
        source=str(SOURCE),coordinates={'entry':audit['coordinates']},conditions=conditions,
        equations='Same full3479-group spatial rate flow, physical graph/delays/privatevariance and locked response. Each original spatialZ held, outsideA native9s, allE M dynamic. Original constant external mean; no future count innovations or parameter refitting.',
        controls='Both begin at the exact original20s complete state; fine history interpolated by physical lag. Passive tangent, condition-specific nominal/replay and centered full-state derivative checks. Coarse10s/fine5s are numerical controls, not independent samples.',
        readout='Full-state growth with fixed cell-weighted norm,100ms renormalization,first2s discarded,all1s blocks. Compare common2--5s window and original local-activity/spatial readout. Growth alone cannot establish asymptotic chaos or a crisis.',
        decision='If expansion is robust here and absent at still-short fields, prioritize loss of stable recurrent dynamics near entry; if already present before long activity, its earlier onset cannot alone explain the prolonged-event transition. Near-zero finite growth does not prove periodic stability.',
        budget='One10s coarse andone5s fine continuation only; root and critical crossing remain necessary before classification.',model_promoted=False))
    log('PROLONGED ENTRY GROWTH REGISTERED',audit['coordinates'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','check','run','audit'])
    p.add_argument('--label',choices=['coarse','fine']);p.add_argument('--device',type=int,default=1)
    a=p.parse_args();{'register':register,'check':lambda:growth.check(a.label,a.device),
        'run':lambda:growth.run(a.label,a.device),'audit':lambda:local.audit(a.label)}[a.command]()
