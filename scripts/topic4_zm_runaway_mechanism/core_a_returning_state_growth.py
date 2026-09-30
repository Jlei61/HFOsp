"""Test the actual returning state before assuming a stable interictal cycle."""
from common import OUT,np,read,write,log
import core_a_type_analysis as growth
import onset_state_continuation as flow
import core_a_resource_branch as local
import argparse

DEST=OUT/'core_a_bifurcation_type_20260924/actual_returning_growth'
SOURCE=OUT/'core_a_transition_continuation_20260924/mid1_from_lower_ext_long'
growth.DEST=DEST;flow.DEST=DEST;local.DEST=DEST


def register():
    DEST.mkdir(exist_ok=True);assert not(DEST/'contract.json').exists()
    audit=read(SOURCE/'local_state_audit.json');assert audit['status']=='AUDIT_PASS'
    assert read(SOURCE/'jobs.json')['status']=='COMPLETE'
    z=np.load(SOURCE/'final_state.npz')['syn'][5];np.savez_compressed(DEST/'fields.npz',returning=z)
    cases={label:dict(label=label,field='returning',initial=str(SOURCE/'final_state.npz'),
        source_dt_ms=.05,dt_ms=dt,duration_ms=duration,previous_elapsed_ms=0,
        prior_same_field_elapsed_ms=30000,tangent=True)
        for label,dt,duration in [('coarse',.05,10000),('fine',.025,5000)]}
    write(DEST/'conditions.json',cases)
    write(DEST/'contract.json',dict(
        question='At the actual D_A=.328121599 returning state after30s, is the complete dynamical flow compatible with a stable periodic attractor? Compare its full-state expansion with the more depleted long-event state without presupposing a Hopf or stable-cycle transition.',
        source=str(SOURCE),coordinates={'returning':audit['coordinates']},conditions=cases,
        equations='Identical full3479-group spatial physical-delay rate field, original constant mean input, fixed native within-A Z pattern and native9s Z outside A. All M dynamic. No future count innovations.',
        checks='Condition-specific nominal bitwise/passive tangent and centered full-state finite differences. Physical-history-preserving step refinement. Fixed full-state norm,100ms renormalization,first2s alignment discarded,all1s growth blocks reported.',
        scope='Finite-time attractor-class diagnostic. Positive growth alone does not establish asymptotic chaos, exclude every stable attractor, or classify a crisis. Neither duration nor perturbation statistic substitutes for a critical invariant-set analysis.',
        budget='One10s coarse continuation and one5s fine control; same actual initial state, not independent replicates.',model_promoted=False))
    log('RETURNING STATE GROWTH REGISTERED',audit['coordinates'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','check','run','audit'])
    p.add_argument('--label',choices=['coarse','fine']);p.add_argument('--device',type=int,default=1);a=p.parse_args()
    {'register':register,'check':lambda:growth.check(a.label,a.device),
        'run':lambda:growth.run(a.label,a.device),'audit':lambda:local.audit(a.label)}[a.command]()
