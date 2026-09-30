"""Classify perturbation growth at the actual60s sustained-A state.

This targets the transition neighborhood rather than extrapolating the
stronger-depletion endpoint's finite-time growth to another Z field.
"""
from common import OUT,np,read,write,log
import core_a_type_analysis as growth
import onset_state_continuation as flow
import core_a_resource_branch as local
import argparse

DEST=OUT/'core_a_bifurcation_type_20260924/actual_sustained_growth'
SOURCE=DEST.parent/'sustained_A_long_window/below_extend_to60s'
growth.DEST=DEST;flow.DEST=DEST;local.DEST=DEST


def register():
    assert read(SOURCE/'jobs.json')['status']=='COMPLETE'
    assert read(SOURCE/'local_state_audit.json')['status']=='AUDIT_PASS'
    assert read(SOURCE.parent/'joined60s_audit.json')['status']=='AUDIT_PASS'
    DEST.mkdir(exist_ok=True);assert not(DEST/'contract.json').exists()
    z=np.load(SOURCE/'final_state.npz')['syn'][5];np.savez_compressed(DEST/'fields.npz',actual_sustained=z)
    coordinates=read(SOURCE.parent/'contract.json')['coordinates']['below']
    cases={label:dict(label=label,field='actual_sustained',initial=str(SOURCE/'final_state.npz'),
        source_dt_ms=.05,dt_ms=dt,duration_ms=duration,previous_elapsed_ms=0,
        prior_same_field_elapsed_ms=60000,tangent=True)
        for label,dt,duration in [('coarse',.05,10000),('fine',.025,5000)]}
    write(DEST/'conditions.json',cases)
    write(DEST/'contract.json',dict(
        question='At the actualD_A=.34315764 sustained-Core-A state after60s, is full-state perturbation growth compatible with a stable periodic attractor, or does persistent expansion require considering an aperiodic invariant set?',
        source=str(SOURCE),coordinates={'actual_sustained':coordinates},conditions=cases,
        equations='Unchanged full3479-group physical-delay spatial rate model, same fixed native-Core-A resource pattern and bitwise identical outside-Core-A Z. All M dynamic. No external future innovations or fitted dynamics.',
        checks='Same actual nominal flow bitwise with/without passive tangent; centered full-state nonlinear finite differences for each step size. Fine source history preserves physical lag coordinates. Renormalize every100ms, discard first2s of tangent alignment, report all subsequent1s blocks and cumulative growth.',
        readout='Finite-time full-state directional growth and unchanged local quiet/activity metrics. Signs at both meshes are a numerical attractor-class diagnostic; not an asymptotic chaos proof or a crisis certificate.',
        scope='Positive growth alone cannot name the transition. Neither periodic root closure nor negative finite-time growth alone proves asymptotic stability. This diagnostic tests whether searching for a stable period is appropriate at the actual sustained state.',
        budget='One10s coarse continuation and one5s fine control from the same original60s full state, not independent replicates.',model_promoted=False))
    log('ACTUAL SUSTAINED GROWTH REGISTERED',coordinates)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','check','run','audit'])
    p.add_argument('--label',choices=['coarse','fine']);p.add_argument('--device',type=int,default=1);a=p.parse_args()
    {'register':register,'check':lambda:growth.check(a.label,a.device),
        'run':lambda:growth.run(a.label,a.device),'audit':lambda:local.audit(a.label)}[a.command]()
