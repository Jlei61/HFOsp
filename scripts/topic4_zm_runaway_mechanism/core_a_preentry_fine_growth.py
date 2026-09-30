"""Resolve whether pre-entry full-state expansion survives a finer time step."""
from common import OUT,np,read,write,log
import core_a_type_analysis as growth
import onset_state_continuation as flow
import core_a_resource_branch as local
import argparse

BASE=OUT/'core_a_bifurcation_type_20260924/reference_stability_gap'
SOURCE=BASE/'actual_native_entry_midpoint/from_interictal_history'
DEST=BASE/'preentry_fine_growth'
growth.DEST=DEST;flow.DEST=DEST;local.DEST=DEST


def register():
    DEST.mkdir(exist_ok=True);assert not (DEST/'contract.json').exists()
    audit=read(SOURCE/'whole_record_audit.json');assert audit['status']=='AUDIT_PASS'
    assert read(BASE/'preentry_growth_comparison/native_midpoint/tangent_result.json')['status']=='COMPLETE'
    initial=SOURCE/'checkpoint10000.npz';z=np.load(initial)['syn'][5]
    np.savez_compressed(DEST/'fields.npz',native_midpoint=z)
    conditions={'fine':dict(label='fine',field='native_midpoint',initial=str(initial),source_dt_ms=.05,
        dt_ms=.025,duration_ms=10000,previous_elapsed_ms=0,prior_same_field_elapsed_ms=10000,tangent=True)}
    write(DEST/'conditions.json',conditions)
    write(DEST/'contract.json',dict(source=str(SOURCE),coordinates={'native_midpoint':audit['coordinates']},
        question='Is expansion of the still-short-event state already robust before the seconds-long activity at Z_A=.700? The coarse pre-entry growth was positive, so numerical refinement is needed before separating loss of periodicity from long-event entry.',
        equations='Unchanged full3479-group physical-delay spatial rate field. Same actual CoreA native9507.453342988ms Z field, outsideA native9s, everyZ held and allE M dynamic; original constant external mean and locked response.',
        controls='Start from the exact same full10s checkpoint as the completed coarse growth run. Only dt is halved, with physical-lag-preserving history interpolation. Original full-state nominal/tangent and centered finite-difference checks required; no stochastic replicate claim.',
        readout='Compare matched2--10s growth and local activities; retain all1s blocks and independently audit the complete rate record. Also report2--5s to expose within-record sign changes. Finite-time expansion does not certify asymptotic chaos, a crisis, or a bifurcation.',
        budget='One10s fine control, matching the completed coarse run. No extension or new parameter sweep in this batch.',model_promoted=False))
    log('PREENTRY FINE GROWTH REGISTERED')


def audit():
    import core_a_local_return_neighborhood as whole
    whole.DEST=DEST;whole.LABEL='fine';whole.flow.DEST=DEST;whole.local.DEST=DEST
    whole.audit()
    coarse=BASE/'preentry_growth_comparison/native_midpoint'
    conditions=read(BASE/'preentry_growth_comparison/conditions.json')
    assert conditions['native_midpoint']['initial']==read(DEST/'conditions.json')['fine']['initial']
    rows=[]
    for name,path in [('coarse',coarse),('fine',DEST/'fine')]:
        assert read(path/'implementation_check.json')['status']=='PASS'
        g=read(path/'tangent_result.json');values=np.array(g['log_growth'])
        assert len(values)==100 and g['duration_ms']==10000
        rows.append(dict(label=name,dt_ms=g['dt_ms'],growth_2_5_s=float(values[20:50].sum()/3),
            growth_2_10_s=float(values[20:].sum()/8),one_second_blocks_per_s=g['one_second_blocks_per_s']))
    write(DEST/'comparison.json',dict(status='MATCHED_DURATION_NUMERICAL_CONTROL_COMPLETE',rows=rows,
        full_fine_activity_audit=str(DEST/'fine/whole_record_audit.json'),
        interpretation='Same physical10s initial checkpoint and matched10s duration. These are numerical controls, not independent realizations. Positive finite-time growth, especially with phase-dependent block variation, does not establish an asymptotic attractor or name the prolonged-event bifurcation.',model_promoted=False))
    log('PREENTRY MATCHED GROWTH',rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','check','run','audit'])
    p.add_argument('--device',type=int,default=0);a=p.parse_args()
    {'register':register,'check':lambda:growth.check('fine',a.device),
     'run':lambda:growth.run('fine',a.device),'audit':audit}[a.command]()
