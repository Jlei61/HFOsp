"""Check the actual Fig.5 state3 Core-A field on the continuous native path."""
from common import OUT,np,read,write,log,model
from core_a_parameter_path_audit import NativeTimeFamily
import core_a_local_return_neighborhood as run
import argparse
from fine_rate_frozen_Z_fields import capture,restore

BASE=OUT/'core_a_bifurcation_type_20260924/reference_stability_gap'
DEST=BASE/'actual_native9420';LABEL='from_interictal_history'
run.DEST=DEST;run.LABEL=LABEL;run.flow.DEST=DEST;run.local.DEST=DEST


def check(device):
    """Retain exact outputs if first-use replay ever fails again."""
    e=run.flow.build(device);c=read(DEST/'conditions.json')[LABEL]
    z=run.flow.initialize(e,c);initial=capture(e)
    source=dict(np.load(c['initial']));A=e.s.E&(e.s.geo['group_region']==0)
    identity={k:bool(np.array_equal(v[:5],source[k][:5]) if k=='syn' else np.array_equal(v,source[k])) for k,v in initial.items()}
    assert all(identity.values()) and np.array_equal(z[~A],source['syn'][5,~A])
    # Crucially do not restore again before the first chunk: test the exact
    # initialize->first-use path that the production run will execute.
    first=e.chunk();terminal=capture(e);restore(e,initial)
    second=e.chunk();repeated=capture(e)
    np.savez_compressed(DEST/f'ordinary_replay_device{device}.npz',first=first,second=second)
    qa=dict(device=device,initial_identity_except_A_Z=identity,
        replay_bitwise=bool(np.array_equal(first,second)),
        max_output_difference=float(np.max(abs(first-second))),
        expected_emitted_bitwise=bool(np.array_equal(first[:,0],first[:,1]) and np.array_equal(second[:,0],second[:,1])),
        finite=bool(np.isfinite(first).all() and np.isfinite(second).all()),
        full_terminal_bitwise={k:bool(np.array_equal(v,repeated[k])) for k,v in terminal.items()},
        Z_held=bool(np.array_equal(repeated['syn'][5],z)),
        prior_failure=('The first ordinary device1 check failed without saving numerical differences. Its log and subsequent diagnostic files are retained; its cause is not established by a later pass.' if (DEST/'implementation_failure_diagnosis_device0.json').exists() else None))
    passed=qa['replay_bitwise'] and qa['expected_emitted_bitwise'] and qa['finite'] and qa['Z_held'] and all(qa['full_terminal_bitwise'].values())
    qa['status']='PASS' if passed else 'FAIL'
    write(DEST/f'ordinary_replay_device{device}.json',qa)
    assert passed,qa
    qa.update(only_core_A_Z_changed=True,full_initial_state_bitwise_except_core_A_Z=True,
              constant_mean_external=True,count_innovations_disabled=True,private_Q_retained=True,M_dynamic=True)
    write(DEST/'implementation_check.json',qa)
    log('NATIVE STATE3 ORDINARY FIRST-USE REPLAY PASS',device)


def register():
    for name in ['actual_D0270','actual_D0300']:
        assert read(BASE/name/LABEL/'whole_record_audit.json')['status']=='AUDIT_PASS'
    DEST.mkdir(exist_ok=True);assert not(DEST/'contract.json').exists()
    source=OUT/'core_a_resource_bifurcation_20260923/reference'
    s=model(40);family=NativeTimeFamily(s);z,D=family.field_at_time(9420.)
    base=dict(np.load(source/'final_state.npz'))
    assert np.array_equal(z[~family.A],base['syn'][5,~family.A])
    k=np.flatnonzero(family.time==9420);assert len(k)==1
    assert np.array_equal(z[family.A],family.fields[k[0],family.A])
    c=dict(label=LABEL,field=LABEL,initial=str(source/'final_state.npz'),source_dt_ms=.05,dt_ms=.05,
           duration_ms=20000,previous_elapsed_ms=0)
    np.savez_compressed(DEST/'fields.npz',**{LABEL:z});write(DEST/'conditions.json',{LABEL:c})
    write(DEST/'contract.json',dict(
        question='Does the native Fig.5 state3 Core-A resource field at9.420s already support prolonged local activity from the same interictal complete state?',
        rationale='AuditedD_A=.270/native9116.474ms yields only<=94ms complete activity in20s, whereasD_A=.300/native9594.907ms yields several complete seconds-long episodes. Native9420 lies between these actual path times and is the user-prioritized figure state. This is a direct relevant point, not an unrelated later equilibrium SN.',
        coordinates={LABEL:dict(D_A=D,Z_A=1-D,native_interpolation_time_ms=9420.)},conditions={LABEL:c},
        path='Continuous original native within-Core-A field indexed by native path time; at9420 take the actual recorded field exactly. MeanD_A is only the plotted coordinate and is not treated as a globally one-to-one parameter. Outside-Core-A Z remains bitwise native9s.',
        physical_scope='Same entire3479-group spatial rate drift, all Z held during each trajectory, all E M dynamic, original constant input mean/private variance. Same complete initial fast/M/delay state as theD_A=.270/.300 interventions. Native path time selects a frozen parameter field; it is not injected future firing or the continuation simulation clock.',
        primary='Whole20s complete/censored local activity durations under unchanged10ms/5Hz/20ms quiet rule, global and spatial readout,1/10/50Hz checks. Local activity is separate from global onset.',
        budget='One20s point at the explicitly requested state3 field, after the two earlier endpoints are independently audited. No automatic parameter grid.',
        acceptance='Exact10ms replay/paired complete-state identity except A Z before launch; independent all-group spatial/readout audit after run. No bifurcation or native global-onset label from a finite record.',model_promoted=False))
    log('NATIVE STATE3 CORE A FIELD REGISTERED',D,1-D)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','check','run','audit']);p.add_argument('--device',type=int,default=1)
    a=p.parse_args();{'register':register,'check':lambda:check(a.device),'run':lambda:run.flow.run(LABEL,a.device),'audit':run.audit}[a.command]()
