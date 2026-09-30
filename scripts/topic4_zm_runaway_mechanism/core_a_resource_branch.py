"""Core A resource intervention, retaining the complete spatial rate model.

Only Core A Z is changed; Core B/surround Z are held at native9s. Every M
variable and all fast spatial interactions remain dynamic. This first paired
screen locates a relevant local state contrast, not a bifurcation certificate.
"""
from common import OUT,np,model,read,write,log
import onset_state_continuation as continuation
from fine_rate_frozen_Z_fields import capture
from datetime import datetime
from scipy.ndimage import uniform_filter1d
import argparse

OLD=continuation.DEST
DEST=OUT/'core_a_resource_bifurcation_20260923'
continuation.DEST=DEST


def register():
    assert read(OLD/'native9000_from_lower/independent_audit.json')['status']=='AUDIT_PASS'
    DEST.mkdir(exist_ok=True);assert not (DEST/'contract.json').exists()
    s=model(40);A=s.E&(s.geo['group_region']==0)
    path=np.load(OUT/'transient_native_Z_path_20260923/native_Z_path.npz')
    def at(tm):
        j=np.flatnonzero(path['time_ms']==tm);assert len(j)==1
        return path['Z'][j[0]].copy()
    baseline=at(9000);late=baseline.copy();late[A]=at(10370)[A]
    assert np.array_equal(late[~A],baseline[~A]) and not np.array_equal(late[A],baseline[A])
    source=OLD/'native9000_from_lower/final_state.npz'
    assert np.max(abs(np.load(source)['syn'][5]-baseline))<1e-12
    fields={'reference9000':baseline,'coreA10370_background9000':late}
    # Use the original checkpoint's exact background values for bitwise
    # matched controls; the native path is independently checked above.
    exact=np.load(source)['syn'][5].copy();fields['reference9000']=exact
    fields['coreA10370_background9000'][~A]=exact[~A]
    np.savez_compressed(DEST/'fields.npz',**fields)
    conditions={label:dict(label=label,field=field,initial=str(source),previous_elapsed_ms=0,duration_ms=10000)
                for label,field in [('reference','reference9000'),('coreA_depleted','coreA10370_background9000')]}
    write(DEST/'conditions.json',conditions)
    coordinates={k:dict(D_A=float(1-np.average(z[A],weights=s.sizes[A])),
        D_global=float(1-z[s.E]@s.mean_weights),Z_A=float(np.average(z[A],weights=s.sizes[A]))) for k,z in fields.items()}
    write(DEST/'contract.json',dict(created_local=datetime.now().astimezone().isoformat(),
        question='Can Core A resource depletion alone change Core A from self-limited activity to sustained local activity while resources outside Core A remain at the same interictal level?',
        priority='User explicitly prioritizes the Core A resource bifurcation over global-mean-Z continuation. Registered global native_mid9420_q pair is deferred, not run.',
        network='Entire unchanged g40/3479group spatial model; original connections, physical delays and locked conditioned39+transient response. No isolated-core replacement or physical parameter refit.',
        controls='Only Core A E-cell-group Z changes. Core B, surround and I Z are bitwise identical in both conditions and held. M is dynamic everywhere. Same full fast/local/M/delay initial state, original constant external mean, no future count innovations, private Q retained.',
        field_family='Actual native Core A spatial Z patterns; all outside-Core-A values held at native9000ms. First screen uses Core A9000ms and10370ms. Native9s is a late-interictal background, not the earlier4.025s figure state; any later background sensitivity is separate.',
        primary='Core A mean E rate in Hz/neuron,10ms-smoothed local rate, local quiet<5Hz, complete local activity episodes bounded by20ms quiet, local50Hz duty. Same readouts for B and surround to identify spread; original global-onset readout remains separate.',
        axes='D_A=1-cell-count-weighted mean Z over Core A only; y=Core A E rate. A field varies internally as recorded, not replaced by a uniformscalar. D_global is audit metadata only.',
        coordinates=coordinates,conditions=conditions,
        budget='Two10s paired continuations, exact replay/non-Core-A field checks and independent local/global readout. No parameter campaign or bifurcation label from these two traces.',
        interpretation='A local state contrast justifies local-parameter continuation. If absent, report that this tested Core-A-only intervention is insufficient at this background; do not claim no local bifurcation anywhere. Regionally localized activity does not prove an isolated-core bifurcation.',
        correspondence='Current conditional drift remains only partially linked to native SNN. Earlier native two-core Z interventions support separating local persistence from global recruitment, but are not an A-only control.',model_promoted=False))
    log('CORE A RESOURCE REGISTERED',coordinates)


def check(device):
    continuation.check(device)
    e=continuation.build(device);states=[];zs=[]
    for c in read(DEST/'conditions.json').values():
        zs.append(continuation.initialize(e,c));states.append(capture(e))
    A=e.s.E&(e.s.geo['group_region']==0)
    assert np.array_equal(zs[0][~A],zs[1][~A])
    for key in states[0]:
        if key=='syn':assert np.array_equal(states[0][key][:5],states[1][key][:5])
        else:assert np.array_equal(states[0][key],states[1][key]),key
    result=read(DEST/'implementation_check.json')
    result.update(only_core_A_Z_changed=True,non_core_A_Z_bitwise=True,paired_full_initial_state_bitwise_except_core_A_Z=True)
    write(DEST/'implementation_check.json',result);log('CORE A PAIRED IMPLEMENTATION PASS')


def local_episodes(rate):
    sm=uniform_filter1d(rate,10,mode='nearest');on=sm>=5
    dif=np.diff(np.r_[False,on,False].astype(int));starts=np.flatnonzero(dif==1);ends=np.flatnonzero(dif==-1)
    complete=[(a,b) for a,b in zip(starts,ends) if a>=20 and b+20<=len(sm) and (sm[a-20:a]<5).all() and (sm[b:b+20]<5).all()]
    return dict(quiet_fraction=float((sm<5).mean()),duty_above50=float((sm>50).mean()),
        range_10ms_hz=[float(sm.min()),float(sm.max())],complete_local_episodes=len(complete),
        episode_durations_ms=[int(b-a) for a,b in complete])


def audit(label):
    import audit_onset_state_continuation as independent
    independent.DEST=DEST;independent.audit(label)
    folder=DEST/label;jobs=read(folder/'jobs.json');assert jobs['status']=='COMPLETE'
    s=model(40);W=continuation.regional_weights(s);rates=[];times=[]
    for b in jobs['completed_blocks']:
        z=np.load(folder/f'block{b:02d}.npz');rates.append(z['group_rate_hz'].astype(float)@W.T)
        times.append(z['elapsed_time_ms'])
    rates=np.concatenate(rates);times=np.concatenate(times);assert len(times)>=5000
    rows=[]
    for j,name in enumerate(['Global E','Core A','Core B','Surround']):
        late=rates[-5000:,j];rows.append(dict(region=name,mean_rate_hz=float(late.mean()),**local_episodes(late)))
    field=read(DEST/'conditions.json')[label]['field'];coordinates=read(DEST/'contract.json')['coordinates'][field]
    write(folder/'local_state_audit.json',dict(status='AUDIT_PASS',label=label,coordinates=coordinates,window_ms=[float(times[-5000]-1),float(times[-1])],rows=rows,
        definition='Local activity episodes apply the explicitly declared5Hz/20ms quiet rule to each regional10ms-smoothed rate; these are not reused global whole-field event counts.',
        scope='Paired finite-time local state evidence only; no certified equilibrium, periodic stability, bistability or bifurcation.',model_promoted=False))
    log('CORE A LOCAL AUDIT',label,coordinates,rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','check','run','audit'])
    p.add_argument('--label',choices=['reference','coreA_depleted']);p.add_argument('--device',type=int,default=0)
    a=p.parse_args();{'register':register,'check':lambda:check(a.device),'run':lambda:continuation.run(a.label,a.device),'audit':lambda:audit(a.label)}[a.command]()
