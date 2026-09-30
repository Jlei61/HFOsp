"""Bidirectional full-state continuation along Core A's native resource path.

Only Core A Z is varied; outside-Core-A Z stays exactly at the original
native9s reference. All M and all spatial fast/delayed states remain dynamic.
Finite-time local activity classes are not bifurcation certificates.
"""
from common import OUT, np, model, read, write, log
from pathlib import Path
from datetime import datetime
import argparse
import core_a_resource_branch as local
import onset_state_continuation as continuation

OLD=local.DEST
DEST=OUT/'core_a_transition_continuation_20260924'
local.DEST=DEST
continuation.DEST=DEST


def register(name,lower,upper,duration_ms=10000):
    assert duration_ms in [10000,20000,30000]
    DEST.mkdir(exist_ok=True)
    lower,upper=Path(lower),Path(upper)
    sources=[lower,upper]
    s=model(40);A=s.E&(s.geo['group_region']==0)
    reference=np.load(OLD/'fields.npz')['reference9000']
    evidence=[];ds=[]
    for source in sources:
        assert read(source/'jobs.json')['status']=='COMPLETE'
        a=read(source/'local_state_audit.json');assert a['status']=='AUDIT_PASS'
        z=np.load(source/'final_state.npz')['syn'][5]
        assert np.array_equal(z[~A],reference[~A])
        da=float(1-np.average(z[A],weights=s.sizes[A]));ds.append(da)
        evidence.append(dict(source=str(source),D_A=da,local_readout=a))
    assert 0<=ds[0]<ds[1]<=1
    target=.5*sum(ds)
    native=np.load(OUT/'transient_native_Z_path_20260923/native_Z_path.npz')
    t=native['time_ms'];d=1-native['Z'][:,A]@(s.sizes[A]/s.sizes[A].sum())
    crossings=np.flatnonzero((t[:-1]>=9000)&(t[1:]<=10370)&(d[:-1]<=target)&(d[1:]>target))
    assert len(crossings)>0
    j=int(crossings[0]);alpha=float((target-d[j])/(d[j+1]-d[j]))
    z=reference.copy();z[A]=(1-alpha)*native['Z'][j,A]+alpha*native['Z'][j+1,A]
    assert np.array_equal(z[~A],reference[~A]) and 0<=z.min()<=z.max()<=1
    assert abs(1-np.average(z[A],weights=s.sizes[A])-target)<1e-14
    coords=dict(D_A=target,Z_A=1-target,D_global=float(1-z[s.E]@s.mean_weights),
                native_time_ms=float(t[j]+alpha*(t[j+1]-t[j])))
    path=DEST/f'{name}_contract.json';assert not path.exists()
    fields=dict(np.load(DEST/'fields.npz')) if (DEST/'fields.npz').exists() else {}
    conditions=read(DEST/'conditions.json') if (DEST/'conditions.json').exists() else {}
    contract=read(DEST/'contract.json') if (DEST/'contract.json').exists() else dict(
        question='Where does Core A cease to return to low activity when only its spatial Z field varies, and does the same field retain distinct full-history outcomes?',
        equations='Unchanged full g40 spatial conditional-drift rate model, original mean input, no future count innovations, physical private Q and locked response, dynamic M everywhere.',
        control='Outside-Core-A Z bitwise native9s reference. Core A follows original native spatial patterns, interpolated only between adjacent5ms native path samples. Entire Z held within each run.',
        readouts='Identical prior regional10ms smoothing, quiet<5Hz, local episodes bounded by20ms quiet,50Hz duty, full-field recurrence and separate original global onset.',
        interpretation='Finite10s history dependence or local duty changes alone are not bistability, an attractor boundary or bifurcation. Full-state stability and step checks must resolve branch changes versus continuous waveform deformation.',
        coordinates={},model_promoted=False)
    assert name not in fields
    fields[name]=z;contract['coordinates'][name]=coords
    new=[]
    for side,source in zip(['lower','upper'],sources):
        label=f'{name}_from_{side}';assert label not in conditions
        c=dict(label=label,field=name,initial=str(source/'final_state.npz'),previous_elapsed_ms=0,duration_ms=duration_ms)
        conditions[label]=c;new.append(c)
    temp=DEST/'fields.next.npz';np.savez_compressed(temp,**fields);temp.replace(DEST/'fields.npz')
    write(DEST/'conditions.json',conditions);write(DEST/'contract.json',contract)
    write(path,dict(created_local=datetime.now().astimezone().isoformat(),coordinates=coords,
        conditions=new,source_evidence=evidence,
        interpolation=dict(native_interval_ms=t[j:j+2].tolist(),fraction=alpha,number_upcrossings=len(crossings)),
        budget=f'Two{duration_ms/1000:g}s full-state continuations. Audit before selecting another field or longer history. No physical parameter refit.',
        initial_M='Carried separately from each full source history; dynamic in both, not fixed or reset.',
        numerical_dt_ms=.05,bifurcation_type='NOT_ESTABLISHED'))
    log('CORE A MIDPOINT REGISTERED',name,coords)


def extend(name,long=False):
    original=read(DEST/f'{name}_{"extension_" if long else ""}contract.json')
    path=DEST/f'{name}_{"long_" if long else ""}extension_contract.json';assert not path.exists()
    conditions=read(DEST/'conditions.json');new=[]
    for source in original['conditions']:
        folder=DEST/source['label'];a=read(folder/'local_state_audit.json')
        assert a['status']=='AUDIT_PASS' and read(folder/'jobs.json')['status']=='COMPLETE'
        assert np.array_equal(np.load(folder/'final_state.npz')['syn'][5],np.load(DEST/'fields.npz')[source['field']])
        label=source['label']+('_long' if long else '_ext');assert label not in conditions
        prior=source.get('prior_same_field_elapsed_ms',0)+source['duration_ms']
        c=dict(label=label,field=source['field'],initial=str(folder/'final_state.npz'),
               previous_elapsed_ms=0,duration_ms=15000 if long else 5000,prior_same_field_elapsed_ms=prior,
               exact_same_Z_continuation=True)
        conditions[label]=c;new.append(c)
    write(path,dict(created_local=datetime.now().astimezone().isoformat(),conditions=new,
        reason=('At15s one history still sustains high CoreA activity, while the other returned to self-limited activity. Test longer persistence versus recurrent switching before attributing distinct attractors.' if long else 'The initial10s contain late sustained segments preceded by self-limited activity. Extend exact histories to15s before attributing an attractor or narrowing from censored activity.'),
        budget=('Two15s same-Z continuations to total30s, allMdynamic, then audit. No state reset. Local elapsed0–15s corresponds to total15–30s.' if long else 'Two5s uninterrupted same-equation, same-Z continuations, allMdynamic. No state reset; local elapsed0–5s corresponds to total10–15s.'),
        bifurcation_type='NOT_ESTABLISHED'))
    write(DEST/'conditions.json',conditions)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','extend','check','run','audit'])
    p.add_argument('--name');p.add_argument('--lower');p.add_argument('--upper')
    p.add_argument('--label');p.add_argument('--device',type=int,default=0)
    p.add_argument('--long',action='store_true')
    p.add_argument('--duration-ms',type=int,default=10000)
    a=p.parse_args()
    if a.command=='register':register(a.name,a.lower,a.upper,a.duration_ms)
    elif a.command=='extend':extend(a.name,a.long)
    elif a.command=='check':continuation.check(a.device)
    elif a.command=='run':continuation.run(a.label,a.device)
    elif a.command=='audit':local.audit(a.label)
