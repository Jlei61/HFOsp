"""Local-resource classification diagnostics on the unchanged spatial equations.

The tangent equations are passive: they do not alter the nominal trajectory.
A finite-time exponent is not a bifurcation certificate.
"""
from common import OUT, np, read, write, log, model
import onset_state_continuation as flow
import core_a_resource_branch as local
from fine_rate_frozen_Z_fields import capture, restore
from onset_tangent_cuda import Tangent
from onset_variational_return import Coordinates
from datetime import datetime
import argparse

DEST=OUT/'core_a_bifurcation_type_20260924'
SOURCE=OUT/'core_a_resource_bifurcation_20260923'
flow.DEST=DEST
local.DEST=DEST


def register():
    DEST.mkdir(exist_ok=True)
    assert not (DEST/'conditions.json').exists()
    source=SOURCE/'coreA_depleted'
    assert read(source/'local_state_audit.json')['status']=='AUDIT_PASS'
    field='coreA10370_background9000'
    z=np.load(SOURCE/'fields.npz')[field]
    assert np.array_equal(np.load(source/'final_state.npz')['syn'][5],z)
    np.savez_compressed(DEST/'fields.npz',**{field:z})
    conditions={}
    for label,dt,duration in [('depleted_coarse',.05,20000),('depleted_fine',.025,10000)]:
        conditions[label]=dict(label=label,field=field,initial=str(source/'final_state.npz'),
            source_dt_ms=.05,dt_ms=dt,duration_ms=duration,previous_elapsed_ms=0,
            prior_same_field_elapsed_ms=10000,tangent=True)
    write(DEST/'conditions.json',conditions)
    write(DEST/'contract.json',dict(created_local=datetime.now().astimezone().isoformat(),
        question='Does the stronger Core A depletion endpoint remain locally active after long continuation and time-step refinement, and is its full-network invariant dynamics compatible with a stable equilibrium or periodic orbit?',
        equations='Unchanged 3479-group spatial rate model, physical private variance and delays, original constant external mean, locked transient response. Entire Z fixed, all M dynamic. Only Core A Z differs from the native9s reference field.',
        coordinates={field:read(SOURCE/'contract.json')['coordinates'][field]},
        continuation='Same exact complete state after10s exposure. Coarse adds20s; fine adds10s with physical-lag-preserving factor-two history interpolation. These are numerical controls, not independent replicates.',
        tangent='Derivative of the actual complete delayed rate map, validated against centered nonlinear full-state differences separately at each step. Passive tangent renormalized every100ms in fixed cell-weighted coordinates; report finite-time growth after2s alignment and consecutive1s blocks. No chaos or crisis label solely from a positive finite-time exponent.',
        classification='Use fixed-point residual/eigenvalue crossing for SN or Hopf; full periodic closure and Floquet crossing for cycle bifurcations; require invariant-set/basin evidence for crisis or global bifurcations. If only a rate threshold changes continuously, do not invent a bifurcation.',
        model_promoted=False))


def check(label,device):
    c=read(DEST/'conditions.json')[label];folder=DEST/label
    folder.mkdir(exist_ok=True);assert not (folder/'implementation_check.json').exists()
    e=flow.build(device,dt=c['dt_ms']);Z=flow.initialize(e,c);base=capture(e)
    nominal=e.chunk();terminal=capture(e)
    restore(e,base);repeat=e.chunk();assert np.array_equal(nominal,repeat)
    assert all(np.array_equal(v,capture(e)[k]) for k,v in terminal.items())
    restore(e,base);t=Tangent(e);t.graph();t.chunk();actual=capture(e)
    assert all(np.array_equal(v,actual[k]) for k,v in terminal.items())
    rng=np.random.default_rng(92401)
    d={k:rng.uniform(-1,1,base[k].shape)*base[k] for k in ['syn','local','history']}
    d['syn'][5]=0;d['syn'][4,~e.s.E]=0
    clock=int(base['clock'][0]);h=base['history'];slack=np.ones(e.s.P)
    for mask,ref in [(e.s.E,2.),(~e.s.E,1.)]:
        used=h[(clock+1-np.arange(1,round(ref/e.dt)))%len(h)][:,mask].sum(0)*e.dt
        slack[mask]=np.clip(1-used,0,1)
    d['history']*=slack
    restore(e,base);t.reset()
    t.syn[:]=e.cp.asarray(d['syn'][:5]);t.local[:]=e.cp.asarray(d['local']);t.history[:]=e.cp.asarray(d['history'])
    e.cp.cuda.get_current_stream().synchronize();t.chunk()
    derivative=dict(syn=t.syn.get(),local=t.local.get(),history=t.history.get());rows=[]
    for eps in [5e-5,2.5e-5,1.25e-5]:
        states=[]
        for sign in [-1,1]:
            p=dict(base)
            for k in d:p[k]=base[k]+sign*eps*d[k]
            restore(e,p);e.chunk();states.append(capture(e))
        errors={}
        for k,v in derivative.items():
            fd=(states[1][k]-states[0][k])/(2*eps)
            if k=='syn':fd=fd[:5]
            errors[k]=float(np.linalg.norm(fd-v)/max(np.linalg.norm(v),1e-12))
        rows.append(dict(epsilon=eps,relative_errors=errors));log('CORE A TANGENT CHECK',label,rows[-1])
    assert max(rows[-1]['relative_errors'].values())<1e-4
    write(folder/'implementation_check.json',dict(status='PASS',dt_ms=e.dt,
        exact_nominal_replay=True,passive_tangent_nominal_bitwise=True,full_state_FD=rows,
        Z_held=True,M_dynamic=True,source=str(c['initial'])))
    # Global run gate is complemented by each condition-specific gate below.
    write(DEST/'implementation_check.json',dict(status='PASS',condition_specific_checks_required=True))


def run(label,device):
    folder=DEST/label;c=read(DEST/'conditions.json')[label]
    assert read(folder/'implementation_check.json')['status']=='PASS'
    original_initialize=flow.initialize
    growth=[];holder={}
    def initialize(e,condition):
        Z=original_initialize(e,condition);base=capture(e);t=Tangent(e);t.graph()
        coords=Coordinates(base,e.s);rng=np.random.default_rng(92402)
        v=rng.normal(size=coords.size);v.reshape(-1,e.s.P)[4,~e.s.E]=0
        v/=np.linalg.norm(v);coords.set_tangent(t,v)
        weights=e.cp.asarray(coords.weight**2/coords.scale**2)
        def norm():
            # Circular history permutation leaves these row-invariant history
            # scales unchanged (Coordinates history floors/RMS can differ).
            tick=int(e.local.clock.get()[0]);idx=(tick-e.cp.arange(coords.depth))%coords.depth
            value=e.cp.sum(t.syn*t.syn*weights[:5])+e.cp.sum(t.local*t.local*weights[5:47])
            value+=e.cp.sum(t.history[idx]**2*weights[47:])
            return float(e.cp.sqrt(value).get())
        n=norm();assert abs(n-1)<1e-12
        calls=0
        def chunk():
            nonlocal calls
            t.chunk();calls+=1
            if calls%10==0:
                n=norm();assert np.isfinite(n) and n>0
                growth.append(float(np.log(n)))
                for a in [t.syn,t.local,t.history]:a/=n
                e.cp.cuda.get_current_stream().synchronize()
                if calls%100==0:
                    values=np.array(growth);usable=values[20:]
                    row=dict(elapsed_ms=calls*10,finite_time_growth_per_s=float(usable.sum()/(len(usable)*.1)) if len(usable) else None,
                        last_1s_growth_per_s=float(values[-10:].sum()),log_growth=values.tolist())
                    write(folder/'tangent_progress.json',row);log('CORE A TANGENT GROWTH',label,{k:v for k,v in row.items() if k!='log_growth'})
            return e.output.get()
        e.chunk=chunk;holder.update(t=t,coords=coords,base=base)
        return Z
    flow.initialize=initialize
    try:flow.run(label,device)
    finally:flow.initialize=original_initialize
    values=np.array(growth);usable=values[20:]
    write(folder/'tangent_result.json',dict(status='COMPLETE',dt_ms=c['dt_ms'],
        alignment_discard_ms=2000,duration_ms=c['duration_ms'],renormalization_ms=100,
        finite_time_growth_per_s=float(usable.sum()/(len(usable)*.1)),
        one_second_blocks_per_s=values.reshape(-1,10).sum(1).tolist(),
        log_growth=values.tolist(),scope='Finite-time full delayed-state directional growth. Not an asymptotic Lyapunov or bifurcation certificate.',model_promoted=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','check','run','audit']);p.add_argument('--label');p.add_argument('--device',type=int,default=0)
    a=p.parse_args()
    {'register':register,'check':lambda:check(a.label,a.device),'run':lambda:run(a.label,a.device),'audit':lambda:local.audit(a.label)}[a.command]()
