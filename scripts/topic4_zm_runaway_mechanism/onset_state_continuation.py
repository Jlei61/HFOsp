"""Follow only the self-limited to sustained-activity neighborhood.

The full spatial Z field is fixed per condition and M remains dynamic.
Continuation of transient states is not by itself a bifurcation certificate.
"""
from common import OUT, model, np, read, write, log
from physical_delay_count_rate import PhysicalDelayCountEngine, projections
from transient_response_network import install, LABEL
from fine_rate_frozen_Z_fields import restore, capture, native_field
from scipy.ndimage import uniform_filter1d
from datetime import datetime
import argparse, os, time

DEST=OUT/'onset_state_continuation_20260923'
OLD=OUT/'transient_autonomous_Z_probe_20260923'
PRESCRIBED=OUT/'transient_native_Z_path_20260923'/LABEL


def regional_weights(s):
    masks=[s.E]+[s.E&(s.geo['group_region']==i) for i in range(3)]
    return np.array([s.sizes*m/(s.sizes*m).sum() for m in masks])


def register():
    assert read(OLD/'independent_audit.json')['status']=='READOUT_AUDIT_PASS'
    DEST.mkdir(exist_ok=True);assert not (DEST/'contract.json').exists()
    s=model(40)
    fields={'D020':np.load(OUT/'transient_equal_D_fields_20260923/fields.npz')['native_shape'],
            'native9000':native_field(s,9000),'native9420':native_field(s,9420),
            'native9870':native_field(s,9870)}
    np.savez_compressed(DEST/'fields.npz',**fields)
    conditions=[dict(label='lower_endpoint',field='D020',initial=str(PRESCRIBED/'checkpoint9000.npz'),
                    previous_elapsed_ms=0,duration_ms=20000),
                dict(label='upper_endpoint',field='native9870',initial=str(OLD/'9870/final_state.npz'),
                    previous_elapsed_ms=5000,duration_ms=15000)]
    write(DEST/'contract.json',dict(created_local=datetime.now().astimezone().isoformat(),
        question='In the onset-relevant spatial Z neighborhood, do self-limited and spatially sustained activities persist under the same autonomous equations, and can their states seed bidirectional continuation?',
        user_priority='Only the actual interictal/self-limited to sustained recruitment transition. UnrelatedD0.02static roots and broad response refits suspended.',
        equations='Unchanged g40 graph, locked transient-corrected conditioned39 response, physical-delay private-Q conditional drift, original constant mean input, no future count innovations. M dynamic. Z entire field held.',
        scope='This is the explicitly declared deterministic conditional drift, not exact stochastic ensemble dynamics. Earlier local and autonomousZ correspondence failures remain; no native-onset attribution from existence alone.',
        path='Actual native spatial Z family; D is only its E-cell-weighted display coordinate. Lower endpoint is actual interpolatedD.20 field; upper is native9870ms fieldD.256344. No uniformZ substitution.',
        conditions=conditions,
        initialization='Lower starts from the original prescribed-Z rate9s full fast/M/history. Upper resumes its already-computed5s deterministic trajectory without resetting any fast/M/delay state. Thus both endpoints reach20s elapsed under fixedZ; old high5s is not rerun.',
        observations='Save1ms fullgroup/20x20E rates,10ms M and exact5s fullstate checkpoints. Original quiet<5Hz20ms, high200Hz200ms and spatialpersistent50Hz/duty90 definitions unchanged. Compare final consecutive5s windows and spatial recurrence before calling an attractor.',
        numerical='dt.05ms. Check exact10ms restored replay and equality of expected/emitted flux. Static or periodic certification requires its own residual/stability/time-step checks.',
        budget='Two endpoint continuations,35seconds of new simulated time. After independent endpoint readout, register only neighboring forward/backward continuations that directly resolve the onset transition. No automatic unrelated branch search.',
        outcomes='If both endpoints lack autonomous self-limited activity, do not invent an interictal limit cycle; inspect excitability/noise-dependent or basin dynamics in this neighborhood. If distinct persistent regimes occur, use them to locate the transition and test coexistence.',
        model_promoted=False))
    write(DEST/'conditions.json',{c['label']:c for c in conditions})


def build(device,dt=.05):
    e=PhysicalDelayCountEngine(dt=dt,seed=1,device=device,count_sampling=False,constant_input=True)
    install(e);e.graph();return e


def regrid_state(source,e,source_dt):
    """Preserve physical lag coordinates when refining a rate-history mesh."""
    state={k:v.copy() for k,v in source.items()}
    if source_dt==e.dt:
        assert state['history'].shape==e.local.history.shape
        return state
    assert source_dt==2*e.dt, 'Only the declared factor-two refinement is implemented'
    old_clock=int(state['clock'][0]);old=state['history']
    old_canonical=old[(old_clock-np.arange(len(old)))%len(old)]
    depth=e.local.history.shape[0]
    assert depth==2*len(old)-1
    canonical=np.empty((depth,e.s.P))
    canonical[::2]=old_canonical
    canonical[1::2]=.5*(old_canonical[:-1]+old_canonical[1:])
    new_clock=2*old_clock
    history=np.empty_like(canonical)
    history[(new_clock-np.arange(depth))%depth]=canonical
    assert np.array_equal(history[(new_clock-2*np.arange(len(old)))%depth],old_canonical)
    assert new_clock*e.dt==old_clock*source_dt
    state['history']=history;state['clock']=np.array([new_clock],dtype=state['clock'].dtype)
    return state


def initialize(e,c):
    source=regrid_state(np.load(c['initial']),e,c.get('source_dt_ms',.05))
    if c.get('M_initial_source'):
        other=np.load(c['M_initial_source'])
        assert np.array_equal(other['syn'][5],source['syn'][5]), 'M swap must use the identical Z field'
        source['syn'][4]=other['syn'][4]
    restore(e,source)
    Z=np.load(DEST/'fields.npz')[c['field']]
    e.syn[5]=e.cp.asarray(Z);e.transport.pars[19].fill(0)
    e.cp.cuda.get_current_stream().synchronize()
    assert np.array_equal(e.syn[4].get(),source['syn'][4])
    assert np.array_equal(e.local.history.get(),source['history'])
    assert np.array_equal(e.local.state.get(),source['local'])
    assert not e.noise and not e.transport.drive_on
    assert np.all(e.transport.pars[20].get()==1)
    return Z


def check(device):
    e=build(device);rows=[]
    for label,c in read(DEST/'conditions.json').items():
        Z=initialize(e,c);state=capture(e);a=e.chunk();terminal=capture(e)
        restore(e,state);b=e.chunk()
        assert np.array_equal(a,b) and np.array_equal(b[:,0],b[:,1])
        repeated=capture(e)
        assert all(np.array_equal(v,repeated[k]) for k,v in terminal.items())
        assert np.array_equal(e.syn[5].get(),Z)
        rows.append(dict(label=label,replayed10ms_bitwise=True,Z_held=True,M_dynamic=True,
                         full_initial_fast_M_history_preserved=True))
    write(DEST/'implementation_check.json',dict(status='PASS',rows=rows,
        constant_mean_external=True,count_innovations_disabled=True,private_Q_retained=True))


def block_summary(t,r,field,M,W,count):
    regional=r@W.T;sm=uniform_filter1d(regional[:,0],10,mode='nearest')
    w=count/count.sum();persistent=float(w[(field>50).mean(0)>=.9].sum())
    return dict(window_ms=[float(t[0]-1),float(t[-1])],mean_rates_global_A_B_surround=regional.mean(0).tolist(),
        global_range_hz=[float(regional[:,0].min()),float(regional[:,0].max())],quiet_fraction=float((sm<5).mean()),
        spatial_persistent_fraction=persistent,
        M_mean_global_A_B_surround=(M@W.T).mean(0).tolist(),
        M_start_end_weighted_abs_change=float(abs(M[-1]-M[0])@W[0]),
        group_rate_relative_variation=float(np.linalg.norm(r-r.mean(0))/max(np.linalg.norm(r),1.)))


def run(label,device):
    assert read(DEST/'implementation_check.json')['status']=='PASS'
    c=read(DEST/'conditions.json')[label];folder=DEST/label;folder.mkdir(exist_ok=True)
    assert not (folder/'jobs.json').exists()
    jobs=dict(status='RUNNING',pid=os.getpid(),condition=c,completed_blocks=[],new_elapsed_ms=0)
    write(folder/'jobs.json',jobs);e=build(device,dt=c.get('dt_ms',.05));Z=initialize(e,c)
    P,count=projections(e.s,e.coarse,e.parent)[20];W=regional_weights(e.s)
    initial_tick=int(e.local.clock.get()[0]);start=time.time();rows=[]
    assert c['duration_ms']%5000==0
    try:
        for block in range(c['duration_ms']//5000):
            R=[];M=[]
            for k in range(500):
                x=e.chunk();assert np.array_equal(x[:,0],x[:,1])
                assert np.isfinite(x).all() and x.min()>=0
                R.append(x[:,0].astype('f4'));M.append(e.syn[4].get())
                if (k+1)%100==0:
                    elapsed=block*5000+(k+1)*10;jobs.update(new_elapsed_ms=elapsed)
                    write(folder/'jobs.json',jobs)
                    log('ONSET STATE',label,elapsed,'ms',round(time.time()-start,1),'s')
            assert np.array_equal(e.syn[5].get(),Z)
            r=np.concatenate(R);m=np.array(M);field=(P@r.astype(float).T).T
            t=c['previous_elapsed_ms']+block*5000+np.arange(1,5001.)
            row=block_summary(t,r.astype(float),field,m,W,count);rows.append(row)
            np.savez_compressed(folder/f'block{block:02d}.npz',elapsed_time_ms=t,group_rate_hz=r,
                field_E_hz=field.astype('f4'),regional_rate_hz=r@W.T,cell_counts=count,
                state_time_ms=t[9::10],M_current=m,Z=Z)
            np.savez_compressed(folder/f'checkpoint{int(t[-1])}.npz',**capture(e))
            jobs['completed_blocks'].append(block);write(folder/'jobs.json',jobs)
            write(folder/'windows.json',rows);log('ONSET STATE WINDOW',label,row)
        assert int(e.local.clock.get()[0])==initial_tick+round(c['duration_ms']/e.dt)
        np.savez_compressed(folder/'final_state.npz',**capture(e))
        write(folder/'result.json',dict(status='COMPLETE',label=label,D=float(1-Z[e.s.E]@e.s.mean_weights),
            conditions=c,windows=rows,seconds=time.time()-start,dt_ms=e.dt,
            scope='Finite-duration state continuation; no certified branch, attractor or bifurcation yet.',model_promoted=False))
        jobs.update(status='COMPLETE');write(folder/'jobs.json',jobs)
    except BaseException as exc:
        jobs.update(status='FAILED',error=repr(exc));write(folder/'jobs.json',jobs);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','check','run']);p.add_argument('--label');p.add_argument('--device',type=int,default=1);a=p.parse_args()
    {'register':register,'check':lambda:check(a.device),'run':lambda:run(a.label,a.device)}[a.command]()
