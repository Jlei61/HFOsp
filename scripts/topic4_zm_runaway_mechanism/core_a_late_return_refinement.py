"""Exact local replay and two step halvings of the64.83s quiet return.

The physical model, full Z field and all dynamic M states are unchanged.
This tests robustness of the observed return, not long-time trajectory
identity or the type of a parameter-dependent invariant-set bifurcation.
"""
from common import OUT,np,read,write,log
from onset_state_continuation import build,regrid_state,regional_weights
from fine_rate_frozen_Z_fields import capture,restore
from scipy.ndimage import uniform_filter1d
import argparse,os,time,gc

BASE=OUT/'core_a_bifurcation_type_20260924'
DEST=BASE/'late_return_counterexample/refinement'
SOURCE=BASE/'sustained_A_long_window/below_extend_to60s/final_state.npz'
RECORD=BASE/'actual_sustained_growth/coarse'


def replay(device):
    assert read(DEST.parent/'result.json')['status']=='AUDIT_PASS'
    DEST.mkdir(exist_ok=True);assert not(DEST/'replay_jobs.json').exists()
    write(DEST/'contract.json',dict(
        question='Does the actual late Core-A termination persist under two local timestep halvings from the same complete pretermination state?',
        replay_source=str(SOURCE),source_total_elapsed_ms=60000,selected_total_elapsed_ms=64650,
        replay='Original dt.05 flow must reproduce every recorded1ms all3479-group float32 rate bitwise from60s to64.65s; exact full state saved at64.65s.',
        controls='Three750ms continuations from that exact physical state, dt=.05,.025,.0125ms. Fine histories are successive physical-lag-preserving linear interpolations; original lag knots and physical clock retained. All Z held, all M dynamic; no tangent, noise, parameter or response change.',
        acceptance='Coarse continuation must match original records bitwise. Qualified quiet retains10ms smoothing,<5Hz for>=20ms. Report each mesh and timing changes; no gate relaxation and no inference of long-time deterministic agreement.',
        scope='Local numerical robustness of a real terminating episode only; not a crisis, periodicity, separatrix or onset-bifurcation certificate.',model_promoted=False))
    write(DEST/'replay_jobs.json',dict(status='RUNNING',pid=os.getpid()))
    e=build(device);base=dict(np.load(SOURCE));restore(e,base);W=regional_weights(e.s)
    expected=np.load(RECORD/'block00.npz')['group_rate_hz'];currents=[];start=time.time()
    for k in range(465):
        r=e.chunk();assert np.array_equal(r[:,0].astype('f4'),expected[k*10:(k+1)*10])
        syn=e.syn.get();mu=e.local.physical.get()[0]
        currents.append(np.vstack([syn[1],syn[3],syn[4],mu])@W.T)
        if (k+1)%100==0:log('LATE RETURN EXACT REPLAY',60000+(k+1)*10,round(time.time()-start,1))
    state=capture(e);assert np.array_equal(state['syn'][5],base['syn'][5])
    assert int(state['clock'][0])-int(base['clock'][0])==round(4650/e.dt)
    np.savez_compressed(DEST/'state64650.npz',**state)
    np.savez_compressed(DEST/'replay_currents.npz',time_ms=np.arange(60010,64651,10),
        AMPA_rawGABA_M_mu_regional=np.array(currents),regional_rate_hz=expected[:4650].astype(float)@W.T)
    write(DEST/'replay_result.json',dict(status='EXACT_REPLAY_PASS',all4650ms_all3479_group_rates_bitwise=True,
        all_Z_held=True,all_M_dynamic=True,seconds=time.time()-start))
    write(DEST/'replay_jobs.json',dict(status='COMPLETE',pid=os.getpid()))


def controls(device):
    assert read(DEST/'replay_result.json')['status']=='EXACT_REPLAY_PASS'
    assert not(DEST/'controls_jobs.json').exists()
    jobs=dict(status='RUNNING',pid=os.getpid(),completed=[]);write(DEST/'controls_jobs.json',jobs)
    original=dict(np.load(DEST/'state64650.npz'));base=original;previous_dt=.05;rows=[]
    expected=np.concatenate([np.load(RECORD/f'block{b:02d}.npz')['group_rate_hz'] for b in [0,1]])[4650:5400]
    for dt in [.05,.025,.0125]:
        e=build(device,dt);base=regrid_state(base,e,previous_dt);previous_dt=dt
        restore(e,base);W=regional_weights(e.s);R=[];M=[];synaptic=[];start=time.time()
        assert np.array_equal(base['syn'][5],original['syn'][5])
        assert int(base['clock'][0])*dt==int(original['clock'][0])*.05
        for k in range(75):
            rr=e.chunk()[:,0].astype('f4')
            if dt==.05:assert np.array_equal(rr,expected[k*10:(k+1)*10])
            R.append(rr);syn=e.syn.get();M.append(syn[4]@W.T)
            synaptic.append(np.vstack([syn[1],syn[3],e.local.physical.get()[0]])@W.T)
        assert np.array_equal(e.syn[5].get(),original['syn'][5])
        r=np.concatenate(R);regional=r.astype(float)@W.T;sm=uniform_filter1d(regional[:,1],10,mode='nearest')
        edge=np.diff(np.r_[False,sm<5,False].astype(int))
        quiet=[[int(a+64650),int(b+64650)] for a,b in zip(np.flatnonzero(edge==1),np.flatnonzero(edge==-1)) if b-a>=20]
        row=dict(dt_ms=dt,qualified_Core_A_quiet_ms=quiet,minimum_Core_A_10ms_hz=float(sm.min()),
            original_coarse_bitwise=dt==.05,all_Z_held=True,all_M_dynamic=True,seconds=time.time()-start)
        rows.append(row);np.savez_compressed(DEST/f'local_dt{dt:g}.npz',time_ms=np.arange(64651,65401),
            regional_rate_hz=regional,Core_A_smoothed_hz=sm,M_time_ms=np.arange(64660,65401,10),
            M_regional=np.array(M),AMPA_rawGABA_mu_regional=np.array(synaptic),Z=original['syn'][5],dt_ms=dt)
        jobs['completed'].append(dt);write(DEST/'controls_jobs.json',jobs);write(DEST/'controls_progress.json',rows)
        log('LATE RETURN MESH CONTROL',row)
        del e;gc.collect()
    passed=all(r['qualified_Core_A_quiet_ms'] for r in rows)
    write(DEST/'result.json',dict(status='LOCAL_RETURN_ROBUST_TO_TWO_STEP_HALVINGS' if passed else 'LOCAL_RETURN_MESH_CHECK_NOT_PASSED',rows=rows,
        first_quiet_timing_ms=[r['qualified_Core_A_quiet_ms'][0][0] if r['qualified_Core_A_quiet_ms'] else None for r in rows],
        scope='Observed late-return numerical robustness only. Finite-time and local history control, not a proof of chaos, crisis or a parameter bifurcation.',model_promoted=False))
    jobs.update(status='COMPLETE');write(DEST/'controls_jobs.json',jobs)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['replay','controls']);p.add_argument('--device',type=int,default=1);a=p.parse_args()
    {'replay':replay,'controls':controls}[a.command](a.device)
