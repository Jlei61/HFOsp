"""Two fixed closure diagnostics under the original stochastic A4 protocol."""
from common import *
from closure_network_sensitivity import setup
from runner import summarize
from native_readouts import readouts, window_stats
from scipy.ndimage import uniform_filter1d
from closure_network_sensitivity_audit import high_entry
import argparse

DEST=OUT/'closure_stochastic_sensitivity'


def record(s,e,duration,label):
    cp=e.cp;acc=cp.zeros(s.P);rates=[];Z=[];M=[];start=time.time()
    for i in range(round(duration/e.dt)):
        acc+=e.step()*e.dt
        if (i+1)%round(1/e.dt)==0:
            rates.append(acc.copy());acc.fill(0)
            if len(rates)%10==0:Z.append(e.y[11].copy());M.append(e.y[10].copy())
        if (i+1)%round(1000/e.dt)==0:
            assert bool(cp.isfinite(e.y).all())
            log('STOCHASTIC CLOSURE',label,'t_ms',(i+1)*e.dt,'seconds',round(time.time()-start,1))
            write(DEST/label/'progress.json',dict(status='RUNNING',time_ms=(i+1)*e.dt))
    return cp.stack(rates).get()*1000,cp.stack(Z).get(),cp.stack(M).get()


def actual_input_rhs_audit(s,e):
    """Independent CPU readout with actual external input and private moments."""
    y0=e.y.copy();h0=e.history.copy()
    point=np.load(BASE/'runs/A4_stoch_seed9108401/checkpoints/t8000ms.npz')['state']
    e.y[:]=e.cp.asarray(point);e.history[:]=e.cp.asarray(np.broadcast_to(s.output(point),e.history.shape).copy())
    rows=[]
    for tick in [0,10013,94209]:
        e.arrivals(tick);e.eval_rhs(e.y,tick,e.f,e.rate)
        nu=e.drive[e.drive_index(tick)].get()
        # Same external physical moments as the frozen CUDA kernel.
        pm=s.tm*s.area[0]*s.jext*nu
        pv=s.tm*s.area[0]**2*s.jext**2*nu
        rhs,r=s.rhs(point,e.arr.get(),dynamic_z=True,pm=pm,pv=pv)
        err=float(abs(rhs-e.f.get()).max());re=float(abs(r-e.rate.get()).max())
        assert np.allclose(rhs,e.f.get(),rtol=1e-9,atol=1e-10),(tick,err)
        assert np.allclose(r,e.rate.get(),rtol=1e-9,atol=1e-12),(tick,re)
        rows.append(dict(tick=tick,drive_index=e.drive_index(tick),RHS_error=err,rate_error=re))
    e.y[:]=y0;e.history[:]=h0;e.tick=0
    return rows


def main(device):
    c=read(OUT/'closure_stochastic_sensitivity_contract.json');DEST.mkdir(exist_ok=True)
    old=BASE/'runs/A4_stoch_seed9108401'
    if not (DEST/'baseline_prefix_audit.json').exists():
        s,e=setup('frozen',c['dt_ms'],device,stochastic=True)
        initial=(e.y.get(),e.history.get())
        np.savez_compressed(DEST/'initial.npz',state=initial[0],history=initial[1])
        R,z,m=record(s,e,c['prefix_duration_ms'],'baseline_prefix')
        ref=np.load(old/'trajectory.npz');n=len(R)
        checks=dict(group_rates_float32=np.array_equal(R.astype('float32'),ref['group_rate_hz'][:n]),
             Z_float32=np.array_equal(z.astype('float32'),ref['Z'][:len(z)]),
             M_float32=np.array_equal(m.astype('float32'),ref['M_current'][:len(m)]))
        assert all(checks.values()),checks
        write(DEST/'baseline_prefix_audit.json',dict(status='BITWISE_PREFIX_PASS',checks=checks,duration_ms=c['prefix_duration_ms']))
        del e,s
    initial=np.load(DEST/'initial.npz')
    rows=[]
    for label in c['variants']:
        folder=DEST/label;folder.mkdir(exist_ok=True)
        if (folder/'result.json').exists():rows.append(read(folder/'result.json'));continue
        s,e=setup(label,c['dt_ms'],device,stochastic=True)
        assert np.array_equal(e.y.get(),initial['state'])
        e.history[:]=e.cp.asarray(initial['history'])
        qa=actual_input_rhs_audit(s,e)
        assert np.array_equal(e.y.get(),initial['state']) and np.array_equal(e.history.get(),initial['history'])
        R,z,m=record(s,e,c['duration_ms'],label)
        _,field,whole,count=summarize(R,s)
        t=np.arange(len(R))+1.;ts=(np.arange(len(z))+1)*10.;d=1-z[:,s.E]@s.mean_weights
        ev,summary,_,_=readouts(t,field,count,label)
        np.savez_compressed(folder/'trajectory.npz',time_ms=t,group_rate_hz=R.astype('float32'),
             field_E_hz=field.astype('float32'),global_E_hz=whole,cell_counts=count,
             Z=z.astype('float32'),M_current=m.astype('float32'),state_time_ms=ts,D=d,
             final_state=e.y.get(),final_history=e.history.get(),final_tick=e.tick,dt_ms=c['dt_ms'])
        summary.update(status='COMPLETE',label=label,qa=qa,initial_bitwise=True,Z_and_M_dynamic=True,
                       D9870=float(d[np.flatnonzero(ts==9870)[0]]),D_final=float(d[-1]),
                       replacement_promoted=False,scope=c['scope'])
        summary['events']=[{k:v for k,v in x.items() if k!='onset'} for x in ev]
        write(folder/'result.json',summary);rows.append(summary)
        write(DEST/'result.json',dict(status='RUNNING',rows=rows))
        log('STOCHASTIC CLOSURE COMPLETE',label,summary['high_onset_ms'],summary['D9870'])
        del e,s
    write(DEST/'result.json',dict(status='DIAGNOSTIC_COMPLETE',rows=rows,scope=c['scope'],replacement_promoted=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0);main(p.parse_args().device)
