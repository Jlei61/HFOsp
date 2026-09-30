"""Generate every dynamic panel from the autonomous rate DDE itself."""
from rate_field import *
import argparse
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def dynamics(rate,start=1000):
    out=[]
    for k in range(3):
        y=rate[start:,k];above=y>=5;edges=np.diff(np.r_[False,above,False].astype(int))
        windows=list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)));merged=[]
        for a,b in windows:
            if merged and a-merged[-1][1]<10:merged[-1][1]=b
            else:merged.append([int(a),int(b)])
        bursts=[(a,b) for a,b in merged if a>0 and b<len(y) and b-a>=4 and y[a:b].max()>=10]
        peak=np.array([a+np.argmax(y[a:b]) for a,b in bursts],int)+start;iei=np.diff(peak)
        out.append(dict(mean_rate_hz=float(y.mean()),minimum_hz=float(y.min()),maximum_hz=float(y.max()),
            quiet_fraction=float(np.mean(y<5)),self_limited_bursts=len(bursts),peak_times_ms=peak+1,
            burst_windows_ms=[[a+start,b+start] for a,b in bursts],
            IEI_CV=float(np.std(iei,ddof=1)/np.mean(iei)) if len(iei)>1 else None,
            median_IEI_ms=float(np.median(iei)) if len(iei) else None,
            oscillatory_peaks=len(find_peaks(y,prominence=1,distance=30)[0])))
    return out


def simulate(s,J,duration,dt,label,device=0):
    dest=RATE_OUT/'runs'/label/f'J{J:.7f}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'result.json').exists():return read(dest/'result.json')
    engine=RateIntegrator(s,J,dt,device=device);cp=engine.cp;steps=round(duration/dt);sample=round(1/dt)
    activity=[];current=[];start=time.time()
    # Readout weights are purely spatial metadata; no native activity is read.
    native_geo=np.load(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz')
    positions=s.geo['original_positions'][:32000];xy=native_geo['contact_xy'];group=s.geo['cell_group'][:32000]
    d=np.linalg.norm(positions[:,None,:]-xy[None,:,:],axis=2);rx=s.p['rx'];Rr=s.p['Rr'];lfp=np.zeros_like(d)
    for j in range(15):
        ids=np.flatnonzero(d[:,j]<=Rr)
        if not len(ids):ids=np.array([np.argmin(d[:,j])])
        dist=np.maximum(d[ids,j],1e-4);ww=np.where(dist<rx,dist**-.5,rx**-.5*(rx/dist)**2);lfp[ids,j]=ww/ww.sum()
    current_weights=np.array([np.bincount(group,weights=lfp[:,j],minlength=s.P) for j in range(15)]).T
    cw=cp.asarray(current_weights);private=cp.asarray(s.private_mu);acc=cp.zeros(s.P)
    write(dest/'contract.json',dict(model_source=str(Path(__file__).with_name('rate_field.py')),closure_source=str(RATE_OUT/'closure.json'),
        model='Autonomous positive two-filter spatial rate DDE',spatial_cells=s.grid**2,rate_groups=s.P,continuous_states=9*s.P,
        J_EE_core=J,duration_ms=duration,dt_ms=dt,initial_state='zero rates and recurrent states; constant private-input moments',
        Z='fixed 1',M='original gain .0005 per spike; tau 1000 ms in rate expectation',
        native_spikes_used=False,noise='No sampled noise. Private Poisson variance enters Phi analytically.',
        scope='Interictal rate-model simulation; nonlinear SNN equivalence remains to be validated'))
    for i in range(steps):
        acc+=engine.step()*dt
        if (i+1)%sample==0:
            activity.append(acc.copy());acc.fill(0)
            current.append((cp.abs(engine.y[3]+private)+cp.abs(engine.y[5]))@cw)
        if (i+1)%round(1000/dt)==0:
            assert bool(cp.isfinite(engine.y).all());assert float(engine.y[:2].min())>=-1e-12
            print(label,J,'t',(i+1)*dt,'sec',round(time.time()-start,1),flush=True)
    activity=cp.stack(activity).get()*1000;current=cp.stack(current).get();size=s.geo['group_size'];E=s.E;reg=s.geo['group_region']
    regional=np.array([np.average(activity[:,E&(reg==k)],axis=1,weights=size[E&(reg==k)]) for k in range(3)]).T
    whole=np.average(activity[:,E],axis=1,weights=size[E]);field=np.zeros((len(activity),s.grid**2));cell=s.geo['group_cell']
    for p in np.flatnonzero(E):field[:,cell[p]]+=activity[:,p]*size[p]
    counts=np.bincount(cell[E],weights=size[E],minlength=s.grid**2);field/=np.maximum(counts,1)
    contact=activity@s.geo['contact_rate_weights'];smooth=gaussian_filter1d(regional,5,axis=0)
    np.savez_compressed(dest/'trajectory.npz',time_ms=np.arange(len(activity))+1,group_rate_hz=activity.astype('float32'),
        regional_rates_hz=regional,all_E_rate_hz=whole,field_E_hz=field.astype('float32'),
        contact_rate_hz=contact,contact_current_proxy=current,contact_names=native_geo['contact_names'],final_state=engine.y.get())
    result=dict(status='COMPLETE',J_EE_core=J,duration_ms=duration,dt_ms=dt,seconds=time.time()-start,
        dynamics=dynamics(smooth,min(1000,int(duration/4))),trajectory=str(dest/'trajectory.npz'),contract=str(dest/'contract.json'),
        observation='Rates in Hz/cell; fields and readouts are functions of this rate trajectory, not sampled spikes',
        native_equivalence='NOT_VALIDATED')
    write(dest/'result.json',result);print('done',J,[(x['self_limited_bursts'],x['IEI_CV'],x['mean_rate_hz']) for x in result['dynamics']],flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--values',type=float,nargs='+',required=True);p.add_argument('--duration',type=int,default=10000)
    p.add_argument('--dt',type=float,default=.1);p.add_argument('--label',default='main');p.add_argument('--device',type=int,default=0)
    args=p.parse_args();s=RateField()
    for J in args.values:simulate(s,J,args.duration,args.dt,args.label,args.device)
