"""Fixed-J initial-history bridge between the displayed weak and burst cycles.

The path interpolates all nine local states and the complete delay history.
It is an initial-condition assay, never a periodic branch continuation.
Every model coefficient remains frozen. Large outputs live on the data disk.
"""
from run_rate_field_long import *
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
import gc

DEST=Path('/data/hfosp/topic4_sef_hfo/interictal_rate_small_burst_connection_20260919')


def recover_cpu(s,path,dt,phase_index=None):
    """Independent CPU harmonic recovery, retaining every stored harmonic."""
    z=np.load(path);r=z['r'];T=float(z['T']);J=float(z['J']);N=len(r)
    regional=np.array([s.regional_rates(x) for x in r])
    shift=int(np.argmin(regional[:,:2].sum(1))) if phase_index is None else int(phase_index)
    r=np.roll(r,-shift,axis=0)
    cf=np.fft.rfft(r,axis=0)/N;K=len(cf);lam=2j*np.pi*np.arange(K)/T
    factors=np.full(K,2.);factors[0]=1
    if N%2==0:factors[-1]=1
    local=np.zeros((9,s.P),complex)
    # Linear recurrent filters from the exact spatial and delay operators.
    for first in range(0,K,32):
        stop=min(K,first+32);ll=lam[first:stop,None];v=cf[first:stop]
        actions=[]
        phase=np.exp(-s.delays[:,None]*lam[None,first:stop])
        for k,(row,col,mask,d) in enumerate(s.raw):
            weights=(d@phase).T
            if k in (0,2):weights*=np.where(mask,J**(1 if k==0 else 2),1.)
            values=weights*v[:,col]
            actions.append(np.array([np.bincount(row,weights=x.real,minlength=s.P)+
                                      1j*np.bincount(row,weights=x.imag,minlength=s.P) for x in values]))
        a,b,qa,qb=actions
        H=s.alpha/(1+ll*s.tf)+(1-s.alpha)/(1+ll*s.ts);target=v/H
        xa=target/(1+ll*s.tf);xb=target/(1+ll*s.ts)
        qav=s.tm*s.area[0]*a/(1+ll*s.rise[0]);iav=qav/(1+ll*s.decay[0])
        qgv=s.tm*s.area[1]*b/(1+ll*s.rise[1]);igv=qgv/(1+ll*s.decay[1])
        va=s.tm*s.area[0]**2*qa/(1+ll*s.tau[0]/2)
        vg=s.tm*s.area[1]**2*qb/(1+ll*s.tau[1]/2);m=.5*s.E*v/(1+1000*ll)
        local+=(np.array([xa,xb,qav,iav,qgv,igv,va,vg,m])*factors[None,first:stop,None]).sum(1)
    D=s.prep['max_delay_steps']*round(.1/dt);depth=D+1
    chronological=(np.exp(-np.arange(depth)[:,None]*dt*lam[None,:])@(cf*factors[:,None])).real
    hist=np.empty_like(chronological);hist[(-np.arange(depth))%depth]=chronological
    y=local.real;defect=float(abs(s.output(y)-hist[0]).max());assert defect<1e-12
    assert min(y[:2].min(),hist.min())>-1e-10
    return y,hist,dict(orbit=str(path),J_EE_core=J,T_ms=T,N=N,phase_index=shift,
        history_endpoint_error=defect,minimum_filter_rate_Hz=float(y[:2].min()*1000),
        minimum_history_rate_Hz=float(hist.min()*1000))


def prepare(s,dt):
    manifest=read(RATE_OUT/'periodic_completion/composite_case_resolution.json')
    folder=DEST/'initial_conditions';folder.mkdir(parents=True,exist_ok=True);out=[]
    for letter in ['b','c']:
        row=next(q for q in manifest['rows'] if q['case']==letter);path=Path(row['orbit'])
        dest=folder/f'{letter}_dt{dt:g}.npz';meta=dest.with_suffix('.json')
        if dest.exists():
            q=read(meta);assert q['orbit']==str(path)
            z=np.load(dest);out.append((z['state'],z['history']));continue
        print('RECOVER',letter,path,flush=True)
        y,h,q=recover_cpu(s,path,dt)
        q['dt_ms']=dt;q['scope']='All nine local states and full physical-delay history; common minimum-core-rate phase for each orbit.'
        np.savez_compressed(dest,state=y,history=h);write(meta,q);out.append((y,h))
        print('RECOVERED',letter,q,flush=True)
    return out


def template_match(s,activity,path):
    z=np.load(path);r=z['r']*1000;T=float(z['T']);N=len(r)
    # Exact one-ms average of the stored Fourier rate, evaluated at bin centres.
    f=np.fft.rfft(r,axis=0);f*=np.sinc(np.arange(len(f))[:,None]/T)
    smooth=np.fft.irfft(f,n=N,axis=0)
    spline=CubicSpline(np.arange(N+1)*T/N,np.vstack([smooth,smooth[0]]),axis=0)
    n=min(len(activity),max(2000,int(3*T)));actual=np.asarray(activity[-n:],float)
    weights=s.geo['group_size']/s.geo['group_size'].sum()
    scale=max(float(np.mean(np.sum((r-r.mean(0))**2*weights,axis=1))),1e-12)
    times=np.arange(n)+.5
    def loss(phase):
        pred=spline((times+phase)%T);return float(np.mean(np.sum((actual-pred)**2*weights,axis=1))/scale)
    grid=np.linspace(0,T,48,endpoint=False);values=[loss(t) for t in grid];best=grid[int(np.argmin(values))]
    fit=minimize_scalar(loss,bounds=(best-T/48,best+T/48),method='bounded')
    return dict(orbit=str(path),period_ms=T,window_ms=n,common_phase_ms=float(fit.x%T),
        relative_full_group_RMS_error=float(np.sqrt(fit.fun)),
        definition='All 935 groups, neuron-weighted RMS normalized by template temporal RMS; one common phase and fixed BVP period; exact one-ms template averages.')


def simulate(s,initials,fraction,dt,duration,device):
    name=f'fraction{fraction:g}_dt{dt:g}';folder=DEST/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return read(folder/'result.json')
    y=(1-fraction)*initials[0][0]+fraction*initials[1][0]
    h=(1-fraction)*initials[0][1]+fraction*initials[1][1]
    assert np.max(abs(h[0]-s.output(y)))<1e-12
    check=read(RATE_OUT/'periodic_completion'/f'long_graph_check_dt{dt:g}.json');assert check['status']=='PASS'
    g=GraphIntegrator(s,.942,dt,device,y,h);chunks=[];begin=time.time()
    blocks=int(np.ceil(duration/g.samples))
    contract=dict(J_EE_core=.942,fraction=fraction,dt_ms=dt,requested_duration_ms=duration,
        actual_duration_ms=blocks*g.samples,block_ms=g.samples,
        initial_path='(1-fraction)*weak-cycle state/history + fraction*burst-cycle state/history',
        parameter_changed=False,noise_added=False,equations_changed=False,spatial_groups=s.P,
        interpretation='Finite initial-condition basin probe; interpolation is not a solution branch or a physical stimulus.')
    write(folder/'contract.json',contract)
    for k in range(blocks):
        chunks.append(g.block().astype('float32'))
        assert bool(g.cp.isfinite(g.engine.y).all())
        assert float(g.engine.y[:2].min())>-1e-10
        if k%5==0 or k+1==blocks:
            write(folder/'status.json',dict(status='RUNNING',pid=os.getpid(),time_ms=g.elapsed_ms,total_ms=blocks*g.samples,elapsed_s=time.time()-begin))
            print('BRIDGE',name,g.elapsed_ms,'/',blocks*g.samples,'ms',round(time.time()-begin,1),'s',flush=True)
    activity=np.concatenate(chunks);regional=np.array([s.regional_rates(x/1000) for x in activity])
    np.savez_compressed(folder/'trajectory.npz',time_ms=np.arange(len(activity))+1,
        group_rate_hz=activity,regional_rates_hz=regional,
        contact_rate_hz=activity@s.geo['contact_rate_weights'],final_state=g.engine.y.get(),
        final_history=g.engine.history.get(),final_history_tick_modulo_depth=0,dt_ms=dt)
    manifest=read(RATE_OUT/'periodic_completion/composite_case_resolution.json')
    matches={letter:template_match(s,activity,next(q['orbit'] for q in manifest['rows'] if q['case']==letter)) for letter in ['b','c']}
    late=regional[-min(len(regional)//2,5000):]
    best=min(matches,key=lambda key:matches[key]['relative_full_group_RMS_error'])
    result=dict(status='COMPLETE',contract=contract,template_matches=matches,
        finite_window_match=best if matches[best]['relative_full_group_RMS_error']<.05 else 'UNRESOLVED',
        tail_mean_Hz=late.mean(0),tail_max_Hz=late.max(0),seconds=time.time()-begin,
        scope='Finite-time full-field template approach only; no global manifold connection, bifurcation or rigorous basin boundary inferred.')
    write(folder/'result.json',result);write(folder/'status.json',dict(status='COMPLETE',source=str(folder/'result.json')))
    print('RESULT',name,result,flush=True)
    del g;gc.collect()
    import cupy as cp
    cp.get_default_memory_pool().free_all_blocks()
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fractions',type=float,nargs='+',default=[0,1,.5])
    p.add_argument('--dt',type=float,default=.1);p.add_argument('--duration',type=int,default=8000)
    p.add_argument('--device',type=int,default=1);p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();s=RateField();initials=prepare(s,a.dt)
    if not a.prepare_only:
        for fraction in a.fractions:
            assert 0<=fraction<=1
            simulate(s,initials,fraction,a.dt,a.duration,a.device)
