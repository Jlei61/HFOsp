"""Autonomous and fixed-Z trajectories of one synchronized rate DDE."""
from model_zm import *
import argparse
import time
from scipy.ndimage import uniform_filter1d


def summary(rate):
    y=uniform_filter1d(rate,10,axis=0,mode='nearest')
    out=[]
    for k in range(y.shape[1]):
        x=y[:,k];edges=np.diff(np.r_[False,x>=5,False].astype(int))
        windows=list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))
        good=[(a,b) for a,b in windows if a>0 and b<len(x) and b-a>=4 and x[a:b].max()>=10]
        out.append(dict(mean_hz=float(x.mean()),minimum_hz=float(x.min()),maximum_hz=float(x.max()),quiet_fraction=float((x<5).mean()),
            self_limited_events=len(good),median_duration_ms=float(np.median([b-a for a,b in good])) if good else None,
            above_200_fraction=float((x>=200).mean()),windows_ms=good))
    return out


def run(s,D,duration,dt,label,dynamic,device=0,z_override=None,z_source=None):
    s.set_D(D);folder=DEST/'runs'/label;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return
    if z_override is not None:
        s.Z=np.array(z_override,dtype=float)
        assert abs(1-s.Z[s.E]@s.mean_weights-D)<1e-6
    e=ZMIntegrator(s,dt=dt,dynamic_z=dynamic,device=device);cp=e.cp
    write(folder/'contract.json',dict(model='Frozen shared two-filter rate DDE plus Z/M',D_initial=D,dynamic_Z=dynamic,dynamic_M=True,
        J_EE_core=1,dt_ms=dt,duration_ms=duration,initial='Zero rates, recurrent filters and M; initial prescribed Z field',
        noise='Deterministic: private Poisson moments retained; shared OU held at its mean',
        Z_closure='Gaussian expectation of native inhibitory-current threshold; not H of mean current',
        native_future_activity_used=False,Z_field_source=z_source or 'Native 9.420 s power path',SNN_equivalence='NOT_VALIDATED'))
    activity=[];zrecord=[];mrecord=[];acc=cp.zeros(s.P);sample=round(1/dt);start=time.time()
    for i in range(round(duration/dt)):
        acc+=e.step()*dt
        if (i+1)%sample==0:
            activity.append(acc.copy());acc.fill(0);zrecord.append(e.y[9].copy());mrecord.append(e.y[8].copy())
        if (i+1)%round(1000/dt)==0:
            assert bool(cp.isfinite(e.y).all())
            print(label,'t_ms',(i+1)*dt,'seconds',round(time.time()-start,1),flush=True)
    a=cp.stack(activity).get()*1000;z=cp.stack(zrecord).get();m=cp.stack(mrecord).get()
    E=s.E;reg=s.geo['group_region'];size=s.sizes
    regional=np.array([np.average(a[:,E&(reg==k)],axis=1,weights=size[E&(reg==k)]) for k in range(3)]).T
    whole=np.average(a[:,E],axis=1,weights=size[E]);Dtrace=1-np.average(z[:,E],axis=1,weights=size[E])
    field=np.zeros((len(a),s.grid*s.grid));cell=s.geo['group_cell']
    for g in np.flatnonzero(E):field[:,cell[g]]+=a[:,g]*size[g]
    counts=np.bincount(cell[E],weights=size[E],minlength=s.grid*s.grid);field/=np.maximum(counts,1)
    np.savez_compressed(folder/'trajectory.npz',time_ms=np.arange(len(a))+1,group_rate_hz=a.astype('float32'),
        regional_rates_hz=regional,global_E_hz=whole,field_E_hz=field.astype('float32'),D=Dtrace,
        Z=z.astype('float32'),M_current=m.astype('float32'),final_state=e.y.get(),final_history=e.history.get(),final_tick=e.tick)
    burn=min(2000,round(duration/4));q=dict(status='COMPLETE',D_initial=D,D_final=Dtrace[-1],dynamic_Z=dynamic,
        duration_ms=duration,dt_ms=dt,seconds=time.time()-start,analysis_start_ms=burn,
        regions=['global','A','B','surround'],dynamics=summary(np.c_[whole,regional][burn:]),
        z_bounds=[float(z.min()),float(z.max())],SNN_equivalence='NOT_VALIDATED',periodic_orbit_stability='NOT_COMPUTED')
    assert q['z_bounds'][0]>=0 and q['z_bounds'][1]<=1
    write(folder/'result.json',q);print('COMPLETE',label,'Dfinal',q['D_final'],'means',[x['mean_hz'] for x in q['dynamics']],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--values',type=float,nargs='+',default=[0.,.18,.20,.22,.24,.26,.30])
    p.add_argument('--duration',type=int,default=8000);p.add_argument('--dt',type=float,default=.1)
    p.add_argument('--dynamic',action='store_true');p.add_argument('--device',type=int,default=0);p.add_argument('--prefix',default='main')
    p.add_argument('--z-snapshot');p.add_argument('--z-index',type=int)
    args=p.parse_args();s=ZMSpatialRate()
    z=None
    if args.z_snapshot:
        assert args.z_index is not None
        z=np.load(args.z_snapshot)['Z'][args.z_index]
        args.values=[float(1-np.average(z[s.E].astype(float),weights=s.sizes[s.E]))]
    for D in args.values:run(s,D,args.duration,args.dt,f'{args.prefix}_'+('dynamic' if args.dynamic else f'D{D:.3f}'),args.dynamic,args.device,
                            z_override=z,z_source=dict(file=args.z_snapshot,index=args.z_index) if z is not None else None)
