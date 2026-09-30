"""Run the v3 dynamic model on the full network (stage A4 / B / C trajectories).

Modes: deterministic skeleton (mean external drive, no finite-size noise) or stochastic contrast
(recorded native external drive per cell per 1 ms and/or Poisson finite-size sampling of each
group's emitted rate). Z dynamic or frozen; M dynamic or frozen. Starts from the common physical
zero state (no synaptic activity, Z=1, M=0) unless an initial state file is given.
Outputs per 1 ms: group rates, global/regional E rates, 20x20 cell field, D, and per-10-ms Z/M.
"""
from dynamics_v3 import *
from native_readouts import readouts,window_stats,cell_xy
import argparse
def group_drive(s,label):
    z=np.load(DEST/f'native_reference/{label}_external_drive.npz');dm=z['drive_mean'];gl=z['glob'];cell=s.geo['group_cell']
    drive=np.empty((dm.shape[0],s.P));drive[:,s.E]=dm[:,cell[s.E]];drive[:,~s.E]=gl[:,None];return drive
def run(a):
    resp=ResponseParams(DEST/'response_closure/closure.json') if (DEST/'response_closure/closure.json').exists() and not a.placeholder_response else ResponseParams()
    s=DynamicModel(resp=resp,quiet=True);folder=DEST/'runs'/a.label;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists() and not a.force:log('exists',a.label);return
    drive=group_drive(s,a.drive) if a.drive!='mean' else None
    if a.D is not None:s.set_D(a.D)
    if a.z_field:
        z=np.load(a.z_field);s.set_Z(z['Z'],source=a.z_field)
    initial=None;history=None
    if a.initial:
        z=np.load(a.initial);initial=z['final_state'] if 'final_state' in z.files else z['state'];history=z['final_history'] if 'final_history' in z.files else z['history']
    split=shared_fraction_operators(s) if a.noise else None
    e=Integrator(s,dt=a.dt,initial=initial,history=history,dynamic_z=not a.frozen_z,dynamic_m=not a.frozen_m,drive=drive,noise=a.noise,seed=a.seed,device=a.device,shared_split=split);cp=e.cp
    write(folder/'contract.json',dict(model='v3 colored-noise MC transfer spatial rate DDE + Z/M',response_closure=getattr(resp,'source','PLACEHOLDER_CONSTANTS'),
        drive=a.drive,finite_size_noise=a.noise,shared_private_split=bool(a.noise),seed=a.seed,dynamic_Z=not a.frozen_z,dynamic_M=not a.frozen_m,D_initial=s.D,z_source=s.z_source,
        dt_ms=a.dt,duration_ms=a.duration,initial=a.initial or 'zero state (no synaptic activity, Z=1, M=0)',native_future_activity_used=False,identity=s.identity()))
    n=round(a.duration/a.dt);sample=round(1/a.dt);acc=cp.zeros(s.P);rates=[];zrec=[];mrec=[];start=time.time()
    ck=sorted(set(int(round(t/a.dt)) for t in (a.checkpoint_ms or [])));ckdir=folder/'checkpoints'
    if ck:ckdir.mkdir(exist_ok=True)
    for i in range(n):
        acc+=e.step()*a.dt
        if (i+1) in ck:
            np.savez_compressed(ckdir/f't{int(round((i+1)*a.dt))}ms.npz',state=e.y.get(),history=e.history.get(),tick=e.tick,time_ms=(i+1)*a.dt,Z=e.y[11].get(),D=float(1-e.y[11][s.E].get()@s.mean_weights))
        if (i+1)%sample==0:
            rates.append((acc/1.).copy());acc.fill(0)
            if len(rates)%10==0:zrec.append(e.y[11].copy());mrec.append(e.y[10].copy())
        if (i+1)%round(1000/a.dt)==0:
            assert bool(cp.isfinite(e.y).all()),'non-finite state'
            log(a.label,'t_ms',(i+1)*a.dt,'s',round(time.time()-start,1),'global %.1f Hz'%(float(rates[-1][s.E]@cp.asarray(s.mean_weights))*1000),'D %.4f'%float(1-e.y[11][s.E]@cp.asarray(s.mean_weights)))
    R=cp.stack(rates).get()*1000;Zr=cp.stack(zrec).get();Mr=cp.stack(mrec).get()
    E=s.E;reg=s.geo['group_region'];size=s.sizes
    regional=np.array([np.average(R[:,E&(reg==k)],axis=1,weights=size[E&(reg==k)]) for k in range(3)]).T
    whole=np.average(R[:,E],axis=1,weights=size[E]);Dtrace=1-np.average(Zr[:,E],axis=1,weights=size[E])
    cell=s.geo['group_cell'];field=np.zeros((len(R),s.grid*s.grid));counts=np.zeros(s.grid*s.grid)
    for g in np.flatnonzero(E):field[:,cell[g]]+=R[:,g]*size[g];counts[cell[g]]+=size[g]
    field/=np.maximum(counts,1)
    t=np.arange(len(R))+1.
    np.savez_compressed(folder/'trajectory.npz',time_ms=t,group_rate_hz=R.astype('float32'),regional_rates_hz=regional,global_E_hz=whole,field_E_hz=field.astype('float32'),
        D=Dtrace,Z=Zr.astype('float32'),M_current=Mr.astype('float32'),final_state=e.y.get(),final_history=e.history.get(),final_tick=e.tick,cell_counts=counts)
    events,summary,allE,sm=readouts(t,field,counts,a.label)
    summary['windows']={f'{x}-{y}':window_stats(events,x,y) for x,y in [(1000,4000),(4000,8000),(8000,9420),(1000,9420),(1000,min(a.duration,12500))]}
    summary.update(status='COMPLETE',D_final=float(Dtrace[-1]),seconds=time.time()-start,regional_mean_hz=regional.mean(0).tolist(),SNN_equivalence='NOT_ASSESSED_HERE')
    np.savez_compressed(folder/'events.npz',onsets=np.array([ev['onset'] for ev in events]) if events else np.zeros((0,400)),start_ms=[ev['start_ms'] for ev in events],duration_ms=[ev['duration_ms'] for ev in events])
    summary['events']=[{k:v for k,v in ev.items() if k!='onset'} for ev in events]
    write(folder/'result.json',summary);log('COMPLETE',a.label,'events',len(events),'high onset',summary['high_onset_ms'],'D final %.4f'%Dtrace[-1])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--drive',default='seed9108401');p.add_argument('--noise',action='store_true')
    p.add_argument('--seed',type=int,default=1);p.add_argument('--frozen-z',action='store_true');p.add_argument('--frozen-m',action='store_true');p.add_argument('--D',type=float)
    p.add_argument('--z-field');p.add_argument('--initial');p.add_argument('--duration',type=float,default=12500.);p.add_argument('--dt',type=float,default=.1)
    p.add_argument('--device',type=int,default=0);p.add_argument('--checkpoint-ms',type=float,nargs='*');p.add_argument('--force',action='store_true');p.add_argument('--placeholder-response',action='store_true');run(p.parse_args())
