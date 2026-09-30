"""Stage B: connect the conditional (frozen-Z) dynamics to the actual Z/M entry trajectory of the
validated deterministic skeleton (A4_det_meandrive) and to the native reference.

B1  Z-field family along the actual entry: rate-model checkpoints (every 250-500 ms from 6.5 s) and native
    projections; D(t); spatial RMS difference to the prescribed power path at equal D.
B2  Controls (all with the same integrator, frozen Z, mean drive, no noise):
    (a) same carried fast state / M / delay history (checkpoint at t_ref), only the Z field replaced by
        the field from another time t_Z of the same trajectory  -> which Z field makes the carried state persist
    (b) same Z field, carried state vs relaxed state (zero state with that Z, 3 s)               -> M/history effect
    (c) conditional attractor at each Z field from the relaxed state, 6 s: self-limited cycle / global bursting /
        sustained broad activity classification + periodic BVP & Floquet where a cycle exists
    (d) alignment with the full Z/M trajectory (entry time, D at entry)
Classification (10-ms all-E rate, last 2 s of the run): SUSTAINED_BROAD = no quiet (<5 Hz) run >= 20 ms and
neuron-weighted fraction of cells > 50 Hz >= 0.75; GLOBAL_SYNCHRONOUS_BURSTS = events with area >= 0.9 separated
by quiet; LOCAL_SELF_LIMITED = events with area < 0.9 separated by quiet; LOW = no events.
"""
from run_network import *
from periodic_v3 import load_model
from native_readouts import readouts,cell_xy
import argparse
OUT=DEST/'stage_b'
def classify(t,field,counts):
    events,summary,allE,sm=readouts(t,field,counts,'x');w=counts/counts.sum();n=len(t);tail=slice(max(0,n-2000),n)
    smt=sm[tail];quiet=smt<5;edges=np.diff(np.r_[0,quiet.astype(int),0]);qs=np.flatnonzero(edges==1);qe=np.flatnonzero(edges==-1);longest_quiet=max([b-a for a,b in zip(qs,qe)],default=0)
    occ=float(((field[tail]>50)@w).mean());ev=[e for e in events if e['start_ms']>=t[tail][0]]
    if longest_quiet<20 and occ>=.75:cat='SUSTAINED_BROAD'
    elif longest_quiet<20:cat='SUSTAINED_PARTIAL'
    elif ev and np.median([e['area_fraction'] for e in ev])>=.9:cat='GLOBAL_SYNCHRONOUS_BURSTS'
    elif ev:cat='LOCAL_SELF_LIMITED'
    else:cat='LOW'
    return dict(category=cat,longest_quiet_ms=int(longest_quiet),occupation_50Hz=occ,tail_mean_hz=float(smt.mean()),n_events_tail=len(ev),
        median_area=float(np.median([e['area_fraction'] for e in ev])) if ev else None,median_duration_ms=float(np.median([e['duration_ms'] for e in ev])) if ev else None,high_onset_ms=summary['high_onset_ms'])
def run_frozen(s,Z,initial,history,duration,label,source):
    s.z_override=Z;s.z_override_source=source;s.set_D(0.)
    e=Integrator(s,initial=initial,history=history,dynamic_z=False,dynamic_m=True);cp=e.cp;n=round(duration/e.dt);acc=cp.zeros(s.P);rates=[];Mrec=[]
    for i in range(n):
        acc+=e.step()*e.dt
        if (i+1)%10==0:rates.append(acc.copy());acc.fill(0)
        if (i+1)%1000==0:Mrec.append(e.y[10].get())
    R=cp.stack(rates).get()*1000;E=s.E;cell=s.geo['group_cell'];size=s.sizes;field=np.zeros((len(R),400));counts=np.zeros(400)
    for g in np.flatnonzero(E):field[:,cell[g]]+=R[:,g]*size[g];counts[cell[g]]+=size[g]
    field/=np.maximum(counts,1);t=np.arange(len(R))+1.;glob=np.average(R[:,E],axis=1,weights=size[E])
    res=classify(t,field,counts);res.update(label=label,duration_ms=duration,D=float(1-Z[E]@s.mean_weights),z_source=source,global_mean_hz=float(glob.mean()),final_state=None)
    np.savez_compressed(OUT/'runs'/f'{label}.npz',time_ms=t,global_E_hz=glob,field_E_hz=field.astype('float32'),group_rate_hz=R.astype('float32'),Z=Z,final_state=e.y.get(),final_history=e.history.get(),M_last=np.array(Mrec))
    s.z_override=None;return res,e.y.get(),e.history.get()
def main(a):
    OUT.mkdir(exist_ok=True);(OUT/'runs').mkdir(exist_ok=True);s=load_model();ck=DEST/'runs'/a.source/'checkpoints';cks=sorted(ck.glob('t*ms.npz'),key=lambda p:int(p.stem[1:-2]))
    res=read(DEST/'runs'/a.source/'result.json');entry=res['high_onset_ms'];log('source',a.source,'entry',entry)
    # ---- B1: Z-field family
    fam={};rows=[]
    for p in cks:
        z=np.load(p);t=int(p.stem[1:-2]);Z=z['Z'];D=float(z['D']);fam[t]=Z
        s.set_D(min(max(D,0),1));rms=float(np.sqrt(np.average((Z[s.E]-s.Z[s.E])**2,weights=s.mean_weights)))
        rows.append(dict(time_ms=t,D=D,spatial_rms_to_power_path_same_D=rms,Z_core_A=float(np.average(Z[s.E&(s.geo['group_region']==0)],weights=s.sizes[s.E&(s.geo['group_region']==0)])),
            Z_core_B=float(np.average(Z[s.E&(s.geo['group_region']==1)],weights=s.sizes[s.E&(s.geo['group_region']==1)])),Z_surround=float(np.average(Z[s.E&(s.geo['group_region']==2)],weights=s.sizes[s.E&(s.geo['group_region']==2)]))))
    npz=dict(np.load(DEST/'native_reference/checkpoint_projections.npz'));nat=[]
    for t in npz['times']:
        Z=np.ones(s.P);Z[s.E]=npz[f'Z_group_{t}'][s.E];D=float(1-Z[s.E]@s.mean_weights);s.set_D(D);rms=float(np.sqrt(np.average((Z[s.E]-s.Z[s.E])**2,weights=s.mean_weights)))
        nat.append(dict(time_ms=int(t),D=D,spatial_rms_to_power_path_same_D=rms));fam[f'native_{t}']=Z
    np.savez_compressed(OUT/'z_fields.npz',**{str(k):v for k,v in fam.items()});write(OUT/'b1_z_family.json',dict(source=a.source,entry_ms=entry,rate_model=rows,native=nat))
    log('B1 done; D at checkpoints',[(r['time_ms'],round(r['D'],4)) for r in rows])
    # ---- B2 (a): carried state at t_ref, Z fields from other times
    tref=a.t_ref;zr=np.load(ck/f't{tref}ms.npz');state=zr['state'];hist=zr['history'];results=dict(a=[],b=[],c=[])
    for tz in a.t_fields:
        Z=fam[tz];r,_,_=run_frozen(s,Z,state,hist,a.duration,f'a_carried{tref}_Z{tz}',f'rate-model Z field at {tz} ms');r.update(t_ref=tref,t_Z=tz);results['a'].append(r);log('B2a',tz,r['category'],round(r['D'],4))
    # ---- B2 (b): same Z field, carried vs relaxed
    for tz in a.t_fields:
        Z=fam[tz];z2=np.load(ck/f't{tz}ms.npz')
        r1,_,_=run_frozen(s,Z,z2['state'],z2['history'],a.duration,f'b_carried{tz}',f'rate-model Z at {tz}');r1.update(t_Z=tz,state='carried')
        r2,_,_=run_frozen(s,Z,None,None,a.duration,f'b_relaxed{tz}',f'rate-model Z at {tz}');r2.update(t_Z=tz,state='relaxed(zero)')
        results['b'].extend([r1,r2]);log('B2b',tz,'carried',r1['category'],'relaxed',r2['category'])
    # ---- B2 (c): conditional attractors from the relaxed state, longer runs, over the family (rate model + native fields)
    for key in a.t_fields+[f'native_{t}' for t in a.native_times]:
        Z=fam[key];r,_,_=run_frozen(s,Z,None,None,a.long_duration,f'c_relaxed_{key}',str(key));r.update(field=str(key));results['c'].append(r);log('B2c',key,r['category'],'D %.4f'%r['D'])
    write(OUT/'b2_controls.json',dict(source=a.source,entry_ms=entry,t_ref=tref,results=results,
        definitions='frozen Z (conditional system), dynamic M, mean drive, no noise; categories from the last 2 s'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='A4_det_meandrive');p.add_argument('--t-ref',type=int,default=7000)
    p.add_argument('--t-fields',type=int,nargs='+',default=[6000,6500,7000,7250,7500,7600,7700,7800,7900,8000,8100,8250,8500,9000])
    p.add_argument('--native-times',type=int,nargs='+',default=[8000,9000,9300,9420,9870,10370]);p.add_argument('--duration',type=float,default=3000.);p.add_argument('--long-duration',type=float,default=6000.)
    main(p.parse_args())
