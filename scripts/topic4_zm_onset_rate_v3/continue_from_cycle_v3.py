"""Stage C3: cross the cycle fold from the cycle itself. Build the full 14-state history from a converged
periodic orbit (orbit_states), then integrate with frozen Z on the prescribed path at the requested D
values (beyond the fold) and classify the post-transition state (stage_b classify + spectral diagnosis)."""
from floquet_v3 import orbit_states
from periodic_v3 import *
from stage_b_entry import classify
from post_transition_diagnosis import diagnose
import argparse
def main(a):
    s=load_model(a.device);z=np.load(a.orbit);sol={k:z[k] for k in z.files};sol['T']=float(sol['T']);sol['D']=float(sol['D'])
    o=PeriodicV3(s,len(sol['r']),a.device);n=int(np.ceil(sol['T']/.1));full,rr=orbit_states(o,sol,n);depth=s.prep['max_delay_steps']+1
    hist=np.array([rr[(-j)%n] for j in range(depth)])[::-1]   # history ring: slot k holds rate at tick k (mod depth) -> approximate with the last `depth` samples
    out=DEST/'post_fold';out.mkdir(exist_ok=True);rows=[]
    for D in a.D:
        s.set_D(D);y0=full[0].copy();y0[11]=s.Z
        e=Integrator(s,initial=y0,history=None,dynamic_z=False,dynamic_m=True);cp=e.cp
        # seed the history ring with the orbit's recent rates
        e.history[:]=cp.asarray(np.broadcast_to(rr[0],(e.depth,s.P)).copy());
        for j in range(1,min(e.depth,n)):e.history[(e.tick-j)%e.depth]=cp.asarray(rr[(-j)%n])
        nsteps=round(a.duration/e.dt);acc=cp.zeros(s.P);rates=[]
        for i in range(nsteps):
            acc+=e.step()*e.dt
            if (i+1)%10==0:rates.append(acc.copy());acc.fill(0)
        Rr=cp.stack(rates).get()*1000;E=s.E;cell=s.geo['group_cell'];field=np.zeros((len(Rr),400));counts=np.zeros(400)
        for g in np.flatnonzero(E):field[:,cell[g]]+=Rr[:,g]*s.sizes[g];counts[cell[g]]+=s.sizes[g]
        field/=np.maximum(counts,1);t=np.arange(len(Rr))+1.;glob=np.average(Rr[:,E],axis=1,weights=s.sizes[E])
        c=classify(t,field,counts);d=diagnose(t,glob,3000);row=dict(D=D,from_orbit=a.orbit,duration_ms=a.duration,category=c['category'],occupation=c['occupation_50Hz'],longest_quiet_ms=c['longest_quiet_ms'],tail_mean_hz=c['tail_mean_hz'],diagnosis=d['state'],period_ms=d.get('period_ms'),return_cv=d.get('return_cv'),relative_drift=d['relative_drift'])
        np.savez_compressed(out/f'fromcycle_D{D:.4f}.npz',time_ms=t,global_E_hz=glob,field_E_hz=field.astype('float32'),D=D,final_state=e.y.get(),final_history=e.history.get());rows.append(row);log(row)
    write(out/'from_cycle.json',dict(rows=rows,orbit=a.orbit))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('orbit');p.add_argument('--D',type=float,nargs='+',default=[.22,.25,.3]);p.add_argument('--duration',type=float,default=6000.);p.add_argument('--device',type=int,default=0);main(p.parse_args())
