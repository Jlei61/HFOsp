"""Frozen-Z/carry-state experiments with canonical delay history and explicit Z transplant."""
from common import *
import argparse
from scipy.ndimage import uniform_filter1d


class GraphChunk:
    """Replay a fixed chunk and rotate its history back to tick zero after each call."""
    def __init__(self, engine, milliseconds=50):
        self.e=engine;cp=engine.cp;P=engine.s.P
        assert engine.tick == 0 and not engine.noise and not engine.drive_on
        self.n=int(round(milliseconds/engine.dt));self.sample=int(round(1/engine.dt))
        assert self.n % self.sample == 0
        self.rates=cp.zeros((self.n//self.sample,P));self.acc=cp.zeros(P)
        self.shift=cp.empty_like(engine.history)
        self.record=cp.RawKernel(r'''
extern "C" __global__ void record(const double* history,double* acc,double* rates,
 int P,int slot,int step,int sample,double dt){
 int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=P)return;
 acc[g]+=history[slot*P+g]*dt;
 if(step%sample==0){rates[(step/sample-1)*P+g]=acc[g];acc[g]=0.;}
}''','record',options=('--fmad=false',))
        self.rotate=cp.RawKernel(r'''
extern "C" __global__ void rotate(const double* input,double* out,int P,int depth,int shift){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=P*depth)return;
 int row=i/P,g=i%P;out[i]=input[((row+shift)%depth)*P+g];
}''','rotate',options=('--fmad=false',))
        # Compile kernels outside capture, then restore the exact initial data.
        y=engine.y.copy();h=engine.history.copy();engine.step()
        self.record(((P+127)//128,),(128,),
            (engine.history,self.acc,self.rates,np.int32(P),np.int32(1),np.int32(1),np.int32(self.sample),engine.dt))
        self.rotate(((P*engine.depth+127)//128,),(128,),
            (engine.history,self.shift,np.int32(P),np.int32(engine.depth),np.int32(0)))
        engine.y[:]=y;engine.history[:]=h;engine.tick=0;self.acc.fill(0)
        cp.cuda.get_current_stream().synchronize()
        self.stream=cp.cuda.Stream(non_blocking=True)
        with self.stream:
            self.stream.begin_capture()
            for i in range(self.n):
                engine.step()
                self.record(((P+127)//128,),(128,),
                    (engine.history,self.acc,self.rates,np.int32(P),np.int32(engine.tick%engine.depth),
                     np.int32(i+1),np.int32(self.sample),engine.dt))
            self.rotate(((P*engine.depth+127)//128,),(128,),
                (engine.history,self.shift,np.int32(P),np.int32(engine.depth),np.int32(self.n%engine.depth)))
            cp.copyto(engine.history,self.shift)
            self.graph=self.stream.end_capture()
        engine.tick=0

    def run(self):
        with self.stream:
            self.graph.launch(stream=self.stream)
        self.stream.synchronize()
        return self.rates.get()*1000


def longest_true(x):
    d=np.diff(np.r_[0,np.asarray(x,dtype=int),0]);return int(max(np.flatnonzero(d==-1)-np.flatnonzero(d==1),default=0))


def summarize(R,s,tail_ms=3000):
    E=s.E;weights=s.sizes;cells=s.geo['group_cell']
    count=np.bincount(cells[E],weights=weights[E],minlength=s.grid*s.grid)
    projection=np.zeros((s.P,s.grid*s.grid))
    projection[np.flatnonzero(E),cells[E]]=weights[E]/np.maximum(count[cells[E]],1)
    field=R@projection
    global_rate=R[:,E]@s.mean_weights
    sm=uniform_filter1d(global_rate,10,mode='nearest');tail=slice(max(0,len(R)-tail_ms),len(R))
    window=sm[tail];occupancy=((field[tail]>50)@(count/count.sum())).mean()
    quiet=longest_true(window<5)
    edges=np.diff(np.r_[0,(window>=5).astype(int),0]);starts=np.flatnonzero(edges==1);stops=np.flatnonzero(edges==-1)
    events=[(int(a),int(b)) for a,b in zip(starts,stops)
            if a>0 and b<len(window) and b-a>=20 and window[a:b].max()>=20]
    if quiet<20:category='SUSTAINED_BROAD' if occupancy>=.75 else 'SUSTAINED_PARTIAL'
    elif events:category='SELF_LIMITED'
    else:category='LOW_OR_UNRESOLVED'
    result=dict(category=category,tail_ms=len(window),tail_mean_hz=float(window.mean()),
        tail_global_cv=float(window.std()/max(window.mean(),1e-12)),longest_quiet_ms=quiet,
        quiet_fraction=float((window<5).mean()),occupation_50Hz=float(occupancy),self_limited_events=events,
        fraction_global_above_200=float((window>=200).mean()),
        group_rate_temporal_relative_std=float(np.linalg.norm(R[tail].std(0))/max(np.linalg.norm(R[tail].mean(0)),1e-12)))
    return result,field,global_rate,count


def checkpoint_checks(device):
    s=model();ck=BASE/'runs/A4_det_meandrive/checkpoints'
    initial,history=checkpoint_initial(ck/'t7000ms.npz')
    target=np.load(ck/'t7250ms.npz')
    s.set_Z(initial[11]);e=Integrator(s,initial=initial,history=history,dynamic_z=True,device=device)
    graph=GraphChunk(e)
    for _ in range(5):graph.run()
    actual=e.y.get();hist=e.history.get();expected_history=np.roll(target['history'],-int(target['tick'])%len(hist),axis=0)
    result=dict(state_error=float(abs(actual-target['state']).max()),history_error=float(abs(hist-expected_history).max()),
                state_bitwise_equal=bool(np.array_equal(actual,target['state'])),
                source='A4_det_meandrive 7000 ms to 7250 ms; all 14 state channels, all history slots')
    assert result['state_error']<1e-9 and result['history_error']<1e-12,result
    # Independent ordinary stepping from the same complete initial condition.
    e2=Integrator(s,initial=initial,history=history,dynamic_z=True,device=device)
    for _ in range(2500):e2.step()
    y2=e2.y.get();h2=np.roll(e2.history.get(),-e2.tick%e2.depth,axis=0)
    result['graph_vs_ordinary_state_error']=float(abs(actual-y2).max())
    result['graph_vs_ordinary_history_error']=float(abs(hist-h2).max())
    assert result['graph_vs_ordinary_state_error']<1e-11
    assert result['graph_vs_ordinary_history_error']<1e-12
    zother=np.load(ck/'t9000ms.npz')['Z'];state_other,_=checkpoint_initial(ck/'t7000ms.npz',zother)
    assert np.array_equal(state_other[11],zother)
    assert np.array_equal(state_other[:11],initial[:11]) and np.array_equal(state_other[12:],initial[12:])
    result['Z_transplant_only_changes_state_channel_11']=True;result['status']='PASS'
    write(OUT/'checkpoint_and_graph_checks.json',result);log(result)


def fields(s,family):
    if family=='rate':
        ck=BASE/'runs/A4_det_meandrive/checkpoints'
        return [(t,np.load(ck/f't{t}ms.npz')['Z']) for t in [7000,7500,7700,7800,8000,8100,8250,8500,9000,10000]]
    z=np.load(BASE/'stage_b/z_fields.npz')
    return [(t,z[f'native_{t}']) for t in [8000,9000,9420,9870,10370]]


def run_condition(s,Z,label,duration,device,initial_path=None,dynamic_z=False,dynamic_m=True,dt=.1):
    out=OUT/'runs'/label;out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():return read(out/'result.json')
    s.set_Z(Z,source=label)
    if initial_path is None:initial,history=None,None
    else:initial,history=checkpoint_initial(initial_path,Z)
    e=Integrator(s,dt=dt,initial=initial,history=history,dynamic_z=dynamic_z,dynamic_m=dynamic_m,device=device)
    assert np.array_equal(e.y[11].get(),Z)
    if dt!=.1 and history is not None:
        source=np.load(initial_path)
        assert 'dt_ms' in source and abs(float(source['dt_ms'])-dt)<1e-12, 'History must have the requested sample spacing'
    start=time.time();graph=GraphChunk(e);records=[];zrec=[];mrec=[]
    write(out/'contract.json',dict(Z_source=label,initial=str(initial_path),history='canonical ring with checkpoint tick accounted for',
        Z_applied_inside_state=True,M='dynamic' if dynamic_m else 'held carried value',Z='dynamic' if dynamic_z else 'held',
        D_initial=float(1-Z[s.E]@s.mean_weights),duration_ms=duration,dt_ms=dt,model='frozen v3, same physical graph',
        rate_history_scheme=getattr(e,'history_scheme','Heun-step average labelled at endpoint (historical)')))
    for chunk in range(int(duration/50)):
        records.append(graph.run());zrec.append(e.y[11].get());mrec.append(e.y[10].get())
        if (chunk+1)%20==0:log(label,'t_ms',(chunk+1)*50,'wall',round(time.time()-start,1))
    R=np.concatenate(records);result,field,glob,count=summarize(R,s)
    yf=e.y.get();hf=e.history.get()
    result.update(label=label,status='COMPLETE',D_initial=float(1-Z[s.E]@s.mean_weights),
                  D_final=float(1-yf[11,s.E]@s.mean_weights),seconds=time.time()-start,
                  Z_actual_final_max_difference=float(abs(yf[11]-Z).max()),duration_ms=duration)
    if not dynamic_z:assert result['Z_actual_final_max_difference']==0
    if not dynamic_m:
        assert np.array_equal(yf[10],initial[10] if initial is not None else np.zeros(s.P))
    np.savez_compressed(out/'trajectory.npz',time_ms=np.arange(len(R))+1,group_rate_hz=R.astype('float32'),
        global_E_hz=glob,field_E_hz=field.astype('float32'),cell_counts=count,Z_source=Z,
        Z_every50ms=np.array(zrec),M_every50ms=np.array(mrec),final_state=yf,final_history=hf,final_tick=0,dt_ms=dt)
    write(out/'result.json',result);log('COMPLETE',label,result['category'],result['tail_mean_hz']);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--family',choices=['rate','native'],default='native')
    p.add_argument('--history',choices=['carried','zero'],default='carried');p.add_argument('--duration',type=int,default=12000)
    p.add_argument('--device',type=int,default=1);a=p.parse_args()
    if a.check:checkpoint_checks(a.device);return
    assert read(OUT/'checkpoint_and_graph_checks.json')['status']=='PASS'
    s=model();initial=BASE/'runs/A4_det_meandrive/checkpoints/t7000ms.npz' if a.history=='carried' else None
    write(OUT/f'jobs_{a.family}_{a.history}.json',dict(status='RUNNING',duration_ms=a.duration))
    results=[]
    for t,Z in fields(s,a.family):
        results.append(run_condition(s,Z,f'{a.family}_Z{t}_{a.history}',a.duration,a.device,initial))
        write(OUT/f'jobs_{a.family}_{a.history}.json',dict(status='RUNNING',completed=len(results),rows=results))
    write(OUT/f'jobs_{a.family}_{a.history}.json',dict(status='COMPLETE',completed=len(results),rows=results))


if __name__=='__main__':main()
