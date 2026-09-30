"""Extract a fine-time cycle from the unchanged endpoint DDE integrator.

This supplies an independent orbit for variational checks when spectral
collocation is sensitive to narrow firing peaks. No periodicity is imposed on
the simulation: closure and consecutive periods are measured before acceptance.
"""
from endpoint_runs import *
from runner import GraphChunk
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq


class DenseChunk:
    def __init__(self,e,ms=25):
        cp=e.cp;self.e=e;self.n=round(ms/e.dt);P=e.s.P
        self.states=cp.empty((self.n,14,P));self.rates=cp.empty((self.n,P));self.shift=cp.empty_like(e.history)
        self.rotate=cp.RawKernel(r'''
extern "C" __global__ void rotate(const double* input,double* out,int P,int depth,int shift){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<P*depth)out[i]=input[(((i/P)+shift)%depth)*P+i%P];
}''','rotate')
        y=e.y.copy();h=e.history.copy();e.step()
        self.rotate(((P*e.depth+127)//128,),(128,),
            (e.history,self.shift,np.int32(P),np.int32(e.depth),np.int32(0)))
        cp.copyto(self.states[0],e.y);cp.copyto(self.rates[0],e.history[e.tick%e.depth])
        e.y[:]=y;e.history[:]=h;e.tick=0;cp.cuda.get_current_stream().synchronize()
        self.stream=cp.cuda.Stream(non_blocking=True)
        with self.stream:
            self.stream.begin_capture()
            for i in range(self.n):
                e.step();cp.copyto(self.states[i],e.y);cp.copyto(self.rates[i],e.history[e.tick%e.depth])
            self.rotate(((P*e.depth+127)//128,),(128,),
                (e.history,self.shift,np.int32(P),np.int32(e.depth),np.int32(self.n%e.depth)))
            cp.copyto(e.history,self.shift);self.graph=self.stream.end_capture()
        e.tick=0

    def run(self):
        with self.stream:self.graph.launch(stream=self.stream)
        self.stream.synchronize()
        return self.states.get(),self.rates.get()


def main(a):
    s=model();(attach_rate_entry_path if a.family=='rate' else attach_native_path)(s)
    z=np.load(a.trajectory);dt=float(z['dt_ms']);state,history=checkpoint_initial(a.trajectory)
    s.set_Z(state[11]);D=s.D
    e=EndpointIntegrator(s,dt=dt,initial=state,history=history,dynamic_z=False,device=a.device)
    if a.warmup:
        warm=GraphChunk(e)
        for k in range(a.warmup//50):
            warm.run()
            if (k+1)%20==0:log('WARMUP',(k+1)*50,'ms')
        del warm
        state=e.y.get();history=e.history.get()
    dense=DenseChunk(e);ys=[state[None]];rs=[s.output(state)[None]];start=time.time()
    if a.disk_record:
        raw=OUT/'physical_cycles'/f'{a.label}_raw';raw.mkdir(parents=True,exist_ok=True)
        size=round(a.duration/dt)+1
        Ydisk=np.lib.format.open_memmap(raw/'states.npy',mode='w+',dtype=np.float64,shape=(size,14,s.P))
        Rdisk=np.lib.format.open_memmap(raw/'rates.npy',mode='w+',dtype=np.float64,shape=(size,s.P))
        Ydisk[0]=state;Rdisk[0]=rs[0][0]
        write(raw/'recording.json',dict(status='RECORDING',dt_ms=dt,D=D,source=a.trajectory))
    for k in range(a.duration//25):
        Y,R=dense.run()
        if a.disk_record:
            lefti=1+k*dense.n;righti=lefti+dense.n
            Ydisk[lefti:righti]=Y;Rdisk[lefti:righti]=R
        else:ys.append(Y);rs.append(R)
        if k==0:
            ordinary=EndpointIntegrator(s,dt=dt,initial=state,history=history,dynamic_z=False,device=a.device)
            for j in range(dense.n):ordinary.step()
            expected=np.roll(ordinary.history.get(),-ordinary.tick%ordinary.depth,axis=0)
            assert np.array_equal(e.y.get(),ordinary.y.get())
            assert np.array_equal(e.history.get(),expected)
            del ordinary
            log('DENSE RECORD PARITY PASS',dense.n,'steps')
        if (k+1)%8==0:log('DENSE RECORD',(k+1)*25,'ms',round(time.time()-start,1),'wall seconds')
    if a.disk_record:
        Ydisk.flush();Rdisk.flush();Y=Ydisk;R=Rdisk
        np.savez(raw/'endpoint.npz',final_state=e.y.get(),final_history=e.history.get(),final_tick=0,dt_ms=dt)
        write(raw/'recording.json',dict(status='DENSE_RECORDING_COMPLETE',dt_ms=dt,D=D,source=a.trajectory,nodes=len(R)))
    else:Y=np.concatenate(ys);R=np.concatenate(rs)
    del ys,rs;t=np.arange(len(R))*dt
    global_r=R[:,s.E]@s.mean_weights*1000
    ix=np.flatnonzero((global_r[:-1]<a.section)&(global_r[1:]>=a.section));accepted=[]
    spline=CubicSpline(t,global_r)
    for i in ix:
        crossing=brentq(lambda u:spline(u)-a.section,t[i],t[i+1],xtol=1e-11)
        if not accepted or crossing-accepted[-1]>150:accepted.append(crossing)
    assert len(accepted)>=3,('not enough comparable cycles',accepted)
    left,right=accepted[-2:];T=right-left;periods=np.diff(accepted)
    # Only the measured final cycle is interpolated. Avoid forming a fourfold
    # cubic-coefficient copy of the entire dense multi-cycle state recording.
    window=(t>=left-5*dt)&(t<=right+5*dt)
    interp=CubicSpline
    if a.local_state:
        from local_cubic import LocalCubic
        assert read(OUT/'local_cubic_check.json')['status']=='PASS'
        interp=LocalCubic
    sy=interp(t[window],Y[window],axis=0);sr=interp(t[window],R[window],axis=0)
    rate_error=float(np.linalg.norm(sr(left)-sr(right))/np.linalg.norm(sr(left)))
    if a.disk_record:
        # Pairwise batch-combined Welford moments: same whole-record standard
        # deviation, without a recording-sized temporary subtraction array.
        count=0;mean=np.zeros((14,s.P));m2=np.zeros_like(mean)
        for j in range(0,len(Y),256):
            block=np.asarray(Y[j:j+256]);nb=len(block);bm=block.mean(0)
            b2=np.sum((block-bm)**2,axis=0);delta=bm-mean;total=count+nb
            m2+=b2+delta**2*(count*nb/total);mean+=delta*(nb/total);count=total
        std=np.sqrt(m2/count)
    else:std=np.std(Y,axis=0)
    state_scale=np.maximum(std,np.maximum(abs(sy(left)),abs(sy(right)))*1e-3+1e-8)
    closure=(sy(right)-sy(left))/state_scale;closure[11]=0.
    nr=int(np.ceil(T/dt));times=np.arange(nr+1)*T/nr
    state_cycle=sy(left+times);rate_cycle=sr(left+times)
    r=sr(left+np.arange(a.N)*T/a.N)
    dest=OUT/'physical_cycles';dest.mkdir(parents=True,exist_ok=True)
    file=dest/f'{a.label}.npz'
    np.savez(file,r=r,T=T,D=D,Z=state[11],state_cycle=state_cycle,rate_cycle=rate_cycle,
                        cycle_times_ms=times,source=a.trajectory,dt_ms=dt,
                        final_state=e.y.get(),final_history=e.history.get(),final_tick=0)
    row=dict(status='CYCLE_EXTRACTED_REQUIRES_VARIATIONAL_CHECK',source=a.trajectory,
        D=D,T_ms=T,successive_periods_ms=periods.tolist(),rate_closure_relative=rate_error,
        state_closure_scaled_rms=float(np.sqrt(np.mean(closure**2))),
        state_closure_scaled_max=float(abs(closure).max()),recording_dt_ms=dt,
        section_global_E_hz=a.section,N_rate=a.N,raw_cycle_nodes=nr+1,
        periodic_bvp_residual='NOT_COMPUTED',recording_graph_parity='BITWISE_PASS',
        warmup_ms=a.warmup,
        state_interpolation='local four-point cubic' if a.local_state else 'cubic spline',
        Z='held',M='dynamic',path=str(file))
    write(file.with_suffix('.json'),row);log('PHYSICAL CYCLE',row)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('trajectory');p.add_argument('--label',required=True)
    p.add_argument('--duration',type=int,default=1200);p.add_argument('--N',type=int,default=4097)
    p.add_argument('--warmup',type=int,default=0)
    p.add_argument('--local-state',action='store_true')
    p.add_argument('--disk-record',action='store_true')
    p.add_argument('--section',type=float,default=80);p.add_argument('--device',type=int,default=0)
    p.add_argument('--family',choices=['rate','native'],default='rate');main(p.parse_args())
