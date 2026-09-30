"""Long autonomous trajectory using the unchanged Heun kernels in a CUDA graph.

The graph period is a common multiple of the delay ring depth and 1-ms
sampling interval. Repeated launches therefore preserve every physical delay.
An independent ordinary-step comparison is required before production use.
"""
from rate_field import *
from math import lcm
import argparse


class GraphIntegrator:
    def __init__(self,s,J,dt,device,initial=None,history=None):
        self.engine=e=RateIntegrator(s,J,dt,initial=initial,history=history,device=device);cp=e.cp;self.cp=cp
        self.sample=round(1/dt);self.steps=lcm(e.depth,self.sample)
        self.samples=self.steps//self.sample;self.output=cp.empty((self.samples,s.P))
        self.acc=cp.RawKernel(r'''
        extern "C" __global__ void sample_rate(const double* hist,double* out,
            int P,int slot,int row,int first,double dt){
            int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=P)return;
            double v=hist[slot*P+g]*dt;
            if(first)out[row*P+g]=v;else out[row*P+g]+=v;
        }''','sample_rate',options=('--fmad=false',))
        y=e.y.copy();h=e.history.copy()
        e.step();self.record(0);cp.cuda.get_current_stream().synchronize()
        e.y[:]=y;e.history[:]=h;e.tick=0
        self.stream=cp.cuda.Stream(non_blocking=True);cp.cuda.get_current_stream().synchronize()
        with self.stream:
            self.stream.begin_capture()
            for i in range(self.steps):e.step();self.record(i)
            self.graph=self.stream.end_capture()
        self.elapsed_ms=0

    def record(self,i):
        e=self.engine
        self.acc(((e.s.P+127)//128,),(128,),
            (e.history,self.output,np.int32(e.s.P),np.int32((i+1)%e.depth),
             np.int32(i//self.sample),np.int32(i%self.sample==0),e.dt))

    def block(self):
        with self.stream:self.graph.launch(stream=self.stream)
        self.stream.synchronize();self.elapsed_ms+=self.samples
        return self.output.get()*1000


def verify(s,J,dt,device):
    graph=GraphIntegrator(s,J,dt,device);e=RateIntegrator(s,J,dt,device=device);cp=e.cp
    rows=[]
    for block in range(2):
        actual=graph.block();expected=[];acc=cp.zeros(s.P)
        for i in range(graph.steps):
            acc+=e.step()*dt
            if (i+1)%graph.sample==0:expected.append(acc.copy());acc.fill(0)
        expected=cp.stack(expected).get()*1000
        rows.append(dict(block=block,sample_max_error_Hz=float(abs(expected-actual).max()),
            local_state_max_error=float(abs(e.y.get()-graph.engine.y.get()).max()),
            complete_history_max_error=float(abs(e.history.get()-graph.engine.history.get()).max())))
    ok=all(max(r[k] for k in ['sample_max_error_Hz','local_state_max_error','complete_history_max_error'])<1e-10 for r in rows)
    result=dict(status='PASS' if ok else 'FAIL',J_EE_core=J,dt_ms=dt,rows=rows,
        block_steps=graph.steps,block_ms=graph.samples,
        method='Two successive graph launches versus ordinary RateIntegrator.step, including all local states, full history and all 1-ms group averages.')
    dest=RATE_OUT/'periodic_completion';write(dest/f'long_graph_check_dt{dt:g}.json',result)
    print('GRAPH CHECK',result,flush=True);assert ok


def simulate(s,J,duration,dt,label,device,initial_state=None):
    check=read(RATE_OUT/'periodic_completion'/f'long_graph_check_dt{dt:g}.json');assert check['status']=='PASS'
    dest=RATE_OUT/'runs'/label/f'J{J:.7f}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'result.json').exists():raise RuntimeError('Completed run exists; choose a new label')
    initial=history=None
    if initial_state:
        z=np.load(initial_state);assert abs(float(z['dt_ms'])-dt)<1e-12
        assert int(z['final_history_tick_modulo_depth'])==0
        assert read(Path(initial_state).parent/'contract.json')['J_EE_core']==J
        initial=z['final_state'];history=z['final_history']
    g=GraphIntegrator(s,J,dt,device,initial,history);e=g.engine;chunks=[];start=time.time()
    blocks=int(np.ceil(duration/g.samples));actual_duration=blocks*g.samples
    write(dest/'contract.json',dict(model='Same positive two-filter spatial rate DDE',model_source=str(Path(__file__).with_name('rate_field.py')),
        closure_source=str(RATE_OUT/'closure.json'),J_EE_core=J,dt_ms=dt,duration_ms=actual_duration,
        requested_duration_ms=duration,initial_state=str(initial_state) if initial_state else 'Zero rate, recurrent states and complete history',
        Z='fixed 1',M='Original dynamic rate-expectation adaptation, tau 1000 ms',
        spatial_cells=s.grid**2,rate_groups=s.P,local_states=9*s.P,
        sampled_noise=False,native_spikes_used=False,readout='Spatially weighted firing rate; not SEEG voltage',
        graph_check=str(RATE_OUT/'periodic_completion'/f'long_graph_check_dt{dt:g}.json')))
    for k in range(blocks):
        chunks.append(g.block().astype('float32'))
        if k%10==0 or k+1==blocks:
            assert bool(e.cp.isfinite(e.y).all()) and float(e.y[:2].min())>=-1e-12
            write(dest/'status.json',dict(status='RUNNING',time_ms=g.elapsed_ms,total_ms=actual_duration,pid=os.getpid(),seconds=time.time()-start))
            print('LONG RATE',J,g.elapsed_ms,'/',actual_duration,'ms',round(time.time()-start,1),'s',flush=True)
    activity=np.concatenate(chunks);del chunks
    size=s.geo['group_size'];cell=s.geo['group_cell'];regional=np.array([
        np.average(activity[:,s.E&(s.geo['group_region']==k)],axis=1,weights=size[s.E&(s.geo['group_region']==k)]) for k in range(3)]).T
    counts=np.bincount(cell[s.E],weights=size[s.E],minlength=s.grid**2)
    weights=np.zeros((s.P,s.grid**2))
    for p in np.flatnonzero(s.E):weights[p,cell[p]]=size[p]/max(counts[cell[p]],1)
    field=(activity@weights).astype('float32');contact=activity@s.geo['contact_rate_weights']
    geo=np.load(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz')
    np.savez_compressed(dest/'trajectory.npz',time_ms=np.arange(len(activity))+1,group_rate_hz=activity,
        regional_rates_hz=regional,field_E_hz=field,contact_rate_hz=contact,
        contact_names=geo['contact_names'],final_state=e.y.get(),final_history=e.history.get(),
        final_history_tick_modulo_depth=0,dt_ms=dt)
    result=dict(status='COMPLETE',J_EE_core=J,duration_ms=actual_duration,dt_ms=dt,seconds=time.time()-start,
        trajectory=str(dest/'trajectory.npz'),meaning='Finite deterministic trajectory; no attractor class inferred from completion.')
    write(dest/'result.json',result);write(dest/'status.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--J',type=float,default=.946);p.add_argument('--duration',type=int,default=200000)
    p.add_argument('--dt',type=float,default=.1);p.add_argument('--label',default='periodic_gap_long')
    p.add_argument('--device',type=int,default=1);p.add_argument('--verify',action='store_true')
    p.add_argument('--initial-state',help='Same-J, same-dt nine-state plus complete-delay checkpoint; no external forcing')
    a=p.parse_args();s=RateField()
    if a.verify:verify(s,a.J,a.dt,a.device)
    else:simulate(s,a.J,a.duration,a.dt,a.label,a.device,a.initial_state)
