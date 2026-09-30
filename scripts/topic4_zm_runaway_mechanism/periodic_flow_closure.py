"""Independently integrate one period from a spectral BVP's full state/history.

This separates nonlinear orbit closure from Floquet-phase propagation failures.
No root, stability or bifurcation is accepted from these diagnostics alone.
"""
from endpoint_runs import *
from streaming_periodic import StreamPeriodic


class StepBlock:
    def __init__(self,e,n):
        cp=e.cp;p=e.s.P;self.e=e
        self.shift=cp.empty_like(e.history)
        rotate=cp.RawKernel(r'''
extern "C" __global__ void rotate(const double* h,double* out,int P,int depth,int shift){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<P*depth)out[i]=h[(((i/P)+shift)%depth)*P+i%P];
}''','rotate')
        state=e.y.copy();hist=e.history.copy();e.step()
        rotate(((p*e.depth+127)//128,),(128,),
               (e.history,self.shift,np.int32(p),np.int32(e.depth),np.int32(0)))
        e.y[:]=state;e.history[:]=hist;e.tick=0
        cp.cuda.get_current_stream().synchronize();self.stream=cp.cuda.Stream(non_blocking=True)
        with self.stream:
            self.stream.begin_capture()
            for _ in range(n):e.step()
            rotate(((p*e.depth+127)//128,),(128,),
                   (e.history,self.shift,np.int32(p),np.int32(e.depth),np.int32(n%e.depth)))
            cp.copyto(e.history,self.shift);self.graph=self.stream.end_capture()
        e.tick=0

    def run(self):
        with self.stream:self.graph.launch(stream=self.stream)
        self.stream.synchronize()


def fourier_value(values,t,T):
    n=len(values);c=np.fft.rfft(values,axis=0)/n;c[1:]*=2
    if n%2==0:c[-1]*=.5
    phase=np.exp(2j*np.pi*np.asarray(t).reshape(-1,1)*np.arange(len(c))[None]/T)
    return (phase@c).real


def main(a):
    s=model();{'native':attach_native_path,'fine':attach_fine_rate_entry_path}[a.family](s)
    z=np.load(a.orbit);assert float(z['residual'])<2e-8
    sol=dict(r=z['r'],T=float(z['T']),D=float(z['D']));s.set_D(sol['D'])
    assert np.max(abs(s.Z-z['Z']))<1e-12
    o=StreamPeriodic(s,len(sol['r']),a.device);o.cache_mean_operators=False
    Y,_=orbit_states(o,sol,len(sol['r']));cp=o.cp
    nsteps=[round(sol['T']/dt) for dt in a.dt]
    offsets=np.array(nsteps)*a.dt-sol['T']
    expected=np.empty((len(a.dt),14,s.P))
    for k in range(14):expected[:,k]=fourier_value(Y[:-1,k],offsets,sol['T'])
    scale=np.maximum(np.std(Y[:-1],axis=0),1e-8)
    initial=Y[0].copy();del Y,o;cp.get_default_memory_pool().free_all_blocks()
    rows=[]
    for index,dt in enumerate(a.dt):
        depth=round(s.delays[-1]/dt)+1
        history=np.empty((depth,s.P))
        history[(-np.arange(depth))%depth]=fourier_value(sol['r'],-np.arange(depth)*dt,sol['T'])
        e=EndpointIntegrator(s,dt=dt,initial=initial,history=history,dynamic_z=False,device=a.device)
        block=StepBlock(e,128);rest=nsteps[index]%128
        finalblock=StepBlock(e,rest) if rest else None
        for _ in range(nsteps[index]//128):block.run()
        if finalblock:finalblock.run()
        final=e.y.get();difference=(final-expected[index])/scale;difference[11]=0
        actual_r=s.output(final);reference_r=s.output(expected[index])
        rate_error=(actual_r-reference_r)*1000
        row=dict(dt_ms=dt,steps=nsteps[index],actual_elapsed_ms=nsteps[index]*dt,
            offset_from_exact_period_ms=float(offsets[index]),
            E_rate_weighted_rms_error_hz=float(np.sqrt(rate_error[s.E]**2@s.mean_weights)),
            E_global_rate_error_hz=float(rate_error[s.E]@s.mean_weights),
            state_component_scaled_rms=np.sqrt(np.mean(difference**2,axis=1)).tolist(),
            final_global_E_hz=s.global_rate(actual_r),expected_global_E_hz=s.global_rate(reference_r),
            initial_output_vs_spectral_rate_max_hz=float(abs(s.output(initial)-sol['r'][0]).max()*1000))
        rows.append(row);log('NONLINEAR PERIOD CLOSURE',row)
        write(OUT/'periodic'/a.label/'result.json',dict(status='RUNNING',source=a.orbit,rows=rows))
        del e,block,finalblock;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    write(OUT/'periodic'/a.label/'result.json',dict(status='DIAGNOSTIC_COMPLETE',source=a.orbit,rows=rows,
        Z='held',M='dynamic',method='Endpoint Heun from complete Fourier state/history; expected endpoint at the actual rounded integration time',
        scope='Forward-flow closure diagnostic; no asymptotic stability or bifurcation classification'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('orbit');p.add_argument('--family',choices=['native','fine'],default='native')
    p.add_argument('--dt',nargs='+',type=float,default=[.05,.025,.0125]);p.add_argument('--device',type=int,default=0)
    p.add_argument('--label',required=True);main(p.parse_args())
