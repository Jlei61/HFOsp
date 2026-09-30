"""Validate the response-unit correction independently of network outcomes."""
from response_voltage_units import *
from transfer_spline import CUDA_DEVICE
from native_cycle_waveform_response import simulate
from lif_mc import condition
import argparse

DEST=OUT/'response_voltage_units'


def response(s,q,f,ch,corrected):
    pop=q['pop'];theta=np.array([q['theta']]);mu=np.array([q['mu']]);ve=np.array([q['ve']]);vi=np.array([q['vi']])
    table=VoltageScaledResponseTable(s.resp.tables[pop]) if corrected else s.resp.tables[pop]
    weights,_=table.evaluate(mu,ve,vi,theta);al,ae,ai,ee,ei=weights[:,0]
    g=s.spline[pop].evaluate(mu,ve,vi,theta);p=s.resp.poles[pop];w=2j*np.pi*f/1000
    if ch==0:return g['d_mu'][0]*1000*(al+(1-al)/(1+w*p['tau_s']))
    label='E' if ch==1 else 'I';a=ae if ch==1 else ai;eta=ee if ch==1 else ei
    return (g['d_ve' if ch==1 else 'd_vi'][0]*1000*(a+(1-a)/(1+w*p['tau_v'+label]))+
            g['d_mu'][0]*1000*eta*w*p['tau_c'+label]/(1+w*p['tau_c'+label]))/(1+w*s.tau[ch-1]/2)


def main(device):
    import cupy as cp
    cp.cuda.Device(device).use();DEST.mkdir(exist_ok=True)
    c=read(OUT/'response_voltage_units_contract.json');s=model();rng=np.random.default_rng(919783)
    results=[]
    kernel=cp.RawKernel(CUDA_DEVICE+CUDA_RESP_VOLTAGE_UNITS+r'''
extern "C" __global__ void evaluate_units(const double* tab,const double* inp,double* out,int n){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=n)return;
 double weights[5],grad[15];resp_weights(tab,inp[j],inp[n+j],inp[2*n+j],inp[3*n+j],weights,grad);
 for(int k=0;k<5;k++)out[k*n+j]=weights[k];for(int k=0;k<15;k++)out[(5+k)*n+j]=grad[k];
}
''','evaluate_units',options=('--fmad=false',))
    for pop in 'EI':
        n=32;theta=rng.uniform(14.2,18,n);sc=theta-11
        mu=11+sc*rng.uniform(-.5,3,n);ve=(sc*rng.uniform(.3,2,n))**2;vi=(sc*rng.uniform(.2,3,n))**2
        tab=VoltageScaledResponseTable(s.resp.tables[pop]);weights,grad=tab.evaluate(mu,ve,vi,theta)
        inputs=np.array([mu,ve,vi,theta]);gpu=cp.empty((20,n))
        kernel((1,),(64,),(cp.asarray(tab.device_block()),cp.asarray(inputs),gpu,np.int32(n)))
        truth=np.r_[weights,grad.reshape(15,n)];gpu=gpu.get()
        error=float(np.max(abs(gpu-truth)));assert np.allclose(gpu,truth,rtol=1e-10,atol=1e-11),error
        fd_errors=[]
        for col in range(3):
            h=1e-5*np.maximum(abs(inputs[col]),1);plus=inputs.copy();minus=inputs.copy()
            plus[col]+=h;minus[col]-=h
            fp=tab.evaluate(*plus)[0];fm=tab.evaluate(*minus)[0];fd=(fp-fm)/(2*h)
            fd_errors.append(float(np.linalg.norm(fd-grad[:,col])/max(np.linalg.norm(grad[:,col]),1e-9)))
        assert max(fd_errors)<1e-5,fd_errors
        # A normalized workpoint must have identical voltage-scaled responses.
        base=dict(pop=pop,theta=18.,mu=19.4,ve=49*.7**2,vi=49*1.4**2)
        scaled=dict(pop=pop,theta=14.5,mu=11+.5*(base['mu']-11),ve=.25*base['ve'],vi=.25*base['vi'])
        comparisons=[]
        for ch in range(3):
            for f in [6.,25.,60.]:
                power=1 if ch==0 else 2
                ref=response(s,base,f,ch,True)
                old=response(s,scaled,f,ch,False)*(.5**power)
                new=response(s,scaled,f,ch,True)*(.5**power)
                assert abs(new-ref)<1e-10*max(abs(ref),1), (pop,ch,f,new,ref)
                comparisons.append(dict(channel=ch,frequency_hz=f,original_relative_error=float(abs(old-ref)/max(abs(ref),1e-12)),corrected_relative_error=float(abs(new-ref)/max(abs(ref),1e-12))))
        results.append(dict(pop=pop,CPU_CUDA_max_error=error,finite_difference_relative_errors=fd_errors,scaling= comparisons))
    # Same colored-LIF law, nonlinear time-varying input, common noise paths.
    z=np.load(OUT/'native_cycle_waveform_response/prepared.npz');original=z['wave'][2]
    original_theta=read(OUT/'native_cycle_waveform_response/preparation.json')['groups'][2]['theta_mv']
    assert original_theta==18.
    wave=np.array([original,original.copy()]);wave[1,0]=11+.5*(wave[1,0]-11);wave[1,1:]*=.25
    pars=[condition(0,18,1,1,'E'),condition(0,14.5,1,1,'E')]
    T=float(z['T_ms']);counts=simulate(pars,wave,c['replicates'],T,.1,round(5*T/.1),round(10*T/.1),c['seed'],128,device)
    assert np.array_equal(counts[0],counts[1]),int(np.max(abs(counts[0].astype(int)-counts[1])))
    q=dict(status='UNIT_CORRECTION_AUDIT_PASS',reference_threshold_mv=18.,reset_mv=11.,rows=results,
           nonlinear_LIF_scaling_counts_bitwise=True,replicates=c['replicates'],statistical_unit='Independent noise path',
           scope='Unit correctness, CPU/GPU parity, derivatives and LIF voltage equivariance only; not a validated replacement network.')
    np.savez_compressed(DEST/'LIF_scaling_counts.npz',counts=counts)
    write(DEST/'implementation_audit.json',q);log('UNIT CORRECTION AUDIT',q)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0)
    main(p.parse_args().device)
