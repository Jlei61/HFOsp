"""Teacher-forced input closure audit with actual native population counts.

No rate is generated here. Native counts and external intensities are inputs;
this diagnostic cannot count as autonomous SNN correspondence.
"""
from common import OUT, ROOT, np, read, write, model
from scipy import sparse
from scipy.linalg import expm
from scipy.special import ndtr
from numba import njit
from pathlib import Path
import argparse, json

DEST = OUT / 'native_input_bridge'
RUN = DEST / 'runs/native_t8000_inputs_observe'


@njit
def filter_inputs(arr, drive, tm, jext, rise, decay, area, initial, covA, covB):
    T, _, P = arr.shape
    # continuous group means(2), discrete group means(2), full covariances(2),
    # private covariance(2); current variances, not effective static sigmas.
    result = np.empty((T, 8, P))
    cont = initial[:4].copy(); disc = cont.copy()
    cov = initial[4:].reshape(2, 3, P).copy(); private = cov.copy()
    for t in range(T):
        for g in range(P):
            for c in range(2):
                ext = jext[g]*drive[t,g] if c == 0 else 0.
                Jrate = arr[t,c,g]+ext
                force = tm[g]*area[c]*Jrate
                a = np.exp(-.1/rise[c]); d = np.exp(-.1/decay[c]); b = rise[c]/(rise[c]-decay[c])*(a-d)
                q, i = cont[2*c,g], cont[2*c+1,g]
                cont[2*c,g] = a*q+(1-a)*force
                cont[2*c+1,g] = b*q+d*i+(1-d-b)*force
                disc[2*c,g] = a*disc[2*c,g]+tm[g]/rise[c]*Jrate*.1
                disc[2*c+1,g] = d*disc[2*c+1,g]+(1-d)*disc[2*c,g]
                result[t,c,g] = cont[2*c+1,g]
                result[t,2+c,g] = disc[2*c+1,g]
                # Same continuous raw shot-noise covariance used by fixed rate.
                for part in range(2):
                    old = cov[c,:,g].copy() if part == 0 else private[c,:,g].copy()
                    extvar = jext[g]**2*drive[t,g] if c == 0 else 0.
                    varforcing = tm[g]*area[c]**2*(arr[t,2+2*part+c,g]+extvar)
                    new = covA[c]@old+covB[c]*tm[g]*varforcing
                    for k in range(3):
                        if part == 0: cov[c,k,g] = new[k]
                        else: private[c,k,g] = new[k]
                    result[t,4+2*part+c,g] = new[2]
    return result


def arrivals(s, spikes, device):
    import cupy as cp
    cp.cuda.Device(device).use()
    T = len(spikes); P = s.P; depth = T+s.prep['max_delay_steps']+2
    history = cp.zeros((depth,P)); history[1:T+1] = cp.asarray(spikes / s.sizes[None,:] / .1)
    # Six operators use identical lag/group column indexing.
    matrices = [sparse.load_npz(s.folder/(name+'.npz')).tocsr() for name in
        ['mean_ampa','mean_gaba','variance_ampa','variance_gaba']]
    private = OUT/'shared_variance_network_sensitivity/units_history_delay_covariance'
    matrices += [sparse.load_npz(private/f'private_{x}.npz').tocsr() for x in ['ampa','gaba']]
    kernel = cp.RawKernel(r'''
extern "C" __global__ void apply(const int* ptr,const int* col,const double* val,const double* hist,double* out,int tick,int depth,int P,int channel){
 int g=blockIdx.x,l=threadIdx.x;double v=0.;
 for(int i=ptr[g]+l;i<ptr[g+1];i+=blockDim.x){int delay=col[i]/P+1;int slot=(tick-delay+depth)%depth;v+=val[i]*hist[(long long)slot*P+col[i]%P];}
 __shared__ double buf[128];buf[l]=v;__syncthreads();for(int k=64;k>0;k/=2){if(l<k)buf[l]+=buf[l+k];__syncthreads();}
 if(l==0)out[channel*P+g]=buf[0];
}''', 'apply', options=('--fmad=false',))
    ops = [(cp.asarray(m.indptr.astype('i4')),cp.asarray(m.indices.astype('i4')),cp.asarray(m.data)) for m in matrices]
    output = np.empty((T,6,P)); block = cp.empty((1000,6,P)); checks=[]
    for start in range(0,T,1000):
        n = min(1000,T-start)
        for k in range(n):
            for ch, op in enumerate(ops): kernel((P,),(128,),(*op,history,block[k],np.int32(start+k+1),np.int32(depth),np.int32(P),np.int32(ch)))
        output[start:start+n] = block[:n].get()
        print('INPUT RECONSTRUCTION',start+n,'/',T,flush=True)
    # Direct CPU sparse calculation, including zero-before-start convention.
    h = np.zeros((depth,P)); h[1:T+1] = spikes/s.sizes/.1
    for k in [0, s.prep['max_delay_steps']+10, T-1]:
        flat=h[(k+1-np.arange(1,s.prep['max_delay_steps']+1))%depth].ravel()
        expected=np.array([m@flat for m in matrices]); error=float(abs(expected-output[k]).max())
        assert error < 1e-8
        checks.append(dict(index=k,maximum_abs_error=error))
    write(DEST/'operator_check.json',dict(status='PASS',rows=checks,
        history='Actual group counts / original N /0.1ms; unknown pre8s history zero; primary starts9s after warmup.',
        mean_operator='Same fixed native physical weight projection',
        covariance='Full Poisson squared weights and previously fixed stationary private subtraction both retained.'))
    return output


def reconstruct(device):
    assert read(DEST/'replay_audit.json')['status']=='PASS'
    assert not (DEST/'reconstructed_moments.npz').exists()
    s=model(); chunks=sorted((RUN/'inputs').glob('*.npz'))
    counts=[];drive=[];times=[]
    for f in chunks:
        with np.load(f) as z:
            counts.append(z['spikes']);drive.append(z['external_rate_per_ms']);times.append(z['time_ms'])
            with np.load(RUN/'chunks'/f.name) as old:
                original=np.array([z['spikes'][:,s.E].sum(1),z['spikes'][:,~s.E].sum(1)]).T
                assert np.array_equal(original,old['population_0p1ms'])
                restrict=sparse.csr_matrix((np.ones(int(s.E.sum())),(s.geo['group_cell'][s.E],np.flatnonzero(s.E))),shape=(400,s.P))
                assert np.array_equal((restrict@z['spikes'].T).T,old['field_0p1ms'])
    counts=np.concatenate(counts);drive=np.concatenate(drive);times=np.concatenate(times)
    arr=arrivals(s,counts,device);initial=np.load(DEST/'initial_moments.npz');states=[];covs=[]
    for c in ['ampa','gaba']:
        q,i=initial[c+'_q'],initial[c+'_i'];states.extend([q,i])
        covs.extend([initial[c+'_q2']-q*q,initial[c+'_qi']-q*i,initial[c+'_i2']-i*i])
    init=np.array(states+covs);As=[];Bs=[]
    for tr,td in zip(s.rise,s.decay):
        G=np.array([[-2/tr,0,0],[1/td,-1/tr-1/td,0],[0,2/td,-2/td]])
        a=expm(G*.1);b=np.linalg.solve(G,(a-np.eye(3))@np.array([1/tr**2,0,0]))
        As.append(a);Bs.append(b)
    observed=filter_inputs(arr,drive,s.tm,s.jext,s.rise,s.decay,s.area,init,np.array(As),np.array(Bs))
    assert np.isfinite(observed).all() and observed[:,4:].min()>-1e-10
    # Independent local covariance normalization against existing implementation.
    from refractory_rate_response import covariance_matrices
    checks=[]
    for pop,tm in [('E',20.),('I',10.)]:
        a,b,_,_,_=covariance_matrices(pop,.1)
        err=max(float(abs(a-np.array(As)).max()),float(abs(b-np.array(Bs)*tm).max()))
        assert err<1e-11;checks.append(dict(pop=pop,error=err))
    np.savez_compressed(DEST/'reconstructed_moments.npz',time_ms=times,moments=observed,
        names=np.array(['ampa_mean_cont','gaba_mean_cont','ampa_mean_native_step','gaba_mean_native_step',
                        'ampa_var_full','gaba_var_full','ampa_var_private','gaba_var_private']),
        covariance_init='Observed within-group covariance at8s; full/private identical initial transient, excluded by1s preparation')
    selected=read(DEST/'contract.json')['selected_groups']
    raw=np.empty((len(times),4,len(selected)))
    for part in range(2):
        for c in range(2):
            force=arr[:,2+2*part+c,selected]
            if c==0:force=force+s.jext[selected]**2*drive[:,selected]
            raw[:,2*part+c]=s.tm[selected]*s.area[c]**2*force
    np.savez_compressed(DEST/'selected_raw_variance_forcing.npz',time_ms=times,groups=selected,
        raw_variances=raw,names=['ampa_full','gaba_full','ampa_private','gaba_private'],
        initial_covariances=init[4:,selected])
    write(DEST/'reconstruction_qa.json',dict(status='PASS',group_count_totals_bitwise=True,spatial_count_fields_bitwise=True,
        covariance_normalization=checks,scope='Teacher-forced inputs only; privatePoisson covariance is an assumption, not measured total spread.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0)
    reconstruct(p.parse_args().device)
