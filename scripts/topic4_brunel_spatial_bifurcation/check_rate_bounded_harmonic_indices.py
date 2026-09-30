"""Check bounded CSR index storage against the full-bank spatial equations."""
from complete_rate_positive_stability import *
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0);a=p.parse_args()
    while float(subprocess.check_output(['nvidia-smi','-i',str(a.device),
        '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))<2.5*1024:
        time.sleep(30)
    path=PERIODIC_OUT/'orbits/LPC_resonance_upper_N512.npz';z=np.load(path)
    s=RateField();N=len(z['r']);T=float(z['T']);J=float(z['J']);rng=np.random.default_rng(61720)
    vector=rng.normal(size=(N//2+1)*s.P)+1j*rng.normal(size=(N//2+1)*s.P)
    reference=[];indices=[];errors=[]
    for capacity in [None,64]:
        o=Periodic(s,N,a.device,harmonic_capacity=capacity);cp=o.cp
        o.low_memory=True;o.harmonic_chunk_size=64;o.derivative_chunk_size=64
        kernels=o.kernels(T,J)
        result=[]
        for bank in kernels[:3]:
            result.extend([(op@cp.asarray(vector)).get() for op in bank if op is not None])
        result.extend([o.moments(cp.asarray(z['r']),kernels,mode).get() for mode in ['normal','T','J']])
        indices.append(sum(index.nbytes+ptr.nbytes for d,mask,index,ptr in o.raw))
        if capacity is None:reference=result
        else:
            errors=[float(np.linalg.norm(v-w)/max(np.linalg.norm(w),1e-15)) for v,w in zip(result,reference)]
            assert max(errors)<5e-12,errors
        o.cache=None;del kernels,o,result;release(a.device)
    # The same local/history variational flow must survive the storage change.
    reference={};flow_errors={};x=None
    for bounded in [False,True]:
        m=Monodromy(s,path,.1,a.device,stream_harmonics=True,bounded_harmonic_indices=bounded)
        if x is None:x=rng.normal(size=m.dim);x/=np.linalg.norm(x)
        current=dict(phase=m.phase_vector(),gains=m.gains.get(),flow=m.matvec(x))
        if not bounded:reference=current
        else:
            flow_errors={k:float(np.linalg.norm(current[k]-reference[k])/max(np.linalg.norm(reference[k]),1e-15)) for k in current}
            assert max(flow_errors.values())<1e-10,flow_errors
        del m,current;release(a.device)
    result=dict(status='PASS',orbit=str(path),N=N,spatial_groups=s.P,
        operator_relative_errors=errors,monodromy_relative_errors=flow_errors,
        original_index_bytes=indices[0],bounded_index_bytes=indices[1],
        scope='Compare all four delay operators and their period/J derivatives with full cached banks, then compare phase, every temporal gain and a full-history monodromy action. All harmonics, spatial groups, delays, equations, central differences and integration steps are unchanged; only repeated CSR index storage is bounded.')
    write(PERIODIC_OUT/'bounded_harmonic_index_monodromy_check.json',result)
    write(DEST/'qa/bounded_harmonic_index_monodromy_check.json',result)
    print('BOUNDED INDEX CHECK',result,flush=True)


if __name__=='__main__':main()
