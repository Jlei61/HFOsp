"""Check coarse-to-fine initial tangents against the unchanged fine BVP."""
from rate_periodic import *
import gc


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    s=RateField();rows=[]
    for label in ['LPC_B_stage3_turn1','LPC_B_stage3_turn2']:
        coarse=read(PERIODIC_OUT/f'{label}_N1024.json')
        fine=read(PERIODIC_OUT/f'{label}_N2048.json');z=np.load(fine['orbit']);r=z['r'];N=len(r)
        core='AB'.index(fine['coordinate'][5]);ww=s.geo['group_size']*s.E*(s.geo['group_region']==core);ww=ww/ww.sum()
        c=np.r_[np.tile(ww/N,N),0.,0.];y=np.r_[(r*1000).ravel(),np.log(float(z['T'])),float(z['J'])*1000]
        o=Periodic(s,N,a.device);o.low_memory=True;o.stream_harmonics=True
        o.harmonic_chunk_size=32;o.derivative_chunk_size=32;cp=o.cp;ref=cp.asarray(r)
        dr=cp.fft.irfft(2j*np.pi*cp.arange(o.K)[:,None]*cp.fft.rfft(ref,axis=0),n=N,axis=0)
        phase=dr/cp.sum(dr*dr)*.001
        _,A,_=o.evaluate(cp.asarray(y),ref,phase,float(z['J']),derivative=True,
                         arc=tuple(cp.asarray(v) for v in [y,c,np.ones_like(c)]))
        rhs=cp.zeros(len(y));rhs[-1]=1;phase_direction=cp.r_[(dr*1000).ravel(),0.,0.]
        response=A@phase_direction;residuals=[]
        for source in [coarse,fine]:
            raw=np.load(PERIODIC_OUT/f'{label}_tangent_N{source["N"]}.npz')['tangent']
            tangent=np.r_[resample(raw[:-2].reshape(source['N'],s.P),N,axis=0).ravel(),raw[-2:]]
            assert abs(c@tangent-1)<1e-5
            candidate=cp.asarray(tangent)
            candidate-=phase_direction*((A@candidate-rhs)[-2]/response[-2])
            residuals.append(float(cp.linalg.norm(A@candidate-rhs)))
        assert residuals[0]<1 and residuals[1]<1e-6,residuals
        rows.append(dict(label=label,source_N=1024,target_N=N,
            zero_guess_residual=1.,coarse_seed_residual=residuals[0],
            independently_solved_fine_tangent_residual=residuals[1]))
        print(rows[-1],flush=True)
        del o,A,ref,dr,phase,phase_direction,response,candidate;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    write(a.output,dict(status='PASS',rows=rows,
        scope='The resampled tangent is a residual-screened initial guess only. The current bordered solve and its acceptance tolerance remain unchanged; this does not certify any additional fold.'))


if __name__=='__main__':main()
