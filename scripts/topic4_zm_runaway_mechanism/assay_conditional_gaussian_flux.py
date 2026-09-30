"""Measure the Gaussian threshold-flux assumption on actual LIF states.

Readout only: moments and spikes of the reference LIF are never fed into a
candidate autonomous predictor. Paired noise paths remain the sampling unit.
"""
from common import OUT, np, read, write, log
from lif_mc import condition, run
from joint_voltage_current_density import voltage_edges
from pathlib import Path
from datetime import datetime
import argparse
import cupy as cp

DEST = OUT / 'conditional_gaussian_flux_diagnostic'

CODE = r'''
#include <curand_kernel.h>
extern "C" __global__ void observe(const double* p, const double* edges,
 double* out, int R, int B, int steps, int burn, unsigned long long seed){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=R)return;
 curandStatePhilox4_32_10_t rng;curand_init(seed,(unsigned long long)k,0,&rng);
 double qa[2]={0,0},ia[2]={0,0},qg[2]={0,0},ig[2]={0,0},v[2]={p[21],p[21]};int ref[2]={0,0};
 for(int t=-burn;t<steps;t++){
  float4 z=curand_normal4(&rng);
  double na=p[11]*z.x,nb=p[12]*z.x+p[13]*z.y,nc=p[14]*z.z,nd=p[15]*z.z+p[16]*z.w;
  for(int side=0;side<2;side++){
   double o=t>=0?(side==0?p[4]:-p[4]):0.;double af=sqrt(1+o);
   ia[side]=p[7]*qa[side]+p[8]*ia[side]+af*nb;qa[side]=p[6]*qa[side]+af*na;
   ig[side]=p[9]*qg[side]+p[10]*ig[side]+nd;qg[side]=p[17]*qg[side]+nc;
   double cur=p[0]+ia[side]-ig[side];ref[side]=max(0,ref[side]-1);
   if(ref[side]==0){
    double before=v[side];double proposal=p[18]*before+(1-p[18])*cur;
    bool fired=proposal>=p[1];
    if(t>=0){
     int lo=0,hi=B;while(lo+1<hi){int mid=(lo+hi)/2;if(before>=edges[mid])lo=mid;else hi=mid;}
     int b=lo;double d=proposal-p[1],d2=d*d;
     long long base=((long long)side*6*B+b)*R+k;
     out[base]+=1.;out[base+(long long)B*R]+=d;out[base+2LL*B*R]+=d2;
     out[base+3LL*B*R]+=d2*d;out[base+4LL*B*R]+=d2*d2;out[base+5LL*B*R]+=(double)fired;
    }
    if(fired){v[side]=p[21];ref[side]=(int)p[19];}else v[side]=proposal;
   }else v[side]=p[21];
  }
 }
}
'''


def register():
    DEST.mkdir(exist_ok=True); path = DEST/'contract.json'; assert not path.exists()
    pilot = read(OUT/'joint_voltage_current_density/independent_audit.json')
    assert pilot['status'] == 'JOINT_LOCAL_PILOT_FAIL'
    original = read(OUT/'conditional_density_linear_response_contract.json')['cases']
    cases = [original[k] for k in [56, 62]]
    assert all(c['channel']==1 and c['frequency_hz']==0 and c['dt_ms']==.05 for c in cases)
    write(path, dict(created_local=datetime.now().astimezone().isoformat(),
          question='At the original E/I mode workpoints, does a Gaussian distribution of the proposed membrane voltage within each pre-step voltage bin reproduce threshold flux and its AMPA-variance gain when supplied the actual LIF moments?',
          selection='The E and I AMPA-variance DC pair: E passed after moment-preserving transport, I failed well beyond reference SEM. Diagnostic selection after seeing errors, not a blind model validation.',
          cases=cases, replicates=8192, duration_ms=4000., burn_ms=1000., dt_ms=.05,
          voltage_nodes=128, seed=920033, variance_relative_amplitude=.025,
          outputs='Per independent path, per pre-step V bin: eligible-step count, four raw moments of proposed V minus threshold, actual spike count. Both paired sides use identical innovations. All recording steps included.',
          checks='Reference-lif kernel per-path spike counts must be exactly identical. Refractory denominator remains all steps. Bootstrap resamples32blocks of256independent noise paths, paired across input sides; time steps are not independent replicates.',
          budget='Two workpoints times paired +/-; one observer and one exact-count reference replay per workpoint. No parameter fit, network or bifurcation launch.',
          interpretation='Actual-state Gaussian flux is a diagnostic reconstruction, not an autonomous response. Failure implicates the Gaussian flux assumption; passing only excludes a large instantaneous flux error and does not validate moment evolution or reset-conditioned distributions.'))


def main(device):
    c = read(DEST/'contract.json'); cp.cuda.Device(device).use()
    kernel = cp.RawKernel(CODE, 'observe', options=('--fmad=false',))
    R = c['replicates']; dt=c['dt_ms']; steps=round(c['duration_ms']/dt); burn=round(c['burn_ms']/dt)
    rows=[]
    for case in c['cases']:
        q=case['workpoint']; pop=q['pop']; prefix=DEST/pop
        assert not prefix.with_suffix('.json').exists()
        p=condition(q['mu'],q['theta'],q['ve'],q['vi'],pop,
                    amplitude=c['variance_relative_amplitude'],channel=1,dt=dt)
        edges=voltage_edges(q['theta'],11.,c['voltage_nodes']); B=len(edges)-1
        output=cp.zeros((2,6,B,R),dtype=cp.float64)
        log('GAUSSIAN FLUX OBSERVER',pop,R,B)
        kernel(((R+127)//128,),(128,),(cp.asarray(p),cp.asarray(edges),output,np.int32(R),np.int32(B),np.int32(steps),np.int32(burn),np.uint64(c['seed'])))
        stats=output.get(); del output
        # Same exact native update, independently implemented in the frozen
        # assay kernel, with no moment accumulator or voltage binning.
        ref=run([p],R,c['duration_ms'],c['burn_ms'],c['seed'],crn=True,dt=dt,device=device)[0]
        observed=stats[:,5].sum(axis=1)
        assert np.array_equal(observed[0],ref[:,2]) and np.array_equal(observed[1],ref[:,3]), 'Observer changed per-path spikes'
        np.save(prefix.with_suffix('.npy'),stats)
        np.savez_compressed(DEST/f'{pop}_reference.npz',raw=ref,pars=p,edges=edges)
        row=dict(population=pop,source_case_id=case['id'],workpoint=q,
                 per_path_spike_count_bitwise_equal=True,counts_per_side=observed.sum(1).tolist(),
                 mean_rate_hz=(observed.mean(1)/c['duration_ms']*1000).tolist())
        write(prefix.with_suffix('.json'),row);rows.append(row);log('GAUSSIAN FLUX OBSERVER COMPLETE',row)
    write(DEST/'execution.json',dict(status='COMPLETE',rows=rows,model_promoted=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--register',action='store_true');parser.add_argument('--device',type=int,default=0)
    a=parser.parse_args();register() if a.register else main(a.device)
