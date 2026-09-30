"""Compare the prespecified native near-turn meshes without inheriting stability."""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.signal import resample
from scipy.optimize import minimize_scalar

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918'


def phase_distance(x,y):
    """All-group L2 after one shared phase shift, with exact Fourier shifts."""
    assert len(y)%2==1
    x=resample(x,len(y),axis=0);n=len(y)
    a=np.fft.rfft(x,axis=0)/n;b=np.fft.rfft(y,axis=0)/n
    factor=np.r_[1.,np.full(len(a)-1,2.)];freq=np.arange(len(a))
    cross=np.sum(a*np.conj(b),axis=1)
    normx=float(np.sum(factor[:,None]*abs(a)**2))
    normy=float(np.sum(factor[:,None]*abs(b)**2))
    def distance(shift):
        return normx+normy-2*float(np.sum(factor*cross*np.exp(2j*np.pi*freq*shift)).real)
    start=float(np.argmax(np.fft.irfft(cross,n=8*n)))/(8*n)
    fit=minimize_scalar(distance,bounds=(start-2/n,start+2/n),method='bounded',
                        options={'xatol':1e-14})
    return dict(phase_aligned_relative_L2=float(np.sqrt(max(0,fit.fun)/normy)),
                phase_shift_cycles=float(fit.x%1),
                unaligned_relative_L2=float(np.linalg.norm(x-y)/np.linalg.norm(y)))


def sanity():
    def signal(t):
        return np.stack([2+np.sin(2*np.pi*t),1+.3*np.cos(6*np.pi*t)],axis=1)
    x=signal(np.arange(129)/129);y=signal(np.arange(257)/257+.037)
    q=phase_distance(x,y)
    assert q['unaligned_relative_L2']>.01 and q['phase_aligned_relative_L2']<1e-7,q
    # A real waveform change must remain after optimizing only one shared phase.
    changed=y.copy();changed[:,1]+=.1*np.cos(4*np.pi*np.arange(257)/257)
    assert phase_distance(x,changed)['phase_aligned_relative_L2']>.02
    return q


def main(contract_name='native_T2644_joint_refinement_contract.json',output_name='periodic/native_T2644_joint_orbit_check.json'):
    sanity()
    contract_path=OUT/contract_name
    contract=json.loads(contract_path.read_text());gates=contract['orbit_criteria']
    source=OUT/contract['source'];target=OUT/contract['target']
    z0=np.load(source);z1=np.load(target)
    summary=json.loads(target.with_suffix('.json').read_text())
    assert len(z0['r'])==contract['coarse']['N'] and len(z1['r'])==contract['fine']['N']
    v=dict(period_difference_ms=abs(float(z0['T'])-float(z1['T'])),
           D_difference=abs(float(z0['D'])-float(z1['D'])),
           max_Z_difference=float(np.max(abs(z0['Z']-z1['Z']))),
           residuals_hz=[float(z0['residual']),float(z1['residual'])],
           minimum_group_rates_hz=[float(z['r'].min()*1000) for z in [z0,z1]],
           **phase_distance(z0['r'],z1['r']))
    checks=dict(residual=max(v['residuals_hz'])<gates['both_max_residual_hz'],
                period=v['period_difference_ms']<gates['max_period_difference_ms'],
                D=v['D_difference']<gates['max_D_difference'],
                Z=v['max_Z_difference']<gates['max_Z_difference'],
                waveform=v['phase_aligned_relative_L2']<gates['max_phase_aligned_rate_relative_L2_difference'],
                prescribed_period=abs(summary['T_ms']-contract['period_ms'])<1e-8)
    q=dict(status='ORBIT_MESH_CHECK_PASS' if all(checks.values()) else 'ORBIT_MESH_CHECK_NOT_MET',
           sources=[str(source),str(target)],contract=str(contract_path),values=v,checks=checks,
           normalization='All-group rate L2; a single common time shift is optimized; separate region shifts are prohibited.',
           scope='Periodic orbit mesh agreement only. Phase invariance, derivatives, Floquet spectrum, branch connection and onset mechanism remain separate checks.')
    path=OUT/output_name
    path.write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--sanity-only',action='store_true')
    parser.add_argument('--contract',default='native_T2644_joint_refinement_contract.json')
    parser.add_argument('--output',default='periodic/native_T2644_joint_orbit_check.json')
    args=parser.parse_args()
    if args.sanity_only:print(sanity())
    else:main(args.contract,args.output)
