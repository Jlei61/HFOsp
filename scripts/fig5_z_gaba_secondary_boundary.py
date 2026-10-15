"""Follow the independently detected slow oscillatory instability in base 2."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import fig5_z_gaba_phase_map as p
import numpy as np


def run(anchor_index=0, output_name='secondary_boundary'):
    cid=p.b.base.rec.IDS[1];out=p.DATA/cid;seeds=p.read(out/'seeds.json')['rows']
    rates=np.load(out/'seeds.npz')['rates'];f=p.b.get_family(cid,'tau_gaba',True)
    anchor=seeds[anchor_index];f.z_anchor=anchor['lambda'];c=p.DelayCharacteristic(f,rates[anchor_index])
    full_source=out/f'spectrum_{anchor_index:03d}_60.0000.json'
    full=p.read(full_source);mode=full['modes'][0]
    initial=dict(mode,parameter=60/f.base.tau_gaba_ms)
    if abs(c.small_eigenvalue(mode['growth_per_ms']+2j*np.pi*mode['frequency_hz']/1000,initial['parameter']))>1e-6:raise RuntimeError('secondary full-map mode disagrees')
    last=initial;discovery=[];cross=None
    for tau in np.arange(59.,31.,-1):
        now,hist=p.advance(c,last,tau/f.base.tau_gaba_ms,.02);discovery.extend(hist)
        if now is None:raise RuntimeError('secondary mode lost')
        if now['growth_per_ms']*last['growth_per_ms']<0:
            cross=c.crossing(now['frequency_hz'],now['parameter']);break
        last=now
    if cross is None or not cross['converged']:raise RuntimeError('secondary crossing not located')
    results=[];steps=[]
    for direction in [-1,1]:
        dose=anchor['lambda'];x=rates[anchor_index].copy();previous=cross
        for seed in sorted([s for s in seeds if (s['lambda']-dose)*direction>=-1e-9],key=lambda r:r['lambda'],reverse=direction<0):
            target=seed['lambda']
            while abs(dose-target)>1e-10:
                dose+=np.sign(target-dose)*min(.005,abs(target-dose));f.z_anchor=dose
                x,ok,err=f.solve(1.,x)
                if not ok:raise RuntimeError('secondary equilibrium continuation failed')
                c=p.DelayCharacteristic(f,x);now=c.crossing(previous['frequency_hz'],previous['scale'])
                if not now['converged'] or abs(now['frequency_hz']-previous['frequency_hz'])>1 or abs(now['scale']-previous['scale'])>.2:
                    p.write(out/(output_name+'.json'),dict(results=results,discovery=discovery,continuation=steps,
                        full_matrix_source=str(full_source),status='CONTINUATION_INCOMPLETE',
                        failure=dict(attempted_lambda=dose,target_lambda=target,reason='ROOT_OR_MODE_CONTINUITY_CHECK_FAILED')))
                    return results
                previous=now;steps.append(dict(**{'lambda':dose},**now))
            if any(abs(r['lambda']-target)<1e-9 for r in results):continue
            f.z_anchor=target;c=p.DelayCharacteristic(f,x)
            left=c.follow_mode(previous['scale']-.005,previous['frequency_hz'])
            right=c.follow_mode(previous['scale']+.005,previous['frequency_hz'])
            native=dict(previous,tau_gaba_ms=previous['scale']*f.base.tau_gaba_ms,
                transverse=bool(left['converged'] and right['converged'] and left['growth_per_ms']*right['growth_per_ms']<0),
                below=left,above=right)
            lim=previous;chain=[]
            for dt in [.09,.08,.07,.06,.05,.04,.03,.025,.02,.01,0.]:
                rr=c.crossing(lim['frequency_hz'],lim['scale'],dt)
                if not rr['converged'] or abs(rr['frequency_hz']-lim['frequency_hz'])>1:break
                lim=rr;chain.append(dict(rr,tau_gaba_ms=rr['scale']*f.base.tau_gaba_ms))
            record=dict(index=seed['index'],**{'lambda':target},native=native,dt_chain=chain,
                continuous_verified=bool(chain and chain[-1]['dt_ms']==0.))
            results.append(record);print('secondary',target,native['tau_gaba_ms'],native['frequency_hz'],flush=True)
            p.write(out/(output_name+'.json'),dict(results=results,discovery=discovery,continuation=steps,full_matrix_source=str(full_source),status='IN_PROGRESS'))
    saved=p.read(out/(output_name+'.json'));saved['status']='REQUESTED_ROWS_COMPLETE';p.write(out/(output_name+'.json'),saved)
    return results


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--anchor-index',type=int,default=0);parser.add_argument('--output-name',default='secondary_boundary');args=parser.parse_args()
    run(args.anchor_index,args.output_name)
