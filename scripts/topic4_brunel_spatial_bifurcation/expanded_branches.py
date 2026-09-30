"""Find and continue stationary branches over an expanded core-coupling range.

Multistart Newton guesses are numerical seeds, not simulated states. All
retained points satisfy the original spatial stationary equations.
"""
from common import *
from model import SpatialBrunel
import argparse

DEST=OUT/'expanded'

def main(args):
    s=SpatialBrunel(response='calibrated_full');dest=DEST/'stationary';dest.mkdir(parents=True,exist_ok=True)
    reg=s.geo['group_region'];branches=[];allrows=[]
    for J in np.round(np.arange(args.end,args.start-.001,-args.step),8):
        existing=[]
        guesses=[(f'continue_{i}',b['rates']) for i,b in enumerate(branches)]
        for label in ('low','A','B','AB'):
            for scale in ([.1,.35] if label!='low' else [0.]):
                r=np.full(s.P,.0001)
                for core in ('A','B'):
                    if core in label:r[s.E&(reg==('AB'.index(core)))]=scale
                r[~s.E]=.01 if scale else .0001
                guesses.append((f'{label}_{scale}',r))
        for label,r in guesses:
            rr,ok,tr=s.solve(float(J),r)
            if not ok:continue
            if any(np.linalg.norm(rr-q['rates'])/np.sqrt(s.P)<1e-6 for q in existing):continue
            rates=s.regional_rates(rr)
            state=('A' if rates[0]>20 else '')+('B' if rates[1]>20 else '') or 'low'
            name=f'J{J:.6f}_root{len(existing)}'
            info=dict(name=name,J_EE_core=J,rates_hz=rates,residual=float(abs(s.residual(rr,J)).max()),
                numerical_seed=label,high_rate_cores=state)
            np.savez_compressed(dest/f'{name}.npz',rates=rr,J=J)
            existing.append(dict(rates=rr,info=info));allrows.append(info)
            print(info,flush=True)
        branches=existing
        write(dest/'result.json',dict(status='RUNNING',rows=allrows,parameter_range=[args.start,args.end],
            spacing=args.step,meaning='Multistart stationary roots; absence of a converged solution is not proof of absence. Temporal stability computed separately.'))
    q=read(dest/'result.json');q['status']='COMPLETE';write(dest/'result.json',q)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--start',type=float,default=.4);p.add_argument('--end',type=float,default=2.0)
    p.add_argument('--step',type=float,default=.05);main(p.parse_args())
