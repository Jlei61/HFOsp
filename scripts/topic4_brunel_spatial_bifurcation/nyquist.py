"""Full spatial characteristic determinant on an adaptively resolved contour.

Counts apply to the analytic shifted-bound response approximation. Empirical
Table-2 corrections depending on Im(lambda) alone are not analytic and cannot
be silently inserted into an argument-principle count.
"""
from common import *
from model import SpatialBrunel
from response import characteristic
from scipy.sparse.linalg import splu
import argparse

def parity(p):
    seen=np.zeros(len(p),bool);cycles=0
    for i in range(len(p)):
        if seen[i]:continue
        cycles+=1;j=i
        while not seen[j]:seen[j]=True;j=p[j]
    return (len(p)-cycles)%2

def main(args):
    s=SpatialBrunel(args.grid,response=args.response);suffix='_'+args.response if args.response!='shifted_white' else ''
    dest=OUT/args.label if getattr(args,'label',None) else OUT/f'g{args.grid}'/('nyquist'+suffix);dest.mkdir(exist_ok=True,parents=True)
    if getattr(args,'state',None):
        state=np.load(args.state);args.values=[float(state['J'])]
    records=[]
    for J in args.values:
        r,ok,_=s.solve(J,state['rates'] if getattr(args,'state',None) else None);assert ok;cache={}
        def evaluate(f):
            f=float(f)
            if f in cache:return cache[f]
            M=characteristic(s,r,J,2j*np.pi*f/1000).tocsc();lu=splu(M)
            phase=np.angle(np.exp(1j*(np.angle(lu.U.diagonal()).sum()+np.pi*(parity(lu.perm_r)+parity(lu.perm_c)))))
            K=sparse.eye(s.P)-M
            bound=float(np.asarray(abs(K).sum(1)).max())
            cache[f]=(phase,bound);return cache[f]
        freq=np.unique(np.r_[0.,np.linspace(.05,12,61),np.linspace(12,40,29),np.geomspace(40,2000,35)])
        for f in freq:evaluate(f)
        for refinement in range(3):
            # Midpoints also catch an entire missed turn whose endpoints align.
            mid=(freq[:-1]+freq[1:])/2
            for f in mid:evaluate(f)
            dense=np.sort(np.r_[freq,mid]);phases=np.unwrap([evaluate(f)[0] for f in dense])
            count=-int(round((phases[-1]-phases[0])/np.pi))
            jump=float(abs(np.diff(phases)).max());print('count',J,refinement,len(dense),count,'jump',jump,flush=True)
            freq=dense
            if refinement>=1 and jump<.5:break
        data=dict(J_EE_core=J,unstable_root_count_candidate=count,frequency_hz=freq,
            unwrapped_phase=phases,maximum_phase_increment=jump,high_frequency_row_sum_bound=evaluate(freq[-1])[1],
            frequency_refinements=refinement+1,meaning='Full determinant count, numerically refined; high-frequency tail checked by loop norm')
        write(dest/f'J{J:.6f}.json',data);records.append({k:v for k,v in data.items() if k not in ['frequency_hz','unwrapped_phase']})
        write(dest/'result.json',dict(status='COMPLETE' if J==args.values[-1] else 'RUNNING',rows=records))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--grid',type=int,default=20);p.add_argument('--values',type=float,nargs='+',default=[.9,.945,.96,.99]);p.add_argument('--response',default='shifted_white')
    p.add_argument('--state');p.add_argument('--label');main(p.parse_args())
