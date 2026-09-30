"""Seed the 0.5 mm (g40) middle equilibrium family from 1 mm branch points: map each g40 group to the
1 mm cell containing its centre (E groups take that cell's E-rate; I groups the cell's I-rate), Newton at
the same D, then launch arclength continuation from the converged seeds in both directions."""
from equilibria_v3 import *
import argparse,subprocess,sys
def main(a):
    s20=SpatialRateV3(grid=20,quiet=True);s40=SpatialRateV3(grid=40,quiet=True)
    pos40=s40.geo['positions'];cell20=(np.floor(pos40[:,0]).astype(int)+20*np.floor(pos40[:,1]).astype(int)).clip(0,399)
    cell_of_20=s20.geo['group_cell'];out=DEST/'equilibria'/'g40_seeds';out.mkdir(exist_ok=True);seeds=[]
    for path in a.points:
        z=np.load(path);r20=z['r'];D=float(z['D'])
        rE=np.zeros(400);rI=np.zeros(400);nE=np.zeros(400);nI=np.zeros(400)
        for g in range(s20.P):
            c=cell_of_20[g]
            if s20.E[g]:rE[c]+=r20[g]*s20.sizes[g];nE[c]+=s20.sizes[g]
            else:rI[c]+=r20[g]*s20.sizes[g];nI[c]+=s20.sizes[g]
        rE/=np.maximum(nE,1);rI/=np.maximum(nI,1);guess=np.where(s40.E,rE[cell20],rI[cell20])
        s40.set_D(D);r,ok,tr=s40.solve(guess,maxiter=120)
        log(Path(path).name,'D',D,'1mm global %.3f'%s20.global_rate(r20),'g40 Newton',ok,'residual %.2e'%tr[-1],'g40 global %.3f'%s40.global_rate(r),'regional',[round(x,2) for x in s40.regional_rates(r)])
        if ok:
            f=out/f'seed_D{D:.5f}.npz';np.savez_compressed(f,r=r,D=D,tangent=np.r_[np.zeros(s40.P),1.]);seeds.append(str(f))
    write(out/'seeds.json',dict(seeds=seeds))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('points',nargs='+');main(p.parse_args())
