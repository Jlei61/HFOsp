"""Pseudo-arclength continuation through folds of full spatial periodic BVPs."""
from rate_periodic import *


def encode(z,N):return np.r_[(resample(z['r'],N,axis=0)*1000).ravel(),np.log(float(z['T'])),float(z['J'])*1000]


def main():
    p=argparse.ArgumentParser();p.add_argument('first');p.add_argument('second');p.add_argument('--N',type=int,default=64)
    p.add_argument('--steps',type=int,default=60);p.add_argument('--ds',type=float,default=.2);p.add_argument('--label',required=True)
    p.add_argument('--device',type=int,default=0);p.add_argument('--start-index',type=int,default=0);a=p.parse_args()
    s=RateField();o=Periodic(s,a.N,a.device);x0=encode(np.load(a.first),a.N);x1=encode(np.load(a.second),a.N)
    weight=np.r_[np.full(a.N*s.P,1/np.sqrt(a.N*s.P)),50.,1.];ds=a.ds;rows=[]
    for i in range(a.start_index,a.start_index+a.steps):
        tangent=x1-x0;tangent/=np.linalg.norm(tangent*weight)
        for retry in range(7):
            pred=x1+ds*tangent;r=pred[:-2].reshape(a.N,s.P)/1000;T=np.exp(pred[-2]);J=pred[-1]/1000
            rr,TT,JJ,err,history=o.solve(r,T,J,arc=(pred,tangent,weight),maxiter=12)
            if err<2e-8:break
            ds*=.5
        else:
            print('CONTINUATION FAILED',i,ds,err,flush=True);break
        path=save_orbit(s,rr,TT,JJ,err,history,f'{a.label}_{i:04d}_N{a.N}');rows.append(dict(path=str(path),ds=ds))
        x0=x1;x1=np.r_[(rr*1000).ravel(),np.log(TT),JJ*1000]
        if len(history)<5:ds=min(ds*1.2,a.ds*2)
        if len(history)>7:ds*=.7
        write(PERIODIC_OUT/f'{a.label}_continuation.json',dict(rows=rows,method='pseudo-arclength',N=a.N))
        if JJ<.5 or JJ>2.1 or TT>5000 or TT<5:break


if __name__=='__main__':main()
