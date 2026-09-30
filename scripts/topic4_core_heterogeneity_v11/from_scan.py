"""Convert an observed periodic candidate into an exact periodic DDE orbit."""
from periodic_boundaries import *
from scipy.interpolate import CubicSpline


def refine(h,g,direction='up',N=1024):
    stem='g'+f'{g:.5f}'.replace('.','p')
    path=OUT/'state_scan_v2'/f'h{h:.5f}'/direction/stem
    meta=read(path.with_suffix('.json'));d=np.load(path.with_suffix('.npz'))
    T=meta['period_ms'];dt=float(d['sample_dt']);r=d['r']
    time=np.arange(len(r))*dt;start=time[-1]-T
    guess=CubicSpline(time,r)(start+np.arange(N)*T/N)
    z=np.r_[(guess/.01).ravel(),np.log(T),g/.01];tan=np.zeros_like(z);tan[-1]=1
    chart=Chart(System(h),z,tan,N,.01);z,t,err,*_=chart.solve(z,0)
    out=OUT/'refined_scan'/f'h{h:.5f}_{direction}_{stem}_N{N}.npz';out.parent.mkdir(exist_ok=True)
    np.savez_compressed(out,r=z[:-2].reshape(N,6)*.01,T=np.exp(z[-2]),g=g,h=h,tangent=t,N=N,residual=err)
    write(out.with_suffix('.json'),dict(h=h,g=g,N=N,T_ms=float(np.exp(z[-2])),orbit_residual=err,source=str(path.relative_to(ROOT)),refined_source=str(out.relative_to(ROOT))))
    print('REFINED',out,'T',np.exp(z[-2]),'res',err,flush=True)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--h',type=float,required=True);p.add_argument('--g',type=float,required=True)
    p.add_argument('--direction',default='up');p.add_argument('--N',type=int,default=1024)
    a=p.parse_args();refine(a.h,a.g,a.direction,a.N)
