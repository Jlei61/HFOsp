"""Additional solved waveforms near a fold; no graphical interpolation."""
from rate_periodic import *


def main():
    p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--device',type=int,default=0)
    p.add_argument('--offsets',type=float,nargs='+',default=[-.2,-.1,-.05,-.02,.02,.05,.1,.2])
    p.add_argument('--tol',type=float,default=1e-9)
    a=p.parse_args();q=read(PERIODIC_OUT/(a.label+'_validation.json'));assert q['status']=='VALIDATED_CYCLE_FOLD'
    critical=q['mesh_checks'][-1];z=np.load(critical['orbit']);s=RateField();N=len(z['r']);o=Periodic(s,N,a.device)
    o.low_memory=True;o.normalize_linear_rhs=True;cp=o.cp
    coordinate_name=critical.get('coordinate')
    assert coordinate_name in ['core_A_mean_Hz','core_B_mean_Hz'], 'This sampler requires the stored core-mean coordinate'
    core='AB'.index(coordinate_name.split('_')[1])
    mask=s.E&(s.geo['group_region']==core);w=s.geo['group_size']*mask;w/=w.sum()
    c=np.r_[np.tile(w/N,N),0.,0.];base=critical['coordinate_value'];cache=[(base,z['r'],float(z['T']),float(z['J']))]
    assert abs(float(z['r'].mean(0)@w*1000)-base)<1e-7
    o.linear_target_aware=a.tol<1e-9
    manifest=PERIODIC_OUT/(a.label+'_neighborhood.json')
    rows=[dict(coordinate=base,path=critical['orbit'])]
    if manifest.exists():
        previous=read(manifest)
        assert previous['coordinate']==coordinate_name
        assert any(Path(v['path']).resolve()==Path(critical['orbit']).resolve() for v in previous['rows'])
        rows=previous['rows']
    for delta in sorted(a.offsets,key=abs):
        coordinate=base+delta;_,r,T,J=min(cache,key=lambda x:abs(x[0]-coordinate))
        pred=np.r_[(r*1000).ravel(),np.log(T),J*1000];pred+=c*(coordinate-c@pred)/(c@c)
        r=pred[:-2].reshape(N,s.P)/1000
        rr,TT,JJ,err,history=o.solve(r,T,J,arc=(pred,c,np.ones_like(c)),maxiter=24,tol=a.tol)
        assert err<2e-8,(a.label,delta,err)
        if a.tol<1e-9:assert err<=a.tol*1.01,(a.label,delta,err)
        assert abs(float(rr.mean(0)@w*1000)-coordinate)<1e-7
        path=save_orbit(s,rr,TT,JJ,err,history,f'{a.label}_local_{delta:+.5f}_N{N}')
        rows=[v for v in rows if Path(v['path']).resolve()!=path.resolve()]
        rows.append(dict(coordinate=coordinate,path=str(path)));cache.append((coordinate,rr,TT,JJ))
        write(PERIODIC_OUT/(a.label+'_neighborhood.json'),dict(rows=sorted(rows,key=lambda x:x['coordinate']),
            coordinate=coordinate_name,status='PARTIAL',scope='Additional converged BVP profiles only; not a stability classification.'))
    result=read(PERIODIC_OUT/(a.label+'_neighborhood.json'));result['status']='COMPLETE';write(PERIODIC_OUT/(a.label+'_neighborhood.json'),result)


if __name__=='__main__':main()
