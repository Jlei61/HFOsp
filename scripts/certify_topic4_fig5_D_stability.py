"""Exact-map winding counts and real-root instability certificates."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,argparse,time
import numpy as np
from topic4_fig5_D_physical_model import Characteristic,OUT
from topic4_fig5_z_characteristic_root_newton import root

def cached_characteristic(q):
    c=Characteristic(q);method=c.weights;cache={}
    def weights(lam):
        if lam not in cache:cache[lam]=method(lam)
        return cache[lam]
    c.weights=weights;return c

def local(q):
    c=cached_characteristic(q);aa=np.load(OUT/f'q{q:g}_Ha.npz');bb=np.load(OUT/f'q{q:g}_Hb.npz')
    Da,Db=float(aa['D']),float(bb['D']);rows=[]
    values=sorted(set([0.,Da-.002,Da+.002,Db-.002,Db+.002]))
    for D in values:
        seed=aa if abs(D-Da)<abs(D-Db) else bb
        r,err,ok=c.eq.solve(seed['r_hz'],D);assert ok
        c.at(r,D);start=time.time();checks=[c.rhp_count(spacing=5.),c.rhp_count(spacing=3.)]
        row=dict(D=D,q_ie=q,counts=checks,residual_hz=err,seconds=time.time()-start)
        assert checks[0]['unstable_multiplier_count']==checks[1]['unstable_multiplier_count']
        rows.append(row);(OUT/f'q{q:g}_local_stability.json').write_text(json.dumps(rows,indent=2)+'\n');print('LOCAL STABILITY',row,flush=True)

def branches(q):
    c=cached_characteristic(q);path=OUT/f'q{q:g}_branch_stability.json';rows=json.loads(path.read_text()) if path.exists() else [];done={(a['branch'],a['index']) for a in rows}
    for name in ['low','middle','high']:
        file=OUT/f'q{q:g}_{name}.npz'
        if not file.exists():continue
        a=np.load(file)
        turns=np.flatnonzero(a['tangent'][:-1,-1]*a['tangent'][1:,-1]<0)
        selected=set(range(0,len(a['s']),8 if name=='high' else 5))|{len(a['s'])-1}
        selected|={int(i) for k in turns for i in (k,k+1)}
        if name=='low':selected=set(range(len(a['s'])))
        for k,(r,D) in enumerate(zip(a['r_hz'],a['s'])):
            if k not in selected or (name,k) in done or D<0 or D>1:continue
            c.at(r,float(D));bracket=None;previous=None;grid=[0.,1.,3.,10.,30.,100.,300.,1000.]
            for lam in grid:
                sign=float(np.linalg.slogdet(c.matrix(lam).real)[0])
                if previous is not None and sign*previous[1]<0:bracket=[previous[0],lam];break
                previous=(lam,sign)
            row=dict(branch=name,index=k,D=float(D),mean_e_hz=float(np.average(r[:c.m.n],weights=c.m.count_e)))
            if bracket:row.update(stability='unstable',method='positive real characteristic root bracket',bracket_per_s=bracket)
            else:
                checks=[c.rhp_count(spacing=5.)]
                # A zero count authorizes a stable line only after contour refinement.
                if checks[0]['unstable_multiplier_count']==0:checks.append(c.rhp_count(spacing=3.))
                assert len({x['unstable_multiplier_count'] for x in checks})==1
                row.update(stability='stable' if checks[0]['unstable_multiplier_count']==0 else 'unstable',method='unit-circle winding',counts=checks)
            rows.append(row);path.write_text(json.dumps(rows,indent=2)+'\n');print('BRANCH STABILITY',row,flush=True)

def old_folds():
    c=cached_characteristic(1.)
    for name in ['old_TP','old_TP2','old_F0']:
        a=np.load(OUT/f'{name}.npz');r=a['r_hz'];D=float(a['D']);c.at(r,D)
        roots=[]
        for guess in [10+150j,10+35j,10+0j]:
            ll,v,er,h=root(c,guess);roots.append(dict(lambda_per_s=[ll.real,ll.imag],residual=er))
            if er<1e-7 and ll.real>1e-4:break
        p=OUT/f'{name}.json';info=json.loads(p.read_text());info['dynamic_roots']=roots
        info['full_stability']='UNSTABLE' if any(x['residual']<1e-7 and x['lambda_per_s'][0]>1e-4 for x in roots) else 'UNRESOLVED';p.write_text(json.dumps(info,indent=2)+'\n');print(name,info['full_stability'],roots,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['local','branches','old_folds']);p.add_argument('--q',type=float,default=1.25);a=p.parse_args()
    old_folds() if a.mode=='old_folds' else (local(a.q) if a.mode=='local' else branches(a.q))
