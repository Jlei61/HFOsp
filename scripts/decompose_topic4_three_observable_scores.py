"""Exact descriptive split of frozen scores; no changed ranking or objective."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
from scripts import analyze_topic4_three_observable_bo as a
from src.topic4_three_observable_objective import signed_statistic

def decompose(device='cuda:0'):
    obj=a.load_objective();dest=a.OUT/'overnight_20260914/score_decomposition';dest.mkdir(exist_ok=True,parents=True)
    for path in sorted((a.OUT/'scores').glob('*.json')):
        r=a.rt.read(path);out=dest/path.name
        if out.exists() or r['J'] is None:continue
        _,t,lab,ids,_=a.load_small(Path(r['source']));terms={};conditional={}
        for g,s in obj.moments(t[ids],lab[ids],device).items():
            d=len(s['sum']);target=obj.targets[g]['psi'];norm=obj.scales[g];n=s['n']
            def off(total,sumsq,target):return signed_statistic(total,sumsq,n,target)['D_off']/norm/3
            part={'global_features':off(s['sum']/np.sqrt(2),s['norm']/2,target[:d]),'mode_frequency':0.}
            for k,name in [(0,'TB'),(1,'TA')]:
                p=obj.proportions[k];start=d+k*(d+1)
                part['mode_frequency']+=off(np.array([s['counts'][k]*.5/p]),s['counts'][k]*.25/p**2,target[start:start+1])
                part[f'{name}_frequency_weighted_features']=off(s['mode_sum'][k]*.5/p,s['mode_norm'][k]*.25/p**2,target[start+1:start+d+1])
            np.testing.assert_allclose(sum(part.values()),r['groups'][g]['scaled']/3,atol=1e-9)
            terms[g]=part;conditional[g]=r['groups'][g]['conditional']
        total={k:sum(p[k] for p in terms.values()) for k in next(iter(terms.values()))}
        np.testing.assert_allclose(sum(total.values()),r['J'],atol=1e-9)
        a.rt.write(out,dict(candidate=r['candidate'],topology=r['topology'],noise=r['noise'],N=r['N'],J=r['J'],contributions_to_J=total,per_group=terms,
            conditional_feature_distances=conditional,source_score=str(path),source_sha256=a.rt.sha(path),objective_sha256=r['objective_sha256'],
            interpretation='Exact algebraic split of the frozen criterion. Mode feature blocks retain frequency weighting; they are not pure mode-shape or independent variance contributions. Conditional distances are reported separately.'))
    return [a.rt.read(p) for p in dest.glob('*.json')]

if __name__=='__main__':
    rows=decompose()
    for cid in ['bridge_circle_out125_xminus075','g1_axis2_minus','g1_axis0_minus','g1_axis3_plus']:
        selected=[r for r in rows if r['candidate']==cid]
        print(cid,{k:float(np.mean([r['contributions_to_J'][k] for r in selected])) for k in selected[0]['contributions_to_J']})
