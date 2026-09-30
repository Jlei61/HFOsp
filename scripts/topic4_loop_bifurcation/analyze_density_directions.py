#!/usr/bin/env python3
"""Direction-conditional contact and field correspondence, frozen old rules."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys,warnings
import numpy as np
from scipy.stats import spearmanr
from campaign import ROOT,REPO,read,write,sha
from compare_density_resolution import sources
sys.path.insert(0,str(REPO/'scripts/topic4_interictal_surrogate'))
from direction_validation_v2 import arrivals, trailing5, describe, score, spatial_comparison, CONTRACT
from interictal_common import safe


def main():
    folder=ROOT/'density_contact_replay';base=read(folder/'comparison.json');assert base['status']=='COMPLETE'
    rows={};maps={};source={r['name']:r for r in sources()}
    geo=np.load(REPO/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz')
    names=geo['contact_names'].tolist();centers=geo['centers_mm'];axis=centers[1]-centers[0];axis/=np.linalg.norm(axis)
    yy,xx=np.mgrid[:20,:20];coordinate=(np.c_[xx.ravel()+.5,yy.ravel()+.5]-centers[0])@axis
    classes=['A_to_B','B_to_A','weak_or_complex','not_estimable']
    for label,readouts in base['rows'].items():
        field=trailing5(source[label]['field'][:8000]);rows[label]={}
        for mode,rec in readouts.items():
            ids=rec['primary_event_ids'];mu=np.array(rec['centroids_ms'],float).reshape(-1,15)
            amap=np.array([arrivals(field,rec['observation']['events'][i]['window_ms']) for i in ids]).reshape(-1,400)
            rho=[]
            for a in amap:
                ok=np.isfinite(a);rho.append(spearmanr(coordinate[ok],a[ok]).statistic if ok.sum()>=5 and len(np.unique(a[ok]))>1 else np.nan)
            conditional={};labels={}
            for cutoff in CONTRACT['field_axis_cutoff_sensitivity']:
                ll=['not_estimable' if not np.isfinite(r) else 'A_to_B' if r>=cutoff else 'B_to_A' if r<=-cutoff else 'weak_or_complex' for r in rho]
                labels[str(cutoff)]=ll
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning)
                    conditional[str(cutoff)]={cl:describe(mu[np.asarray(ids)[np.asarray(ll)==cl]],names) for cl in classes}
            rows[label][mode]=dict(primary_count=len(ids),axis_rho=rho,labels=labels,conditional=conditional,
                counts={c:{cl:v['N'] for cl,v in d.items()} for c,d in conditional.items()})
            maps[(label,mode)]=amap
    comparisons=[]
    for mode in ['firing','current_hfo']:
        pairs=[('native9108402','native9108401','native_noise_reference')]+[(l,r,'density_native')
            for l in rows if l.startswith('R') and mode in rows[l] for r in ['native9108401','native9108402']]
        for left,right,role in pairs:
            a,b=rows[left][mode],rows[right][mode]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                contact={c:{cl:score(a['conditional'][c][cl],b['conditional'][c][cl],names) for cl in classes} for c in a['conditional']}
                spatial=spatial_comparison(maps[(left,mode)],maps[(right,mode)],a['labels']['0.3'],b['labels']['0.3'],geo['cell_e_counts'],labels=classes)
            comparisons.append(dict(left=left,right=right,role=role,readout=mode,contacts=contact,spatial=spatial))
    write(folder/'direction_comparison.json',safe(dict(status='COMPLETE',rows=rows,comparisons=comparisons,
        producer_sha256=sha(__file__),original_direction_rule=CONTRACT,
        field_observer='Original trailing5ms, >=50Hz sustained5ms, omit left-censored cells; rank gradient on physical coreA→B axis. This differs explicitly from A4 late-centroid direction used in the baseline event plot.',
        scope='Original fixed primary contact windows; all weak, complex and unestimable cases retained. Two native trajectories only; within-direction counts are small, events and cross-event pairs are nested, no equivalence inference.',
        model_promoted=False,formal_bifurcation_allowed=False)))
    print({k:{m:v['counts']['0.3'] for m,v in d.items()} for k,d in rows.items()},flush=True)


if __name__=='__main__':main()
