"""A-dominated bifurcations of the full coupled system, original EE axis."""
from common import *
from threshold_node_scan import MeanSystem,MEAN,STD
from periodic_boundaries import Chart,load_seed
import argparse,csv

DEST=OUT/'core_a_focus';DEST.mkdir(exist_ok=True)

def audit():
    p=ROOT/'results/topic4_sef_hfo/core_bifurcation_types_v8_20260916/critical_mode_components.json'
    rows=[]
    for r in read(p):
        v=np.array(r['right_percent']);shares=[float(v[[0,3]].sum()),float(v[[1,4]].sum()),float(v[[2,5]].sum())]
        # Explicit display convention; it is not an anatomical causality test.
        largest=int(np.argmax(shares));dominant='ABS'[largest] if shares[largest]>=80 else 'shared'
        rows.append(dict(label=r['label'],J_EE_core=r['J'],A_percent=shares[0],B_percent=shares[1],S_percent=shares[2],
                         classification=dominant,show_in_A_panel=(dominant=='A')))
    for k in ['PD2','PD3']:
        data=[r for r in read(OUT/'bifurcation_curves.json') if r['label']==k]
        assert all(r['critical_mode_percent'][0]+r['critical_mode_percent'][3]>80 for r in data)
        md=read(OUT/'threshold_nodes'/f'{k}.json')['points']
        assert all(r['critical_mode_percent'][0]+r['critical_mode_percent'][3]>80 for r in md)
    write(DEST/'mode_audit.json',dict(convention='At least 80 percent of squared rate-mode norm in A_E and A_I; equal per-neuron-rate coordinates, not cell-count weighting',
         full_feedback_retained=True,x_axis='Original common AA and BB EE multiplier',rows=rows))
    with (DEST/'mode_audit.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def pair(axis,value,branch):
    label='PD2' if branch=='burst' else 'PD3'
    if axis=='spread':
        groups={k:[r for r in read(OUT/'bifurcation_curves.json') if r['label']==k] for k in ['PD2','PD3']}
        system=lambda:System(value)
        for k in groups:groups[k]=sorted(groups[k],key=lambda r:r['h'])
        xy='h'
    else:
        groups={k:sorted(read(OUT/'threshold_nodes'/f'{k}.json')['points'],key=lambda r:r['mean_A_mV']) for k in ['PD2','PD3']}
        system=lambda:MeanSystem(value);xy='mean_A_mV'
    lo,hi=[float(np.interp(value,[r[xy] for r in groups[k]],[r['g'] for r in groups[k]])) for k in ['PD3','PD2']]
    g=(lo+hi)/2
    seed=min(groups[label],key=lambda r:abs(r[xy]-value));s=System(seed[xy]) if axis=='spread' else MeanSystem(seed[xy]);N=1024
    q,t=load_seed(ROOT/seed['source'],N);normal=np.zeros_like(q);normal[-1]=1
    q,t,err,*_=Chart(s,q,normal,N,.01).solve(q,0)
    start=float(q[-1]*.01)
    for gg in np.linspace(start,g,max(2,int(np.ceil(abs(g-start)/.0003))+1))[1:]:
        step=(gg-q[-1]*.01)/.01
        predicted=q+step*t/t[-1];predicted[-1]=gg/.01
        q,t,err,*_=Chart(s,predicted,normal,N,.01).solve(predicted,0)
    s=system()
    q,t,err,*_=Chart(s,q,normal,N,.01).solve(q,0)
    path=DEST/f'{axis}_{value:.6f}_{branch}_N1024.npz'
    np.savez_compressed(path,r=q[:-2].reshape(N,6)*.01,T=np.exp(q[-2]),g=g,N=N,tangent=t)
    q2,t2=load_seed(path,2048);normal=np.zeros_like(q2);normal[-1]=1
    q2,t2,err2,*_=Chart(s,q2,normal,2048,.01).solve(q2,0)
    path2=path.with_name(path.name.replace('N1024','N2048'))
    r=q2[:-2].reshape(2048,6)*.01;T=float(np.exp(q2[-2]))
    assert abs(T-np.exp(q[-2]))<.01 and err2<1e-8
    np.savez_compressed(path2,r=r,T=T,g=g,N=2048,tangent=t2)
    sys.path.append(str(ROOT/'scripts/topic4_core_network_bifurcation_v7'))
    import analytic_poincare as pc
    pc.System=system;pc.OUT=DEST
    vals=[pc.compute(path2,dtmax=dt,method='rk4',nev=4) for dt in [.05,.025]]
    stable=all(x['max_transverse']<1 and x['orbit_tangent_defect']<.02 for x in vals)
    assert stable
    row=dict(axis=axis,value=value,g=g,branch=branch,rate_min_hz=(r.min(0)*1000).tolist(),rate_max_hz=(r.max(0)*1000).tolist(),
             T_ms=T,stable=stable,grid_period_difference_ms=float(T-np.exp(q[-2])),residual=err2,
             spectra=vals,source=str(path2.relative_to(ROOT)),parameter_interval=[lo,hi])
    write(path2.with_suffix('.json'),row)
    print('COEXISTENCE CHECK',axis,value,branch,g,T,stable,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['audit','spread','mean']);a.add_argument('--value',type=float);a.add_argument('--branch',choices=['burst','high'])
    x=a.parse_args()
    if x.action=='audit':audit()
    else:pair(x.action,x.value,x.branch)
