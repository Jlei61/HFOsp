"""Realized weighted EE displacements for the three already-run axis canaries.

Immutable cached graph + the executor's block multipliers, checked against the
executor's applied block counts and doses. This does not launch simulations.
"""
from pathlib import Path
import sys,json,pickle,gc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scripts import analyze_topic4_shape_output_response as an
BASE=Path('/data/hfosp/topic4_sef_hfo/global_axis_residual_probe_20260913')
OUT=BASE/'applied_geometry_audit'

def main():
    OUT.mkdir(exist_ok=True);result=[];histograms=[];identities=[];input_totals={};checks=[]
    edges=np.arange(0,20.05001,.05)
    for offset in [0,-15,15]:
        cid=f'global_axis_{offset:+d}';folder=BASE/'canary/units'/cid/'2511_847101';ap=an.rt.read(folder/'applied_physics.json');prm=ap['candidate']['parameters']
        assert not ap['graph']['stage_audits']['local_output'].get('enabled',False)
        assert ap['graph']['stage_audits']['degree']['exact_noop']
        with np.load(folder/'workers/trajectory.npz') as a:pos=a['positions_E'].astype(float);core=a['core_index_E'];mixture=a['core_ou_mixture_values'].copy();input_audit=a['input_rate_audit_values'].copy()
        if offset==0:refmixture=mixture;refinput=input_audit;refid=ap['identity']
        else:
            for key,value in refid.items():
                if not key.startswith('ampa_'):assert ap['identity'][key]==value
            assert np.array_equal(mixture,refmixture) and np.array_equal(input_audit,refinput)
            assert ap['identity']['ampa_topology_sha256']!=refid['ampa_topology_sha256']
        identities.append(dict(candidate=cid,nominal_angle_deg=ap['graph']['kernel']['theta_deg'],identity=ap['identity'],core_ou_prefix_identical=True,expected_input_audit_identical=True))
        with Path(ap['graph']['global_cache']).open('rb') as f:bins,_=pickle.load(f)
        ne=len(pos);block={};totalinput=np.zeros(ne);indeg=np.zeros(ne,dtype=np.int64)
        selected=['ALL_EE','EA->EA','EB->EB','EA->EO','EB->EO','EO->EO'];mom={key:np.zeros(11) for key in selected};hist={key:np.zeros(len(edges)-1) for key in selected}
        for delay,mat in enumerate(bins):
            if not mat.nnz:continue
            coo=mat.tocoo(copy=False);keep=coo.row<ne;rows=coo.row[keep];cols=coo.col[keep];w=coo.data[keep].astype(float,copy=True)
            if not len(w):continue
            tc,sc=core[rows],core[cols]
            same=(tc>=0)&(sc==tc);out=(tc<0)&(sc>=0);outside=(tc<0)&(sc<0)
            w[same]*=prm['EE_same_core_scale'];w[out]*=prm['EE_core_to_out_scale'];w[outside]*=prm['EE_out_to_out_scale']
            totalinput+=np.bincount(rows,weights=w,minlength=ne);indeg+=np.bincount(rows,minlength=ne)
            disp=pos[rows]-pos[cols];xx,yy=disp.T;distance=np.hypot(xx,yy)
            masks={'ALL_EE':np.ones(len(w),bool)}
            for pre,s in [('A',0),('B',1),('O',-1)]:
              for post,t in [('A',0),('B',1),('O',-1)]:
                key=f'E{pre}->E{post}';mask=(sc==s)&(tc==t);n=int(mask.sum())
                if n:
                    v=block.setdefault(key,np.zeros(3));v+=np.array([n,w[mask].sum(),n*delay*.1])
                if key in selected:masks[key]=mask
            for key,mask in masks.items():
                ww=w[mask];dx=xx[mask];dy=yy[mask];dd=distance[mask]
                mom[key]+=np.array([len(ww),ww.sum(),ww@dx,ww@dy,ww@(dx*dx),ww@(dx*dy),ww@(dy*dy),ww@dd,ww@(dd*dd),ww.sum()*delay*.1,len(ww)*delay*.1])
                hist[key]+=np.histogram(dd,bins=edges,weights=ww)[0]
        assert np.all(indeg==800)
        for key,values in block.items():
            expected=ap['graph']['block_summary'][key]
            assert int(values[0])==expected['n_edges'],key
            assert np.isclose(values[1],expected['total_weight'],rtol=1e-10),key
            assert np.isclose(values[2]/values[0],expected['mean_delay_ms'],rtol=1e-10),key
        input_totals[cid]=totalinput
        for key,v in mom.items():
            n,sw,sx,sy,sxx,sxy,syy,sd,sdd,sdelay,udelay=v;mean=np.array([sx,sy])/sw
            cov=np.array([[sxx,sxy],[sxy,syy]])/sw-np.outer(mean,mean);vals,vec=np.linalg.eigh(cov);axis=vec[:,-1];angle=(np.rad2deg(np.arctan2(axis[1],axis[0]))+90)%180-90
            result.append(dict(candidate=cid,offset_deg=offset,block=key,edges=int(n),weight_sum=sw,mean_dx_mm=mean[0],mean_dy_mm=mean[1],weighted_axis_deg=angle,weighted_sd_ratio=np.sqrt(vals[-1]/vals[0]),sd_major_mm=np.sqrt(vals[-1]),sd_minor_mm=np.sqrt(vals[0]),mean_distance_mm=sd/sw,rms_distance_mm=np.sqrt(sdd/sw),mean_delay_ms=udelay/n,weighted_mean_delay_ms=sdelay/sw,nominal_angle_deg=ap['graph']['kernel']['theta_deg']))
            histograms.extend(dict(candidate=cid,block=key,distance_lo_mm=float(edges[i]),distance_hi_mm=float(edges[i+1]),weight_mass=float(value/sw)) for i,value in enumerate(hist[key]))
        checks.append(dict(candidate=cid,EE_indegree_min=int(indeg.min()),EE_indegree_max=int(indeg.max()),applied_block_counts_weights_delays='PASS',cache=str(ap['graph']['global_cache']),coordinate_precision='saved float32 E positions; sufficient for displacement summaries, not exact weight-array reconstruction identity'))
        print(cid,'checked',flush=True);del bins;gc.collect()
    pd.DataFrame(result).to_csv(OUT/'weighted_ee_displacements.csv',index=False);pd.DataFrame(histograms).to_csv(OUT/'weighted_distance_distributions.csv',index=False)
    delta=[]
    for cid,values in input_totals.items():
        diff=values-input_totals['global_axis_+0'];delta.append(dict(candidate=cid,mean_input=float(values.mean()),input_q05_q50_q95=np.quantile(values,[.05,.5,.95]).tolist(),paired_input_change_q05_q50_q95=np.quantile(diff,[.05,.5,.95]).tolist(),paired_input_change_rms=float(np.sqrt(np.mean(diff**2)))))
    an.rt.write(OUT/'audit.json',dict(status='PASS',unit_scope='actual 500ms canaries, shared base topology 2511 and noise 847101',identities=identities,checks=checks,target_total_input=delta,definition='weighted displacement covariance about its own mean; principal axis is axial modulo 180 degrees, not a propagation direction or causal graph proof',source=str(Path(__file__)),source_sha256=an.rt.sha(Path(__file__))))

if __name__=='__main__':main()
