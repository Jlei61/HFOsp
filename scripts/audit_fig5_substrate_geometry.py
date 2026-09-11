#!/usr/bin/env python3
"""Read-only structural audit of the frozen Fig.5 graph and rev22 geometry."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import types

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_topic4_rev21_fig5_checkpoints import (
    load_selected_contract, build_selected_substrate)
from src.topic4_manual_dual_core import budget_matched_dual_core_h
from src.topic4_local_connectivity import local_pair_features


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def axis_angle(vector):
    return float(np.degrees(np.arctan2(vector[1], vector[0])) % 180)


def axial_gap(a, b):
    return float(abs((a - b + 90) % 180 - 90))


def retained_disk_fraction(center, radius, L=20):
    x = np.linspace(max(0, center[0]-radius), min(L, center[0]+radius), 20001)
    half = np.sqrt(np.maximum(0, radius**2-(x-center[0])**2))
    height = np.maximum(0, np.minimum(L, center[1]+half)-np.maximum(0, center[1]-half))
    return float(np.trapz(height, x)/(np.pi*radius**2))


def graph_audit(net, h, centers):
    n_e = int(net['NE'])
    pos = np.asarray(net['pos'])
    nearest = np.argmin(np.linalg.norm(pos[:n_e,None]-centers[None], axis=2),axis=1)
    groups = {
        'core_A': (h > .5) & (nearest == 0),
        'core_B': (h > .5) & (nearest == 1),
        'central_patch': np.linalg.norm(pos[:n_e]-[10,10],axis=1)<2,
        'upper_patch': np.linalg.norm(pos[:n_e]-[10,17],axis=1)<2,
        'all_E': np.ones(n_e, bool),
    }
    sums = {k:np.zeros(7) for k in groups}
    ee_self = 0; ii_self = 0; total_ee = 0; ee_self_weight = 0.; ee_weight = 0.
    for matrix in net['ampa_by_delay']:
        coo = matrix.tocoo(copy=False)
        mask = coo.row < n_e
        row, col, w = coo.row[mask],coo.col[mask],coo.data[mask]
        delta = pos[col]-pos[row]
        diag = row==col
        ee_self += int(diag.sum()); total_ee += len(row)
        ee_self_weight += float(w[diag].sum()); ee_weight += float(w.sum())
        for k, target_mask in groups.items():
            take=target_mask[row]; d=delta[take]; wt=w[take]
            sums[k] += [wt.sum(), np.dot(wt,d[:,0]), np.dot(wt,d[:,1]),
                        np.dot(wt,d[:,0]**2),np.dot(wt,d[:,0]*d[:,1]),
                        np.dot(wt,d[:,1]**2),int(take.sum())]
    for matrix in net['gaba_by_delay']:
        coo=matrix.tocoo(copy=False)
        ii_self += int(np.sum((coo.row>=n_e) & ((coo.row-n_e)==coo.col)))
    result={}
    for k,s in sums.items():
        moment=np.array([[s[3],s[4]],[s[4],s[5]]])/s[0]
        eig, vec=np.linalg.eigh(moment)
        result[k]={'n_targets':int(groups[k].sum()),'n_edges':int(s[6]),
                   'axis_deg':axis_angle(vec[:,-1]),
                   'rms_aspect_ratio':float(np.sqrt(eig[-1]/eig[0])),
                   'mean_source_minus_target_mm':(s[1:3]/s[0]).tolist(),
                   'rms_distance_mm':float(np.sqrt(np.trace(moment)))}
    result['autapses']={'E_to_E_count':ee_self,'I_to_I_count':ii_self,
                       'E_to_E_edges':total_ee,'E_to_E_edge_fraction':ee_self/total_ee,
                       'E_to_E_weight_fraction':ee_self_weight/ee_weight}
    return result


def main():
    art=Path('/home/honglab/leijiaxin/HFOsp')
    rev22=art/'.worktrees/topic4-dual-core-mechanism-scan/config/topic4_rev22_dci_dual_core_interictal_identifiability.json'
    proposals=art/'results/topic4_sef_hfo/data_driven_dual_core_interictal_identifiability/response_fit/final_execution_candidate_manifest.json'
    cfg22=json.loads(rev22.read_text())
    proposal=next(r for r in json.loads(proposals.read_text())['candidates'] if r['candidate_id']=='dci_p030')
    historical_blob=subprocess.check_output(['git','show','26bc4338:config/topic4_dual_core_ood_node_pathways.json'],cwd=ROOT)
    historical_config=json.loads(historical_blob)
    historical=historical_config['two_core_family']['historical_anchor']
    generator_blob=subprocess.check_output(['git','show','26bc4338:src/topic4_dual_core_ood.py'],cwd=ROOT)
    historical_module=types.ModuleType('historical_dual_core_ood')
    exec(compile(generator_blob,'git:26bc4338:src/topic4_dual_core_ood.py','exec'),historical_module.__dict__)
    regenerated=next(r for r in historical_module.generate_sobol_candidates(historical_config['two_core_family']) if r['candidate_id']=='dualcore_s39')
    base=art/'results/topic4_sef_hfo/data_driven_dual_core_zm_transition/timescale'
    source=base/'workers/rev21_ts_tz3000_ta500_topology_2542_dynamics_2642.json'
    config_path=ROOT/'config/topic4_rev21_dual_core_zm_transition.json'
    config,candidate,meta,npz=load_selected_contract(config_path,base/'candidate_manifest.json',source,art)
    _,sub=build_selected_substrate(config,candidate,topology_seed=2542,dynamics_seed=2642,artifact_root=art)
    print('Frozen substrate reconstructed; no dynamics executed.',flush=True)
    centers=np.array(cfg22['dual_core_anchor']['centers_mm'])
    old=np.array(historical['centers_mm'])
    assert np.array_equal(centers,np.array(candidate['node_field']['centers_mm']))
    assert np.array_equal(centers,np.array(regenerated['centers_mm']))
    assert regenerated['target_count']==candidate['node_field']['target_count']
    old_h,old_audit=budget_matched_dual_core_h(sub.positions_e,old,target_count=historical['target_count'])
    new_h,new_audit=budget_matched_dual_core_h(sub.positions_e,centers,target_count=candidate['node_field']['target_count'])
    assert np.array_equal(new_h,sub.h_e)
    figmeta_path=ROOT/'results/paper-ready-figure/fig5_transition_susceptibility_v3/figures/fig5-transition-susceptibility-v3-metadata.json'
    figmeta=json.loads(figmeta_path.read_text())
    event_vec=np.array(figmeta['panel_C']['reference_axis_unit_xy'])
    graph_angle=cfg22['graph_kernel']['theta_EE_deg']
    new_angle=axis_angle(centers[1]-centers[0])
    old_angle=axis_angle(old[1]-old[0])
    proposal_angle=proposal['mechanisms']['ellipse_angle_deg']
    measured=graph_audit(sub.net,sub.h_e,centers)
    # Tiny independent reproductions of the exclusion error: no graph mutation.
    import connectivity
    import connectivity_rot
    points=np.array([[0.,0.],[.1,0.],[0.,.1],[.1,.1]])
    sample_iso=connectivity._sample_partners(points[0],points,2,.38,0,np.random.default_rng(7),self_local=0)
    sample_rot=connectivity_rot._sample_partners_rot(points[0],points,2,.38*np.sqrt(2),.38/np.sqrt(2),0,np.random.default_rng(7),self_local=0)
    evidence={
        'status':'STRUCTURE_AUDITED_NOT_FINAL_SUBSTRATE_FREEZE',
        'historical_field':{**historical,**old_audit},
        'frozen_field':{**candidate['node_field'],**new_audit},
        'disk_fraction_inside_sheet':[retained_disk_fraction(c,new_audit['distance_cutoff_mm']) for c in centers],
        'angles_deg':{'historical_core_line':old_angle,'frozen_core_line':new_angle,
                      'graph_kernel':graph_angle,'rev22_p030_kernel':proposal_angle,
                      'fig5_event_axis':axis_angle(event_vec),
                      'core_vs_graph_gap':axial_gap(new_angle,graph_angle),
                      'core_vs_p030_kernel_gap':axial_gap(new_angle,proposal_angle),
                      'event_vs_graph_gap':axial_gap(axis_angle(event_vec),graph_angle),
                      'event_vs_final_graph_gap':axial_gap(axis_angle(event_vec),measured['all_E']['axis_deg'])},
        'final_fig5_graph':measured,
        'kernel':{'ell_mm':.38,'AR':2.,'parallel_mm':.38*np.sqrt(2),'perpendicular_mm':.38/np.sqrt(2),
                  'meaning':'relative sampling priority, not marginal connection probability with fixed in-degree',
                  'scope':'all E targets, including both cores; no central corridor mask; nonperiodic sheet'},
        'outside_core_features_for_source_dx_0p38':local_pair_features(np.array([[0.,0.]]),np.array([[.38,0.]]),np.array([0.]),np.array([0.]),length_scale=.38).tolist(),
        'self_exclusion_toy':{'isotropic_indices':sample_iso.tolist(),'rotated_indices':sample_rot.tolist(),
                              'self_selected_in_both':bool(0 in sample_iso and 0 in sample_rot)},
        'slow_parameters':candidate['slow_variables'],
        'network_cache':sub.network_cache,
        'source_hashes':{str(p):digest(p) for p in (rev22,proposals,source,config_path,figmeta_path,Path(__file__))},
        'historical_config_git_revision':'26bc4338',
        'historical_config_blob_sha256':hashlib.sha256(historical_blob).hexdigest(),
        'sobol_regeneration':{'exact_centers_and_budget_match':True,
                              'accepted_index':regenerated['sobol_accepted_index'],
                              'draw_index':regenerated['sobol_draw_index'],
                              'historical_source_sha256':hashlib.sha256(generator_blob).hexdigest()},
        'threshold_audit':{'base_mV':sub.engine['v_base'],
                           'core_target_mean_mV':sub.engine['core_mean'],
                           'core_target_sd_mV':sub.engine['core_std'],
                           'core_actual_mean_mV':float(sub.vtheta[:sub.n_e][new_h>.5].mean()),
                           'core_above_base_count':int(np.sum(sub.vtheta[:sub.n_e][new_h>.5]>sub.engine['v_base'])),
                           'core_below_base_count':int(np.sum(sub.vtheta[:sub.n_e][new_h>.5]<sub.engine['v_base']))},
        'claim_boundary':'Geometry and code audit. No simulation changes, no new dynamical or anisotropy-causal claim. rev22 p030 is a train-selected proposal.'}
    out=ROOT/'results/topic4_sef_hfo/fig5_substrate_geometry_audit'
    figures=out/'figures';figures.mkdir(parents=True,exist_ok=True)
    (out/'geometry_audit.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(out/'geometry_arrays.npz',positions_e=sub.positions_e,
                        contact_xy=sub.contact_xy,contact_names=np.array(sub.contact_names))
    render_figure(evidence,sub.positions_e,sub.contact_xy,sub.contact_names,out)
    evidence['geometry_arrays_sha256']=digest(out/'geometry_arrays.npz')
    evidence['figure_outputs']={str(f):digest(f) for f in figures.glob('core_geometry*')}
    (out/'geometry_audit.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'angles':evidence['angles_deg'],'cutoff':new_audit['distance_cutoff_mm'],'retained_disk_fraction':evidence['disk_fraction_inside_sheet'],'realized_graph':measured,'toy':evidence['self_exclusion_toy'],'output':str(out)},indent=2),flush=True)


def render_figure(evidence, positions_e, contact_xy, contact_names, out):
    figures=out/'figures'
    old_audit=evidence['historical_field'];new_audit=evidence['frozen_field']
    old=np.array(old_audit['centers_mm']);centers=np.array(new_audit['centers_mm'])
    old_h,_=budget_matched_dual_core_h(positions_e,old,target_count=old_audit['target_count'])
    new_h,_=budget_matched_dual_core_h(positions_e,centers,target_count=new_audit['target_count'])
    graph_angle=evidence['angles_deg']['graph_kernel']
    proposal_angle=evidence['angles_deg']['rev22_p030_kernel']
    new_angle=evidence['angles_deg']['frozen_core_line']
    ea=np.deg2rad(evidence['angles_deg']['fig5_event_axis']+180)
    event_vec=np.array([np.cos(ea),np.sin(ea)])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axs=plt.subplots(1,3,figsize=(15,6),gridspec_kw={'width_ratios':[1,1,1]})
    fig.subplots_adjust(left=.055,right=.975,bottom=.26,top=.84,wspace=.27)
    fig.suptitle('Frozen core locations and the E-to-E connection axis',fontsize=18,weight='bold',y=.96)
    axis_vec=np.array([np.cos(np.deg2rad(graph_angle)),np.sin(np.deg2rad(graph_angle))])
    for i,(cs,h,audit,title) in enumerate([(old,old_h,old_audit,'A   Historical hand-placed cores'),(centers,new_h,new_audit,'B   Frozen whole-sheet search result')]):
        ax=axs[i];ax.set_title(title,loc='left',fontsize=12,pad=14,weight='bold')
        for x in [2,6,10,14,18]:
            for y in [2,6,10,14,18]:
                p=np.array([x,y]);seg=np.array([p-axis_vec*.75,p+axis_vec*.75])
                ax.plot(seg[:,0],seg[:,1],color='#245C3F',alpha=.5,lw=1.5)
        ax.scatter(positions_e[h>.5,0],positions_e[h>.5,1],s=2,c='#8B4BB1',alpha=.3,rasterized=True)
        for k,c in enumerate(cs):
            ax.add_patch(Circle(c,audit['distance_cutoff_mm'],fill=False,ec='#79409B',lw=1.5))
            ax.scatter(*c,s=35,color='#79409B',zorder=5)
        ax.plot(cs[:,0],cs[:,1],color='#79409B',lw=1.4,ls='--')
        for shaft,color in [('ICL','#EE8A3A'),('SCL','#2EAFC4')]:
            mask=np.array([n.startswith(shaft) for n in contact_names])
            ax.scatter(contact_xy[mask,0],contact_xy[mask,1],s=24,facecolors='white',edgecolors=color,zorder=4)
        ax.set(xlim=(0,20),ylim=(0,20),xticks=[0,10,20],yticks=[0,10,20],xlabel='x (mm)',ylabel='y (mm)',aspect='equal')
    # This event direction is a measured output, separate from the structural axis.
    start=np.array([13.,13.]);end=start+event_vec*5
    axs[1].annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','color':'#8B5AA9','lw':2})
    axs[1].text(11.2,14.8,'Fig. 5 event axis',fontsize=9,color='#8B5AA9')
    ax=axs[2];ax.set_title('C   Same local kernel across the sheet',loc='left',fontsize=12,pad=14,weight='bold')
    x=np.linspace(-1.5,1.5,301);xx,yy=np.meshgrid(x,x)
    u=axis_vec[0]*xx+axis_vec[1]*yy;v=-axis_vec[1]*xx+axis_vec[0]*yy
    q=np.sqrt((u/(.38*np.sqrt(2)))**2+(v/(.38/np.sqrt(2)))**2)
    levels=[np.exp(-3),np.exp(-2),np.exp(-1)]
    ax.contourf(xx,yy,np.exp(-q),levels=np.linspace(0,1,21),cmap='Greens',alpha=.9)
    ax.contour(xx,yy,np.exp(-q),levels=levels,colors='#245C3F',linewidths=.8)
    ax.plot([-1.3*axis_vec[0],1.3*axis_vec[0]],[-1.3*axis_vec[1],1.3*axis_vec[1]],color='#245C3F',lw=2,label='Graph axis: -22.8°')
    pv=np.array([np.cos(np.deg2rad(proposal_angle)),np.sin(np.deg2rad(proposal_angle))])
    ax.plot([-1.3*pv[0],1.3*pv[0]],[-1.3*pv[1],1.3*pv[1]],ls='--',color='#476CAD',lw=1.5,label='rev22 proposal: -31.4°')
    ax.scatter(0,0,s=25,c='black')
    ax.set(aspect='equal',xlabel='Source minus target: Δx (mm)',ylabel='Δy (mm)',xlim=(-1.5,1.5),ylim=(-1.5,1.5))
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.19),frameon=False,fontsize=9)
    fig.text(.055,.09,'Green marks: E→E axis at every location\nPurple disks: core membership; circles: electrode contacts',fontsize=10)
    fig.text(.39,.09,f'Core-line / graph-axis gap: {axial_gap(new_angle,graph_angle):.1f}°\nEvent-axis / graph-axis gap: {axial_gap(axis_angle(event_vec),graph_angle):.1f}°',fontsize=10)
    stem=figures/'core_geometry_and_connection_axis'
    for ext in ('png','pdf','svg'):fig.savefig(stem.with_suffix('.'+ext),dpi=220)
    plt.close(fig)
    (figures/'README.md').write_text('### core_geometry_and_connection_axis.png / .pdf / .svg\n左图是历史手放几何，中图是当前冻结的全平面搜索双核，均在本次 Fig. 5 的同一神经元位置和电极坐标下显示；历史图只是几何对比，没有重跑动力学。绿色短线表示全平面一致的 E→E 方向，紫色箭头是 Fig. 5 单个事件测出的传播方向；右图按真实代码绘制相对采样权重核，蓝色虚线仅标 rev22 训练选择候选的重加权方向。圆盘实际成员由固定神经元数量确定，边界处被非周期平面裁切。\n**关注点**：核连线、连接长轴和事件传播轴是三个不同的量；当前核连线与长轴并未保持历史对齐关系。\n')


if __name__=='__main__':
    if '--render-only' in sys.argv:
        out=ROOT/'results/topic4_sef_hfo/fig5_substrate_geometry_audit'
        evidence=json.loads((out/'geometry_audit.json').read_text())
        with np.load(out/'geometry_arrays.npz',allow_pickle=False) as data:
            render_figure(evidence,data['positions_e'],data['contact_xy'],data['contact_names'],out)
    else:
        main()
