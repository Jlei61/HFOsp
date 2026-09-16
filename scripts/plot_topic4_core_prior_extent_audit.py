"""Audit how equal circular cores represent the already adopted endpoint prior.

Contact coverage is not a physical necessity or a new loss. The endpoint identity
audit is reused as the adopted coarse prior, not called a causal source finding.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scripts import analyze_topic4_propagation_recovery_night as s

ANCHOR='/data/hfosp/topic4_sef_hfo/geometry_threshold_refinement_20260909/anchor_audit.json'


def main():
    anchor=s.rt.read(ANCHOR);plan=s.rt.read(s.an.run.OUT/'plan.json');base=next(c for c in plan['candidates'] if c['id']=='endpoint__baseline')
    p=s.an.run.result_path('screen',base['id'],2511,847101)
    with np.load(p.with_suffix('.npz')) as z:names=z['contact_names'].astype(str);xy=z['contact_xy_mm']
    rows=[]
    for k,a in enumerate(anchor['rows']):
        ix=[list(names).index(name) for name in a['names']];points=xy[ix];center=np.asarray(base['centers_mm'][k]);radius=base['radii_mm'][k]
        assert np.linalg.norm(points.mean(0)-center)<1e-5
        d=np.linalg.norm(points-center,axis=1)
        rows.append(dict(core='A' if k==0 else 'B',anchor_names=a['names'],center_mm=center.tolist(),radius_mm=radius,
            distances_mm=d.tolist(),edge_gaps_mm=np.maximum(d-radius,0).tolist(),inside_count=int((d<=radius).sum()),
            pairwise_max_distance_mm=float(np.linalg.norm(points[:,None]-points[None,:],axis=2).max()),
            projected_x_range_mm=float(np.ptp(points[:,0])),projected_y_range_mm=float(np.ptp(points[:,1]))))
    variants=[('端点原位',base['centers_mm'][0],base['radii_mm'][0])]
    for cid,label in [('refine_mid_EE075','上移3mm'),('refine_midpoint_EE075','两位置中点'),('refine_near_EE075','靠近上部SCL')]:
        c=s.rt.read(s.an.run.OUT/'candidates'/f'{cid}.json');variants.append((label,c['centers_mm'][0],c['radii_mm'][0]))
    cloud=xy[[list(names).index(n) for n in rows[0]['anchor_names']]]
    gaps=np.array([np.maximum(np.linalg.norm(cloud-np.asarray(center),axis=1)-radius,0) for _,center,radius in variants])
    dest=s.night.OUT/'core_prior_extent_audit';F=dest/'figures';F.mkdir(parents=True,exist_ok=True)
    s.rt.write(dest/'geometry.json',dict(rows=rows,variants=[dict(label=label,center_mm=center,radius_mm=radius) for label,center,radius in variants],A_anchor_edge_gaps_mm=gaps.tolist(),
        anchor_source=ANCHOR,anchor_sha256=s.rt.sha(ANCHOR),actual_layout_source=str(p),arrays_sha256=s.rt.read(p)['arrays_sha256'],
        provenance='Already adopted endpoint geometry, verified by prior nearest-contact identity audit; not a new derivation of physiological onset sources.',
        interpretation='Equal core radii encode very different coverage of the two adopted endpoint clouds. Contacts need not lie within cores: propagation and readout can recruit them outside. This audit motivates comparing extent and outgoing connectivity, not a hard coverage requirement.',
        not_training=True,not_independent_validation=True,producer=__file__,producer_sha256=s.rt.sha(__file__)))
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'pdf.fonttype':42})
    fig,(ax,bx)=plt.subplots(1,2,figsize=(11.5,5.2),layout='constrained',gridspec_kw={'width_ratios':[1,1.25]})
    for shaft,color in [('ICL','#dc8722'),('SCL','#42a9b5')]:
        ix=np.char.startswith(names,shaft);ax.plot(xy[ix,0],xy[ix,1],'-o',c=color,ms=4,mfc='white',lw=1)
    for k,(r,color) in enumerate(zip(rows,['#bd3934','#2679b0'])):
        center=np.asarray(r['center_mm']);ax.add_patch(Circle(center,r['radius_mm'],fill=False,edgecolor=color,lw=1.5))
        ax.scatter(*center,c=color,s=28,marker='+')
        for name,d in zip(r['anchor_names'],r['distances_mm']):
            point=xy[list(names).index(name)];ax.plot([center[0],point[0]],[center[1],point[1]],c=color,lw=.8,ls=':')
            ax.scatter(*point,s=44,facecolors=color,edgecolors='white',zorder=4)
            offsets={'ICL11':(-3,-14),'ICL9':(0,-14),'ICL3':(-10,16),'ICL2':(0,5),'ICL1':(3,16)}
            ax.annotate(name,point,xytext=offsets.get(name,(3,7)),ha='center' if name.startswith('ICL') else 'left',textcoords='offset points',fontsize=8)
        label_xy=(8.8,8.6) if k==0 else (16.4,1.2)
        ax.text(*label_xy,f"核{r['core']}：圈内{r['inside_count']}/3",ha='center',c=color,fontsize=10)
    ax.set(xlim=(0,20),ylim=(0,20),aspect='equal',xlabel='x (mm)',ylabel='y (mm)',title='已采用端点摘要与初始双核')
    im=bx.imshow(gaps,cmap='YlOrBr',vmin=0,vmax=8,aspect='auto')
    bx.set(xticks=range(3),xticklabels=rows[0]['anchor_names'],yticks=range(len(variants)),yticklabels=[v[0] for v in variants],title='左核平移后的端点—core边缘距离')
    for i in range(gaps.shape[0]):
        for j in range(3):bx.text(j,i,f'{gaps[i,j]:.2f}',ha='center',va='center',c='white' if gaps[i,j]>4.8 else 'black')
    fig.colorbar(im,ax=bx,location='bottom',pad=.13,label='几何间距 (mm)；圈内记为0')
    fig.suptitle('相同半径，没有保留两组端点相同程度的空间范围\n两核半径均约1.75mm；红/蓝点是已采用的A/B几何先验，不是本事件的起燃定位',fontsize=12)
    for ext in ['png','pdf']:fig.savefig(F/f'endpoint_cloud_and_core_extent.{ext}',dpi=190)
    plt.close(fig)
    (F/'README.md').write_text('\n\n'.join(f'### endpoint_cloud_and_core_extent.{ext}\n\n左图保留实际2D电极布局，核中心来自已采用的三触点几何摘要，两核使用同一约1.75mm半径；右图列出左核几种已测平移下，各先验端点到core边缘的距离。距离只反映当前空间近似，不要求触点必须位于core内，也不是患者真实源范围估计。**关注点**：质心初值保留了位置而没有保留端点云的分散范围；下一步应区分core范围与向外连接能否补齐招募，不能据此硬编码具体传播路线。' for ext in ['png','pdf'])+'\n')
    print({'output':str(dest),'rows':rows},flush=True)


if __name__=='__main__':main()
