#!/usr/bin/env python3
"""Fig5 display revision: actual matched electrodes, selected clinical example."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.colors import Normalize, TwoSlopeNorm, ListedColormap, to_rgba
from matplotlib.cm import ScalarMappable
from matplotlib.text import Annotation
import plot_topic4_fig5_native_dense_v11 as v11
from analyze_topic4_fig5_patient_readout import power

prior=v11.prior; BASE=v11.BASE; ROOT=v11.ROOT
OUT=BASE/'layout_v12'; FIG=OUT/'figures'; COMP=BASE/'patient_readout_v1'


def save(fig,name):
    FIG.mkdir(parents=True,exist_ok=True)
    fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight',pad_inches=.15)
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight',pad_inches=.15)
    plt.close(fig)


def all_axes(fig):
    axes=[]
    def visit(ax):
        if ax in axes:return
        axes.append(ax)
        for child in ax.child_axes:visit(child)
    for ax in fig.axes:visit(ax)
    return axes


def typography(fig):
    for ax in all_axes(fig):
        ax.tick_params(axis='both',labelsize=16,pad=4)
        ax.xaxis.label.set_fontsize(18);ax.yaxis.label.set_fontsize(18)
        if hasattr(ax,'zaxis'):
            ax.tick_params(axis='z',labelsize=16,pad=4)
            ax.zaxis.label.set_fontsize(18)
            ax.xaxis.labelpad=13;ax.yaxis.labelpad=15;ax.zaxis.labelpad=10
        ax.title.set_fontsize(18);ax._left_title.set_fontsize(20)
        for txt in ax.texts:
            if isinstance(txt,Annotation):continue
            txt.set_fontsize(max(txt.get_fontsize(),15))
        legend=ax.get_legend()
        if legend:
            for text in legend.get_texts():text.set_fontsize(15)


def position_e2_labels(fig):
    """Place heading/footer relative to final text extents at either export size."""
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    header,footer,maps=fig._e2_layout
    inv=fig.transFigure.inverted()
    top=max(inv.transform(ax.title.get_window_extent(renderer).corners())[:,1].max() for ax in maps)
    bottom=min(inv.transform(ax.xaxis.label.get_window_extent(renderer).corners())[:,1].min() for ax in maps)
    gap=.16/fig.get_figheight()
    header.set_transform(fig.transFigure);header.set_position((maps[0].get_position().x0,top+gap));header.set_va('bottom')
    x=(maps[0].get_position().x0+maps[-1].get_position().x1)/2
    footer.set_transform(fig.transFigure);footer.set_position((x,bottom-gap));footer.set_va('top')


def strengthen_left(axes,windows):
    maps=[ax for ax in axes if ax.images]
    for ax in axes:
        for line in ax.lines:
            if len(line.get_xdata())>100:
                rgba=np.array(to_rgba(line.get_color()));rgba[:3]*=.70
                line.set_color(rgba);line.set_alpha(1.)
                line.set_linewidth(max(1.05,line.get_linewidth()*1.2))
        for coll in ax.collections:
            if isinstance(coll,PathCollection) and len(coll.get_offsets())>100:
                colors=coll.get_facecolors().copy();colors[:,:3]*=.67;colors[:,3]=1.
                coll.set_facecolors(colors);coll.set_sizes([1.65]);coll.set_alpha(1.)
        for txt in list(ax.texts):
            if txt.get_text()=='M off':txt.remove()
    colors=v11.RATE_CMAP(np.linspace(0,1,256));colors[:,:3]*=.86
    darker=ListedColormap(colors,name='activity_contrast_v12')
    labels=['Self-limited','Entry','High rate','After refill','High rate']
    for k,(ax,w) in enumerate(zip(maps,windows)):
        ax.images[0].set_cmap(darker)
        ax.set_title(f'{k+1}  {labels[k]}\n{w["time"]:.2f} s',color=w['color'],pad=10)
    return maps


def electrode_map(ax,xy,values,norm,title,centers=None,ylabel=True):
    grid=np.linspace(0,20,201);xx,yy=np.meshgrid(grid,grid)
    d2=(xx[...,None]-xy[:,0])**2+(yy[...,None]-xy[:,1])**2
    w=np.exp(-d2/(2*2.5**2));support=w.sum(-1)
    field=np.sum(w*values,axis=-1)/np.maximum(support,1e-15)
    alpha=np.clip(support/(.32*support.max()),0,1)
    ax.imshow(field,origin='lower',extent=[0,20,0,20],cmap='Blues',norm=norm,
              alpha=alpha,interpolation='nearest',rasterized=True)
    ax.scatter(xy[:,0],xy[:,1],c=values,cmap='Blues',norm=norm,s=55,
               edgecolors='#121212',linewidths=1,zorder=5)
    # Anatomical shaft identity is fixed before looking at power.
    ax.text(7.0,17.4,'SCL',fontsize=15,ha='center',color='#173a51')
    ax.text(11.5,2.0,'ICL',fontsize=15,ha='center',color='#173a51')
    if centers is not None:prior.core_outlines(ax,centers,1.5,labels=True,color='#d35c21')
    ax.set(xlim=(0,20),ylim=(0,20),xticks=[0,10,20],yticks=[0,10,20])
    ax.set_xlabel('x (model mm)');ax.set_ylabel('y (model mm)' if ylabel else '')
    ax.set_title(title,pad=12)
    if not ylabel:ax.set_yticklabels([])
    return field


def comparison(fig,spec,a,summary):
    rows=spec.subgridspec(3,1,height_ratios=[.13,1,.10],hspace=.20)
    head=fig.add_subplot(rows[0]);head.axis('off')
    header=head.text(0,.6,'E2  Early energy · 1–150 Hz',fontsize=20,weight='bold')
    gs=rows[1].subgridspec(1,5,width_ratios=[1,.045,.24,1,.045],wspace=.13)
    selected=summary['selected'];xy=a['contact_xy']
    details=[];map_axes=[]
    for k,(values,title,label) in enumerate([
        (a['model_dB'],'Model readout\n10–11 s','Change (dB)'),
        (a['patient_robust_z'],f'E1146 · {selected["public_seizure"]}\n0–10 s from onset','Power (robust z)')]):
        norm=Normalize(0,float(np.ceil(values.max())))
        ax=fig.add_subplot(gs[0 if k==0 else 3])
        map_axes.append(ax)
        field=electrode_map(ax,xy,values,norm,title,a['centers_mm'] if k==0 else None,ylabel=k==0)
        cb=fig.colorbar(ScalarMappable(norm=norm,cmap='Blues'),cax=fig.add_subplot(gs[1 if k==0 else 4]),
                       ticks=[0,norm.vmax/2,norm.vmax])
        cb.set_label(label,labelpad=8)
        details.append(dict(values=values.tolist(),color_limits=[0,norm.vmax],field_range=[float(field.min()),float(field.max())]))
    footer=fig.add_subplot(rows[2]);footer.axis('off')
    footer_text=footer.text(.5,.3,f'Selected example · 15 contacts · ρ = {selected["rho"]:.2f}',ha='center',fontsize=16)
    fig._e2_layout=(header,footer_text,map_axes)
    return details


def native_check(a,summary):
    native=np.load(v11.NATIVE/'native_fields.npz')
    ampa=native['cell_mean_ampa'];gaba=native['cell_mean_applied_gaba']
    ncols=ampa.shape[1]
    native_db=np.empty(ncols)
    for start in range(0,ncols,100):
        x=np.add(ampa[:,start:start+100],gaba[:,start:start+100],dtype=np.float64)
        p,t=power(x,2000.)
        base=(t>=1)&(t<=7.5);early=np.isclose(t,10.5)
        native_db[start:start+100]=10*np.log10(p[early][0]/p[base].mean(0))
    np.savez_compressed(OUT/'native_readout_comparison.npz',native_dB=native_db,
                        model_dB=a['model_dB'],patient_robust_z=a['patient_robust_z'])
    fig=plt.figure(figsize=(19,6.0))
    gs=fig.add_gridspec(1,8,left=.065,right=.955,bottom=.21,top=.80,
        width_ratios=[1,.045,.23,1,.045,.23,1,.045],wspace=.16)
    limit=float(np.ceil(np.nanmax(abs(native_db))/5)*5)
    ax=fig.add_subplot(gs[0]);im=v11.native_map(ax,native_db,a,
        TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),'RdBu_r',core_labels=True)
    ax.set_title('Native E-cell current\n10–11 s',fontsize=18)
    ax.scatter(a['contact_xy'][:,0],a['contact_xy'][:,1],s=35,facecolors='none',edgecolors='black',lw=.8)
    cb=fig.colorbar(im,cax=fig.add_subplot(gs[1]),ticks=[-limit,0,limit]);cb.set_label('Change (dB)')
    for k,(vals,title,label) in enumerate([
        (a['model_dB'],'Model electrode readout\n10–11 s','Change (dB)'),
        (a['patient_robust_z'],f'E1146 · {summary["selected"]["public_seizure"]}\n0–10 s from onset','Power (robust z)')]):
        j=3+3*k;norm=Normalize(0,np.ceil(vals.max()))
        ax=fig.add_subplot(gs[j]);electrode_map(ax,a['contact_xy'],vals,norm,title,
            a['centers_mm'] if k==0 else None,ylabel=False)
        cb=fig.colorbar(ScalarMappable(norm=norm,cmap='Blues'),cax=fig.add_subplot(gs[j+1]),ticks=[0,norm.vmax/2,norm.vmax])
        cb.set_label(label)
    typography(fig);save(fig,'native_field_and_electrode_readout')
    return dict(native_band_hz=[1,150],native_limits_dB=[-limit,limit],native_grid=[40,40],
                estimator_matches_electrode=True,projection_not_equal_to_native_power=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    a=np.load(COMP/'comparison_arrays.npz');summary=json.loads((COMP/'summary.json').read_text())
    assert summary['model_positive_contacts']==15 and summary['selected']['positive_contacts']==15
    metadata=json.loads((prior.previous.OUT/'figure_metadata.json').read_text())
    original,run,*_=prior.previous.previous.prior.old.previous.source.load_main()
    t,xyz,_=prior.previous.previous.current_coordinates(original)
    paths=prior.previous.previous.previous.previous.complete_paths(t,xyz,run,metadata['windows'])
    saved=np.load(BASE/'layout_v8/trajectory_arrays.npz')
    assert all(np.array_equal(p['coords'],saved[f'path{k}_Z_H_E']) and
        np.array_equal(p['time'],saved[f'path{k}_time_s']) for k,p in enumerate(paths))
    d=np.load(v11.DENSE/'analysis_arrays.npz')
    assert json.loads((v11.DENSE/'analysis_summary.json').read_text())['runs']==147
    fig=plt.figure(figsize=(27,19))
    outer=fig.add_gridspec(1,2,width_ratios=[1.32,1],left=.065,right=.945,top=.965,bottom=.055,wspace=.21)
    before=len(fig.axes)
    left=prior.left_with_boundaries(fig,outer[0],original,run,metadata['windows'],1.5,[10.,11.])
    maps=strengthen_left(fig.axes[before:],metadata['windows'])
    right=outer[1].subgridspec(3,1,height_ratios=[1.08,1.03,.96],hspace=.35)
    phase=right[0].subgridspec(1,3,width_ratios=[.06,1,.22])[1]
    ax=fig.add_subplot(phase,projection='3d')
    prior.previous.plot_summary(ax,t,xyz,paths,run,metadata['windows'],panel_letter=False)
    ax.set_ylabel(r'$H_E$ (mV equiv.)')
    for artist in list(ax.get_children()):
        if isinstance(artist,Annotation):
            if not artist.get_text().isdigit():artist.remove()
            else:artist.xyann=tuple(.68*np.array(artist.xyann));artist.set_fontsize(16)
    ax.set_title('E1  State trajectory',weight='bold',loc='left',pad=15)
    e2=comparison(fig,right[1],a,summary)
    v11.latency(fig,right[2],d)
    typography(fig)
    # Smaller D headings avoid collisions while retaining large axis labels.
    for ax in maps:ax.title.set_fontsize(15)
    for ax in all_axes(fig):
        if ax.get_title(loc='left').startswith('A '):
            for txt in ax.texts:
                if txt.get_text() in ['Refill Z','Release Z']:txt.set_fontsize(15)
    position_e2_labels(fig);fig.canvas.draw()
    qa=dict(trajectory_and_time_bitwise_unchanged=True,continuous_raster=left,
            exact_contact_alignment=summary['exact_name_alignment_pass'],
            clinical_checkpoint_reproduction=summary['checkpoint_reproduction_pass'],
            n_parameter_runs=147,parameter_grid_shape=[7,7],
            overall_title_absent=fig._suptitle is None,
            tick_font_min=min(t.get_fontsize() for ax in all_axes(fig) for t in ax.get_xticklabels()+ax.get_yticklabels()),
            axis_label_font_min=min(ax.xaxis.label.get_fontsize() for ax in all_axes(fig)),
            human_acceptance='PENDING_USER_REVIEW',agent_visual_review='PENDING')
    save(fig,'fig5_manual_core_release_layout_v12')
    fig=plt.figure(figsize=(12.5,7.2))
    spec=fig.add_gridspec(1,1,left=.09,right=.94,top=.95,bottom=.06)[0]
    comparison(fig,spec,a,summary);typography(fig);position_e2_labels(fig);save(fig,'early_energy_model_vs_E1146')
    native=native_check(a,summary)
    doc=dict(source_run=str(BASE/'runs/continuous_refill_release.npz'),
        clinical_analysis=str(COMP),selected=summary['selected'],
        selection_rule=json.loads((COMP/'selection_protocol.json').read_text()),
        clinical_n_candidates=summary['n_candidates'],
        positive_candidates=summary['n_positive_enhancement_candidates'],
        model_window_s=[10,11],clinical_window_s=[0,10],band_hz=[1,150],
        electrode_display='Same fixed contact embedding, 2.5 mm Gaussian interpolation for display only; rho uses unsmoothed 15-contact values.',
        common_amplitude_scale=False,E2=e2,native=native,
        physical_core_radius_mm=1.5,parameter_analysis=str(v11.DENSE),
        original_native_diagnostics=str(v11.FIG),
        scientific_scope='Current-proxy electrode illustration; not independent clinical validation, a full seizure reproduction, or proof of Hopf.')
    (OUT/'figure_metadata.json').write_text(json.dumps(doc,indent=2)+'\n')
    (OUT/'artifact_qa.json').write_text(json.dumps(qa,indent=2)+'\n')
    (OUT/'producer_manifest.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__),ROOT/'scripts/analyze_topic4_fig5_patient_readout.py']},indent=2)+'\n')
    (FIG/'README.md').write_text('''### fig5_manual_core_release_layout_v12.png / .pdf
同一手放双核连续SNN，保留原SEEG、连续raster、Z和原生放电快照，将左侧颜色加深、刻度与轴名统一放大，并去掉整图标题。E2以同一15触点几何展示模型10–11秒1–150 Hz增强和E1146 SZ13临床起始0–10秒的增强；F为完整7×7×3条模拟。
**关注点**：SZ13为25例中17例全部触点增强候选里相关最高的例子，触点Spearman ρ=0.70，属于最佳例示而不是独立检验。模型为dB、患者为robust-z，两者幅度不直接等同；补回Z仍是外部操作。

### early_energy_model_vs_E1146.png / .pdf
放大E2的真实电极读出对照，圆点为原始15通道值，背景仅为固定2.5毫米核的显示插值，未依据功率改变电极位置。模型显示真实1.5毫米core边界，患者图不添加虚构的core。
**关注点**：所有25例的相关性与功率正负均保留在patient_readout_v1；无筛选最高相关的SZ16实际全部低于基线，不能用来展示增强，因此明确另选正增强的SZ13。相关性在未插值触点上计算，不在彩色像素上计算。

### native_field_and_electrode_readout.png / .pdf
同一10–11秒窗并列原生0.5毫米E神经元电流功率、模型电极读出和所选患者示例，均使用1–150 Hz，并让原生场和模型电极使用相同PSD估计器。原生场保留正负dB，不将电极投影后全部正增强推回成神经元处处增强。
**关注点**：相似的电极模式仍可能掩盖不同的底层频谱，原生场及其单细胞高率持续放电的诊断保留在layout_v11中。此图与主图均待用户目视验收。
''')
    print(json.dumps(qa),flush=True)


if __name__=='__main__':main()
