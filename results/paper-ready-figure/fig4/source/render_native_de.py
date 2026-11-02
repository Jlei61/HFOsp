"""Reusable, native-axes D/E panels. No global rcParams or figure resizing.

rectD/rectE are (left, bottom, width, height) in normalized figure coordinates.
Designed for 144 x 78 mm each; physical column anchors align the middle and bottom rows.
The default shows a fixed post-burn-in interval containing the first three local bursts, every E/I neuron in the fixed local ROI,
the same native current-LFP site. LFP vertical limits are explicit and independent to resolve sparse background.
"""
from pathlib import Path
import importlib.util
import json
import numpy as np
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

DATA = Path(__file__).with_name('native_de_data')


def render_de(fig, rectD, rectE, *, data_dir=None, window_ms=None, fontsize=10.5, letters=False):
    data_dir=Path(data_dir) if data_dir else DATA
    info=json.loads((data_dir/'observables.json').read_text())
    candidate=json.loads((data_dir/'candidate.json').read_text())
    begin,end=(1500.,2900.) if window_ms is None else map(float,window_ms)
    if begin<0 or end<=begin or end>json.loads((data_dir/'design.json').read_text())['duration_ms']:
        raise ValueError('Requested window outside actual native data')
    with np.load(data_dir/'substrate.npz') as s:
        pos=s['positions_E'].copy();sites=s['lfp_sites_mm'].copy()
        roi_E=np.flatnonzero(s['input_patch_E']>=0)
        roi_I=np.flatnonzero(s['input_patch_I']>=0)
    arrays={}
    for arm in ['D','E']:
        with np.load(data_dir/f'{arm}_native.npz') as z:
            arrays[arm]={k:z[k].copy() for k in ['spike_steps','spike_ids','dt_ms','lfp_proxy','vtheta']}
    ne,ni=info['n_E'],info['n_I']
    selected_ids=np.r_[roi_E,ne+roi_I]
    roi_order=np.full(ne+ni,-1,int);roi_order[selected_ids]=np.arange(len(selected_ids))
    nre,nri=len(roi_E),len(roi_I)
    lfp_limits=[]
    for z in arrays.values():
        dt=float(z['dt_ms']); y=z['lfp_proxy'][round(begin/dt):round(end/dt),0]
        lfp_limits += [float(y.min()),float(y.max())]
    low,high=min(lfp_limits),max(lfp_limits)
    pad=max((high-low)*.07,.05)
    low,high=low-pad,high+pad
    raster_marker_area_pt2={'D':5.0,'E':3.0}
    axes={}
    spec=importlib.util.spec_from_file_location('shared_overview',Path(__file__).with_name('neuron_overview.py'))
    overview=importlib.util.module_from_spec(spec);spec.loader.exec_module(overview)
    width_mm,height_mm=fig.get_size_inches()*25.4
    for arm,rect in [('D',rectD),('E',rectE)]:
        x,y,w,h=map(float,rect)
        def ax_at_mm(u,v,ww,hh):
            ax=fig.add_axes([x+u/width_mm,y+v/height_mm,ww/width_mm,hh/height_mm])
            ax.tick_params(labelsize=9.5,length=2.5,width=.6,pad=2)
            for sp in ax.spines.values():sp.set_linewidth(.6)
            return ax
        # Physical column anchors match the visible F/G/H/I axes below.
        field_left_mm=10. if arm=='D' else 4.
        readout_left_mm=86. if arm=='D' else 84.
        field_bottom_mm=13.14;field_size_mm=60.
        field=ax_at_mm(field_left_mm,field_bottom_mm,field_size_mm,field_size_mm)
        field.set_facecolor('white')
        lfp=ax_at_mm(readout_left_mm,field_bottom_mm+38.,57.,22.)
        raster=ax_at_mm(readout_left_mm,field_bottom_mm,57.,30.)
        z=arrays[arm]
        center=candidate['centers_mm'][0];radius=candidate['radii_mm'][0]
        # Only E's core region is shaded; the surrounding sheet stays white.
        if arm=='E':
            field.add_patch(Circle(center,radius,facecolor='#B8A1CF',edgecolor='none',zorder=1))
        neuron_layout=overview.draw_neurons(field,shift_mm=10.)
        if arm=='E':
            field.add_patch(Circle(center,radius,fill=False,edgecolor='#624B7D',lw=1.0,zorder=4))
            field.annotate('Core',xy=(center[0],center[1]+radius),xytext=(0,5),textcoords='offset points',
                ha='center',va='bottom',fontsize=fontsize,color='#4E3B64',zorder=6)
        else:
            # The same sampled ROI exists in D, but D has no core.
            field.add_patch(Circle(center,radius,fill=False,edgecolor='#7E718B',lw=.9,ls=(0,(3,2)),zorder=4))
        field.plot(sites[0,0],sites[0,1],marker='+',ms=5,mew=.9,color='#333333',zorder=7)
        field.set(xlim=(0,20),ylim=(0,20),xticks=[0,10,20],yticks=[0,10,20])
        field.set_xlabel('x (mm)',fontsize=fontsize,labelpad=2)
        field.set_ylabel('y (mm)',fontsize=fontsize,labelpad=2)
        field.set_aspect('equal')
        field.set_title('No core' if arm=='D' else 'One core',fontsize=fontsize,loc='left',pad=3)
        cbax=None
        dt=float(z['dt_ms']);lo,hi=round(begin/dt),round(end/dt)
        t=np.arange(lo,hi)*dt/1000.
        lfp.plot(t,z['lfp_proxy'][lo:hi,0],color='#202020',lw=.5,rasterized=True)
        lfp.set_xlim(begin/1000,end/1000)
        if arm=='D':
            native=z['lfp_proxy'][lo:hi,0];span=native.max()-native.min();pad=max(.1,.10*span)
            lfp.set_ylim(native.min()-pad,native.max()+pad)
        else:lfp.set_ylim(-65.3618721606485,1143.7893516127006)
        lfp.set_ylabel('LFP (a.u.)',fontsize=fontsize,labelpad=2)
        lfp.yaxis.set_major_locator(MaxNLocator(nbins=3,integer=True))
        lfp.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
        lfp.tick_params(labelbottom=False)
        lfp.spines[['top','right']].set_visible(False)
        st=z['spike_steps'].astype(float)*dt;ids=z['spike_ids']
        selected=(st>=begin)&(st<end)&(roi_order[ids]>=0)
        for kind,color in [(ids<ne,'#c84040'),(ids>=ne,'#386c9d')]:
            m=selected&kind
            raster.scatter(st[m]/1000.,roi_order[ids[m]],s=raster_marker_area_pt2[arm],marker='.',color=color,linewidths=0,rasterized=True)
        if not np.any(selected):
            raster.text(.5,.5,'0 spikes',transform=raster.transAxes,ha='center',va='center',
                        fontsize=fontsize,color='#666666')
        raster.axhline(nre-.5,color='#999999',lw=.4)
        raster.set(xlim=(begin/1000,end/1000),ylim=(-.5,nre+nri-.5),yticks=[nre/2,nre+nri/2],
                   yticklabels=['E','I'])
        raster.set_ylabel('Neuron ID',fontsize=fontsize,labelpad=2)
        raster.set_xlabel('Time (s)',fontsize=fontsize,labelpad=2)
        raster.tick_params(axis='y',length=0,pad=6.5,labelsize=11.5)
        for label,color in zip(raster.get_yticklabels(),['#D84840','#347FB3']):
            label.set_color(color);label.set_fontweight('bold')
        raster.spines[['top','right']].set_visible(False)
        if letters:fig.text(x,y+h,arm,fontsize=fontsize+4,fontweight='bold',va='top')
        axes[arm]=dict(field=field,lfp=lfp,raster=raster,colorbar=cbax)
    # Anchor each raster label to its LFP label after measuring the tick widths.
    # Figure coordinates keep alignment fixed through PNG/PDF and panel exports.
    fig.canvas.draw()
    label_positions={}
    for arm,group in axes.items():
        label=group['lfp'].yaxis.label
        pixel_x=label.get_transform().transform(label.get_position())[0]
        label_x=fig.transFigure.inverted().transform((pixel_x,0))[0]
        for key in ('lfp','raster'):
            ax=group[key];box=ax.get_position()
            ax.yaxis.set_label_coords(label_x,(box.y0+box.y1)/2,transform=fig.transFigure)
        label_positions[arm]=label_x*width_mm
    return dict(axes=axes,raster_group_labels={'font_size_pt':11.5,'font_weight':'bold','pad_pt':6.5,'colors':{'E':'#D84840','I':'#347FB3'}},core_colorbar_shown=False,core_mask_shown=True,core_outline_shown=True,D_readout_ROI_outline_shown=True,field_background={'D':'white','E':'white'},E_core_fill='#B8A1CF',shared_neuron_layout=neuron_layout,readout_ylabel_x_mm=label_positions,core_fill_meaning='Categorical purple fill inside E core only; complement unshaded',field_raster_baselines_aligned=True,axis_label_font_pt=10.5,axis_tick_font_pt=9.5,raster_marker_area_pt2=raster_marker_area_pt2,window_ms=[begin,end],lfp_ylim={k:list(v['lfp'].get_ylim()) for k,v in axes.items()},lfp_axes_independently_scaled=True,
                neuron_counts=dict(E=nre,I=nri),whole_network_counts=dict(E=ne,I=ni),
                raster_neuron_ids=selected_ids.tolist(),all_ROI_spikes_displayed=True,
                same_neuron_order=True,display_window_rule='Fixed 1500-2900 ms window after burn-in; first three complete local bursts in E; identical D/E interval',full_analysis_window_ms=info['analysis_and_display_window_ms'],field_overlay='Shared spaced E/I schematic in B/D/E; D unchanged white background and dashed ROI; E purple core interior with white complement',colors=dict(E='#c84040',I='#386c9d'),
                measured_lowering_range_mV={a:[float((18-z['vtheta'][:ne]).min()),float((18-z['vtheta'][:ne]).max())] for a,z in arrays.items()},
                metadata=info)


if __name__=='__main__':
    import argparse
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    fig=plt.figure(figsize=(320/25.4,96/25.4))
    result=render_de(fig,(9/320,10/96,144/320,78/96),(171/320,10/96,144/320,78/96),letters=True)
    for suffix in ['png','pdf','svg']:
        fig.savefig(a.output_dir/f'native_de_preview.{suffix}',dpi=240,facecolor='white')
    (a.output_dir/'preview_metadata.json').write_text(json.dumps({k:v for k,v in result.items() if k not in ['axes','metadata']},indent=2)+'\n')
