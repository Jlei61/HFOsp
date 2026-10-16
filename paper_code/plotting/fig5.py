"""New native trajectories supplied to the Fig5 reference-layout painters."""
import numpy as np
import matplotlib.pyplot as plt
from plotting.paper import fig5 as paper
from modeling.src.topic4_node_dualmode import sheet_bin_indices
from analysis.display_field import _smooth_rank_field_mm


def prepare(run,eta,onset):
    # The notebook keeps both variables; the 2026-10-11 panel uses eta*M.
    dt=run['parameters']['dt'];ne=len(run['ec']);slow=run['slow'];n=len(run['rate_E'])//round(1/dt)
    counts=np.rint(run['rate_E'][:n*round(1/dt)].reshape(n,-1).sum(1)*ne*dt/1000).astype(int)
    bins,_=sheet_bin_indices(run['positions'][:ne],bin_mm=1.,sheet_mm=20.)
    z=np.zeros((len(slow['time_ms']),6));z[:,0]=slow['z_mean'];z[:,5]=slow['z_core_a']
    m=np.column_stack([slow['m_mean'],slow['m_core_a']])
    a=dict(contact_names=run['contact_names'],lfp_native_time_ms=run['current_time_ms'],lfp_native=run['contact_current'].T,
        slow_time_ms=slow['time_ms'],Z=z,M=m,spikes_1ms=np.column_stack([counts,counts*0]),
        field_1ms=run['field_1ms'].reshape(-1,400),cell_e_counts=np.bincount(bins,minlength=400),
        centers_mm=run['centers'],core_radius_mm=run['radius_mm'])
    np.testing.assert_array_equal(a['field_1ms'].sum(1),counts)
    duration=n/1000;center=onset if onset is not None else duration-1
    # Preserve the reference display's offsets from its measured t120=15.0508 s.
    # New dynamics set the absolute times; numbered samples do not assert new state labels.
    offsets=np.array([14.6,15.045,15.075,15.11,15.3])-15.0508
    times=np.clip(center+offsets,.05,duration-.05)
    samples=[dict(number=i+1,time_s=float(t),color=c,label='') for i,(t,c) in enumerate(zip(times,['#267ba8','#9b8d32','#bd6c35','#ab4255','#ba263c']))]
    paper.ETA=eta;paper.RECRUIT=max(.05,float(center)-.1508)
    paper.WINDOW=(max(0,center-.7508),min(duration,center+.3492))
    paper.TRAJECTORY_WINDOW=(0,n//5*.005)
    # Original rate binning requires complete 5-ms groups; only the trailing incomplete bin is omitted.
    a['spikes_1ms']=a['spikes_1ms'][:n//5*5]
    return dict(a=a,samples=samples,onset=float(center),window=paper.WINDOW,entry_observed=onset is not None)


def waveform(context):
    fig,ax=plt.subplots(figsize=(11,5));fig.subplots_adjust(left=.1,right=.98,bottom=.12,top=.85)
    paper.draw_a(ax,context['a'],context['samples'],context['onset']);ax.tick_params(axis='x',labelbottom=True);ax.set_xlabel('Time (s)')
    return fig


def slow_variables(context):
    fig,axes=plt.subplots(2,1,figsize=(10,4),sharex=True);fig.subplots_adjust(left=.12,right=.98,bottom=.16,top=.78,hspace=.2)
    paper.draw_slow(*axes,context['a'],context['samples'],context['onset']);return fig


def spatial_activity(context):
    fig=plt.figure(figsize=(12,3));spec=fig.add_gridspec(1,1,left=.06,right=.9,bottom=.2,top=.78)[0]
    paper.draw_native_maps(fig,spec,context['a'],context['samples']);return fig


def state_trajectory(context):
    fig=plt.figure(figsize=(7,5));spec=fig.add_gridspec(1,1,left=.0,right=.8,bottom=.15,top=.98)[0]
    end=paper.TRAJECTORY_WINDOW[1];start=context['onset']
    paper.draw_zm_trajectory(fig,spec,context['a'],context['samples'],(max(0,min(start,end-.01)),min(start+1,end)))
    return fig


def onset_field(run,result,geometry,*,normalized=True):
    from plotting.paper.onset_field import _draw_field,_normalize_minmax,_display_geometry
    fig,ax=plt.subplots(figsize=(5,4),layout='constrained')
    if result['status']!='OK':
        ax.text(.5,.5,result['reason'],ha='center',transform=ax.transAxes);return fig
    scorer=geometry['scorer']
    values=np.asarray(result['robust_z'])[geometry['contact_mask']];support=np.asarray(scorer['support'])
    fz=_display_geometry(geometry)
    mapped=_draw_field(ax,fz,_normalize_minmax(values),support,cmap='Blues',
        colorbar_values=np.array([0.,1.]) if normalized else values,title='Model onset',title_color='black',show_y=True)
    cb=fig.colorbar(mapped,ax=ax,fraction=.04)
    cb.set_label('Normalized z field' if normalized else 'Baseline robust z')
    return fig
