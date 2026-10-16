"""Original Fig2 rose/readout/frame painters with new computed input arrays."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from analysis.display_field import _smooth_rank_field_mm
from analysis.envelope_normalization import values_at
from plotting.paper import fig2_rose as rose,fig2_dynamic as dynamic


def projection(xyz,geometry):
    fig=plt.figure(figsize=(9,4));ax=fig.add_subplot(121,projection='3d');p=geometry['plane'];points=geometry['points_mm']
    for rod in np.unique(geometry['shafts']):
        ix=np.flatnonzero(geometry['shafts']==rod);ax.plot(*xyz[ix].T,'o-',c='#347D89',ms=3,lw=.7)
    span=max(np.ptp(points,axis=0));U,V=np.meshgrid([-span*.65,span*.65],[-span*.45,span*.45]);P=p['origin']+U[...,None]*p['u']+V[...,None]*p['w'];ax.plot_surface(P[:,:,0],P[:,:,1],P[:,:,2],color='#BED8DC',alpha=.2)
    ax.set(xlabel='x (mm)',ylabel='y (mm)',zlabel='z (mm)');ax.view_init(25,-65)
    ax=fig.add_subplot(122)
    for rod in np.unique(geometry['shafts']):
        ix=np.flatnonzero(geometry['shafts']==rod);ax.plot(*points[ix].T,'o-',c='#347D89',mec='white',ms=5,lw=.7)
        ax.text(*points[ix[0]],rod,ha='right',fontsize=10,color='#48656D')
    ax.add_patch(Circle(points[-1],6,ec='#347D89',fc='#BED8DC',alpha=.4));ax.text(*(points[-1]+[1,7]),'σ = 6 mm',fontsize=10)
    ax.set(xlim=geometry['display_xlim_mm'],ylim=geometry['display_ylim_mm'],aspect='equal',xlabel='shared TA axis (mm)',ylabel='y (mm)');fig.tight_layout();return fig


def direction_scores(payload,gain):
    fig=plt.figure(figsize=(9,4.2));ax=fig.add_axes([.05,.15,.32,.72],projection='polar');edges=np.linspace(-np.pi,np.pi,19)
    rose.draw_combined_probability_rose(ax,payload,edges=edges,rmax=rose._rose_max([payload],edges),row_label='E1146')
    ax=fig.add_axes([.50,.22,.46,.65]);x=[gain['timing_only_score'],gain['timing_plus_space_score']]
    ax.plot(x,[0,0],c='#D2D2D2',lw=.8);ax.scatter(x,[0,0],c=['white','#C75D3A'],edgecolors=['#5B7894','#C75D3A'],s=35,zorder=3)
    ax.axvline(0,color='.65',ls='--',lw=.8);ax.set(yticks=[0],yticklabels=['E1146'],xlabel='Held-out direction score',ylim=(-.7,.7),xlim=(-.1,max(.8,max(x)+.05)))
    fig.legend(handles=[Line2D([],[],color='#B2182B',lw=2,label='TA'),Line2D([],[],color='#2166AC',lw=2,label='TB'),Line2D([],[],color='.3',ls='--',label='Timing only'),Line2D([],[],color='.3',label='Timing + 3D')],ncol=4,loc='upper center',frameon=False,fontsize=10)
    return fig


def event_fields(payload,geometry):
    fig=plt.figure(figsize=(13,6));gs=fig.add_gridspec(2,6,width_ratios=[1,1,1,1,1,.04],left=.08,right=.94,bottom=.12,top=.9,hspace=.6,wspace=.30)
    order=np.argsort(geometry['points_mm'][:,0]);points=geometry['points_mm']
    for row,event in enumerate(payload['events']):
        label=['TA','TB'][row];ax=fig.add_subplot(gs[row,0]);xlim=(max(-100,event['tile_lo_ms']),min(150,event['tile_hi_ms']))
        dynamic._readout(ax,event,geometry,order,xlim,None,template=label,row_label=label,show_xlabel=row==1)
        for j,t in enumerate(payload['times_ms']):
            raw=values_at(event['env_z'],event['t'],event['t0'],t,avg_ms=3.)
            values,_=dynamic._static_frame_relative_values(raw,event['part'])
            X,Y,F,_,_=_smooth_rank_field_mm(*points.T,values,event['part'].astype(float),geometry['display_xlim_mm'],geometry['display_ylim_mm'],6.)
            ax=fig.add_subplot(gs[row,j+1]);dynamic._panel(ax,(X,Y),F,points,values,event['part'],1,f'{t:+.0f} ms',show_y=j==0,show_x=row==1 and j==1)
        cb=fig.colorbar(ScalarMappable(norm=dynamic.FIELD_DISPLAY_NORM,cmap=dynamic.CMAP),cax=fig.add_subplot(gs[row,5]));cb.set_label('Relative HFO envelope')
    return fig


def template_fields(fields,geometry):
    fig,axes=plt.subplots(2,1,figsize=(4.5,6),layout='constrained');p=geometry['points_mm']
    for k,(ax,f) in enumerate(zip(axes,fields)):
        im=ax.imshow(f['field'],extent=(f['X'].min(),f['X'].max(),f['Y'].min(),f['Y'].max()),origin='lower',cmap='viridis',vmin=0,vmax=1)
        use=f['support']>0;ax.scatter(*p[use].T,c=f['values'][use],cmap='viridis',vmin=0,vmax=1,s=35,edgecolor='white',lw=1.2)
        ax.set(xlabel='shared TA axis (mm)' if k==1 else '',ylabel='y (mm)');ax.text(-.3,.5,['TA field','TB field'][k],transform=ax.transAxes,rotation=90,ha='center',va='center',color=['#B2182B','#2166AC'][k],weight='bold')
    cb=fig.colorbar(im,ax=axes,fraction=.035,pad=.03,ticks=[0,.5,1]);cb.ax.set_yticklabels(['0 early','0.5','1 late']);return fig
