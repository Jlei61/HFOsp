"""Fifteen real contacts, anatomical shaft order, unchanged fitting ranks/times."""
from pathlib import Path
import json
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib import patheffects
from matplotlib.path import Path as MplPath
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyArrowPatch, ConnectionPatch, PathPatch
HERE=Path(__file__).resolve().parent


def _edges(x):
    x=np.asarray(x)
    return np.r_[x[0]-(x[1]-x[0])/2,(x[:-1]+x[1:])/2,x[-1]+(x[-1]-x[-2])/2]


def render_c(fig,rect):
    x0,y0,w,h=map(float,rect)
    with np.load(HERE/'c_panel_data.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    meta=json.loads((HERE/'c_panel_metadata.json').read_text());axes=[];artists=[]
    def box(a,b,c,e):
        ax=fig.add_axes([x0+a*w,y0+b*h,c*w,e*h]);axes.append(ax);return ax
    def text(a,b,s,**kw):
        defaults=dict(fontsize=8,ha='center',va='center');defaults.update(kw)
        t=fig.text(x0+a*w,y0+b*h,s,**defaults);artists.append(t);return t
    n=len(d['contact_names']);assert n==15
    row_height=.040
    rows=[.827-.044*i-(.016 if i>=4 else 0) for i in range(n)]
    frequency_scalebars=[];time_scalebars=[]
    for side,sx,rx,title,subtitle,col in [
        ('model',.070,.292,'Model Events','','#C63D3A'),
        ('patient',.733,.635,'Data Events','','#000000')]:
        text(sx+.1025,.952,title,fontsize=10,fontweight='bold',color=col).set_gid('event_title_'+side)
        if subtitle:
            text(sx+.1025,.905,subtitle,fontsize=8,color=col).set_gid('event_subtitle_'+side)
        text(rx+.016,.905,'Rank',fontsize=8)
        lo,hi=(0,80) if side=='model' else (50,300)
        spectra=d[side+'_tfr_relative'];f=d[side+'_tfr_frequency_hz'];tt=d[side+'_tfr_time_ms']
        spectral=[]
        for i,(name,yy) in enumerate(zip(d['contact_names'],rows)):
            ax=box(sx,yy,.205,row_height);spectral.append(ax)
            ax.pcolormesh(_edges(tt),_edges(f),spectra[i],shading='flat',cmap='coolwarm',vmin=0,vmax=1,rasterized=True,antialiased=False)
            ax.scatter([d[side+'_tfr_centroid_ms'][i]],[d[side+'_tfr_centroid_hz'][i]],s=7,fc='#ffb000',ec='#333333',lw=.35,zorder=5)
            ax.axvline(d[side+'_native_centroid_ms'][i],ymin=.76,ymax=.99,c='#222222',lw=.65,zorder=6)
            # Show actual STFT cell support to both edges. No white padding, no time warp.
            ax.set_xlim(_edges(tt)[[0,-1]])
            ax.set_ylim((0,80) if side=='model' else (50,300))
            ax.set_xticks([]);ax.set_yticks([])
            for sp in ax.spines.values():sp.set_linewidth(.4)
            ax.text(-.045,.50,str(name),transform=ax.transAxes,ha='right',va='center',fontsize=6.3)
            rank=float(d[side+'_native_rank'][i])
            ra=box(rx,yy,.032,row_height)
            ra.imshow([[rank]],cmap='viridis',vmin=1,vmax=15,aspect='auto')
            ra.text(0,0,f'{rank:g}',ha='center',va='center',fontsize=6.5,color='white' if rank<=7 else '#111111')
            ra.set_axis_off()
        # Put the shared ruler inside the left blue region of the last (ICL1) heatmap.
        # A narrow white keyline keeps the black marks readable without adding a box.
        scale_ax=spectral[-1];bx=.055;cap=.018
        for suffix,xx,yy in [('stem',[bx,bx],[0,1]),
                             ('bottom',[bx-cap,bx+cap],[0,0]),
                             ('top',[bx-cap,bx+cap],[1,1])]:
            line=Line2D(xx,yy,transform=scale_ax.transAxes,color='black',lw=1.0,
                        solid_capstyle='butt',clip_on=True,zorder=9,
                        path_effects=[patheffects.withStroke(linewidth=1.8,foreground='white')])
            line.set_gid('frequency_scale_'+side+'_'+suffix)
            scale_ax.add_line(line);artists.append(line)
        label=f'{lo}–{hi} Hz'
        label_artist=scale_ax.text(.100,.5,label,transform=scale_ax.transAxes,
            ha='left',va='center',fontsize=5.0,color='black',zorder=9,clip_on=True,
            path_effects=[patheffects.withStroke(linewidth=.50,foreground='white')])
        label_artist.set_gid('frequency_scale_'+side+'_label');artists.append(label_artist)
        frequency_scalebars.append(dict(side=side,frequency_limits_hz=[lo,hi],
            span_hz=hi-lo,height_in_panel=row_height,label=label,
            location='Inside ICL1 heatmap, left blue baseline region',
            anchor_axes_fraction=[bx,0,1],label_axes_fraction=[.100,.5],
            meaning='A single channel TFR height; not the full electrode stack',
            color='black',outline='thin white keyline; no opaque label background'))
        # Join a calibrated horizontal time ruler to the frequency ruler's foot.
        # Data x coordinates keep the length exactly 50 ms for each native time extent.
        time_span_ms=50.0
        xmin,xmax=scale_ax.get_xlim()
        tx0=xmin+bx*(xmax-xmin);tx1=tx0+time_span_ms
        time_transform=scale_ax.get_xaxis_transform()
        for suffix,xx,yy in [('stem',[tx0,tx1],[0,0]),
                             ('end',[tx1,tx1],[0,.16])]:
            line=Line2D(xx,yy,transform=time_transform,color='black',lw=1.0,
                        solid_capstyle='butt',clip_on=False,zorder=10,
                        path_effects=[patheffects.withStroke(linewidth=1.8,foreground='white')])
            line.set_gid('time_scale_'+side+'_'+suffix)
            scale_ax.add_line(line);artists.append(line)
        time_label=scale_ax.text((tx0+tx1)/2,-.25,'50 ms',transform=time_transform,
            ha='center',va='top',fontsize=5.0,color='black',zorder=10,clip_on=False)
        time_label.set_gid('time_scale_'+side+'_label');artists.append(time_label)
        time_scalebars.append(dict(side=side,span_ms=time_span_ms,
            data_endpoints_ms=[tx0,tx1],axis_limits_ms=[xmin,xmax],
            length_axes_fraction=time_span_ms/(xmax-xmin),
            location='Joined to lower foot of ICL1 frequency ruler in left blue region',
            label='50 ms',label_position='below horizontal ruler',color='black'))
        for i in range(n-1):
            if str(d['contact_names'][i])[:3]!=str(d['contact_names'][i+1])[:3]:continue
            line=ConnectionPatch(xyA=(d[side+'_tfr_centroid_ms'][i],d[side+'_tfr_centroid_hz'][i]),xyB=(d[side+'_tfr_centroid_ms'][i+1],d[side+'_tfr_centroid_hz'][i+1]),coordsA=spectral[i].transData,coordsB=spectral[i+1].transData,color='#9c5a00',lw=.5,alpha=.8,zorder=4,clip_on=False)
            fig.add_artist(line);artists.append(line)
    text(.480,.770,'Loss',fontsize=15,fontweight='bold',color='#483D8B')
    arrow=FancyArrowPatch((x0+.330*w,y0+.698*h),(x0+.630*w,y0+.698*h),transform=fig.transFigure,arrowstyle='<->',mutation_scale=11,color='#7367BE',lw=1.9)
    fig.add_artist(arrow);artists.append(arrow)
    # Three larger rows preserve the exact objective and leave the spectra fixed.
    # The prefactor applies to all three rows; a left curly brace groups them.
    text(.333,.427,r'$\mathrm{Loss}=\frac{1}{3}$',fontsize=10.5,color='#483D8B',ha='left').set_gid('loss_equation_lhs')
    outer_verts=[(.455,.654),(.443,.654),(.444,.622),(.444,.602),
                 (.444,.490),(.444,.436),(.434,.4245),
                 (.444,.413),(.444,.360),(.444,.247),
                 (.444,.227),(.443,.195),(.455,.195)]
    outer_brace=PathPatch(MplPath([(x0+a*w,y0+b*h) for a,b in outer_verts],
                                 [MplPath.MOVETO]+[MplPath.CURVE4]*12),
                         transform=fig.transFigure,fc='none',ec='#483D8B',lw=.9)
    outer_brace.set_gid('loss_outer_brace_left')
    fig.add_artist(outer_brace);artists.append(outer_brace)
    loss_terms=[]
    for i,(g,label,color,yy) in enumerate([('r','Rank','#3D5C6F',.615),
                                         ('o','Order','#E47159',.460),
                                         ('p','Participation','#6B7546',.305)]):
        cx=.530
        symbol=rf'$\mathcal{{L}}_{g}(\hat{{{g}}},{g})$'
        text(cx,yy,symbol,fontsize=11,color=color).set_gid('loss_term_'+label)
        if i>0:text(.464,yy,'+',fontsize=11,color=color).set_gid('loss_plus_'+g)
        half=.049;top=yy-.030;shoulder=yy-.042;cusp=yy-.056
        verts=[(cx-half,top),(cx-half,shoulder),(cx-.014,top),(cx-.007,shoulder),
               (cx-.004,shoulder),(cx-.004,cusp),(cx,cusp),
               (cx+.004,cusp),(cx+.004,shoulder),(cx+.007,shoulder),
               (cx+.014,top),(cx+half,shoulder),(cx+half,top)]
        brace=PathPatch(MplPath([(x0+a*w,y0+b*h) for a,b in verts],
                               [MplPath.MOVETO]+[MplPath.CURVE4]*12),
                        transform=fig.transFigure,fc='none',ec=color,lw=.9)
        brace.set_gid('loss_underbrace_'+label);fig.add_artist(brace);artists.append(brace)
        text(cx,yy-.083,label,fontsize=9,color=color).set_gid('loss_label_'+label)
        loss_terms.append(dict(symbol=symbol,label=label,color=color,center=cx,
                               row_y=yy,term_font_pt=11,label_font_pt=9))
    ca=box(.961,rows[-1],.011,rows[0]+row_height-rows[-1])
    cb=fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap='coolwarm'),cax=ca,orientation='vertical',ticks=[0,1])
    cb.ax.tick_params(labelsize=7,length=2,pad=1);cb.outline.set_linewidth(.4)
    text(.966,.887,'Rel.',fontsize=6.5)
    meta['display']=dict(event_titles={'model':'Model Events','patient':'Data Events'},event_subtitles={},patient_title_color='#000000',patient_title_weight='bold',all_15_contacts=True,vertical_colorbar=True,frequency_limits_hz={'model':[0,80],'patient':[50,300]},no_horizontal_white_padding=True,TFR_centroid_legend=False,rank_is_frozen_fitting_rank=True,patient_rank_left_of_heatmap=True,arrows_point_to_rank_columns=True,colorbar_location='right of patient heatmaps',signal_and_frequency_subtitles=False,frequency_scalebars=frequency_scalebars,time_scalebars=time_scalebars,scalebar_label_font_pt=5.0,time_axis_style="compact 50-ms ruler",loss_display='Loss = (1/3) [L_r(r_hat,r) + L_o(o_hat,o) + L_p(p_hat,p)]; the same actual normalized training components displayed in G and I',formula_location='Three vertical rows below central Loss arrow, between rank columns',formula_layout='vertical',formula_enclosure='left curly brace only',formula_plus_positions='left of second and third rows',formula_prefactor=1/3,formula_term_font_pt=11,formula_label_font_pt=9,loss_terms=loss_terms)
    return dict(axes=axes,artists=artists,metadata=meta,rect=list(rect))
