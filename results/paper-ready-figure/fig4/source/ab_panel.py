"""Vector schematic with geometric neuron/terminal anchors and frozen sheet/readout."""
from pathlib import Path
import importlib.util
import numpy as np
from matplotlib.patches import Polygon, Circle, FancyArrowPatch, Rectangle, ConnectionPatch
from matplotlib.lines import Line2D
HERE=Path(__file__).parent
RED,BLUE='#D84840','#347FB3'


def draw_a(fig, rect):
    ax=fig.add_axes(rect);ax.set(xlim=(0,76),ylim=(0,70));ax.set_axis_off()
    def p(u,v,upper=True):return np.array([6+55*np.asarray(u)+13*np.asarray(v),(39 if upper else 10)+16*np.asarray(v)+0*np.asarray(u)])
    def excite(a,b,rad=0):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=7,
            shrinkA=0,shrinkB=0,connectionstyle=f'arc3,rad={rad}',color=RED,lw=.7,zorder=5))
    bars=[]
    def inhibit(a,b,rad=0,tangent=None):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-',shrinkA=0,shrinkB=0,
            connectionstyle=f'arc3,rad={rad}',color=BLUE,lw=.75,zorder=6))
        d=np.asarray(tangent if tangent is not None else np.asarray(b)-a,float);d/=np.linalg.norm(d)
        orth=np.array([-d[1],d[0]])*.72
        ax.plot([b[0]-orth[0],b[0]+orth[0]],[b[1]-orth[1],b[1]+orth[1]],color=BLUE,lw=.8,zorder=7)
        bars.append(1.44)
    for upper in [False,True]:
        corners=np.array([p(0,0,upper),p(1,0,upper),p(1,1,upper),p(0,1,upper)])
        ax.add_patch(Polygon(corners,fc='#FAFAFA',ec='#9BA2A7',lw=.65,zorder=0))
        for v in [.25,.5,.75]:ax.plot(*p(np.array([0,1]),v,upper),c='#E4E7E9',lw=.4,zorder=1)
        for u in [.25,.5,.75]:ax.plot(*p(u,np.array([0,1]),upper),c='#E4E7E9',lw=.4,zorder=1)
    t=np.linspace(0,2*np.pi,200)
    for upper,rx,ry,col in [(True,.40,.28,RED),(False,.17,.42,BLUE)]:
        for size in np.linspace(1,.1,14):
            xy=p(.5+rx*size*np.cos(t),.5+ry*size*np.sin(t),upper).T
            ax.add_patch(Polygon(xy,fc=col,ec='none',alpha=.022,zorder=2))
        ax.plot(*p(.5+rx*np.cos(t),.5+ry*np.sin(t),upper),color=col,alpha=.55,lw=.7,zorder=3)
    es=[(.12,.5),(.30,.5),(.50,.5),(.70,.5),(.88,.5),(.25,.18),(.50,.18),(.75,.18),(.25,.82),(.50,.82),(.75,.82)]
    ips=[(.32,.5),(.50,.5),(.68,.5),(.50,.18),(.50,.82)]
    ec=p(.5,.5);ic=p(.5,.5,False);radius=.90
    # Triangle vertices are exact data-coordinate geometry, not scatter marker bounds.
    tri=np.array([[-1.05,-.68],[1.05,-.68],[0,1.28]])
    for u in [.12,.30,.70,.88]:
        endpoint=p(u,.5)+[0,1.28]
        excite(ec+[0,1.28],endpoint,rad=.20 if u<.5 else -.20)
    # Only two lateral I-I interactions; each T is shorter than a circle diameter.
    for u in [.32,.68]:
        dst=p(u,.5,False);vec=(dst-ic)/np.linalg.norm(dst-ic)
        inhibit(ic+radius*vec,dst-(radius+.2)*vec)
    # Reciprocal projections: exact E base midpoint -> exact I rim.
    e_base=ec+tri[:2].mean(axis=0)
    i_target=ic+radius*np.array([-.60,.80])
    excite(e_base,i_target,rad=.12)
    inhibit(ic+radius*np.array([.60,.80]),ec+[1.20,-.58],rad=.10,tangent=[-.18,1.])
    for u,v in es:
        xy=p(u,v);ax.add_patch(Polygon(xy+tri,fc=RED if (u,v)==(.5,.5) else 'white',ec=RED,lw=.7,zorder=10))
    for u,v in ips:
        ax.add_patch(Circle(p(u,v,False),radius,fc=BLUE if (u,v)==(.5,.5) else 'white',ec=BLUE,lw=.7,zorder=10))
    ax.text(38,66,'Local E/I circuit',ha='center',fontsize=10,weight='bold')
    ax.text(3,56,'E',color=RED,weight='bold',fontsize=11)
    ax.text(3,27,'I',color=BLUE,weight='bold',fontsize=11)
    ax.text(39,59,'Anisotropic E→E',color=RED,ha='center',fontsize=10.)
    ax.text(40,6,'Isotropic I→I',color=BLUE,ha='center',fontsize=10.)
    ax.text(47,33,'Local I→E',color=BLUE,ha='left',fontsize=9.5)
    # Outline is the magnified region corresponding to the small rectangle in B.
    ax.add_patch(Rectangle((1,3),74,60,fill=False,ec='#9DA2A5',lw=.55,ls=(0,(2,2)),zorder=-1))
    ax.set_gid('fig4_a_circuit')
    return dict(representation='Two tilted E/I sheets; regular neuron rows; schematic only',E_to_I_start=e_base.tolist(),E_triangle_base_midpoint=e_base.tolist(),E_to_I_target=i_target.tolist(),I_center=ic.tolist(),I_radius_mm=radius,I_I_edges=2,inhibitory_T_lengths_mm=bars,neuron_distribution='regular rows and symmetric cross',zoom_region_in_B_mm=[-9.3,-.4,3.0,3.0])


def draw_b(fig, rect):
    ax=fig.add_axes(rect);ax.set_gid('fig4_b_sheet')
    spec=importlib.util.spec_from_file_location('shared_overview',HERE/'neuron_overview.py')
    overview=importlib.util.module_from_spec(spec);spec.loader.exec_module(overview)
    neuron_layout=overview.draw_neurons(ax)
    with np.load(HERE/'rigid_contact_geometry.npz') as z:names=z['names'];xy=z['rigid_xy_mm']
    for rod in ['SCL','ICL']:
        ix=sorted(np.flatnonzero(np.char.startswith(names,rod)),key=lambda i:int(names[i][len(rod):]))
        ax.plot(*xy[ix].T,color='#494949',lw=.8,zorder=4)
        ax.scatter(*xy[ix].T,s=9,fc='white',ec='#494949',lw=.6,zorder=5)
        for j in [ix[0],ix[-1]]:
            ax.annotate(str(names[j]),xy[j],xytext=((10,14) if str(names[j])=='SCL6' else (0,-7 if rod=='ICL' else 5)),textcoords='offset points',fontsize=7.0,ha='center',zorder=6)
    ax.set(xlim=(-10,10),ylim=(-10,10),xticks=[-10,0,10],yticks=[-10,0,10],xlabel='x (mm)',ylabel='y (mm)',aspect='equal')
    ax.tick_params(labelsize=9.5,length=2.5,width=.6,pad=2);ax.xaxis.label.set_size(10.5);ax.yaxis.label.set_size(10.5)
    ax.xaxis.labelpad=ax.yaxis.labelpad=2
    ax.set_title('Spatial SNN',fontsize=10,weight='bold',pad=7)
    handles=[Line2D([],[],marker=m,ls='none',color=c,mfc='white',ms=4,label=l) for m,c,l in [('^',RED,'E neuron'),('o',BLUE,'I neuron')]]
    ax.legend(handles=handles,fontsize=8.2,loc='upper right',handletextpad=.3,labelspacing=.2,borderpad=.2,frameon=True,framealpha=.96,edgecolor='none')
    region=(-9.3,-.4,3.,3.)
    ax.add_patch(Rectangle(region[:2],*region[2:],fill=False,ec='#858B91',lw=.75,ls=(0,(2,2)),zorder=6))
    local=next(a for a in fig.axes if a.get_gid()=='fig4_a_circuit')
    for axy,bxy in [((75,63),(region[0],region[1]+region[3])),((75,3),(region[0],region[1]))]:
        patch=ConnectionPatch(axy,bxy,coordsA=local.transData,coordsB=ax.transData,color='#9DA2A5',lw=.55,ls=(0,(2,2)),zorder=-2,clip_on=False)
        fig.add_artist(patch)
    # Restore the accepted local electrode geometry and Gaussian sampling operator.
    inset=ax.inset_axes([.38,.30,.60,.30]);inset.set_facecolor('white');inset.set_zorder(10)
    # Cover the overview below the inset before drawing its transparent weight layers.
    inset.add_patch(Rectangle((0,0),1,1,transform=inset.transAxes,fc='white',ec='none',zorder=-3))
    spec=importlib.util.spec_from_file_location('fig4_sampling',HERE/'local_sampling_panel.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sampling=module.draw_sampling(inset)
    center=np.array(sampling['center_overview_mm']);fov=np.array(sampling['field_of_view_mm']);lo=center-fov/2;hi=center+fov/2
    ax.add_patch(Rectangle(lo,*fov,fc='#93C4A7',alpha=.18,ec='none',zorder=2))
    ax.add_patch(Rectangle(lo,*fov,fill=False,ec='#647A6B',lw=.85,ls=(0,(2,2)),zorder=7))
    for origin,dest in [((lo[0],hi[1]),(0,0)),((hi[0],hi[1]),(1,0))]:
        line=ConnectionPatch(origin,dest,coordsA=ax.transData,coordsB=inset.transAxes,color='#84988B',lw=.6,ls=(0,(2,2)),clip_on=False,zorder=8)
        ax.add_artist(line)
    return dict(**neuron_layout,zoom_region_mm=list(region),sampling=sampling,local_sampling_inset=True,readout_waveforms=False,
        removed=['purple regions','OPTM','MTA/MTB waveform inset'])
