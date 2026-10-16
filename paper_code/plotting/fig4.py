"""Adapters from newly simulated arrays to the paper's Figure 4 painters."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from types import SimpleNamespace
from scipy.stats import spearmanr
from plotting.paper import fig4_search,fig4_errors,fig4_matching,fig4_similarity,fig4_density
from plotting.paper.fig4_circuit import _draw_integrated_mechanism

RED,BLUE='#C63D3A','#287FA1'
ERROR_COLORS={'rank_error':'#3D5C6F','within_rod_order_error':'#E47159','participation_error':'#6B7546'}
METRICS=('mean_rank_error','within_shaft_order_error','participation_error')
DISPLAY=fig4_matching.DISPLAY
FONT=dict(label=10,tick=9,legend=8.8)


def model_definition(run,radius):
    fig=plt.figure(figsize=(13,4.3))
    a=fig.add_axes([.01,.10,.29,.8]);_draw_integrated_mechanism(a,central_offset=.11,z_linewidth=.6,z_dashes=(1.5,1.5));a.set_title('Local E/I circuit')
    a=fig.add_axes([.36,.14,.28,.76]);ne=len(run['ec']);pos=run['positions'];ix=np.arange(0,ne,40);ii=np.arange(ne,len(pos),10)
    a.scatter(*(pos[ix]-10).T,s=4,c='#ed6e75',marker='^',alpha=.6,label='E neuron')
    a.scatter(*(pos[ii]-10).T,s=3,c='#74acd7',alpha=.65,label='I neuron')
    for center in run['centers']:a.add_patch(Circle(center-10,radius,fc='#8a6dae',ec='#8a6dae',alpha=.45))
    names=run['contact_names'];xy=run['contact_xy']-10
    for rod in ['SCL','ICL']:
        ids=sorted([i for i,n in enumerate(names) if n.startswith(rod)],key=lambda i:int(names[i][3:]))
        a.plot(*xy[ids].T,'o-',mfc='white',mec='.4',color='.4',ms=3,lw=.8)
    a.set(xlim=(-10,10),ylim=(-10,10),aspect='equal',xlabel='x (mm)',ylabel='y (mm)');a.legend(fontsize=8,loc='upper right')
    a=fig.add_axes([.71,.54,.27,.36]);contact=run['contact_xy'][names.index('ICL4')];d=pos[:ne]-contact
    use=(abs(d[:,0])<1.9)&(abs(d[:,1])<.95)
    weights=np.exp(-np.sum(d[use]**2,axis=1)/(2*.25**2))
    a.scatter(*d[use].T,c=weights,cmap='Greens',vmin=0,vmax=1,s=15,marker='^')
    a.set(xlim=(-1.9,1.9),ylim=(-.95,.95),aspect='equal',xlabel='x (mm)',ylabel='y (mm)',title='Local sampling, σ = 0.25 mm')
    a=fig.add_axes([.71,.14,.27,.25]);t=np.arange(run['envelope'].shape[1])*run['envelope_dt_ms'];ids=[names.index(n) for n in ['ICL3','ICL4','ICL5']]
    valid=[i for i in run['observed']['primary_event_indices'] if run['observed']['events'][i]['window_ms'][0]>=1500]
    if valid:
        center=np.nanmin(run['observed']['centroid_ms'][valid[0]]);take=(t>=center-60)&(t<=center+60)
        x=run['envelope'][ids][:,take];scale=max(float(np.max(x)),1e-12)
        for j,row in enumerate(x):a.plot(t[take]-(center-60),row/scale+j,color=RED,lw=.9)
        a.set(yticks=range(3),yticklabels=['ICL3','ICL4','ICL5'],xlabel='Time (ms)',xlim=(0,120))
    return fig


def errors(rows,*,angle=False):
    fig,ax=plt.subplots(figsize=(4.5,4),layout='constrained')
    values=[[r.get('descriptive',{}).get(m,np.nan) if r.get('descriptive',{}).get('full_contact_pair_support',False) else np.nan for r in rows] for m in METRICS]
    values=np.asarray([[np.nan if x is None else x for x in v] for v in values])
    if angle:
        plate=SimpleNamespace(base=SimpleNamespace(KEYS=tuple(ERROR_COLORS)),ERROR_COLORS=ERROR_COLORS,FONT=FONT,MIDDLE_LABEL_PT=10,axis_type=lambda ax: ax.tick_params(labelsize=9))
        fig4_errors.error_curves(plate,ax,[r['angle'] for r in rows],values,'E→E angle (°)',[-20,0,20],[])
    else:
        cmap=LinearSegmentedColormap.from_list('iteration_purple',['#B1A9DA','#7367BE','#483D8B']);epochs=np.array([r['epoch'] for r in rows]);norm=Normalize(1,max(2,epochs.max()))
        for epoch,v in sorted(zip(epochs,values.T),key=lambda x:x[0]):ax.plot([0,1,2],v,color=cmap(norm(epoch)),lw=.65)
        ax.set(xlim=(0,2),ylim=(0,max(.5,float(np.nanmax(values))*1.08) if np.isfinite(values).any() else .5),xticks=[0,1,2],xticklabels=['Mean rank\n$ε_r$','Order\n$ε_o$','Participation\n$ε_p$'],ylabel='Error')
        for text,color in zip(ax.get_xticklabels(),ERROR_COLORS.values()):text.set_color(color)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax,pad=.03,fraction=.035,label='Iteration order')
    if np.isfinite(values).any() and np.nanmax(values)>.42:ax.set_ylim(0,float(np.nanmax(values))*1.1)
    return fig


def optimization_curves(searches):
    arrays={};prefix={}
    for search in searches:
        rows=search['rows'];k=rows[0]['k'];best=np.inf;curve=[]
        for row in rows:
            if row['J'] is not None:best=min(best,row['J'])
            curve.append([best if np.isfinite(best) else np.nan])
        arrays.setdefault(k,[]).append(curve);prefix.setdefault(k,[]).append(len(rows))
    case=dict(arrays={k:np.asarray(v) for k,v in arrays.items()},summary=dict(common_epoch=min(n for v in prefix.values() for n in v),prefix_by_restart=prefix,display_id='E1146'))
    fig,ax=plt.subplots(figsize=(4.5,4),layout='constrained')
    fig4_search.draw(ax,case,show_title=False,paper_style=FONT,show_legend=True,summary_style='mean_std')
    return fig


def core_density(rows,contacts,names):
    rows=[r for r in rows if r['k']==2 and r['J'] is not None]
    if not rows:raise ValueError('No scorable two-core candidates')
    rows=sorted(rows,key=lambda r:r['J']);selected=rows[:max(1,int(np.ceil(.2*len(rows))))]
    def xy(r):
        p=np.asarray(r['vector'][:4]).reshape(2,2);return p[np.argsort(p[:,0])]
    allxy=np.asarray([xy(r) for r in rows]);good=np.asarray([xy(r) for r in selected])
    grid=np.linspace(0,20,321);X,Y=np.meshgrid(grid,grid);components=[];contours={}
    for k in range(2):
        z=np.mean([np.exp(-((X-p[0])**2+(Y-p[1])**2)/(2*.35**2))/(2*np.pi*.35**2) for p in good[:,k]],axis=0)
        components.append(z);order=np.sort(z.ravel())[::-1];level=order[np.searchsorted(np.cumsum(order)/order.sum(),.9)]
        temp,ax=plt.subplots();cs=ax.contour(X,Y,z,levels=[level]);paths=[v.tolist() for v in cs.allsegs[0]];plt.close(temp)
        contours['AB'[k]]={'0.9':dict(paths_mm=paths,threshold_per_mm2=float(level))}
    straight=np.asarray(contacts).copy()
    for rod in ['SCL','ICL']:
        ids=[i for i,n in enumerate(names) if n.startswith(rod)];p=straight[ids];center=p.mean(0);axis=np.linalg.svd(p-center)[2][0];straight[ids]=center+((p-center)@axis)[:,None]*axis
    arrays=dict(x_mm=grid,y_mm=grid,density_per_mm2=np.mean(components,axis=0),core_components_per_mm2=components,all_positions_mm=allxy,selected_positions_mm=good)
    pixels,cmap,norm,labels,extent=fig4_density.completed_density_layer(arrays,dict(density_contours=contours,contact_names=names,straight_contact_xy_mm=straight))
    fig,ax=plt.subplots(figsize=(4.5,4.8),layout='constrained');ax.imshow(pixels,extent=extent,origin='upper');ax.set(xlabel='x (mm)',ylabel='y (mm)',xticks=[5,10,15],yticks=[5,10,15])
    for p,label in zip(labels,['Core A','Core B']):ax.text(*p,label,ha='center',fontsize=9,color='#403949')
    handles=[Line2D([],[],ls='none',marker='o',ms=3,color='#B6B2BF',label='Search'),
             Line2D([],[],ls='none',marker='o',ms=4,mfc='none',mec='#65509A',label='Outlier'),
             Line2D([],[],color='#9784B6',lw=.8,label='90% mass')]
    ax.legend(handles=handles,loc='upper left',frameon=False,fontsize=8,handlelength=1.,labelspacing=.25)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax,label='Position density (mm⁻²)',pad=.02,fraction=.035)
    return fig


def rank_profiles(result,names):
    order=np.array([names.index(n) for n in DISPLAY]);fig=fig4_matching.rank_panel(order,result['model_profiles'][::-1],result['patient_profiles'][::-1])
    # Match the accepted panel G legend: model solid, patient dashed.
    fig.legends[0].remove();ax=fig.axes[0]
    ax.legend(handles=[Line2D([],[],color=c,ls=ls,marker=m,lw=1.2,ms=2.8,label=s) for c,ls,m,s in [(RED,'-','o','MTA'),(RED,'--',None,'TA'),(BLUE,'-','o','MTB'),(BLUE,'--',None,'TB')]],ncol=2,fontsize=7,loc='lower right',bbox_to_anchor=(1,1.02),borderaxespad=0,frameon=False)
    return fig


def crossfit_matrix(result):
    def stars(p):return '' if p is None else '***' if p<=.001 else '**' if p<=.01 else '*' if p<=.05 else 'n.s.'
    tests={key:dict(stars=stars(p)) for key,p in zip(['MTA_vs_TA','MTB_vs_TB'],result['diagonal_p'])}
    return fig4_matching.matrix_panel({'matrix':np.asarray(result['matrix'])[::-1,::-1]},tests)


def template_similarity(result):
    values=[float(spearmanr(a,b,nan_policy='omit').statistic) for a,b in zip(result['patient_profiles'],result['model_profiles'])]
    summary=dict(subjects=[dict(subject='E1146',matched_rhos=values)],statistics=[dict(q25_q75=[v,v],median=v) for v in values],mode_order=['TA–MTA','TB–MTB'])
    fig,ax=plt.subplots(figsize=(4.5,4),layout='constrained');fig4_similarity.draw_modes(ax,summary,legend_loc='upper right')
    return fig
