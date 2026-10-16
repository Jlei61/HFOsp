"""Adapters from newly simulated arrays to the paper's Figure 4 painters."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from types import SimpleNamespace
from scipy.stats import spearmanr
from plotting.paper import fig4_search,fig4_errors,fig4_matching,fig4_similarity
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


def core_positions(rows, contacts, names):
    # One candidate contributes one A/B pair. Core order is physical identity,
    # including its independent OU stream; never swap cores by their x coordinate.
    rows = [r for r in rows if r['k'] == 2]
    finite = [r['J_joint'] for r in rows if r.get('J_joint') is not None]
    if not finite:
        raise ValueError('No scorable two-core candidates')
    cutoff = np.sort(finite)[int(np.ceil(.2*len(finite)))-1]
    norm = Normalize(min(finite), max(finite))
    fig, ax = plt.subplots(figsize=(5.3, 5), layout='constrained')
    xy = np.asarray(contacts)
    for rod in ['SCL', 'ICL']:
        ids = sorted([i for i, n in enumerate(names) if n.startswith(rod)],
                     key=lambda i: int(names[i][len(rod):]))
        ax.plot(*xy[ids].T, 'o-', color='.6', mfc='white', ms=3, lw=.7, zorder=1)
    for row in rows:
        positions = np.asarray(row['vector'][:4]).reshape(2, 2)
        loss = row.get('J_joint')
        for k, marker in enumerate(['o', '^']):
            color = '.8' if loss is None else plt.cm.Purples_r(norm(loss))
            selected = loss is not None and loss <= cutoff
            ax.scatter(*positions[k], marker=marker, c=[color], s=45,
                       edgecolors='black' if selected else '.6', linewidths=1.1 if selected else .4, zorder=3)
    handles = [Line2D([], [], ls='none', marker=m, color='.5', label=f'Core {n}')
               for m, n in zip(['o', '^'], ['A', 'B'])]
    handles.append(Line2D([], [], ls='none', marker='o', mfc='none', mec='black', label='Lowest 20%'))
    missing = any(r.get('J_joint') is None for r in rows)
    if missing:
        handles.append(Line2D([], [], ls='none', marker='o', mfc='.8', mec='.6', label='Unscorable'))
    ax.legend(handles=handles, frameon=False, loc='upper left', bbox_to_anchor=(0, 1.17), ncol=2 if missing else 3, fontsize=8)
    ax.set(xlim=(0, 20), ylim=(0, 20), aspect='equal', xlabel='x (mm)', ylabel='y (mm)')
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='Purples_r'), ax=ax,
                 label=r'$J_{\mathrm{joint}}$ (lower is better)', fraction=.04, pad=.03)
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
