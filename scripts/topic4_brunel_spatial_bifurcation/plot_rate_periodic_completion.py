"""Show continued periodic branches, critical modes and same-J coexistence.

Branch geometry is separate from stability certification. Uncomputed portions
are never turned into solid 'stable cycles' from a time-domain extrema scan.
"""
from rate_periodic import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
from matplotlib.colors import PowerNorm
import re

F=PERIODIC_OUT/'figures';COL=['#2166ac','#ad42a4'];FAMILY={'A':'#1b9e77','B':'#8d65b4','single':'#d17c00','double':'#cf4446','Bleading':'#795548','PDchild':'#e377c2','PDupperchild':'#00897b'}
FAMILY['PDreturnchild']='#0072b2'
FAMILY.update({f'H{i}':c for i,c in enumerate(['#238b45','#66a61e','#e6550d','#7570b3','#b15928','#1f78b4'],3)})
FAMILY.update({'H9':'#e7298a','H10':'#636363','H11':'#e6ab02'})
FAMILY.update(dict(zip([f'H{i}' for i in range(12,20)],
    ['#542788','#a6611a','#01665e','#c51b7d','#4d9221','#b2182b','#2166ac','#666600'])))
CRITICAL_NAMES=['LPC_resonance_upper','LPC_resonance_lower','LPC_A1','LPC_B1','LPC_burst_low','LPC_burst_high','LPC_Bleading_low','LPC_double_high']
CRITICAL_NAMES += [f'LPC_A{i}' for i in range(2,6)]+[f'LPC_B{i}' for i in range(2,9)]
CRITICAL_NAMES += ['LPC_double_low']
CRITICAL_NAMES += [f'LPC_double_secondary{i}' for i in range(1,5)]
new_fold_check=PERIODIC_OUT/'LPC_A_low_extension_validation.json'
if new_fold_check.exists() and read(new_fold_check)['status']=='VALIDATED_CYCLE_FOLD':
    CRITICAL_NAMES += ['LPC_A_low_extension']
for name in ['LPC_A_return_exchange','LPC_A_return_recruitment']:
    check=PERIODIC_OUT/(name+'_validation.json')
    if check.exists() and read(check)['status']=='VALIDATED_CYCLE_FOLD':CRITICAL_NAMES.append(name)
EXTENSION_FOLD_LABELS={**{f'LPC_A_large_return{i}':f'LPC{27+i}' for i in range(1,4)},
    **{f'LPC_single_upper{i}':f'LPC{30+i}' for i in range(1,5)},
    'LPC_A_global_turn1':'LPC35',
    **{f'LPC_single_upper{i}':f'LPC{31+i}' for i in range(5,9)},
    **{f'LPC_B_burst_turn{i}':f'LPC{39+i}' for i in range(1,7)},
    **{f'LPC_B_further_turn{i}':f'LPC{45+i}' for i in range(1,11)}}
EXTENSION_FOLD_LABELS.update({**{f'LPC_A_connection{i}':f'LPC{55+i}' for i in range(1,3)},
    **{f'LPC_B_connection{i}':f'LPC{57+i}' for i in range(1,3)}})
EXTENSION_FOLD_LABELS['LPC_A_next1']='LPC60'
EXTENSION_FOLD_LABELS['LPC_A_stage4_turn1']='LPC61'
EXTENSION_FOLD_LABELS['LPC_A_stage4_turn2']='LPC62'
EXTENSION_FOLD_LABELS['LPC_A_stage4_turn3']='LPC63'
EXTENSION_FOLD_LABELS['LPC_B_stage3_turn1']='LPC64'
EXTENSION_FOLD_LABELS['LPC_B_stage3_turn2']='LPC65'
EXTENSION_FOLD_LABELS['LPC_B_stage4_turn1']='LPC66'
EXTENSION_FOLD_LABELS['LPC_B_stage4_turn2']='LPC67'
for name in EXTENSION_FOLD_LABELS:
    check=PERIODIC_OUT/(name+'_validation.json')
    if check.exists() and read(check)['status']=='VALIDATED_CYCLE_FOLD':CRITICAL_NAMES.append(name)
CRITICAL_LABELS={name:f'LPC{i+1}' for i,name in enumerate(CRITICAL_NAMES)}
# Reserve the extension labels independently of the order in which checks
# finish, so an asynchronous completion cannot rename an existing point.
CRITICAL_LABELS.update(EXTENSION_FOLD_LABELS)
CRITICAL_LABELS['TR_A_B']='TR1'
CRITICAL_LABELS['TR_A_return']='TR2'
CRITICAL_LABELS['PD_double_low']='PD1'
CRITICAL_LABELS['PD_double_upper']='PD2'
CRITICAL_LABELS['PD_A_return']='PD3'
CRITICAL_LABELS['PD_H2_after_LPC13']='PD4'


def save(fig,name):
    F.mkdir(exist_ok=True,parents=True)
    for ext in ['png','pdf','svg']:fig.savefig(F/f'{name}.{ext}',dpi=190,bbox_inches='tight')
    plt.close(fig)


def update_readme(entries):
    """Replace these figure entries while preserving other existing figures."""
    path=F/'README.md';txt=path.read_text() if path.exists() else ''
    for name,body in entries.items():
        pattern=r'^### '+re.escape(name)+r'(?:\.png)?\s*\n.*?(?=^### |\Z)'
        txt=re.sub(pattern,'',txt,flags=re.M|re.S).rstrip()
        txt+='\n\n### '+name+'.png\n'+body.strip()+'\n'
    path.write_text(txt.lstrip())


def row(path):
    q=read(Path(path).with_suffix('.json'));return q if q['status']=='CONVERGED' else None


def additional_hopfs():
    out=[]
    for path in (PERIODIC_OUT/'stationary_root_counts').glob('H*_validation.json'):
        check=read(path)
        if check.get('status')!='VALIDATED_LOCAL_HOPF':continue
        q=read(path.with_name(check['label']+'_highorder_half.json'))
        q.update(criticality=check['criticality'],validation=str(path),validation_status=check['status'])
        FAMILY.setdefault(check['label'],plt.get_cmap('tab20')((int(check['label'][1:])-3)%20))
        out.append(q)
    return sorted(out,key=lambda q:int(q['label'][1:].split('_')[0]))


def families():
    folder=PERIODIC_OUT/'orbits';out={}
    def records(pattern):
        return [q for f in sorted(folder.glob(pattern)) if '_accuracy_' not in f.stem
                for q in [read(f)] if q['status']=='CONVERGED']
    for core,last in [('A',3.),('B',2.6)]:
        fs=sorted(folder.glob(f'H{core}_a*_N*.json'));by={}
        for f in fs:
            if '_accuracy_' in f.stem:continue
            q=read(f);amp=float(f.stem.split('_a')[1].split('_N')[0])
            if q['status']=='CONVERGED' and amp<=last and (amp not in by or q['N']>by[amp]['N']):by[amp]=q
        out[core]=[by[k] for k in sorted(by)]
    out['A'] += records('resonanceA_*.json')
    for amp in [3.5,3.6,3.7,3.8]:out['A'].append(read(folder/f'HA_a{amp:.5f}_N64.json'))
    out['A'] += records('arcA_*_N64.json')
    out['A'] += records('arcAextended_*_N128.json')
    acheck=PERIODIC_OUT/'arcAlowExtension_accuracy.json'
    if acheck.exists() and read(acheck)['status']=='PASS':
        out['A'] += records('arcAlowExtension_*_N256.json')
    out['B'] += records('arcB_*_N64.json')
    out['B'] += records('arcBextended_*_N128.json')
    out['B'] += records('arcBsecond_*_N128.json')
    for segment in ['arcBtoBurst','arcBtoBurstFurther','arcBconnectionFurther','arcBconnectionNext','arcBconnectionStage3','arcBconnectionStage4_20260920','arcBconnectionStage5_20260920']:
        bcheck=PERIODIC_OUT/(segment+'_accuracy.json')
        if bcheck.exists() and read(bcheck)['status']=='SAMPLED_PASS':
            out['B'] += [read(Path(path).with_suffix('.json')) for path in read(bcheck)['included_orbits']]
    down=records('arcSingleDown_*.json')[::-1]
    up=records('arcSingleUp_*.json')
    out['single']=down+[read(folder/'branch_J1.300000000_N256.json')]+up
    for family,label in [('A','arcAreturnStrong'),('A','arcAglobalConnection'),('A','arcAconnectionFurther'),('A','arcAconnectionNext'),('A','arcAconnectionStage3'),('A','arcAconnectionStage4'),('A','arcAconnectionStage5'),('single','arcSingleUpperConnect')]:
        check=PERIODIC_OUT/(label+'_accuracy.json')
        if check.exists() and read(check)['status']=='SAMPLED_PASS':
            # Include only the frozen checked snapshot, while continuation may
            # still be writing additional, not yet inspected waveforms.
            out[family]+=[read(Path(path).with_suffix('.json')) for path in read(check)['included_orbits']]
    out['double']=records('arcDoubleLow_*.json')[::-1]
    out['double'] += records('arcDoubleDown_*.json')[::-1]
    out['double']+=[read(folder/'lowerburst_J0.941000000_N512.json'),read(folder/'refined_J0.942000000_N1536.json')]
    out['double']+=records('arcDoubleUp_*.json')
    out['double']+=records('arcDoubleRef_*.json')
    out['double']+=records('arcDoubleFine_*.json')
    out['Bleading']=[read(folder/'branch095_J0.950000000_N512.json')]
    out['Bleading']+=records('arc095Down_*.json')
    out['Bleading']+=records('arcBleadingExtended_*_N1024.json')
    bridge=PERIODIC_OUT/'arcBleadingBridge_accuracy.json'
    if bridge.exists() and read(bridge)['status']=='PASS':
        out['Bleading']+=records('arcBleadingBridge_*_N1024.json')
    connection=PERIODIC_OUT/'arcBleadingConnection_20260920_accuracy.json'
    if connection.exists() and read(connection)['status']=='SAMPLED_PASS':
        out['Bleading'] += [read(Path(f).with_suffix('.json')) for f in read(connection)['included_orbits']]
    pd1=PERIODIC_OUT/'PD_double_low_validation.json'
    pd1child=PERIODIC_OUT/'PD_child_classification.json'
    if (pd1.exists() and read(pd1).get('status')=='VALIDATED_PD' and
        pd1child.exists() and read(pd1child).get('status')=='SUBCRITICAL_PD'):
        children={}
        for f in folder.glob('PDchild_a*_N*.json'):
            if '_accuracy_' in f.stem:continue
            q=read(f);amp=float(f.stem.split('_a')[1].split('_N')[0])
            if q['status']=='CONVERGED' and (amp not in children or q['N']>children[amp]['N']):children[amp]=q
        if children:out['PDchild']=[children[a] for a in sorted(children)]
    physical_pd1=Path('/data/hfosp/topic4_sef_hfo/interictal_rate_branch_completion_20260920/physical_children/PD_double_low/result.json')
    if physical_pd1.exists():
        checked=read(physical_pd1)
        if checked.get('full_physical_child_checks'):
            # Fine physical children supersede the old coarse departure
            # profiles. This admits actual periodic solutions, without
            # promoting their pending parent-side/criticality evidence.
            rows=checked['rows']
            assert rows and all(q['physical_pass'] and
                q['physical_check']['maximum_group_defect_Hz']<.001 and
                q['physical_check']['filter_state_check']['positive'] for q in rows)
            out['PDchild']=[read(Path(q['orbit']).with_suffix('.json'))
                            for q in sorted(rows,key=lambda v:v['amplitude_hz'])]
    upper=PERIODIC_OUT/'PD_upper_child_classification.json'
    if upper.exists() and read(upper)['status'] in ['SUPERCRITICAL_PD','SUBCRITICAL_PD']:
        # Only the positive, fine-mesh child profiles enter the plotted branch.
        # The coarse mesh is retained separately as a departure diagnostic.
        out['PDupperchild']=sorted(records('PDupperchild_a*_N4096.json'),
            key=lambda q:float(Path(q['path']).stem.split('_a')[1].split('_N')[0]))
        extension=PERIODIC_OUT/'PD2_extended_child_accuracy.json'
        if extension.exists() and read(extension).get('status')=='PASS':
            out['PDupperchild'] += [read(Path(q['orbit']).with_suffix('.json')) for q in read(extension)['checks']]
    physical=Path('/data/hfosp/topic4_sef_hfo/interictal_rate_branch_completion_20260920/PD3_child_followup/physical_profiles.json')
    if physical.exists() and read(physical).get('status')=='ALL_FOUR_PROFILES_PASS':
        # Use corrected positive profiles. Old departure and instability
        # records do not certify constituent filter-state positivity.
        out['PDreturnchild']=[]
        for row in read(physical)['rows']:
            assert row['physical']['filter_state_check']['positive']
            out['PDreturnchild'].append(read(Path(row['orbit']).with_suffix('.json')))
    for h in additional_hopfs():
        name=h['label'].split('_')[0];children={}
        for f in folder.glob(name+'_a*_N*.json'):
            if '_accuracy_' in f.stem:continue
            q=read(f);amp=float(f.stem.split('_a')[1].split('_N')[0])
            if q['status']=='CONVERGED' and (amp not in children or q['N']>children[amp]['N']):children[amp]=q
        if children:out[name]=[children[a] for a in sorted(children)]
    h5check=PERIODIC_OUT/'arcH5resolved_accuracy.json'
    if 'H5' in out and h5check.exists() and read(h5check)['status']=='PASS':
        out['H5']+=records('arcH5resolved_*_N64.json')
    return out


def critical():
    names=['TR_A_B']+CRITICAL_NAMES
    if (PERIODIC_OUT/'TR_A_return_validation.json').exists():names+=['TR_A_return']
    for pdname in ['PD_double_low','PD_double_upper','PD_A_return','PD_H2_after_LPC13']:
        pdcheck=PERIODIC_OUT/(pdname+'_validation.json')
        if pdcheck.exists() and read(pdcheck).get('status')=='VALIDATED_PD':names.append(pdname)
    rows=[]
    for name in names:
        if name=='TR_A_B' and (PERIODIC_OUT/'TR_A_B_validation.json').exists():
            validation=read(PERIODIC_OUT/'TR_A_B_validation.json')
            q=validation['critical_point'].copy();q.update(label=name,criticality='locally subcritical',validation_status=validation['status'])
            rows.append(q);continue
        fs=list(PERIODIC_OUT.glob(f'{name}_N*.json'))
        candidates=[read(f) for f in fs]
        candidates=[q for q in candidates if q['label'].startswith(('TR','PD')) or abs(q.get('dJ_dcoordinate',q.get('dJ_dlogT',float('inf'))))<1e-7]
        if candidates:
            q=max(candidates,key=lambda q:q['N'])
            if name=='TR_A_return' and (PERIODIC_OUT/'TR_A_return_nonlinear_validation.json').exists():
                q.update(criticality='locally supercritical',nonlinear_validation=str(PERIODIC_OUT/'TR_A_return_nonlinear_validation.json'))
            if name.startswith('PD'):
                q.update(type='period doubling',validation_status='VALIDATED_PD')
                validation=read(PERIODIC_OUT/(name+'_validation.json'))
                if 'accepted_parent_orbit' in validation:
                    q['orbit']=validation['accepted_parent_orbit']
                    q['accepted_parent_mesh_N']=validation.get('filter_state_followup',{}).get('N',validation['N'])
            rows.append(q)
    return rows


def style(ax):ax.spines[['top','right']].set_visible(False);ax.tick_params(direction='out')


def continuation_breaks(rows):
    """Do not join a refined restart to the last overlapping coarse sample.

    arcDoubleRef restarts near coarse point 13, while arcDoubleUp includes
    points through 18. Both remain indexed for the frozen Floquet survey.
    Their list adjacency is not an additional continuation step or fold.
    """
    return [i for i in range(1,len(rows))
            if Path(rows[i-1]['path']).stem.startswith('arcDoubleUp_')
            and Path(rows[i]['path']).stem.startswith('arcDoubleRef_')]


def plot_branch(ax,k,fs,xlim=(.68,1.68),near=False):
    evidence=PERIODIC_OUT/'cycle_fold_evidence_inventory.json'
    checked_folds={q['internal_label'] for q in read(evidence)['rows'] if q['status']=='INDEPENDENT_MODE_CHECKED'} if evidence.exists() else set()
    profile_audit=PERIODIC_OUT/'rate_filter_state_positivity_audit.json'
    physical={str(Path(q['orbit']).resolve()):q['positive'] for q in read(profile_audit)['rows']} if profile_audit.exists() else {}
    z=np.load(RATE_OUT/'equilibrium_branch.npz');ax.plot(z['J'],z['regional'][:,k],':',color='#7d7d7d',lw=.9)
    spectrum={q['index']:q for q in read(RATE_OUT/'branch_spectrum.json')['rows']};indices=sorted(spectrum);segments=[]
    for i,j in zip(indices[:-1],indices[1:]):
        if spectrum[i]['positive'] and spectrum[j]['positive']:
            if segments and segments[-1][1]==i:segments[-1][1]=j
            else:segments.append([i,j])
    for i,j in segments:ax.plot(z['J'][i:j+1],z['regional'][i:j+1,k],'--',color='#7d7d7d',lw=.9)
    h1=read(RATE_OUT/'hopfs.json')['rows'][0];cut=np.flatnonzero(z['J']>=h1['J_EE_core'])[0]
    ax.plot(np.r_[z['J'][:cut],h1['J_EE_core']],np.r_[z['regional'][:cut,k],h1['rates_hz'][k]],color='#333333',lw=1.3)
    if not near:
        for q in read(OUT/'critical_revision/fold_audit.json')['rows']:
            ax.plot(q['J_EE_core'],q['rates_hz'][k],'o',mfc='white',mec='#777777',ms=3,zorder=4)
    for name,rr in fs.items():
        x=np.array([r['J_EE_core'] for r in rr]);mean=np.array([r['mean_rates_hz'][k] for r in rr])
        low=np.array([r['min_rates_hz'][k] for r in rr]);high=np.array([r['max_rates_hz'][k] for r in rr])
        # Colored geometry; stability is stated in the companion multiplier panel.
        boundaries=[0,*continuation_breaks(rr),len(rr)]
        for start,end in zip(boundaries[:-1],boundaries[1:]):
            ax.plot(x[start:end],mean[start:end],color=FAMILY[name],lw=1.6)
            if not near:
                ax.plot(x[start:end],low[start:end],color=FAMILY[name],lw=.6,alpha=.45)
                ax.plot(x[start:end],high[start:end],color=FAMILY[name],lw=.6,alpha=.45)
    for i,h in enumerate(read(RATE_OUT/'hopfs.json')['rows']):
        ax.plot(h['J_EE_core'],h['rates_hz'][k],'o',color=COL[i],ms=5)
        if near:ax.annotate(f'H{i+1}',(h['J_EE_core'],h['rates_hz'][k]),xytext=(-8,-16),textcoords='offset points',fontsize=9)
    for h in additional_hopfs():
        name=h['label'].split('_')[0]
        ax.plot(h['J_EE_core'],h['rates_hz'][k],'o',mfc=FAMILY[name],mec=FAMILY[name],ms=4,zorder=6)
        if not near and name in ['H20','H22','H23','H24']:
            offsets={'H20':(-8,15),'H22':(-10,12),'H23':(-5,12),'H24':(8,-17)}
            ax.annotate('H20/21' if name=='H20' else name,(h['J_EE_core'],h['rates_hz'][k]),
                xytext=offsets[name],textcoords='offset points',fontsize=8)
    for q in critical():
        meta=read(Path(q['orbit']).with_suffix('.json'));v=meta['mean_rates_hz'][k]
        marker='D' if q['label'].startswith('TR') else ('v' if q['label'].startswith('PD') else 's')
        mode_checked=q['label'] in checked_folds or q['label'].startswith('PD')
        face='black' if mode_checked and physical.get(str(Path(q['orbit']).resolve()),False) else 'white'
        ax.plot(q['J_EE_core'],v,marker,mfc=face,mec='black',ms=5 if marker=='v' else 4,zorder=5)
        if not near and q['label'].startswith('PD'):
            ax.annotate(CRITICAL_LABELS[q['label']],(q['J_EE_core'],v),xytext=(25,-16),textcoords='offset points',fontsize=9,arrowprops=dict(arrowstyle='-',lw=.6))
        if not near and q['label']=='LPC_A_low_extension':
            ax.annotate('LPC25',(q['J_EE_core'],v),xytext=(10,10),textcoords='offset points',fontsize=8)
        if not near and q['label']=='LPC_A_next1':
            ax.annotate('LPC60',(q['J_EE_core'],v),xytext=(18,-24),textcoords='offset points',fontsize=8,arrowprops=dict(arrowstyle='-',lw=.6))
        if not near and q['label'] in ['LPC_A_stage4_turn1','LPC_A_stage4_turn2','LPC_A_stage4_turn3']:
            offset={'1':(24,24),'2':(30,-12),'3':(14,-18)}[q['label'][-1]]
            ax.annotate(CRITICAL_LABELS[q['label']],(q['J_EE_core'],v),xytext=offset,
                textcoords='offset points',fontsize=8,arrowprops=dict(arrowstyle='-',lw=.6))
        if (near and q['label'] in ['LPC_A1','LPC_B1']) or (not near and q['label']=='LPC_burst_high'):
            ax.annotate(CRITICAL_LABELS[q['label']],(q['J_EE_core'],v),xytext=(-35,8),textcoords='offset points',fontsize=8)
    pd_review=PERIODIC_OUT/'PD_resolution_review.json'
    if pd_review.exists():
        for q in read(pd_review)['rows']:
            if q['status']!='RESOLUTION_REVIEW_PENDING':continue
            meta=read(Path(q['orbit']).with_suffix('.json'));v=meta['mean_rates_hz'][k]
            ax.plot(meta['J_EE_core'],v,'v',mfc='white',mec='black',ms=5,zorder=5)
            if not near:ax.annotate(q['label']+'?',(meta['J_EE_core'],v),xytext=(-48,18),textcoords='offset points',fontsize=9,arrowprops=dict(arrowstyle='-',lw=.6))
    ax.set(xlim=xlim,xlabel=r'$J_{\mathrm{EE,core}}$',ylabel=f'Core {"AB"[k]} rate (Hz / cell)');style(ax)
    if not near:ax.set_yscale('symlog',linthresh=.1);ax.set_ylim(.005,550);ax.set_yticks([.01,.1,1,10,100,400]);ax.set_yticklabels(['0.01','0.1','1','10','100','400'])


def main():
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'});fs=families();s=RateField();cr=critical()
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    for k in [0,1]:plot_branch(axes[0,k],k,fs);axes[0,k].set_title(f'Core {"AB"[k]}: periodic branch geometry')
    plot_branch(axes[1,0],0,fs,(.936,.975),True);axes[1,0].set_ylim(.72,1.7);axes[1,0].set_title('Hopf branches and periodic folds')
    ax=axes[1,1];rs=[read(f) for f in sorted((PERIODIC_OUT/'orbits').glob('resonanceA_*.json'))]
    ax.plot([r['J_EE_core'] for r in rs],[r['mean_rates_hz'][1] for r in rs],color=FAMILY['A'],lw=1.8)
    marks=[('TR_A_B','TR1',(-68,2)),('LPC_resonance_upper','LPC1',(-52,12)),('LPC_resonance_lower','LPC2',(-64,-12))]
    if any(q['label']=='TR_A_return' for q in cr):marks.append(('TR_A_return','TR2',(20,-14)))
    for name,label,off in marks:
        q=next(x for x in cr if x['label']==name);meta=read(Path(q['orbit']).with_suffix('.json'));y=meta['mean_rates_hz'][1]
        ax.plot(q['J_EE_core'],y,'D' if label.startswith('TR') else 's',mfc='white',mec='black',ms=6)
        ax.annotate(label,(q['J_EE_core'],y),xytext=off,textcoords='offset points',arrowprops=dict(arrowstyle='-',lw=.6),fontsize=10)
    h2=read(RATE_OUT/'hopfs.json')['rows'][1]['J_EE_core'];ax.axvline(h2,color=COL[1],lw=.8,ls=':');ax.text(h2,.7482,'H2',color=COL[1],ha='center',fontsize=9)
    ax.set(xlim=(.94568,.94589),ylim=(.743,.7485),xticks=[.94570,.94578,.94586],xlabel=r'$J_{\mathrm{EE,core}}$',ylabel='Core B period mean (Hz / cell)',title='Resolved A–B interaction near H2');ax.xaxis.set_major_formatter(FormatStrFormatter('%.5f'));style(ax)
    handles=[Line2D([0],[0],color=FAMILY[n],label=l) for n,l in [('A','From H1'),('B','From H2'),('double','Alternating lead order'),('single','A-leading bursts'),('Bleading','B-leading bursts')]]
    if 'PDchild' in fs:handles.append(Line2D([0],[0],color=FAMILY['PDchild'],label='From PD1: doubled period'))
    if 'PDupperchild' in fs:handles.append(Line2D([0],[0],color=FAMILY['PDupperchild'],label='From PD2: doubled period'))
    if 'PDreturnchild' in fs:handles.append(Line2D([0],[0],color=FAMILY['PDreturnchild'],label='From PD3: doubled period'))
    fig.legend(handles=handles,loc='outside lower center',ncol=3,frameon=False)
    save(fig,'periodic_bifurcation_branches')
    # Same-parameter coexistence from two independently solved periodic BVPs.
    names=['smallA_J0.942000000_N64','refined_J0.942000000_N1536'];fig=plt.figure(figsize=(18,8.4));grid=fig.add_gridspec(2,3,width_ratios=[1,1.8,1],left=.055,right=.97,top=.86,bottom=.17,wspace=.3,hspace=.55)
    geometry=np.load(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz');xy=geometry['contact_xy']
    from src.snn_contact_display import CONTACT_ORDER,contact_indices,SHAFT_COLORS
    order=contact_indices(geometry['contact_names'].tolist());groupcell=s.geo['group_cell'];sz=s.geo['group_size'];cellcount=np.bincount(groupcell[s.E],weights=sz[s.E],minlength=400)
    for i,name in enumerate(names):
        z=np.load(PERIODIC_OUT/f'orbits/{name}.npz');r=z['r'];T=float(z['T']);regional=np.array([s.regional_rates(v) for v in r]);t=np.arange(len(r))*T/len(r)
        ax=fig.add_subplot(grid[i,0]);
        for k in [0,1]:ax.plot(t,regional[:,k],color=COL[k],lw=1.2,label=f'Core {"AB"[k]}')
        ax.set(xlabel='Time within one period (ms)',ylabel='Rate (Hz / cell)',title='Stable small oscillation' if i==0 else 'Stable two-burst cycle');style(ax)
        contact=r@s.geo['contact_rate_weights']*1000;ax=fig.add_subplot(grid[i,2])
        im=ax.imshow(contact[:,order].T,origin='upper',aspect='auto',extent=(0,T,14.5,-.5),cmap='magma',norm=PowerNorm(.5,vmin=0,vmax=200))
        ax.set(yticks=np.arange(15),yticklabels=CONTACT_ORDER,xlabel='Time within one period (ms)',title='SEEG-site firing rate')
        ax.tick_params(axis='y',labelsize=8,length=2);ax.axhline(3.5,color='white',lw=.5)
        for tick,n in zip(ax.get_yticklabels(),CONTACT_ORDER):tick.set_color(SHAFT_COLORS[n[:3]])
        sub=grid[i,1].subgridspec(1,4,wspace=.08)
        if i==0:ids=[int(x*len(r)) for x in [.0,.25,.5,.75]]
        else:
            pk=find_peaks(regional[:,1],height=20,distance=len(r)//4)[0];base=int(pk[-1])
            ids=[int((base+offset/T*len(r))%len(r)) for offset in [-20,20,70,120]]
        for j,ix in enumerate(ids):
            field=np.bincount(groupcell[s.E],weights=sz[s.E]*r[ix,s.E]*1000,minlength=400)/np.maximum(cellcount,1)
            ax=fig.add_subplot(sub[j]);imf=ax.imshow(field.reshape(20,20),origin='lower',extent=(0,20,0,20),cmap='inferno',norm=PowerNorm(.55,0,500))
            for center in s.geo['centers_mm']:ax.add_patch(plt.Circle(center,1.5,fill=False,color='white',lw=.8))
            ax.scatter(xy[:,0],xy[:,1],s=7,facecolors='none',edgecolors='cyan',linewidths=.6)
            ax.set(xticks=[0,20],yticks=[0,20],title=f'{t[ix]:.0f} ms');ax.tick_params(labelsize=8)
            if j>0:ax.tick_params(labelleft=False)
    fig.suptitle(r'Same spatial rate model, same $J_{\mathrm{EE,core}}=0.942$: two stable periodic states',fontsize=15)
    fig.legend(handles=[Line2D([0],[0],color=COL[k],label=f'Core {"AB"[k]}') for k in [0,1]],loc='upper left',bbox_to_anchor=(.055,.94),ncol=2,frameon=False)
    fig.colorbar(imf,cax=fig.add_axes([.405,.035,.25,.015]),orientation='horizontal',label='E rate (Hz / cell)')
    fig.colorbar(im,cax=fig.add_axes([.78,.035,.15,.015]),orientation='horizontal',label='Contact-weighted rate (Hz / cell)')
    save(fig,'same_parameter_periodic_coexistence')
    write(PERIODIC_OUT/'figure_coverage.json',dict(periodic_points={k:len(v) for k,v in fs.items()},critical_points=cr,
        line_meaning='Colored lines are converged periodic BVP geometry; global stability has not been certified along every segment.',
        unresolved=['Global nonlinear fate beyond the local TR1 torus branch','Connections of large-burst families to Hopf-born families','All other Floquet crossings on traced families','Completeness of all possible branches'],
        first_torus_validation=str(PERIODIC_OUT/'TR_A_B_validation.json'),
        native_SNN_equivalence='Local descriptive support at J=.942; full-range dynamical equivalence not established'))
    update_readme(dict(
        periodic_bifurcation_branches='展示同一空间 rate DDE 的固定点、已延拓周期轨道均值与极值，以及临界附近的局部放大。彩色曲线表示周期边值解的几何连接，并不表示整段稳定性均已确认。**关注点**：Hopf、周期折叠和环面分岔属于不同对象；仍需结合 Floquet 结果阅读。',
        same_parameter_periodic_coexistence='同一 J=0.942 下，小振荡与双 burst 周期解具有不同的二维场及触点读出。全部来自 rate 周期边值解，场和接触点采用跨行共享物理量色标。**关注点**：这是模型内共存结果，不等价于原生 SNN 已验证共存；弱活动在统一场色标下可能不可见。'))


if __name__=='__main__':main()
