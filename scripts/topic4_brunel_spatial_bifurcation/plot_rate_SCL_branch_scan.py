"""Show SCL readout by periodic family without filling unverified gaps."""
from plot_rate_branch_completion import *


def main():
    folder=DATA/'SCL_branch_scan';summary=read(folder/'summary.json');manifest=read(folder/'manifest.json')
    assert summary['status']=='CHECKED_POINT_OBSERVER_SCAN_COMPLETE'
    names=summary['SCL_names'];assert names==['SCL9','SCL8','SCL7','SCL6']
    colors=['#0072b2','#e69f00','#009e73','#cc79a7']
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,3,figsize=(14.5,8.5))
    fig.subplots_adjust(left=.075,right=.98,bottom=.09,top=.87,hspace=.48,wspace=.32)
    main=read(DATA/'SCL_case_and_branch_readout_summary.json')
    from check_rate_Bleading_return_witness import verified_return_witness
    witness=verified_return_witness()
    marked_witness=None
    case_family={'b':'A','c':'double','d':'Bleading','e':'single'}
    labels={'A':'H1 family','B':'H2 family','double':'Alternating bursts',
            'Bleading':'B-leading bursts','single':'A-leading bursts'}
    families=['A','B','double','Bleading','single'];report=[]
    for family,ax,letter in zip(families,axes.ravel(),'ABCDE'):
        rows=[r for r in summary['rows'] if r['family']==family]
        assert rows and [r['index'] for r in rows]==sorted(r['index'] for r in rows)
        groups=[];start=0
        for i in range(1,len(rows)):
            if rows[i]['index']!=rows[i-1]['index']+1 or rows[i]['index'] in manifest['continuation_breaks'][family]:
                groups.append(rows[start:i]);start=i
        groups.append(rows[start:])
        maximum=max(max(r['SCL_peak_threshold_ratio_max']) for r in rows)
        minimum=min(min(r['SCL_peak_threshold_ratio_min']) for r in rows)
        for group in groups:
            j=[r['J_EE_core'] for r in group]
            lo=np.array([r['SCL_peak_threshold_ratio_min'] for r in group]);hi=np.array([r['SCL_peak_threshold_ratio_max'] for r in group])
            for k,color in enumerate(colors):
                y=(lo[:,k]+hi[:,k])/2
                if len(group)>1:
                    ax.plot(j,y,color=color,lw=1.25)
                    ax.fill_between(j,lo[:,k],hi[:,k],color=color,alpha=.18,lw=0)
                else:
                    ax.errorbar(j,y,yerr=np.array([y-lo[:,k],hi[:,k]-y]),fmt='o',color=color,ms=5,capsize=2,lw=1)
        ax.axhline(1,color='black',ls='--',lw=1)
        if family=='Bleading' and witness is not None:
            evidence=witness['evidence']
            candidates=[r for r in rows if Path(r['orbit']).resolve()==Path(evidence['orbit']).resolve()]
            assert len(candidates)==1 and evidence['status']=='UNSTABLE'
            point=candidates[0]
            peak=max(point['SCL_peak_threshold_ratio_max'])
            ax.plot(point['J_EE_core'],peak,'x',color='black',ms=8,mew=1.5,zorder=9)
            marked_witness=dict(source=witness['source'],orbit=evidence['orbit'],
                J_EE_core=point['J_EE_core'],status=evidence['status'],
                numerical_unstable_dimension=evidence['classification']['numerical_unstable_dimension'])
        for case in main['rows']:
            if case_family.get(case['label'])!=family:continue
            scl=np.array([n.startswith('SCL') for n in main['contact_names']])
            peak=max(np.array(r['contact_peak_to_threshold'])[scl].max() for r in case['records'])
            ax.plot(case['J_EE_core'],peak,'*',color='black',ms=8,zorder=8)
            ax.annotate(case['label'],(case['J_EE_core'],peak),xytext=(6,7),textcoords='offset points',weight='bold')
        j=np.array([r['J_EE_core'] for r in rows]);pad=max(.0003,np.ptp(j)*.07)
        ax.set(xlim=(j.min()-pad,j.max()+pad),ylim=(min(.0008,minimum*.7),max(4.,maximum*1.7)),yscale='log',
            xlabel=r'$J_{\mathrm{EE,core}}$',ylabel='Filtered peak / contact threshold',
            title=f'{letter}   {labels[family]} ({len(rows)} checked points)')
        ax.yaxis.set_major_locator(FixedLocator([.001,.01,.1,1,10,100]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'))
        ax.yaxis.set_minor_locator(FixedLocator([]))
        if np.ptp(j)<.04:
            ax.xaxis.set_major_locator(plt.MaxNLocator(4));ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        style(ax)
        counts=[r['SCL_count'] for r in rows]
        report.append(dict(family=family,checked_points=len(rows),J_range=[float(j.min()),float(j.max())],
            points_with_SCL_at_all_bin_origins=sum(min(c)>0 for c in counts),
            SCL_count_range=[min(map(min,counts)),max(map(max,counts))],
            all_bin_origins_agree=all(len(set(c))==1 for c in counts),
            contiguous_plotted_runs=len(groups),unverified_family_samples=len(manifest['unverified_family_samples'][family])))
    ax=axes.ravel()[-1];ax.axis('off')
    handles=[Line2D([0],[0],color=c,lw=2,label=n) for c,n in zip(colors,names)]
    handles += [Line2D([0],[0],color='black',ls='--',label='Individual contact threshold'),
                Line2D([0],[0],color='black',marker='*',ls='',ms=8,label='Main-figure case (b–e)')]
    if marked_witness is not None:
        handles.append(Line2D([0],[0],color='black',marker='x',ls='',ms=8,label='Unstable cycle: paired spectrum'))
    ax.legend(handles=handles,loc='upper left',frameon=False,fontsize=12)
    ax.text(.04,.28 if marked_witness is not None else .38,'No line across unverified gaps.\n\nRecruitment also requires ≥ 4 ms\nabove the original threshold.',
            transform=ax.transAxes,fontsize=11,va='top',linespacing=1.4)
    fig.suptitle('SCL readout along physically checked periodic branches',fontsize=15,y=.97)
    name='SCL_recruitment_by_periodic_family';save_new(fig,name)
    write(folder/'figure_summary.json',dict(source=str(folder/'summary.json'),families=report,
        individually_classified_return_witness=marked_witness,
        readout='Four original SCL contact thresholds; ranges span four deterministic bin origins, not uncertainty across realizations.',
        scope='Periodic solutions, including unstable or unclassified branches. No complete parameter window or new dynamical bifurcation is established.'))
    path=OUTPUT/'figures/README.md';body=path.read_text()
    body=re.sub(r'^### '+name+r'\.png\s*\n.*?(?=^### |\Z)','',body,flags=re.M|re.S).rstrip()
    body+='\n\n### '+name+'.png\n在通过完整波形检查的五类周期分支上，逐点展示四个 SCL 触点滤波峰值与原检测阈值之比；b–e 与主图代表状态一致。只连接延拓顺序上相邻且已验证的点，保留未经波形检查的空缺；色带仅表示四个 bin 起点的差别。黑色叉号标出返回分支上经两套时间步长验证的不稳定周期样本。**关注点**：单触点招募还要求持续至少4 ms，不等于合格群体事件；本图含不稳定或稳定性待定的周期解，不能视为完整的稳定吸引子参与窗口。\n'
    path.write_text(body)
    print('SCL FAMILY SUMMARY',report,flush=True)


if __name__=='__main__':main()
