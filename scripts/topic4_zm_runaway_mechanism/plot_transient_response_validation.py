"""All fresh-profile errors plus the recurring weak E and late I examples."""
from transient_response_bias import DEST,np,read,write,log
from validate_transient_response_correction import VDIR
from common import OUT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    assert read(VDIR/'independent_audit.json')['status']=='PASS'
    assert read(VDIR/'implementation_audit.json')['status']=='PASS'
    result=read(VDIR/'result.json');rows=result['rows']
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),gridspec_kw={'width_ratios':[.8,1.25,1.25]})
    fig.subplots_adjust(left=.065,right=.985,bottom=.22,top=.87,wspace=.37)
    ax=axes[0]
    labels=set()
    for r in rows:
        c=r['comparisons'][1];pop='E' if r['pop']==0 else 'I';key=(pop,r['kind'])
        label=f'{pop}, {r["kind"]} inputs' if key not in labels else None;labels.add(key)
        ax.scatter(c['parent_L2'],c['L2'],s=32,c='#ba536d' if pop=='E' else '#28848c',marker='o' if r['kind']=='early' else 's',label=label,zorder=3)
    ax.plot([0,.3],[0,.3],color='.55',lw=.8,ls=':');ax.axhline(.15,color='black',lw=.8,ls='--')
    ax.set(xlim=(0,.3),ylim=(0,.3),xlabel='Parent response error',ylabel='Corrected response error',xticks=[0,.1,.2,.3],yticks=[0,.1,.2,.3])
    ax.legend(frameon=False,fontsize=8,loc='upper left',handletextpad=.3)
    selections=[1,12]
    for ax,case in zip(axes[1:],selections):
        r=next(r for r in rows if r['id']==case);j=r['index'];kind=r['kind']
        mc=np.load(VDIR/f'{kind}_lif_dt0.05.npz');pred=np.load(VDIR/f'{kind}_prediction_dt0.05.npz')
        reference=mc['counts'][j,:int(mc['replicates'][j])].mean(0)*20
        times=(r['burn_ms']+np.arange(r['bins'])*50+25)/1000
        for values,label,color in [(reference,'LIF reference','#202020'),(pred['parent_counts'][j]*20/r['N'],'Parent response','#865ba6'),(pred['counts'][j]*20/r['N'],'Transient correction','#27867b')]:
            ax.plot(times,values,color=color,lw=1.05,label=label)
        ax.set(xlim=(r['burn_ms']/1000,(r['burn_ms']+50*r['bins'])/1000),ylim=(0,None),xlabel='Time (s)',ylabel='Rate (Hz / neuron)')
        ax.text(.5,1.06,'E, faster input' if case==1 else 'I, slower input',ha='center',transform=ax.transAxes)
    for j,ax in enumerate(axes):ax.text(-.22,1.08,'ABC'[j],fontsize=14,fontweight='bold',transform=ax.transAxes)
    handles,labels=axes[1].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.65,.005))
    folder=OUT/'figures';stem='fig_transient_response_validation'
    for ext in ['png','pdf','svg']:fig.savefig(folder/f'{stem}.{ext}',dpi=200)
    plt.close(fig)
    write(folder/f'{stem}.json',dict(source=str(VDIR/'result.json'),reference_dt_ms=.05,all_profiles=14,examples=selections,
        example_selection='RecurringweakEfailure (group531 variant1) andpreviouslateIovercorrection location (group594 variant0); illustrative, not evidence in place of all14 rows.',
        panel_A='Everyfreshcondition. Dotteddiagonal=noimprovement; horizontal.15 is waveformcriterion only. Countbiascriterionalsoapplies inreport, omittedfromthisaxis.',
        panel_B_C='Newprescribedsyntheticinput histories andindependentGaussianLIF reference, localclockfrominitialization; notnativeSNNtrajectories or bifurcationorbits.',
        parameters_locked_before_new_references=True,agent_PNG_PDF_check='PENDING',human_visual_acceptance='PENDING',model_promoted=False))
    p=folder/'README.md';heading=f'### {stem}.png / .pdf / .svg';assert heading not in p.read_text()
    with p.open('a') as f:f.write('\n'+heading+'\nA显示14个新输入条件的原率响应与瞬态修正误差，B/C分别展示重复失败的较弱核外E位置和此前受统一偏移影响的晚期I位置。全部预测与参数在新LIF目标生成前锁定，LIF只作局部参考；新验证13/14通过，失败和原有模型缺口保留。**关注点**：瞬态修正能保护晚期响应但未解决较弱核外位置；本图不是原生轨迹或分岔图。\n')
    log('TRANSIENT RESPONSE FIGURE',folder/f'{stem}.png')


if __name__=='__main__':main()
