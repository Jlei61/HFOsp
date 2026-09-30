"""Same-clock response of all nine geometrically selected surround groups."""
from common import OUT,np,read,write,log
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEST=OUT/'native_early_surround_inputs'


def main():
    assert read(DEST/'independent_response_audit.json')['status']=='PASS'
    assert read(DEST/'local_lif_reference/result.json')['numerical_all_pass']
    z=np.load(DEST/'fixed_readout.npz');groups=read(DEST/'contract.json')['selected_groups']
    mc=np.load(DEST/'local_lif_reference/dt0.1.npz')
    reference=np.column_stack([mc['counts'][j,:int(mc['replicates'][j])].mean(0)*groups[j]['N'] for j in range(16)])
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,3,figsize=(10.8,8.1));fig.subplots_adjust(left=.085,right=.975,top=.94,bottom=.14,wspace=.28,hspace=.38)
    series=[('native_counts','Native SNN','#202020','-'),('counts_free_corrected_rate','Free rate model','#237d88','-'),
        ('counts_native_mean_private_variance','Fixed rate response','#865ba6','-'),
        ('conditional_LIF','Conditional LIF reference','#c38b26','--')]
    t=(z['bin_start_ms']+25)/1000
    for j,ax in enumerate(axes.ravel()):
        N=groups[j]['N'];scale=1000/(50*N)
        for key,label,color,style in series:
            values=reference if key=='conditional_LIF' else z[key]
            ax.plot(t,values[:,j]*scale,color=color,ls=style,lw=1,label=label)
        p=groups[j]['position_mm'];ax.text(.5,1.04,f'({p[0]:.1f}, {p[1]:.1f}) mm',ha='center',transform=ax.transAxes)
        ax.text(-.16,1.04,'ABCDEFGHI'[j],fontweight='bold',fontsize=14,transform=ax.transAxes)
        ax.set(xlim=(.5,3),ylim=(0,None),xticks=[.5,1,2,3])
        if j%3==0:ax.set_ylabel('E rate (Hz / neuron)')
        if j>=6:ax.set_xlabel('Time (s)')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.01))
    folder=OUT/'figures';stem='fig_native_early_surround_response'
    for ext in ['png','pdf','svg']:fig.savefig(folder/f'{stem}.{ext}',dpi=190)
    plt.close(fig)
    write(folder/f'{stem}.json',dict(source=str(DEST/'input_response_result.json'),group_ids=[r['group'] for r in groups[:9]],
        meaning='Allnine geometry-selectedsurroundEtargets, same50msclock. Currentprivate-Qfixedreadout andindependentGaussianLIF underexactlythesamenative groupnetmean/privatevariance/Z. Native andfreecorrectedratecounts are references.',
        scope='Supplied-input diagnostic, not a fullsurroundpopulation estimate, autonomouspropagation validation or bifurcation.',
        conditional_reference='local_lif_reference/dt0.1.npz; all16conditions also passed0.05ms numericalsensitivity. LIF is an inputresponse reference, not a replacementnetwork.',
        omitted_from_figure='Projectedmean andnativemarginalvariance conditions,core/Icontroltargets remain inall-resultJSON.',
        agent_PNG_PDF_check='PENDING',human_visual_acceptance='PENDING'))
    p=folder/'README.md';heading=f'### {stem}.png / .pdf / .svg'
    description='\n'+heading+'\n九格显示预先按几何选择的核外E群体，在相同0.5–3秒时钟与50ms窗下比较原生、自由率模型，以及同一原生输入驱动的固定率响应和独立高斯LIF参考。后两者共用原生净电流均值、修复后的私有方差和Z，LIF只作局部输入响应参考；这里用了原生放电重建输入，不能作为自主率网络验收。**关注点**：固定率响应是否系统低估条件LIF及原生短事件，进而为检验闭环累积误差提供方向；九个局部群体不代表整个核外总体。\n'
    old=p.read_text()
    if heading in old:
        start=old.index(heading);end=old.find('\n### ',start+len(heading));end=len(old) if end<0 else end
        p.write_text(old[:start].rstrip()+description+old[end:])
    else:p.write_text(old+description)
    log('EARLY INPUT FIGURE',folder/f'{stem}.png')


if __name__=='__main__':main()
