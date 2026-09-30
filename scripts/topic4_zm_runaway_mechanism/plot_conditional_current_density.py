"""Local response validation; not a network trajectory or bifurcation panel."""
from common import OUT,np,read,write
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    folder=OUT/'conditional_current_density'
    assert read(folder/'independent_audit.json')['full_waveform_gate_pass']
    reference=np.load(OUT/'factorial_waveform/response.npz')
    info=read(OUT/'factorial_waveform/preparation.json')['rows']
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(10.0,6.9))
    fig.subplots_adjust(left=.085,right=.985,bottom=.18,top=.93,hspace=.43,wspace=.23)
    rows=[]
    for letter,ax,index in zip('ABCD',axes.ravel(),[0,3,6,9]):
        z=np.load(folder/f'index{index:02d}_grid256.npz')
        time=z['phase_centres']*float(z['T_ms'])
        mean=reference['measured_hz'][index];sem=reference['sem_hz'][index]
        ax.fill_between(time,mean-2*sem,mean+2*sem,color='black',alpha=.18,lw=0)
        ax.plot(time,mean,color='black',lw=1.8,label='LIF ensemble')
        ax.plot(time,reference['predicted_hz'][index,1],color='#D18A2C',lw=1.4,ls='--',label='Previous rate response')
        ax.plot(time,z['predicted_hz'],color='#7651A8',lw=1.5,label='Population response')
        ax.set_xlim(0,float(z['T_ms']));ax.set_xticks([0,100,200]);ax.set_ylim(bottom=0)
        ax.set_xlabel('Cycle phase (ms)');ax.set_ylabel('Rate (Hz / neuron)')
        ax.text(-.14,1.06,letter,transform=ax.transAxes,fontweight='bold',fontsize=15)
        ax.text(.5,1.04,info[index]['source_label'],transform=ax.transAxes,ha='center')
        q=read(folder/f'index{index:02d}_grid256.json')
        rows.append(dict(label=info[index]['source_label'],source_index=index,waveform_L2=q['waveform_L2'],relative_mean_error=q['relative_mean_error']))
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.52,.025))
    stem=OUT/'figures/fig_population_response_local_validation'
    for extension in ['png','pdf','svg']:fig.savefig(stem.with_suffix('.'+extension),dpi=180)
    plt.close(fig)
    write(stem.with_suffix('.json'),dict(source='conditional_current_density',rows=rows,
        reference='8192 independent colored-LIF paths per input, original factorial assay',
        prediction='Own computed threshold flux, 256 requested voltage nodes; no fitted coefficients or observed spikes supplied',
        uncertainty='LIF plus/minus2SEM, repeated cycles aggregated within paths',
        scope='Four local imposed-input responses, not autonomous spatial dynamics or a bifurcation figure',
        human_visual_acceptance='PENDING'))
    path=OUT/'figures/README.md';text=path.read_text()
    if '### fig_population_response_local_validation.png' not in text:
        text+='''\n\n### fig_population_response_local_validation.png / .pdf / .svg
四格分别为已固定的核A、核B、外围E子群及I子群，输入取旧条件周期的实际三通道波形。黑线与色带为8192条LIF噪声路径的均值和±2SEM，橙虚线为此前历史率响应，紫线为保留复位、不应期及条件突触电流矩的新群体响应；预测没有使用观测放电，也未拟合波形系数。**关注点**：强输入后的恢复是否正确，尤其是此前失败的外围子群；这是局部输入验证图，不是自主网络或分岔图，新响应尚未通过完整模型验收。\n'''
        path.write_text(text)


if __name__=='__main__':main()
