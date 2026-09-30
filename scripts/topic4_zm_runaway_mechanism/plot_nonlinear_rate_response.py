"""Fixed four-source validation display, never a bifurcation diagram."""
from nonlinear_rate_response import DEST,OUT,np,read,write
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    result=read(DEST/'validation/result.json')
    reference=np.load(OUT/'factorial_waveform/response.npz')
    info=read(OUT/'factorial_waveform/preparation.json')['rows']
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                        'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(10,6.8))
    fig.subplots_adjust(left=.09,right=.98,bottom=.17,top=.94,hspace=.38,wspace=.25)
    for letter,ax,index in zip('ABCD',axes.ravel(),[0,3,6,9]):
        data=np.load(DEST/f'validation/factorial_waveform_{index:02d}.npz')
        T=float(data['period_ms']);time=(np.arange(128)+.5)*T/128
        rate=reference['measured_hz'][index];sem=reference['sem_hz'][index]
        ax.fill_between(time,rate-2*sem,rate+2*sem,color='black',alpha=.15,lw=0)
        ax.plot(time,rate,color='black',lw=1.5,label='LIF ensemble')
        ax.plot(time,data['predicted_hz'],color='#7651A8',lw=1.5,label='Finite-state rate candidate')
        ax.set_xlim(0,T);ax.set_xticks([0,100,200]);ax.set_ylim(bottom=0)
        ax.set_xlabel('Input phase (ms)');ax.set_ylabel('Rate (Hz / neuron)')
        ax.text(-.14,1.04,letter,transform=ax.transAxes,fontweight='bold',fontsize=15)
        ax.text(.48,.93,info[index]['source_label'],transform=ax.transAxes,va='top',ha='center')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.52,.03))
    stem=OUT/'figures/fig_finite_state_rate_response_validation'
    for extension in ['png','pdf','svg']:fig.savefig(stem.with_suffix('.'+extension),dpi=180)
    plt.close(fig)
    write(stem.with_suffix('.json'),dict(source=str(DEST/'validation/result.json'),source_indices=[0,3,6,9],
        selection='Same four full input profiles specified before this candidate; no best-example selection.',
        observed='8192localcoloredLIFnoise paths; shade is plus/minus2SEM. Repeated periods aggregated within each path.',
        predicted='36linear history states and calibrated bounded nonlinear rate readout; independent inputs, no native future spikes.',
        validation_status=result['status'],scope='Local imposed-input test. Not autonomous spatial output, a periodic orbit branch or a bifurcation figure.',
        human_visual_acceptance='PENDING'))
    path=OUT/'figures/README.md';text=path.read_text()
    if '### fig_finite_state_rate_response_validation.png' not in text:
        text+='''\n\n### fig_finite_state_rate_response_validation.png / .pdf / .svg
四格沿用此前已固定的核A、核B、核外E和I子群输入，黑线为局部LIF集合平均率与±2SEM，紫线为36个连续历史状态加非线性率读出的独立预测。该候选只使用原训练表和224条新训练输入拟合，图示输入与另64条新验证输入均未用于拟合；完整验证仍失败。**关注点**：强输入后的率峰及恢复误差仍存在，这使候选不能支撑当前网络的分岔认证；本图不表示自主网络轨迹或分岔分支。\n'''
        path.write_text(text)


if __name__=='__main__':main()
