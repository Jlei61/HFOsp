"""Paired local validation display, explicitly separate from bifurcation."""
from refractory_rate_response import DEST,OUT,np,read,write
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    result=read(DEST/'validation/result.json');reference=np.load(OUT/'factorial_waveform/response.npz')
    info=read(OUT/'factorial_waveform/preparation.json')['rows']
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(10,6.8));fig.subplots_adjust(left=.09,right=.98,bottom=.17,top=.94,hspace=.38,wspace=.25)
    for letter,ax,index in zip('ABCD',axes.ravel(),[0,3,6,9]):
        data=np.load(DEST/f'validation/factorial_waveform_{index:02d}.npz');T=float(data['period_ms']);t=(np.arange(128)+.5)*T/128
        rate=reference['measured_hz'][index];sem=reference['sem_hz'][index]
        ax.fill_between(t,rate-2*sem,rate+2*sem,color='black',alpha=.15,lw=0)
        ax.plot(t,rate,color='black',lw=1.5,label='LIF ensemble')
        ax.plot(t,data['predicted_hz'],color='#B25F23',lw=1.5,label='Refractory rate candidate')
        ax.set_xlim(0,T);ax.set_xticks([0,100,200]);ax.set_ylim(bottom=0)
        ax.set_xlabel('Input phase (ms)');ax.set_ylabel('Rate (Hz / neuron)')
        ax.text(-.14,1.04,letter,transform=ax.transAxes,fontweight='bold',fontsize=15)
        ax.text(.48,.93,info[index]['source_label'],transform=ax.transAxes,va='top',ha='center')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.52,.03))
    stem=OUT/'figures/fig_refractory_rate_response_validation'
    for extension in ['png','pdf','svg']:fig.savefig(stem.with_suffix('.'+extension),dpi=180)
    plt.close(fig)
    write(stem.with_suffix('.json'),dict(source=str(DEST/'validation/result.json'),indices=[0,3,6,9],
        selection='Same prespecified four full-input cases as preceding response comparison; not selected by fit quality.',
        observation='8192localLIFpaths per condition; shade2SEM; average over repeated periods within each path.',
        prediction='Own refractory history plus6current covariance and36input history states. No observed firing supplied.',
        validation_status=result['status'],scope='Imposed-input local check, not autonomous spatial trajectory or bifurcation.',human_visual_acceptance='PENDING'))
    path=OUT/'figures/README.md';text=path.read_text()
    if '### fig_refractory_rate_response_validation.png' not in text:
        text+='''\n\n### fig_refractory_rate_response_validation.png / .pdf / .svg
四格继续使用事先固定的核A、核B、核外E和I群体强输入；黑线为局部LIF集合平均及±2SEM，棕线为带连续不应期反馈的率模型独立预测。预测中的不应期占用完全来自模型自己的放电，未输入观测放电历史；原24条波形通过21条，新64条独立波形通过50条，当前候选仍未验收。**关注点**：核外E的恢复时序仍有偏差，小扰动频率响应也未通过；本图是局部响应诊断，不是自主网络或分岔图。\n'''
        path.write_text(text)

if __name__=='__main__':main()
