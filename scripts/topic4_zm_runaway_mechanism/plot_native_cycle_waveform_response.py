"""Local response diagnostic; these are selected groups, not whole-core rates."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    source = OUT/'native_cycle_waveform_response'
    assert read(source/'independent_audit.json')['status']=='PASS'
    result = read(source/'result.json');z=np.load(source/'response.npz')
    t=z['phase_centres']*float(z['T_ms'])
    plt.rcParams.update({'font.size':12,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(9,6.5),sharex=True)
    fig.subplots_adjust(left=.10,right=.98,bottom=.14,top=.86,hspace=.32,wspace=.28)
    for i,(ax,row) in enumerate(zip(axes.flat,result['rows'])):
        ax.plot(t,z['predicted_hz'][i],color='black',lw=1.5,label='Rate model')
        ax.plot(t,z['measured_hz'][i],color='#9b4f96',lw=1.5,label='Driven LIF population')
        ax.fill_between(t,z['measured_hz'][i]-2*z['sem_hz'][i],
                        z['measured_hz'][i]+2*z['sem_hz'][i],color='#9b4f96',alpha=.22,lw=0)
        ax.text(0,1.05,row['label']+' subgroup',transform=ax.transAxes)
        ax.text(-.15,1.05,'ABCD'[i],transform=ax.transAxes,fontweight='bold',fontsize=16)
        ax.set_xlim(0,float(z['T_ms']));ax.set_xticks([0,100,200])
        ax.set_ylim(bottom=0);ax.spines[['top','right']].set_visible(False)
        ax.set_ylabel('Rate (Hz / neuron)')
        if i>=2:ax.set_xlabel('Time in cycle (ms)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.54,.99),ncol=2,frameon=False)
    dest=OUT/'figures';name='fig_native_cycle_local_response_validation'
    for ext in ['png','pdf','svg']:fig.savefig(dest/f'{name}.{ext}',dpi=190)
    plt.close(fig)
    write(dest/f'{name}.json',dict(source=str(source/'response.npz'),
          audit=str(source/'independent_audit.json'),groups=result['rows'],
          meaning='One selected subgroup per class; open-loop response to the same candidate-cycle input.',
          uncertainty='Mean plus/minus 2 SEM across 2048 independent noise paths; 20 cycles within each replicate.',
          human_visual_acceptance='PENDING'))
    path=dest/'README.md';text=path.read_text();heading=f'### {name}.png / .pdf / .svg'
    if heading not in text:
        path.write_text(text+'\n'+heading+'\n'
          '把原生空间Z路径候选周期的输入均值与方差波形送给同参数的有色高斯输入LIF群体，与冻结rate模型的输出比较。'
          '四格分别是核A、核B、外围E和I中预先按周期族形变能量选取的一个群体，不是整个核的平均；色带为2048条独立噪声路径的均值±2 SEM。'
          '**关注点**：全部四组未通过预定波形验证，E组平均放电高估约21%–23%；这是局部开放输入检验，不是自主SNN仿真或分岔图。\n')
    print(dest/f'{name}.png')


if __name__=='__main__':main()
