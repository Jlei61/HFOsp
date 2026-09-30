"""Paired local prediction comparison for the conditioning-only intervention."""
from conditioned_refractory_rate import DEST,PARENT,OUT,np,read,write
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    result=read(DEST/'validation/result.json');ref=np.load(OUT/'factorial_waveform/response.npz');meta=read(OUT/'factorial_waveform/preparation.json')['rows']
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(10.5,6.8));fig.subplots_adjust(left=.09,right=.98,bottom=.17,top=.94,hspace=.38,wspace=.25)
    for letter,ax,k in zip('ABCD',axes.ravel(),[0,3,6,9]):
        z=np.load(DEST/f'validation/factorial_waveform_{k:02d}.npz');old=np.load(PARENT/f'validation/factorial_waveform_{k:02d}.npz');T=float(z['period_ms']);t=(np.arange(128)+.5)*T/128
        r=ref['measured_hz'][k];sem=ref['sem_hz'][k];ax.fill_between(t,r-2*sem,r+2*sem,color='black',alpha=.15,lw=0)
        ax.plot(t,r,color='black',lw=1.5,label='LIF ensemble')
        ax.plot(t,old['predicted_hz'],color='#B25F23',ls='--',lw=1.1,label='Previous fit')
        ax.plot(t,z['predicted_hz'],color='#7651A8',lw=1.5,label='Conditioned fit')
        ax.set_xlim(0,T);ax.set_xticks([0,100,200]);ax.set_ylim(bottom=0);ax.set_xlabel('Input phase (ms)');ax.set_ylabel('Rate (Hz / neuron)')
        ax.text(-.14,1.04,letter,transform=ax.transAxes,fontweight='bold',fontsize=15);ax.text(.48,.93,meta[k]['source_label'],transform=ax.transAxes,va='top',ha='center')
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.53,.03))
    stem=OUT/'figures/fig_conditioned_refractory_rate_validation'
    for ext in ['png','pdf','svg']:fig.savefig(stem.with_suffix('.'+ext),dpi=180)
    plt.close(fig)
    write(stem.with_suffix('.json'),dict(source_indices=[0,3,6,9],selection='Unchanged prespecified full-input examples',
        black='8192localLIFpaths,meanand2SEM',brown='Previous rejected refractory readout',purple='Samefunctionclass after conditioning-only refit',
        line_style_scope='Dashed distinguishes previous fit here; this is a local time/phase diagnostic, not a stable/unstable bifurcation branch.',
        scientific_status=result['status'],human_visual_acceptance='PENDING'))
    p=OUT/'figures/README.md';text=p.read_text()
    if '### fig_conditioned_refractory_rate_validation.png' not in text:
        text+='''\n\n### fig_conditioned_refractory_rate_validation.png / .pdf / .svg
固定四个群体的同一输入，比较LIF集合平均（黑色及±2SEM）、前一率模型拟合（棕色虚线）及只改善输入坐标后的拟合（紫色实线）。物理状态、训练数据、模型大小和训练步数均相同；原24波形通过数由21变23，但完整局部验收仍失败。**关注点**：核外E的强输入响应仍有偏差；本图虚实线只区分两版拟合，不代表分岔稳定性，也不是自主空间网络图。\n'''
        p.write_text(text)

if __name__=='__main__':main()
