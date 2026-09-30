"""Whole-network sensitivity to fixed response perturbations, not branches."""
from common import OUT,np,read,write,log
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

DEST=OUT/'early_response_bias_diagnostic/causal_counterfactual'


def main():
    assert read(DEST/'result.json')['status']=='READOUT_AUDIT_PASS'
    z=np.load(DEST/'audit_readouts.npz')
    geo=np.load(OUT/'native_early_surround_inputs/membership.npz')
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(10.8,6.1),gridspec_kw={'height_ratios':[.8,1.]})
    fig.subplots_adjust(left=.07,right=.86,bottom=.11,top=.92,hspace=.38,wspace=.3)
    labels=[('native','Native SNN','#202020'),('parent','Rate model','#237d88'),('bias_counterfactual','Response perturbation','#ad5c28')]
    vmax=np.ceil(max(z[label+'_mean_field_hz'].max() for label,_,_ in labels)/10)*10
    rate_max=max(uniform_filter1d(z[label+'_global_hz'],10,mode='nearest').max() for label,_,_ in labels)*1.05
    for j,(label,title,color) in enumerate(labels):
        ax=axes[0,j];smooth=uniform_filter1d(z[label+'_global_hz'],10,mode='nearest')
        ax.plot(z[label+'_time_ms']/1000,smooth,color=color,lw=.85)
        ax.set(xlim=(.5,3),ylim=(0,rate_max),xlabel='Time (s)',xticks=[.5,1,2,3])
        ax.text(.5,1.08,title,ha='center',transform=ax.transAxes)
        ax.text(-.22,1.08,'ABC'[j],fontsize=15,fontweight='bold',transform=ax.transAxes)
        if j==0:ax.set_ylabel('Global E rate (Hz)')
        ax=axes[1,j]
        im=ax.imshow(z[label+'_mean_field_hz'].reshape(20,20),origin='lower',extent=[0,20,0,20],vmin=0,vmax=vmax,cmap='magma',interpolation='nearest')
        for k,c in enumerate(geo['centers_mm']):
            ax.add_patch(Circle(c,1.75,fill=False,color='#16c5c7',lw=1.2));ax.text(c[0],c[1]+2.1,'AB'[k],color='#16c5c7',ha='center',fontsize=10)
        ax.set(xlabel='x (mm)',xticks=[0,10,20],yticks=[0,10,20])
        if j==0:ax.set_ylabel('y (mm)')
        ax.text(-.22,1.03,'DEF'[j],fontsize=15,fontweight='bold',transform=ax.transAxes)
    cax=fig.add_axes([.9,.13,.018,.31]);fig.colorbar(im,cax=cax,label='Mean E rate (Hz)')
    stem='fig_early_response_bias_counterfactual';folder=OUT/'figures'
    for ext in ['png','pdf','svg']:fig.savefig(folder/f'{stem}.{ext}',dpi=190)
    plt.close(fig)
    write(folder/f'{stem}.json',dict(source=str(DEST/'result.json'),window_ms=[500,3000],
        top='Actualclock10mssmoothedglobalE rates, sameinitialization/exogenousdrive. No eventalignment.',
        bottom='Cellcountweighted20x20 spatial mean rates over0.5-3s, not instantaneoussnapshots or bifurcationeigenmodes.',
        counterfactual='Two alreadylocked localLIF-derived E/Ihazard offsets only. OriginallocalgateFAILretained; separatelyregistered sensitivity, noacceptedmodel.',
        field_vmax_hz=float(vmax),core_circle_radius_mm=1.75,model_promoted=False,agent_PNG_PDF_check='PENDING',human_visual_acceptance='PENDING'))
    p=folder/'README.md';heading=f'### {stem}.png / .pdf / .svg'
    assert heading not in p.read_text()
    with p.open('a') as f:
        f.write('\n'+heading+'\n上排比较原生、未改率模型和仅加入固定E/I响应偏差的3秒自由运行，全局率以同一10ms定义平滑。下排是0.5–3秒同窗的二维时间平均活动，非瞬时快照；所有Z/M动态，连接、输入、种子和初态保持原样。**关注点**：局部响应变化能否恢复核外招募及事件间低活动；修正的局部验证失败仍保留，该图不能作为模型验收或分岔图。\n')
    log('EARLY RESPONSE BIAS FIGURE',folder/f'{stem}.png')


if __name__=='__main__':main()
