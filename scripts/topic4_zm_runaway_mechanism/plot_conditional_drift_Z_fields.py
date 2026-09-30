"""Matched count/no-count rate and spatial comparison; not a bifurcation plot."""
from common import OUT,model,np,read,write,log
from conditional_drift_Z_fields import DEST,SOURCE
from audit_fine_rate_frozen_Z_fields import moving_mean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def main():
    a=read(DEST/'independent_comparison.json');assert a['status']=='READOUT_AUDIT_PASS' and len(a['rows'])==3
    s=model(40);fig=plt.figure(figsize=(10,8.4));gs=fig.add_gridspec(3,3,left=.12,right=.88,bottom=.08,top=.92,hspace=.38,wspace=.22,height_ratios=[.8,1,1])
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'})
    meta=[];handles=[]
    for col,row in enumerate(a['rows']):
        label=row['label'];count=np.load(SOURCE/label/'trajectory.npz');drift=np.load(DEST/label/'trajectory.npz')
        ax=fig.add_subplot(gs[0,col]);t=count['time_ms'];assert np.array_equal(t,drift['time_ms'])
        for z,color,name in [(count,'#bb762d','Count sampling'),(drift,'#222222','Sampling off')]:
            line,=ax.plot(t/1000,moving_mean(z['global_E_hz'],10),color=color,lw=.7,label=name)
            if col==0:handles.append(line)
        ax.set(xlim=(9,12.5),ylim=(0,510),xticks=[9,10,11,12],yticks=[0,250,500],xlabel='Time (s)')
        ax.spines[['top','right']].set_visible(False);ax.axvline(12,color='black',lw=.6,ls=':')
        tm=int(label.split('_')[1][1:])/1000
        ax.text(.5,1.10,f'Z field: {tm:.2f} s\n'+r'$\langle Z_E\rangle='+f'{1-row["D"]:.3f}'+r'$',ha='center',va='bottom',transform=ax.transAxes)
        if col==0:
            ax.set_ylabel('Global E rate (Hz)');ax.text(-.32,1.15,'A',transform=ax.transAxes,fontsize=17,fontweight='bold')
            ax.legend(loc='upper right',frameon=False,fontsize=9)
        else:ax.tick_params(labelleft=False)
        for k,(z,name) in enumerate([(count,'Count sampling'),(drift,'Sampling off')],1):
            ax=fig.add_subplot(gs[k,col]);sel=(t>=11975)&(t<12025);assert sel.sum()==50
            im=ax.imshow(z['field_E_hz'][sel].mean(0).reshape(20,20),extent=[0,20,0,20],origin='lower',cmap='magma',vmin=0,vmax=500)
            ax.set(xticks=[0,10,20],yticks=[0,10,20])
            if k==2:ax.set_xlabel('x (mm)')
            else:ax.tick_params(labelbottom=False)
            if col==0:
                ax.set_ylabel(name+' (12.00 s)\ny (mm)');ax.text(-.32,1.02,'BC'[k-1],transform=ax.transAxes,fontsize=17,fontweight='bold')
            else:ax.tick_params(labelleft=False)
            for center,letter in zip(s.geo['centers_mm'],'AB'):
                ax.add_patch(Circle(center,1.75,fill=False,color='#20c4cf',lw=.9))
                ax.text(center[0],center[1]+2.2,letter,color='#20c4cf',ha='center',fontsize=9)
            meta.append(dict(condition=label,sampling=name,snapshot_time_ms=12000,window_ms=[11975,12025]))
    cax=fig.add_axes([.92,.08,.014,.54]);bar=fig.colorbar(im,cax=cax,ticks=[0,250,500]);bar.set_label('E rate (Hz)')
    folder=OUT/'figures';name='fig_conditional_drift_Z_fields'
    for ext in ['png','pdf','svg']:fig.savefig(folder/f'{name}.{ext}',dpi=190)
    plt.close(fig)
    write(folder/f'{name}.json',dict(source=str(DEST/'independent_comparison.json'),snapshots=meta,
        states='Both modes use the same9000ms fullrate history, heldfullZfield, dynamicM andfutureexogenousinput. OnlyBinomial samplingremoved; privateQretained.',
        lineage='Legacy-private variance version for a pairednoise-removal diagnostic. Physical-delay unit repair is a separate version; these data not overwritten.',
        scientific_scope='Finite-window conditionalnoisecomparison, no bifurcation orstability claim.',agent_PNG_PDF_check='PENDING',human_visual_acceptance='PENDING'))
    f=folder/'README.md';heading=f'### {name}.png / .pdf / .svg'
    if heading not in f.read_text():
        with f.open('a') as h:h.write('\n'+heading+'\n三列对应原生9.00、9.42、9.87秒完整Z场，均从同一率模型9秒历史出发；橙色保留计数采样，黑色只关闭采样，私有方差和M动态规则保持相同。下两行显示共同12秒时刻的50ms空间活动，统一0–500Hz色标。这是修复延迟方差单位之前的配对基线，不能作为已通过物理修复或分岔认证的图。**关注点**：移除计数创新后，早期自限与晚期大范围持续的区别是否保留。\n')
    log('PRIVATE DRIFT FIGURE',folder/f'{name}.png')


if __name__=='__main__':main()
