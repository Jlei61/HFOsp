"""Resource path and common-time spatial fields; diagnostic, not bifurcation plot."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from native_readouts import centers
import argparse


def main(stochastic=False,variance_split=False):
    if variance_split:stochastic=True
    parent=OUT/('closure_stochastic_sensitivity' if stochastic else 'closure_network_sensitivity')
    review=OUT/'shared_variance_network_sensitivity' if variance_split else parent
    q=read(review/'independent_comparison.json')
    assert q['status']==('READOUT_AUDIT_PASS' if stochastic else 'MATCHED_INITIAL_AND_READOUT_AUDIT_PASS')
    t_ms=9870.;half=25.
    labels={'native':'Native SNN','frozen':'Original closure','units':'Voltage scaling','units_history':'History weights'}
    colors={'frozen':'#555555','units':'#d28a1e','units_history':'#8755a5'}
    labels['units_history_delay_covariance']='History + variance correction'
    colors['units_history_delay_covariance']='#238467'
    plt.rcParams.update({'font.size':12,'pdf.fonttype':42,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(11.8,5.5))
    grid=fig.add_gridspec(2,3,width_ratios=[1.65,1,1],left=.08,right=.885,bottom=.12,top=.91,
                         wspace=.33,hspace=.5)
    ax=fig.add_subplot(grid[:,0]);snapshots=[]
    keys=['native','frozen','units_history','units_history_delay_covariance'] if variance_split else ['native','frozen','units','units_history']
    for j,key in enumerate(keys):
        if key=='native':
            z=np.load(BASE/'native_reference/seed9108401_readouts.npz');t=z['t'];field=z['rate_cells']
            d=read(BASE/'native_reference/checkpoint_projections.json');times=np.array(sorted(map(int,d)))
            ax.plot(times/1000,[d[str(t)]['D'] for t in times],marker='o',linestyle='none',color='black',ms=4,label=labels[key])
        else:
            source=BASE/'runs/A4_stoch_seed9108401' if stochastic and key=='frozen' else parent/key
            if key=='units_history_delay_covariance':source=review/key
            z=np.load(source/'trajectory.npz');t=z['time_ms'];field=z['field_E_hz']
            ts=z['state_time_ms'] if 'state_time_ms' in z else (np.arange(len(z['D']))+1)*10.
            ax.plot(ts/1000,z['D'],color=colors[key],lw=1.5,label=labels[key])
        selected=(t>=t_ms-half)&(t<t_ms+half);assert selected.sum()==50
        state=field[selected].mean(0)
        spatial=fig.add_subplot(grid[j//2,j%2+1])
        im=spatial.imshow(state.reshape(20,20),origin='lower',extent=[0,20,0,20],cmap='magma',vmin=0,vmax=500)
        spatial.set_xticks([0,10,20]);spatial.set_yticks([0,10,20])
        spatial.set_xlabel('x (mm)')
        if j%2==0:spatial.set_ylabel('y (mm)')
        else:spatial.tick_params(labelleft=False)
        panel_label='Variance corrected' if key=='units_history_delay_covariance' else labels[key]
        spatial.text(0,1.08,panel_label,transform=spatial.transAxes,fontsize=11)
        spatial.text(-.28,1.08,'BCDE'[j],transform=spatial.transAxes,fontweight='bold',fontsize=16)
        for center,name in zip(centers,'AB'):
            spatial.add_patch(Circle(center,1.5,fill=False,ec='#16c1ce',lw=1.0))
            spatial.text(center[0],center[1]+2,name,ha='center',color='#16c1ce',fontsize=9)
        snapshots.append(dict(label=key,window_ms=[t_ms-half,t_ms+half],sample_count=int(selected.sum()),
                              sample_times_range=[float(t[selected][0]),float(t[selected][-1])]))
    ax.axvline(t_ms/1000,color='black',lw=.7,ls=':')
    ax.set(xlim=(0,12.7),ylim=(0,.8 if stochastic else .6),xlabel='Time (s)',ylabel=r'$D=1-\langle Z_E\rangle$')
    ax.set_xticks([0,4,8,12]);ax.set_yticks([0,.2,.4,.6,.8] if stochastic else [0,.2,.4,.6])
    ax.text(-.17,1.04,'A',transform=ax.transAxes,fontsize=16,fontweight='bold')
    ax.legend(frameon=False,loc='upper left',fontsize=10,handlelength=2.2)
    fig.text(.63,.975,'9.870 s',ha='center',fontsize=12)
    cb=fig.add_axes([.913,.21,.013,.54]);bar=fig.colorbar(im,cax=cb,ticks=[0,250,500])
    bar.set_label('E rate (Hz)',labelpad=3)
    folder=OUT/'figures';name='fig_closure_stochastic_sensitivity' if stochastic else 'fig_closure_network_sensitivity'
    if variance_split:name='fig_variance_split_stochastic_sensitivity'
    for ext in ['png','pdf','svg']:fig.savefig(folder/f'{name}.{ext}',dpi=190)
    plt.close(fig)
    write(folder/f'{name}.json',dict(source=str(review/'independent_comparison.json'),snapshots=snapshots,
        earlier_source=str(parent/'independent_comparison.json'),
        resource='E-cell count weighted D, measured native checkpoints shown as separate dots without interpolation',
        rate_conditions=('All Z/M dynamic; original recorded external input, shared/private variance split, same group Philox seed1 and .1ms step. State-dependent Poisson counts are not identical.' if stochastic else 'All Z/M dynamic; common deterministic meaninput and .05ms step. Native reference has its original stochastic input.'),
        scope='Closure sensitivity diagnostic, no accepted replacement or bifurcation point.',
        human_visual_acceptance='PENDING',agent_PNG_PDF_check='PENDING'))
    readme=folder/'README.md';text=readme.read_text();heading=f'### {name}.png / .pdf / .svg'
    if heading not in text:
        condition=('rate沿用原A4的逐位置外部输入与有限群体随机发放；噪声键一致，但Poisson计数随各组自身率变化，不能称放电完全相同。' if stochastic else 'rate使用平均外驱，不能以此单独宣称恢复原生随机传播。')
        introduction=('左侧比较原响应、读取历史系数、以及再修正公共/私有方差分配的rate轨迹；三组从同一初态和延迟历史出发，Z与M均动态。方差修正由原图的物理延迟与突触核计算，恢复平稳Poisson方差恒等式，不代表已经恢复动态协方差频谱。' if variance_split else '左侧比较原响应、方差到均值系数的电压量纲修正、以及再改为读取历史输入的rate模型；三组从同一初态和完全相同延迟历史出发，Z与M都动态。')
        readme.write_text(text+'\n'+heading+'\n'
          +introduction+'黑点是原生SNN真实检查点，未在检查点间插值。'
          '右侧为同一9.870秒、50ms窗口的原生及三种rate空间活动，统一0–500Hz色标；'+condition+
          '**关注点**：已诊断的响应近似会不会改变资源路径、进入时刻与同一时刻的空间募集；此图不是完成的分岔图，新响应尚未通过模型验收。\n')
    log('PLOT',folder/f'{name}.png')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stochastic',action='store_true')
    p.add_argument('--variance-split',action='store_true');a=p.parse_args();main(a.stochastic,a.variance_split)
