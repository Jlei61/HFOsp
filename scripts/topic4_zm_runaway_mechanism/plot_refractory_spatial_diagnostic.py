"""Common-clock activity, both slow variables and spatial fields of all four runs."""
from common import OUT,BASE,read,write,np,model,log
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import argparse

DEST=OUT/'conditioned_refractory_spatial_diagnostic'

def main(filtered=False,count_consistent=False,fine_grid=False,fine_expected=False,fine_forcing=False):
    audit=read(DEST/'independent_comparison.json');assert audit['status']=='READOUT_AUDIT_PASS'
    s=model();native=np.load(BASE/'native_reference/seed9108401_readouts.npz')
    checkpoints=read(BASE/'native_reference/checkpoint_projections.json')
    labels=['native']+read(DEST/'jobs.json')['completed']
    names=['Native SNN','Mean input','Recorded input','Recorded + noise 1','Recorded + noise 2']
    sources=[None]+[DEST/label for label in labels[1:]]
    if filtered:
        paired=OUT/'conditioned_refractory_external_filter_pair'
        assert read(paired/'independent_comparison.json')['status']=='READOUT_AUDIT_PASS'
        labels=['native','expected_unfiltered','expected_filtered','noise1_unfiltered','noise1_filtered']
        names=['Native SNN','Recorded input','Recorded + filter','Noise 1','Noise 1 + filter']
        sources=[None,DEST/'recorded_drive_expected',paired/'recorded_drive_expected',DEST/'recorded_drive_poisson_seed1',paired/'recorded_drive_poisson_seed1']
    if count_consistent:
        paired=OUT/'conditioned_refractory_external_filter_pair';count=OUT/'conditioned_refractory_count_consistency'
        assert read(count/'independent_comparison.json')['status']=='READOUT_AUDIT_PASS'
        labels=['native','expected_filtered','poisson_filtered','binomial_filtered']
        names=['Native SNN','Expected rates','Poisson counts','Refractory counts']
        sources=[None,paired/'recorded_drive_expected',paired/'recorded_drive_poisson_seed1',count/'recorded_drive_binomial_seed1']
    if (fine_grid or fine_expected) and not fine_forcing:
        fine=OUT/'conditioned_refractory_spatial_resolution';coarse=OUT/'conditioned_refractory_external_filter_pair';count=OUT/'conditioned_refractory_count_consistency'
        audit_name='partial_comparison.json' if fine_expected else 'independent_comparison.json'
        assert read(fine/audit_name)['status'] in ['READOUT_AUDIT_PASS','PARTIAL_READOUT_AUDIT_PASS']
        labels=['native','expected_g20','expected_g40','binomial_g20','binomial_g40']
        names=['Native SNN','Rate 1 mm','Rate 0.5 mm','Counts 1 mm','Counts 0.5 mm']
        sources=[None,coarse/'recorded_drive_expected',fine/'recorded_drive_expected',count/'recorded_drive_binomial_seed1',fine/'recorded_drive_binomial_seed1']
        if fine_expected:labels=labels[:3];names=names[:3];sources=sources[:3]
    if fine_forcing:
        paired=OUT/'conditioned_refractory_fine_forcing';parent=OUT/'conditioned_refractory_spatial_resolution'
        audit_name='partial_comparison.json' if fine_expected else 'independent_comparison.json'
        assert read(paired/audit_name)['status'] in ['READOUT_AUDIT_PASS','PARTIAL_READOUT_AUDIT_PASS']
        labels=['native','expected_parent_input','expected_fine_input','counts_parent_input','counts_fine_input']
        names=['Native SNN','Rate, 1 mm input','Rate, 0.5 mm input','Counts, 1 mm input','Counts, 0.5 mm input']
        sources=[None,parent/'recorded_drive_expected',paired/'recorded_drive_expected',parent/'recorded_drive_binomial_seed1',paired/'recorded_drive_binomial_seed1']
        if fine_expected:labels=labels[:3];names=names[:3];sources=sources[:3]
    colors=['#151515','#987544','#8064a2','#248b75','#ca6a38']
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False})
    ncols=len(labels);fig=plt.figure(figsize=(10.4 if fine_expected else 14.2*ncols/5,9.1))
    grid=fig.add_gridspec(4,ncols,left=.10 if fine_expected else (.075 if count_consistent else .067),right=.855 if fine_expected else (.89 if count_consistent else .924),bottom=.07,top=.945,
                         height_ratios=[.86,.72,1,1],wspace=.29,hspace=.48)
    metadata=[];images=[]
    for col,(label,name,color,source) in enumerate(zip(labels,names,colors,sources)):
        if label=='native':
            t=native['t'];field=native['rate_cells'];rate=native['allE']
            ts=np.array(sorted(map(int,checkpoints)))
            d=np.array([checkpoints[str(k)]['D'] for k in ts]);m=np.array([checkpoints[str(k)]['mean_M']*.0005 for k in ts])
            ls='none';marker='o'
        else:
            z=np.load(source/'trajectory.npz');t=z['time_ms'];field=z['field_E_hz'];rate=z['global_E_hz']
            ms=model(40) if 'parent_g20' in z.files else s
            ts=z['state_time_ms'];d=z['D'];m=z['M_current'][:,ms.E]@ms.mean_weights
            ls='-';marker=None
        ax=fig.add_subplot(grid[0,col]);ax.plot(t/1000,uniform_filter1d(rate,10,mode='nearest'),color=color,lw=.65)
        ax.set(xlim=(0,12.5),ylim=(0,510),xticks=[0,4,8,12],yticks=[0,250,500],xlabel='Time (s)')
        ax.text(.5,1.12,name,ha='center',transform=ax.transAxes)
        if col==0:ax.set_ylabel('Global E rate (Hz)');ax.text(-.31,1.12,'A',transform=ax.transAxes,fontweight='bold',fontsize=16)
        else:ax.tick_params(labelleft=False)
        for snap in [4.025,9.870]:ax.axvline(snap,c='black',ls=':',lw=.55)
        ax=fig.add_subplot(grid[1,col]);line,=ax.plot(ts/1000,d,color='#242424',lw=1.1,ls=ls,marker=marker,ms=3)
        other=ax.twinx();other.spines['right'].set_visible(True)
        adapt,=other.plot(ts/1000,m,color='#9053a2',lw=1.1,ls=ls,marker=marker,ms=3)
        ax.set(xlim=(0,12.5),ylim=(0,1),xticks=[0,4,8,12],yticks=[0,.5,1],xlabel='Time (s)')
        other.set(ylim=(0,.3),yticks=[0,.15,.3]);other.tick_params(axis='y',colors='#9053a2')
        if col==0:ax.set_ylabel(r'$D=1-\langle Z_E\rangle$');ax.text(-.31,1.1,'B',transform=ax.transAxes,fontweight='bold',fontsize=16)
        else:ax.tick_params(labelleft=False)
        if col==ncols-1:other.set_ylabel(r'$\eta_M\langle M\rangle$ (mV)',color='#9053a2')
        else:other.tick_params(labelright=False)
        for row,snap in enumerate([4025.,9870.],start=2):
            ax=fig.add_subplot(grid[row,col]);selected=(t>=snap-25)&(t<snap+25);assert selected.sum()==50
            values=field[selected].mean(0)
            im=ax.imshow(values.reshape(20,20),origin='lower',extent=[0,20,0,20],cmap='magma',vmin=0,vmax=500)
            ax.set(xticks=[0,10,20],yticks=[0,10,20]);images.append(im)
            if row==3:ax.set_xlabel('x (mm)')
            else:ax.tick_params(labelbottom=False)
            if col==0:
                ax.set_ylabel(f'{snap/1000:.3f} s\ny (mm)')
                ax.text(-.31,1.05,'CD'[row-2],transform=ax.transAxes,fontweight='bold',fontsize=16)
            else:ax.tick_params(labelleft=False)
            for center,core in zip(s.geo['centers_mm'],'AB'):
                ax.add_patch(Circle(center,1.75,fill=False,ec='#20c4cf',lw=.9))
                ax.text(center[0],center[1]+2.2,core,ha='center',color='#20c4cf',fontsize=9)
            metadata.append(dict(label=label,time_ms=snap,window_ms=[snap-25,snap+25],samples=int(selected.sum())))
    cb=fig.add_axes([.90 if fine_expected else (.916 if count_consistent else .946),.075,.01,.391]);bar=fig.colorbar(images[-1],cax=cb,ticks=[0,250,500]);bar.set_label('E rate (Hz)')
    folder=OUT/'figures';name='fig_conditioned_refractory_spatial_diagnostic'
    if filtered:name='fig_refractory_external_mean_filter_pair'
    if count_consistent:name='fig_refractory_count_consistency'
    if fine_grid:name='fig_refractory_spatial_resolution'
    if fine_expected:name='fig_refractory_spatial_resolution_expected'
    if fine_forcing:name='fig_refractory_fine_forcing_expected' if fine_expected else 'fig_refractory_fine_forcing'
    for ext in ['png','pdf','svg']:fig.savefig(folder/f'{name}.{ext}',dpi=190)
    plt.close(fig)
    write(folder/f'{name}.json',dict(source=str(DEST/'independent_comparison.json'),trajectory_sources=[str(x) if x is not None else str(BASE/'native_reference/seed9108401_readouts.npz') for x in sources],labels=labels,snapshots=metadata,
        Z_and_M='Both dynamic in every trajectory; native slow variables are exact checkpoint observations shown as unconnected dots, no interpolation.',
        scope='Common physical time comparison, no phase or onset alignment. Diagnostic rate runs, local validation FAIL, no accepted replacement or bifurcation. Countvariant retains stationaryPoisson privatevarianceapproximation despite changing finiteoutput; limitations remain.' if count_consistent else ('Common physical time comparison, no phase or onset alignment. Diagnostic rate runs, local validation FAIL, no accepted replacement or bifurcation. Filterpair changesonlyexternalAMPAmeanpath.' if filtered else 'Common physical time comparison, no phase or onset alignment. Four diagnostic rate runs, local validation FAIL, no accepted replacement or bifurcation.'),
        conventions='All curves are time courses; no branch stability encoding. No overall title, no narrative gray annotations, same0-500Hz spatial scale.',
        agent_PNG_PDF_check='PENDING',human_visual_acceptance='PENDING'))
    if (fine_grid or fine_expected) and not fine_forcing:
        meta=read(folder/f'{name}.json');meta['source']=str(OUT/'conditioned_refractory_spatial_resolution'/audit_name)
        meta['scope']='Same native physical graph,thresholdmembers,delays,Z/M,lockedresponse andsame1mm externaldrive lifted to0.5mm. Common20x20count-weightedreadout; samephysicalclock, no phase/onset alignment. LocalresponsevalidationFAIL; no modelpromotion or bifurcationclassification.'
        meta['runs']='Expectedflux pair only; registeredfineactual-count run not included.' if fine_expected else 'Two meshresolutions timesexpected oractual-count flux.'
        write(folder/f'{name}.json',meta)
    if fine_forcing:
        meta=read(folder/f'{name}.json');meta['source']=str(paired/audit_name)
        meta['scope']='Same0.5mmratefield,originalgraph,thresholdmembers,response,Z/Mandcountnoise. OnlyexternalspatialOUprojectionchanges:lifted1mmcellmeans versusactual0.5mmgroupmeans. Same1msinputclockandcommon20x20readout; no phase/onset alignment. Scientificacceptancepending,notabifurcationfigure.'
        meta['runs']='Completedexpectedfluxpaironly; fineinputcountarmnotincluded.' if fine_expected else 'Expectedfluxandactualcountpairsatfixed0.5mmdynamicalgrid.'
        write(folder/f'{name}.json',meta)
    p=folder/'README.md';text=p.read_text();heading=f'### {name}.png / .pdf / .svg'
    first='从左到右为原生SNN，以及平均输入、原记录输入、原记录输入加两套有限群体噪声的四条率模型轨迹。'
    if filtered:first='从左到右为原生SNN、原记录输入下外驱均值未滤波/已滤波的一对率轨迹，以及同一有限群体噪声种子下的对应一对。唯一改变是恢复外部输入均值的AMPA两级滤波，率响应、连接、Z/M和随机数键保持不变。'
    if count_consistent:first='从左到右为原生SNN、完整外驱滤波下的期望率模型、原Poisson群体输出、以及按实际可发放人数抽样并用同一计数更新不应期和M的群体率模型。最后两列使用同一种子，但抽样规律改变，不是相同脉冲实现；确定性方程保持不变。私有输入方差仍沿用此前平稳Poisson近似，不能称已完成精确有限群体噪声闭合。'
    if fine_grid:first='从左到右为原生SNN、1mm/0.5mm期望率模型，以及1mm/0.5mm有限群体计数模型。保留同一原生连接图、物理参数和局部率响应；原1mm外部驱动按细胞成员映射给细网格，未同时改变输入分辨率。空间图均按原细胞数权重汇回同一20×20读出网格；细分后有限群体数改变，同一种子不意味着同一放电实现。'
    if fine_expected:first='从左到右为原生SNN、1mm期望率模型、0.5mm期望率模型；只展示已完成的两种空间分辨率确定性率对照，不包含另外注册的细网格计数模型。两者使用同一原生连接图、阈值成员、物理参数和冻结局部响应，原1mm外部驱动按成员关系提升到0.5mm群体。所有空间图均汇回相同20×20细胞加权读出。'
    if fine_forcing:first='固定0.5mm动力学网格，比较继承1mm外部空间输入均值与恢复原始0.5mm群体输入的结果；连接、阈值成员、率响应、Z/M和计数规则均不变。输入仍按1ms采样保持，期望率列也受原有OU驱动，不是恒定输入的自主分岔系统。'+('仅展示原生与已完成的期望率对照三列；计数对照尚未纳入。' if fine_expected else '从左到右是原生、两种输入分辨率的期望率、两种输入分辨率的有限群体计数，共五列。')
    if heading not in text:p.write_text(text+'\n'+heading+'\n'+first+'上两排显示全局活动与D/M，原生慢变量只画真实检查点；下两排是在共同4.025秒和9.870秒的50ms空间活动，未按各自事件或onset对齐。所有运行Z、M均动态，候选局部响应尚未通过验收。**关注点**：局部响应修正能否同时保留自主间期事件和同一资源路径上的全局招募；这是直接对应诊断，不是分岔图。\n')
    log('PLOT',folder/f'{name}.png')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--filtered',action='store_true');p.add_argument('--count-consistent',action='store_true');p.add_argument('--fine-grid',action='store_true');p.add_argument('--fine-expected',action='store_true');p.add_argument('--fine-forcing',action='store_true');a=p.parse_args();main(a.filtered,a.count_consistent,a.fine_grid,a.fine_expected,a.fine_forcing)
