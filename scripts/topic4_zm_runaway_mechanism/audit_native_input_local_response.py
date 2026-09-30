"""Independent bin/count and numerical-envelope audit of local input response."""
from pathlib import Path
import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918'
DEST=OUT/'native_input_bridge'


def main():
    fine=np.load(DEST/'local_lif/dt0.05.npz');coarse=np.load(DEST/'local_lif/dt0.1.npz')
    fixed=np.load(DEST/'fixed_readout.npz');src=np.load(DEST/'selected_input_history.npz')
    starts=fixed['bin_start_ms'];groups=src['groups'];G=len(groups);t=src['time_ms'];rows=[];draw=[]
    for j in range(2*G):
        g=j%G;N=int(src['group_size'][g]);nr=int(fine['replicates'][j]);assert nr%N==0
        a=[d['counts'][j,:nr].astype(float).reshape(-1,N,27).sum(1) for d in [coarse,fine]]
        assert all(np.isfinite(x).all() and x.min()>=0 for x in a)
        if g in [2,3] and j>=G:assert np.array_equal(fine['counts'][j],fine['counts'][g]),'Uniform native threshold must give identical MC'
        expected=[];actual=[]
        for lo in starts:
            keep=(t>=lo)&(t<lo+50)
            expected.append(fixed['projected_private'][keep,g].sum()*.1*N/1000)
            actual.append(src['spikes'][keep,g].sum())
        expected=np.array(expected);actual=np.array(actual)
        assert np.max(abs(expected-fixed['counts_projected_private'][:,g]))<1e-10
        assert np.array_equal(actual,fixed['native_counts'][:,g])
        bounds=[np.quantile(x,[.025,.975],axis=0) for x in a]
        low=np.minimum(bounds[0][0],bounds[1][0]);high=np.maximum(bounds[0][1],bounds[1][1])
        for lo,hi in [(9000,9420),(9420,9868.5),(9868.5,10370)]:
            keep=(starts>=lo)&(starts+50<=hi);native=actual[keep];pred=expected[keep]
            means=[x[:,keep].mean(0) for x in a]
            baseline=max(np.linalg.norm(means[0]),1)
            rows.append(dict(group=int(groups[g]),threshold_case='group_mean' if j<G else 'native_mixture',
                window_ms=[lo,hi],included_complete_50ms_bins=int(keep.sum()),native_counts=int(native.sum()),
                rate_counts=float(pred.sum()),MC_dt01_counts=float(means[0].sum()),MC_dt005_counts=float(means[1].sum()),
                rate_vs_MC_dt01_L2=float(np.linalg.norm(pred-means[0])/baseline),
                native_vs_MC_dt01_L2=float(np.linalg.norm(native-means[0])/baseline),
                native_outside_both_step_predictive_bounds=int(((native<low[keep])|(native>high[keep])).sum())))
        if j<G:draw.append((actual,expected,a[0].mean(0),low,high,N))
    result=dict(status='LOCAL_READOUT_AUDIT_PASS',rows=rows,
        scope='Independent count/bin reproduction and constant-threshold identities. No scientific model acceptance. Primarynativeclock is0.1ms;0.05ms is numericalsensitivity.',
        intervals='Central95percent finiteN predictive ranges conditional on independentGaussianinput at eachstep; their union is a descriptive numerical envelope, not a calibrated95percent native interval. Strongcross-bin/condition dependence; no significance tests.',
        window_policy='Only predeclared50msbins wholly within each original statewindow are included; boundarycrossing bins are reported in wholewindow source but excluded from stage summaries.')
    (DEST/'local_lif/independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(12,7));fig.subplots_adjust(left=.08,right=.98,top=.92,bottom=.13,wspace=.25,hspace=.38)
    names=['Core A E','Core B E','Surround E','I','Core A E, low threshold','Core B E, low threshold']
    keep=starts+50<=9420;x=(starts[keep]+25)/1000
    for j,(ax,values,name) in enumerate(zip(axes.ravel(),draw,names)):
        native,pred,mc,low,high,N=values;scale=1000/(50*N)
        ax.fill_between(x,low[keep]*scale,high[keep]*scale,color='#d9c69c',alpha=.6,label='LIF numerical envelope')
        ax.plot(x,native[keep]*scale,'o-',color='#202020',ms=4,label='Native SNN')
        ax.plot(x,mc[keep]*scale,'--',color='#b07522',label='LIF reference')
        ax.plot(x,pred[keep]*scale,'s-',color='#8e5ca8',ms=3,label='Locked rate response')
        ax.text(.5,1.05,name,transform=ax.transAxes,ha='center');ax.text(-.17,1.05,'ABCDEF'[j],transform=ax.transAxes,fontweight='bold',fontsize=15)
        ax.set_xlim(9,9.42);ax.set_xticks([9,9.2,9.4]);ax.set_ylim(bottom=0)
        if j%3==0:ax.set_ylabel('Rate (Hz / neuron)')
        if j>=3:ax.set_xlabel('Time (s)')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,fontsize=10,bbox_to_anchor=(.5,.005))
    folder=OUT/'figures';stem='fig_native_input_local_response_preentry'
    for ext in ['png','pdf','svg']:fig.savefig(folder/f'{stem}.{ext}',dpi=190)
    plt.close(fig)
    (folder/f'{stem}.json').write_text(json.dumps(dict(source=str(DEST/'local_lif/independent_audit.json'),groups=groups.tolist(),
        counts='Fixed nonoverlapping50msbins entirely before9.42s. Same nativeclock; nophase shifts.',
        reference='Supplied native groupcount-derived means/privatevariances,observedZ/M,mean groupthreshold. ConditionalGaussianLIF reference only, not newSNN/network.',
        envelope=result['intervals'],human_visual_acceptance='PENDING',agent_visual_check='PENDING'),indent=2)+'\n')
    p=folder/'README.md';text=p.read_text();heading=f'### {stem}.png'
    if heading not in text:p.write_text(text+'\n'+heading+'\n六格保留预先选定的核A、核B、核外、I及两核低阈值群体，比较9–9.42秒内完整50ms窗的原生发放、同输入高斯LIF响应与固定率读出。LIF输入来自原生群体放电经过固定延迟算子和私有方差近似的重建；不是自主生成的新网络。阴影为0.1/0.05ms两步长下有限原生群体大小的条件模型预测范围的并集，不能当原生置信区间。**关注点**：进入前短事件的局部响应误差，避免长时间高活动把误差平均掉；本图不认证分岔。\n')
    print('LOCAL READOUT AUDIT PASS',len(rows),flush=True)


if __name__=='__main__':main()
