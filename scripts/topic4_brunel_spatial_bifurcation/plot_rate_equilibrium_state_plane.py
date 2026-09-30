"""Distinguish equilibria that nearly overlap in a one-core projection."""
from plot_rate_periodic_completion import *
from matplotlib.collections import LineCollection


def main():
    s=RateField();z=np.load(RATE_OUT/'equilibrium_branch.npz');r=z['regional'][:330,:2];J=z['J'][:330]
    points=r[:,None,:];segments=np.concatenate([points[:-1],points[1:]],axis=1)
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(8,6.5));fig.subplots_adjust(left=.12,right=.85,bottom=.12,top=.91)
    line=LineCollection(segments,cmap='viridis',norm=plt.Normalize(.934,1.035),lw=1.6);line.set_array((J[:-1]+J[1:])/2);ax.add_collection(line)
    offsets={'H3':(-38,-6),'H4':(-38,3),'H5':(-24,10),'H6':(-5,14),'H7':(10,9),'H8':(10,2),'H9':(10,1),'H10':(12,-2),'H11':(5,-21)}
    for h in additional_hopfs():
        label=h['label'].split('_')[0];x,y=h['rates_hz'][:2]
        if label not in offsets:continue
        ax.plot(x,y,'o',mfc='white',mec=FAMILY[label],ms=6)
        ax.annotate(label,(x,y),xytext=offsets[label],textcoords='offset points',color=FAMILY[label],
            arrowprops=dict(arrowstyle='-',lw=.6,color=FAMILY[label]))
    for a,b in [('H3','H9'),('H4','H8'),('H5','H7')]:
        hs={h['label'].split('_')[0]:h for h in additional_hopfs()}
        if a not in hs or b not in hs:continue
        ax.plot([hs[a]['rates_hz'][0],hs[b]['rates_hz'][0]],
            [hs[a]['rates_hz'][1],hs[b]['rates_hz'][1]],':',color='#777777',lw=.8,zorder=0)
    ax.set(xlim=(.6,3.7),ylim=(.5,8.3),xlabel='Core A equilibrium rate (Hz / cell)',ylabel='Core B equilibrium rate (Hz / cell)',
        title='Different equilibria behind overlapping branch projections')
    style(ax);fig.colorbar(line,cax=fig.add_axes([.89,.17,.025,.65]),label=r'$J_{\mathrm{EE,core}}$')
    save(fig,'equilibrium_two_core_state_plane')
    update_readme(dict(equilibrium_two_core_state_plane='用两个 core 的平衡放电率共同定位同一条延拓分支，颜色为 J。虚点连线连接空间特征模态相近的 Hopf 对，显示在 core B 单轴投影中几乎重叠的点实际上具有不同的 core A 状态。**关注点**：图中新增 Hopf 都位于已经不稳定的平衡分支；此图不表示这些平衡态是可稳定观测到的网络状态。'))


if __name__=='__main__':main()
