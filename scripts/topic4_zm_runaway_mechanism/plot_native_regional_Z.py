"""Common-clock 50 ms spatial activity in native regional Z interventions."""
from native_regional_Z_audit import DEST, N, OUT, np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def main():
    q = N.read(DEST / 'result.json'); assert q['status'] == 'COMPLETE'
    z = np.load(DEST / 'fields.npz')
    arms = [('all_dynamic', 'All Z dynamic'), ('all_held', 'All Z held'),
            ('cores_dynamic', 'Core Z dynamic'), ('surround_dynamic', 'Surround Z dynamic')]
    times = [10390, 12000]
    plt.rcParams.update({'font.size': 12, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 4, figsize=(11.8, 6.3), sharex=True, sharey=True)
    fig.subplots_adjust(left=.085, right=.87, top=.91, bottom=.11, wspace=.16, hspace=.17)
    records = []
    for i, t in enumerate(times):
        for j, (arm, label) in enumerate(arms):
            ax = axes[i, j]; field = z[arm + '_field_Hz']
            sample = field[t - 9000 - 25:t - 9000 + 25]; assert len(sample) == 50
            rate = sample.mean(0)
            im = ax.imshow(rate.reshape(20, 20), origin='lower', extent=[0,20,0,20], cmap='magma', vmin=0, vmax=500)
            ax.set_xticks([0,10,20]); ax.set_yticks([0,10,20])
            if i == 0: ax.text(.5,1.07,label,ha='center',transform=ax.transAxes)
            if i == 1: ax.set_xlabel('x (mm)')
            if j == 0:
                ax.set_ylabel('y (mm)')
                ax.text(-.33,.5,f'{t/1000:.3f} s',rotation=90,va='center',ha='center',transform=ax.transAxes)
                ax.text(-.31,1.05,'AB'[i],fontsize=17,fontweight='bold',transform=ax.transAxes)
            for center, name in zip(z['centers_mm'],'AB'):
                ax.add_patch(Circle(center,1.75,fill=False,ec='#20c4cf',lw=1.1))
                ax.text(center[0],center[1]+2.2,name,color='#20c4cf',ha='center',fontsize=10)
            records.append(dict(arm=arm,window_ms=[t-25,t+25],global_E_Hz=float(np.average(rate,weights=z['cell_counts']))))
    cax = fig.add_axes([.905,.28,.016,.40]); fig.colorbar(im,cax=cax,label='E rate (Hz)',ticks=[0,250,500])
    name = 'fig_native_regional_Z_feedback'; folder = OUT/'figures'
    for ext in ['png','pdf','svg']: fig.savefig(folder/f'{name}.{ext}',dpi=190)
    plt.close(fig)
    N.write(folder/f'{name}.json',dict(source=str(DEST/'result.json'),panels=records,M='Dynamic in all arms',
            circles='Original native core membership radius1.75mm, not the smaller1.5mm legacy readout circles.',
            common_clock_selection='Predeclared original broad-entry10.39s and12s, centered50ms windows.',
            human_visual_acceptance='PENDING',scope=q['statistical_unit']))
    readme = folder/'README.md'; text = readme.read_text(); heading = f'### {name}.png / .pdf / .svg'
    if heading not in text:
        readme.write_text(text+'\n'+heading+'\n原生SNN从同一个9秒完整状态和相同未来输入出发，比较全部Z动态、全部固定、仅两核动态及仅核外动态；四组M都动态。两行取共同10.390与12.000秒的50ms空间窗口，共享0–500Hz色标，圆圈标实际1.75mm核成员边界。**关注点**：核内资源反馈与核外资源反馈对局部持续和全局募集是否可区分；这是同一历史的有限窗干预，不能据此指定分岔类型。\n')
    print(folder/f'{name}.png',flush=True)


if __name__ == '__main__': main()
