"""Additional fully validated Hopfs on higher folded spatial equilibria."""
from plot_rate_periodic_completion import *


def main():
    s=RateField();hh=[h for h in additional_hopfs() if 12<=int(h['label'].split('_')[0][1:])<=19]
    assert len(hh)==8,'All eight nonlinear validations must precede this figure'
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(15,11));grid=fig.add_gridspec(3,4,height_ratios=[1.25,1,1],
        left=.07,right=.96,top=.94,bottom=.12,wspace=.26,hspace=.46)
    z=np.load(RATE_OUT/'equilibrium_branch.npz');ids=np.arange(284,343)
    offsets=[{'H12':(-18,-29),'H13':(-15,22),'H14':(35,-9),'H15':(-42,-15),
        'H16':(-26,10),'H17':(-12,22),'H18':(28,10),'H19':(12,5)},
        {'H12':(-48,12),'H13':(18,0),'H14':(12,20),'H15':(-32,13),
        'H16':(-26,10),'H17':(-10,22),'H18':(22,-13),'H19':(12,5)}]
    for k in [0,1]:
        ax=fig.add_subplot(grid[0,k*2:k*2+2])
        ax.plot(z['J'][ids],z['regional'][ids,k],'--',color='#555555',lw=1.2)
        for h in hh:
            label=h['label'].split('_')[0];x=h['J_EE_core'];y=h['rates_hz'][k]
            ax.plot(x,y,'o',mfc='white',mec=FAMILY[label],ms=6)
            ax.annotate(label,(x,y),xytext=offsets[k][label],textcoords='offset points',color=FAMILY[label],
                arrowprops=dict(arrowstyle='-',color=FAMILY[label],lw=.6))
        ax.set(xlim=(1.001,1.044),xlabel=r'$J_{\mathrm{EE,core}}$',ylabel=f'Core {"AB"[k]} equilibrium rate (Hz / cell)',
            title=f'{"AB"[k]}  Higher folded equilibria: Core {"AB"[k]}');style(ax)
    cell=s.geo['group_cell'];sz=s.geo['group_size'];count=np.bincount(cell[s.E],weights=sz[s.E],minlength=400)
    geo=np.load(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz');xy=geo['contact_xy'];rows=[]
    for i,h in enumerate(hh):
        label=h['label'].split('_')[0];q=np.load(PERIODIC_OUT/f'stationary_root_counts/{label}_highorder_half.npz')['vector']
        field=np.sqrt(np.bincount(cell[s.E],weights=sz[s.E]*abs(q[s.E])**2,minlength=400)/np.maximum(count,1));field/=field.max()
        ax=fig.add_subplot(grid[1+i//4,i%4]);im=ax.imshow(field.reshape(20,20),origin='lower',extent=(0,20,0,20),cmap='inferno',vmin=0,vmax=1)
        for center in s.geo['centers_mm']:ax.add_patch(plt.Circle(center,1.5,fill=False,color='white',lw=.8))
        ax.scatter(xy[:,0],xy[:,1],s=9,facecolors='none',edgecolors='cyan',linewidths=.6)
        ax.set(title=f'{label}: {h["frequency_hz"]:.2f} Hz\n{h["criticality"]}',xlabel='x (mm)',xticks=[0,10,20],yticks=[0,10,20])
        if i%4==0:ax.set_ylabel('y (mm)')
        else:ax.tick_params(labelleft=False)
        mass=sz*abs(q)**2*s.E;mass/=mass.sum()
        rows.append(dict(label=label,J_EE_core=h['J_EE_core'],frequency_hz=h['frequency_hz'],criticality=h['criticality'],
            E_mode_energy_by_region=[float(mass[s.geo['group_region']==k].sum()) for k in range(3)],validation=h['validation']))
    fig.colorbar(im,cax=fig.add_axes([.35,.057,.32,.018]),orientation='horizontal',label='Normalized E-rate eigenfunction amplitude')
    save(fig,'higher_equilibrium_hopf_modes')
    write(PERIODIC_OUT/'higher_equilibrium_hopf_figure.json',dict(rows=rows,
        scope='H12-H19 on already unstable equilibria. Local supercritical/subcritical classification does not establish full-system cycle stability. These are linear mode amplitudes, not propagation snapshots.'))
    update_readme(dict(higher_equilibrium_hopf_modes='展示 H12–H19 在两个 core 平衡率投影中的位置及其完整空间特征模态，标注由正规形与非线性周期子分支共同验证的局部超／亚临界类型。上下图使用同一 935 群体模型，未将不同平衡态合并为单一 core 分支。**关注点**：母平衡态已经不稳定，超临界不意味着新生周期轨道在全系统中稳定；模态幅度不是传播事件快照。'))


if __name__=='__main__':main()
