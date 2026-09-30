"""Visual diagnostic of the candidate TR2 saddle-cycle connection."""
from plot_rate_periodic_completion import *


def main():
    q=read(PERIODIC_OUT/'TR2_same_parameter_saddle_approach.json')
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained',
                          gridspec_kw={'width_ratios':[1.25,1.1,1.]})
    colors=['#8c510a','#35978f','#762a83'];rows=q['rows']
    for c,r in zip(colors,rows):
        d=np.asarray(r['distances_Hz']);d=np.roll(d,len(d)//2-np.argmin(d))
        axes[0].semilogy(np.arange(len(d))/len(d)-.5,d,color=c,lw=1.5,
                        label=f'{r["slow_period_s"]:.1f} s')
    axes[0].set(xlabel=r'Slow phase relative to closest approach / $2\pi$',
                ylabel='Distance to same-J saddle cycle (Hz)',
                title='A  Longer residence near a saddle cycle')
    axes[0].legend(frameon=False,title='Modulation period',fontsize=9)
    x=np.array([-np.log(r['distance_range_Hz'][0]) for r in rows]);y=np.array([r['slow_period_s'] for r in rows])
    prediction=q['reference_log_scaling_coefficient_s']*(x-x[0])+y[0]
    axes[1].plot(x,prediction,'--',color='black',lw=1,label='Saddle-exponent prediction')
    for i,(xx,yy,c) in enumerate(zip(x,y,colors)):axes[1].plot(xx,yy,'o',color=c,ms=6)
    axes[1].set(xlabel=r'$-\ln(d_{\min}/1\,\mathrm{Hz})$',ylabel='Modulation period (s)',
                title='B  Period growth matches saddle slowing')
    axes[1].legend(frameon=False,fontsize=8)
    spec=read(Path(q['eigenvalue_source']));mu=np.array([complex(*v) for v in spec['multipliers']])
    near=mu[abs(mu-1)<.02];theta=np.linspace(-.008,.008,301)
    axes[2].plot(np.cos(theta),np.sin(theta),'--',color='black',lw=.8)
    axes[2].axhline(0,color='#aaaaaa',lw=.6)
    for v in near:
        c='#b2182b' if abs(v)>1 else '#2166ac'
        axes[2].plot(v.real,v.imag,'o',color=c,ms=6)
        axes[2].annotate(f'{v.real:.6f}',(v.real,v.imag),xytext=(0,12 if abs(v)>1 else -20),
                         textcoords='offset points',ha='center',color=c)
    axes[2].set(xlim=(.992,1.006),ylim=(-.002,.002),xlabel=r'Re $\mu$',ylabel=r'Im $\mu$',
                title='C  Saddle multipliers near +1')
    axes[2].ticklabel_format(axis='x',useOffset=False)
    for ax in axes:style(ax)
    save(fig,'torus_saddle_cycle_approach')
    update_readme(dict(torus_saddle_cycle_approach='比较三条已做加密网格检查的 TR2 环面与完全相同参数下的鞍周期轨道，显示接近距离、慢调制周期增长及移除相位后的 Poincaré 乘子。周期增长与鞍点指数给出的对数标度一致。**关注点**：这是鞍周期连接候选的数值证据，尚未完成完整状态空间的稳定／不稳定流形连接验证；不等同于稳定 irregular burst 或已确定的全局分岔类型。'))


if __name__=='__main__':main()
