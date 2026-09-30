"""Separate the small Hopf cycle from finite-amplitude burst-cycle birth."""
from plot_rate_periodic_completion import *


def main():
 plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'});fs=families();fig,axs=plt.subplots(1,2,figsize=(13,5.2),layout='constrained')
 ax=axs[0];z=np.load(RATE_OUT/'equilibrium_branch.npz');h=read(RATE_OUT/'hopfs.json')['rows'][0];cut=np.flatnonzero(z['J']>=h['J_EE_core'])[0]
 ax.plot(z['J'],z['regional'][:,0],'--',color='#999999',lw=1);ax.plot(np.r_[z['J'][:cut],h['J_EE_core']],np.r_[z['regional'][:cut,0],h['rates_hz'][0]],color='black',lw=1.4)
 rr=[q for q in fs['A'] if q['mean_rates_hz'][0]<1.05];ax.plot([q['J_EE_core'] for q in rr],[q['mean_rates_hz'][0] for q in rr],color=FAMILY['A'],lw=2,label='Small-cycle mean')
 rr=[read(PERIODIC_OUT/f'orbits/arcDoubleLow_{i:04d}_N512.json') for i in range(12)]
 rr=[read(PERIODIC_OUT/'orbits/arcDoubleDown_0079_N512.json')]+rr
 ax.plot([q['J_EE_core'] for q in rr],[q['mean_rates_hz'][0] for q in rr],color=FAMILY['double'],lw=2,label='Two-burst cycle mean')
 q=next(q for q in critical() if q['label']=='LPC_double_low');m=read(Path(q['orbit']).with_suffix('.json'))
 ax.plot(h['J_EE_core'],h['rates_hz'][0],'o',color=COL[0],ms=6);ax.annotate('H1',(h['J_EE_core'],h['rates_hz'][0]),xytext=(-15,-23),textcoords='offset points')
 ax.plot(q['J_EE_core'],m['mean_rates_hz'][0],'s',mfc='white',mec='black',ms=6);ax.annotate('Burst-cycle fold',(q['J_EE_core'],m['mean_rates_hz'][0]),xytext=(30,-30),textcoords='offset points',arrowprops=dict(arrowstyle='-',lw=.8))
 ax.set(xlim=(.93798,.93834),ylim=(.5,18),yscale='log',yticks=[.5,1,2,5,10],yticklabels=['0.5','1','2','5','10'],xticks=[.9380,.9381,.9382,.9383],xlabel=r'$J_{\mathrm{EE,core}}$',ylabel='Core A period mean (Hz / cell)',title='Small oscillation and large burst have distinct onsets');ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'));ax.legend(frameon=False,loc='upper right',fontsize=9);style(ax)
 ax=axs[1];th=np.linspace(0,2*np.pi,501);ax.plot(np.cos(th),np.sin(th),'k--',lw=.8)
 data=[('Small cycle, J=0.942','smallA_J0.942000000_N64_dt0.1',FAMILY['A']),
       ('Two-burst cycle, J=0.942','refined_J0.942000000_N1536_dt0.05',FAMILY['double']),
       ('B-leading, J=0.95','branch095_J0.950000000_N512_dt0.1',FAMILY['Bleading']),
       ('A-leading, J=1.3','branch_J1.300000000_N256_dt0.1',FAMILY['single'])]
 for i,(label,filename,c) in enumerate(data):
  q=read(PERIODIC_OUT/'floquet'/f'{filename}.json');v=np.array([complex(*x) for x in q['multipliers']]);neutral=np.argmin(abs(v-1))
  assert abs(v[neutral]-1)<.002
  v=np.delete(v,neutral);v=np.r_[v,np.conj(v[abs(v.imag)>1e-8])]
  ax.scatter(v.real,v.imag,s=45,marker=['o','s','^','v'][i],facecolors='none',edgecolors=c,label=label)
 ax.plot(1,0,'+',color='black',ms=9);ax.annotate('Neutral phase',(1,0),xytext=(-76,-22),textcoords='offset points',fontsize=9)
 ax.set(aspect='equal',xlim=(-1.1,1.1),ylim=(-1.1,1.1),xlabel=r'Re $\mu$',ylabel=r'Im $\mu$',title='Sampled cycles: nontrivial multipliers lie inside');ax.legend(frameon=False,loc='lower left',fontsize=8);style(ax)
 save(fig,'burst_onset_and_stable_coexistence')
 with (F/'README.md').open('a') as f:f.write('\n### burst_onset_and_stable_coexistence\n左图放大 H1 与双 burst 周期下端折点，区分小振荡起始和有限振幅周期折叠。右图展示四个实际周期解的非平凡 Floquet 乘子，自治相位的 +1 单独标出。**关注点**：这是指定轨道的稳定性证据，不代表整条彩色分支已逐段认证。\n')

if __name__=='__main__':main()
