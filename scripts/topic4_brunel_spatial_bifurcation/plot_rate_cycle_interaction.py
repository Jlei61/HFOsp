"""Display the core phase/amplitude structure behind secondary cycle folds."""
from plot_rate_periodic_completion import *


def main():
 s=RateField();fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained');records=[]
 for row,(fam,span,region,limits) in enumerate([('arcA',(65,113),1,(.9381,.9410)),('arcB',(101,157),0,(.94625,.9478))]):
  files=[PERIODIC_OUT/f'orbits/{fam}_{i:04d}_N64.npz' for i in range(*span)];points=[]
  for f in files:
   z=np.load(f);reg=np.array([s.regional_rates(v) for v in z['r']]);cf=np.fft.rfft(reg,axis=0);harm=np.argmax(abs(cf[1:,:2]),axis=0)+1
   locked_phase=float(np.angle(np.exp(1j*(np.angle(cf[harm[0],0])-harm[0]/harm[1]*np.angle(cf[harm[1],1])))))
   q=dict(orbit=str(f),J_EE_core=float(z['J']),T_ms=float(z['T']),regional_mean_hz=reg.mean(0),regional_std_hz=reg.std(0),dominant_harmonic_A_B=harm,locked_phase_radians=locked_phase)
   records.append(q);points.append(q)
  ax=axes[row,0];ax.plot([q['J_EE_core'] for q in points],[q['regional_mean_hz'][region] for q in points],color=FAMILY['AB'[row]])
  for q in critical():
   allowed={f'LPC_A{i}' for i in range(2,6)} if row==0 else {f'LPC_B{i}' for i in range(2,9)}
   if q['label'] not in allowed:continue
   m=read(Path(q['orbit']).with_suffix('.json'));ax.plot(q['J_EE_core'],m['mean_rates_hz'][region],'s',mfc='white',mec='black',ms=4)
  ax.set(xlim=limits,xlabel=r'$J_{\mathrm{EE,core}}$',ylabel=f'Core {"AB"[region]} period mean (Hz)',title='H1-born branch: both cores join the oscillation' if row==0 else 'H2-born branch: mixed core oscillations');style(ax)
  ax.xaxis.set_major_formatter(FormatStrFormatter('%.5f'))
  chosen=[74,87,106] if row==0 else [102,116,150];ax=axes[row,1]
  for j,ii in enumerate(chosen):
   z=np.load(PERIODIC_OUT/f'orbits/{fam}_{ii:04d}_N64.npz');reg=np.array([s.regional_rates(v) for v in z['r']]);T=float(z['T']);tt=np.arange(len(reg)+1)/len(reg)
   for k in [0,1]:ax.plot(tt,np.r_[reg[:,k],reg[0,k]]+j*3.5,color=COL[k],lw=1)
   ax.text(1.025,j*3.5+.8,f'J={float(z["J"]):.6f}\nT={T:.1f} ms',fontsize=9,va='center')
  ax.set(xlim=(0,1.28),xticks=[0,.5,1],xlabel='Fraction of the full-network period',ylabel='Rate (Hz), successive examples offset by 3.5 Hz',title='A and B: 1:1 dominant harmonics' if row==0 else 'A and B: 2:1 dominant harmonics');style(ax)
 axes[0,1].legend(handles=[Line2D([0],[0],color=COL[k],label=f'Core {"AB"[k]}') for k in [0,1]],frameon=False)
 save(fig,'secondary_cycle_folds_and_core_interaction')
 write(PERIODIC_OUT/'secondary_cycle_harmonics.json',dict(rows=records,interpretation='Dominant 1:1 or 2:1 harmonics and changing locked phase/amplitude on the computed periodic branches. This does not prove torus stability, period doubling, or native SNN correspondence.'))
 with (F/'README.md').open('a') as f:f.write('\n### secondary_cycle_folds_and_core_interaction\n展开两条 Hopf 周期分支继续回折后的密集转弯，右侧展示对应 A/B 波形。H1 后段为两核主谐波 1:1，H2 后段为 A/B 主谐波 2:1；方块只标已通过导数条件精化的周期折点。**关注点**：这些是模型周期解的核间振幅与锁相关系，不能仅据 2:1 谐波认定倍周期起源；整段稳定性尚未穷尽。\n')

if __name__=='__main__':main()
