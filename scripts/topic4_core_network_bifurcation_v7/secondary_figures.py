from common import *
from figures import save,COL,J,FIG
from scipy.signal import resample,find_peaks
import numpy as np
import matplotlib.pyplot as plt

def samples(path,m):
 z=np.load(path);r=resample(z['r'],m*8192,axis=0);N=len(r);pk=np.argmax(r[:N//m,0]);r=np.roll(r,-pk,axis=0)
 return float(z['g']),r.reshape(m,N//m,6).mean(1)*1000

def main():
 p0=V6/'flips/surround_micro_flip_N4096.npz';p2=OUT/'flips/surround_2T_flip_N8192.npz';z0=np.load(p0);z2=np.load(p2);g0=float(z0['g']);g2=float(z2['g'])
 fig,axes=plt.subplots(2,2,figsize=(12,10));fig.subplots_adjust(left=.1,right=.97,bottom=.09,top=.94,wspace=.26,hspace=.3)
 ax=axes[0,0]
 pp=[p0,V6/'periodic/surround_period2/amp0.0001_N4096.npz',V6/'periodic/surround_period2/amp0.0002_N4096.npz',p2,V6/'periodic/surround_period2/amp0.0004_N4096.npz']
 xy=[]
 for k,p in enumerate(pp):
  g,y=samples(p,1 if k==0 else 2)
  if len(y)==1:y=np.repeat(y,2,axis=0)
  xy.append((g,y[:,2]))
 for k in range(2):
  xx=[(g-g0)*1e12 for g,y in xy];yy=[y[k] for g,y in xy]
  ax.plot(xx[:4],yy[:4],'o-',color=COL[0],lw=1.6,ms=5);ax.plot(xx[3:],yy[3:],'o--',mfc='white',color=COL[0],lw=1.6,ms=5)
 ax.axvline((g2-g0)*1e12,color='#333333',ls='--',lw=1)
 ax.set(xlabel=r'$\Delta J_{\mathrm{EE,core}}\times10^{12}$ (from PD0)',ylabel='Surround E mean per burst cycle (Hz)',title='T → 2T → loss of 2T stability')
 ax=axes[0,1];xy=[]
 for p in [p2,OUT/'periodic/surround_period4_refined/amp2e-05_N8192.npz',OUT/'periodic/surround_period4_refined/amp4e-05_N8192.npz']:
  g,y=samples(p,2 if p==p2 else 4)
  if len(y)==2:y=np.tile(y,(2,1))
  xy.append((g,y[:,2]))
 for k in range(4):ax.plot([(g-g2)*1e13 for g,y in xy],[y[k] for g,y in xy],'o-',color=COL[k%2],lw=1.6,ms=6)
 ax.set(xlabel=r'$\Delta J_{\mathrm{EE,core}}\times10^{13}$ (from PD0(2T))',ylabel='Surround E mean per burst cycle (Hz)',title='Four distinct cycles on the 4T branch');ax.ticklabel_format(axis='y',style='plain',useOffset=False)
 ax=axes[1,0];rows=[]
 for p in (V6/'poincare/surround_period2').glob('*/rk4_orthogonal_dt0.025.json'):
  q=read(p);rows.append(q)
 for p in (OUT/'poincare/flips/surround_2T_flip_N4096').glob('*.json'):rows.append(read(p))
 rows=[r for r in rows if abs(r['g']-g2)<2e-11];rows.sort(key=lambda r:r['g'])
 ax.plot([(r['g']-g2)*1e12 for r in rows],[r['multipliers'][0][0] for r in rows],'o-',color=COL[0],label='2T mother branch')
 child_label=True
 for folder in ['surround_period4','surround_period4_refined']:
  for p in (OUT/'poincare'/folder).glob('*/rk4_orthogonal_dt0.025.json'):
   q=read(p);ax.plot((q['g']-g2)*1e12,q['multipliers'][0][0],'s',color=COL[2],ms=7,label='4T child branch' if child_label else None);child_label=False
 ax.axhline(-1,color='#222222',ls='--');ax.axhline(1,color='#222222',ls=':');ax.set(xlim=(-4.2,12),ylim=(-7,2),xlabel=r'$\Delta J_{\mathrm{EE,core}}\times10^{12}$ (from PD0(2T))',ylabel='Leading real transverse multiplier',title='Independent return-map stability');ax.legend(frameon=False)
 ax=axes[1,1];v=z2['mode'];w=z2['left_mode'];fr=(v*v).sum(0);fr/=fr.sum();fl=(w*w).sum(0);fl/=fl.sum()
 ax.bar(np.arange(6)-.18,100*fr,.36,color=COL[0],label='Right mode');ax.bar(np.arange(6)+.18,100*fl,.36,color=COL[3],label='Adjoint mode');ax.set(xticks=np.arange(6),xticklabels=['A E','B E','S E','A I','B I','S I'],ylabel='Rate-mode squared norm (%)',title='Critical 2T → 4T mode');ax.legend(frameon=False)
 save(fig,'secondary_period_doubling','新增2T→4T分岔。上排每点为按A峰对齐后、单个core burst周期内的周边E均值；不同分支使用真实2T/4T轨道，不用复制波形冒充周期加倍。下排为独立返回乘子与临界左右率模，稳定性应以乘子判断，颜色仅区分分支/模态。')
 fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True);fig.subplots_adjust(left=.12,right=.97,bottom=.09,top=.94,hspace=.2)
 z=np.load(OUT/'periodic/surround_period4_refined/amp4e-05_N8192.npz');r=z['r']*1000;delta=r-np.roll(r,len(r)//2,axis=0);t=np.arange(len(r))*float(z['T'])/len(r)/1000
 for k,ax in enumerate(axes):ax.plot(t,delta[:,k],color=COL[k],lw=1.7);ax.axhline(0,color='#555555',lw=.6);ax.set(title=['Core A E','Core B E','Surround E'][k],ylabel=r'$r(t)-r(t+2T)$ (Hz)')
 axes[-1].set_xlabel('Time within the 4T orbit (s)');save(fig,'secondary_four_cycle_difference','真实4T轨道与其平移2T后的差值。普通率轴下两者几乎重合，此图显示可数值分辨的破缺；这些微小确定性差别没有被称为原生SNN的不规则burst。')
if __name__=='__main__':main()
