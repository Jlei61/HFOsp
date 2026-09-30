"""Late-window recurrence diagnostics; does not identify chaos from IEI CV."""
from plot_rate_periodic_completion import *
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar


def main():
 path=RATE_OUT/'runs/periodic_gap/J0.9460000/trajectory.npz';z=np.load(path);reg=z['regional_rates_hz'];g=z['group_rate_hz'];n=len(reg)
 rows=[];peaks=[]
 for k in [0,1]:
  yy=gaussian_filter1d(reg[:,k],5);pk=find_peaks(yy,height=10,prominence=10,distance=60)[0];peaks.append(pk)
  for start in [10000,20000,25000]:
   p=pk[pk>=start];iei=np.diff(p);amp=yy[p]
   rows.append(dict(core='AB'[k],window_ms=[start,n],peaks_ms=p+1,peak_count=len(p),peak_amplitudes_hz=amp,IEI_ms=iei,IEI_CV=float(np.std(iei)/np.mean(iei)) if len(iei)>1 else None))
 # Test whole-field recurrence at lags suggested by the two core traces.
 x=reg[20000:,:2];x=x-x.mean(0);errors=[]
 for lag in range(100,2501):errors.append(np.linalg.norm(x[lag:]-x[:-lag])/np.linalg.norm(x[:-lag]))
 minima=find_peaks(-np.asarray(errors),distance=80)[0]+100;chosen=sorted(minima,key=lambda lag:errors[lag-100])[:8];checks=[]
 for lag in chosen:
  times=np.arange(20000,n-2502);a=reg[times,:2]
  def shifted(arr,delay):
   k=int(np.floor(delay));frac=delay-k;return arr[times+k]*(1-frac)+arr[times+k+1]*frac
  opt=minimize_scalar(lambda delay:np.linalg.norm(shifted(reg,delay)[:,:2]-a)/np.linalg.norm(a),bounds=(lag-2,lag+2),method='bounded')
  aa=g[times];bb=shifted(g,opt.x)
  checks.append(dict(lag_ms=int(lag),core_relative_recurrence_error=errors[lag-100],refined_lag_ms=float(opt.x),refined_core_relative_recurrence_error=float(opt.fun),full_group_relative_recurrence_error=float(np.linalg.norm(bb-aa)/np.linalg.norm(aa)),interpolation='Linear interpolation of original 1-ms integrated rates; common comparison window across candidate lags'))
 write(PERIODIC_OUT/'gap_J0p946_diagnostics.json',dict(source=str(path),rows=rows,recurrence=checks,interpretation='Finite late-window test only. Neither nonzero IEI CV nor failure of these recurrence lags proves an irregular attractor or chaos.'))
 fig,axs=plt.subplots(2,2,figsize=(14,8),layout='constrained')
 ax=axs[0,0]
 for k in [0,1]:ax.plot(np.arange(20000,n)/1000,reg[20000:,k],color=COL[k],lw=.75,label=f'Core {"AB"[k]}')
 ax.set(xlabel='Time from zero-state start (s)',ylabel='Rate (Hz / cell)',title='J=0.946: late autonomous rate trajectory');ax.legend(frameon=False);style(ax)
 ax=axs[0,1]
 for k in [0,1]:
  p=peaks[k][peaks[k]>=20000];ax.plot(p[1:]/1000,np.diff(p),'.-',color=COL[k],ms=4,lw=.8)
 ax.set(xlabel='Time (s)',ylabel='Consecutive burst interval (ms)',title='Intervals: 20–30 s window');style(ax)
 ax=axs[1,0]
 for k in [0,1]:
  p=peaks[k][peaks[k]>=20000];amp=gaussian_filter1d(reg[:,k],5)[p];ax.plot(amp[:-1],amp[1:],'o',color=COL[k],ms=4,alpha=.8)
 ax.set(xlabel='Burst peak n (Hz / cell)',ylabel='Burst peak n+1 (Hz / cell)',title='Finite-window return map');style(ax)
 ax=axs[1,1];ax.plot(np.arange(100,2501),errors,color='#333333');ax.set(xlabel='Candidate recurrence lag (ms)',ylabel='Relative core-trajectory mismatch',title='Tested recurrence lags; no attractor label inferred');style(ax)
 save(fig,'rate_gap_J0p946_late_dynamics')
 with (F/'README.md').open('a') as f:f.write('\n### rate_gap_J0p946_late_dynamics\n展示 J=0.946 从零初态自主仿真的末段波形、burst 间隔、峰值返回图及候选重复周期误差。使用 20–30 s 窗口，同时保存多个末段窗口统计和完整 935 群体的重复误差。**关注点**：有限时窗不规则外观、非零 CV 或此范围无重复，均不足以证明稳定 irregular attractor 或混沌。\n')
 print('late rows',[(q['core'],q['window_ms'],q['peak_count'],q['IEI_CV']) for q in rows],flush=True);print('recurrence',checks,flush=True)

if __name__=='__main__':main()
