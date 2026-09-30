"""Large standalone spatial/readout panels for each confirmed burst example."""
from plot_rate_periodic_composite import *


def main():
 plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'});s=RateField();geo=np.load(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz');xy=geo['contact_xy'];order=contact_indices(geo['contact_names'].tolist())
 cell=s.geo['group_cell'];sz=s.geo['group_size'];ct=np.bincount(cell[s.E],weights=sz[s.E],minlength=400)
 for letter,name,J,title in CASES[2:]:
  r,T=loadcase(s,name,J);N=len(r);reg=np.array([s.regional_rates(v) for v in r]);shift=int(np.argmin(reg[:,:2].sum(1)));rr=np.roll(r,-shift,axis=0);rg=np.roll(reg,-shift,axis=0);t=np.arange(N)*T/N
  if letter=='c':
   ids=[]
   for k in [0,1]:
    pk=find_peaks(np.tile(rg[:,k],3),height=20,distance=N//4)[0];ids.extend((pk[(pk>=N)&(pk<2*N)]-N).tolist())
   ids=sorted(ids)
  else:
   pk=int(np.argmax(rg[:,1 if letter=='d' else 0]));ids=[int((pk+off/T*N)%N) for off in [-20,0,40,80]]
  fig=plt.figure(figsize=(13,8));grid=fig.add_gridspec(2,3,width_ratios=[1.2,1,1],left=.07,right=.965,bottom=.15,top=.86,wspace=.35,hspace=.5)
  ax=fig.add_subplot(grid[0,0]);
  for k in [0,1,2]:ax.plot(t,rg[:,k],color=COL[k] if k<2 else '#555555',lw=1.4,label=f'Core {"AB"[k]}' if k<2 else 'Surround')
  for ix in ids:ax.axvline(t[ix],ls=':',color='#888888',lw=.6)
  ax.set(xlim=(0,T),xlabel='Time within full period (ms)',ylabel='Rate (Hz / cell)');ax.legend(frameon=False,fontsize=9);style(ax)
  contact=rr@s.geo['contact_rate_weights']*1000;ax=fig.add_subplot(grid[1,0]);im=ax.imshow(contact[:,order].T,origin='upper',aspect='auto',extent=(0,T,14.5,-.5),cmap='magma',norm=PowerNorm(.5,0,200))
  ax.set(yticks=np.arange(15),yticklabels=CONTACT_ORDER,xlabel='Time within full period (ms)');ax.tick_params(labelsize=9);ax.axhline(3.5,color='white',lw=.6)
  for tick,n in zip(ax.get_yticklabels(),CONTACT_ORDER):tick.set_color(SHAFT_COLORS[n[:3]])
  for j,ix in enumerate(ids):
   fld=np.bincount(cell[s.E],weights=sz[s.E]*rr[ix,s.E]*1000,minlength=400)/np.maximum(ct,1);ax=fig.add_subplot(grid[j//2,1+j%2]);imf=ax.imshow(fld.reshape(20,20),origin='lower',extent=(0,20,0,20),cmap='inferno',norm=PowerNorm(.55,0,500))
   for k,center in enumerate(s.geo['centers_mm']):
    ax.add_patch(plt.Circle(center,1.5,fill=False,color='white',lw=1.2));ax.text(center[0],center[1],"AB"[k],color='white',ha='center',va='center',fontsize=10)
   ax.scatter(xy[:,0],xy[:,1],s=20,facecolors='none',edgecolors='cyan',linewidths=.9)
   ax.set(xlabel='x (mm)',ylabel='y (mm)',xticks=[0,10,20],yticks=[0,10,20],title=f'{t[ix]:.0f} ms')
  fig.suptitle(f'{letter}  {title} | '+rf'$J_{{\mathrm{{EE,core}}}}={J:g}$'+f' | full period {T:.2f} ms',fontsize=15)
  fig.colorbar(im,cax=fig.add_axes([.085,.045,.25,.018]),orientation='horizontal',label='Contact-weighted rate (Hz / cell)')
  fig.colorbar(imf,cax=fig.add_axes([.57,.045,.25,.018]),orientation='horizontal',label='E rate (Hz / cell)')
  save(fig,f'rate_cycle_{letter}_space')
  with (F/'README.md').open('a') as f:f.write(f'\n### rate_cycle_{letter}_space\n主图 {letter} 状态的独立放大版，包括一个完整周期的核内/周围活动、触点放电率和四帧二维空间场。所有面板来自同一个精化周期轨道；白圈为核，青色点为触点，三种状态共享物理量色标。**关注点**：时刻为周期相位；触点亮起不自动等于满足原事件检测阈值。\n')

if __name__=='__main__':main()
