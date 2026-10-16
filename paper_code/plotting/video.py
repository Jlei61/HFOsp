"""Fresh events in the accepted paired-row Video2 layout and fixed scales."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from matplotlib.patches import Circle
from plotting.paper import contact_display as display


def selected_propagation_movie(runs,selection,path,radius):
    if selection['status']!='READY':
        print('No complete pair of template classes in both new simulations.')
        return
    propagation_movie(runs,path,radius,event_indices=selection['events'])
    selection['status']='GENERATED_FROM_NEW_SPIKES'
    from IPython.display import Image,display as show
    show(Image(filename=str(path)))


def propagation_movie(runs,path,radius,*,event_indices):
    fig=plt.figure(figsize=(13,8))
    gs=fig.add_gridspec(2,6,width_ratios=[.6,.6,.035,1,1,.035],left=.08,right=.94,bottom=.08,top=.89,hspace=.65,wspace=.35)
    images=[];cursors=[];titles=[];contexts=[]
    for row,run in enumerate(runs):
        order=display.contact_indices(run['contact_names'])
        for j,ei in enumerate(event_indices[row]):
            event=run['observed']['events'][int(ei)];lo,hi=event['window_ms']
            start,stop=round(lo/2),round(hi/2)
            if stop-start!=125 or start<0 or stop>len(run['movie']):raise ValueError('A complete native 250-ms event window is required')
            times=np.asarray(run['observed']['centroid_ms'][int(ei)],float);zero=float(np.nanmin(times))
            env=run['envelope'][:,start:stop].astype(float).copy();env/=np.maximum(env.max(1,keepdims=True),1e-20);env[~np.isfinite(times)]=np.nan
            contexts.append(dict(row=row,j=j,run=run,order=order,env=env,times=times,zero=zero,start=start,lo=lo,hi=hi))
    limits=(min(c['lo']-c['zero'] for c in contexts),max(c['hi']-c['zero'] for c in contexts))
    for row,run in enumerate(runs):
        y=.96 if row==0 else .46
        fig.text(.04,y,['Higher error / Poorer fit','Lower error / Better fit'][row],fontsize=15,weight='bold',color=['#7D5C30','#28665A'][row])
        for c in [c for c in contexts if c['row']==row]:
            j=c['j'];color=['#C63D3A','#287FA1'][j];mode=['MTA','MTB'][j]
            ax=fig.add_subplot(gs[row,j]);cmap=plt.get_cmap('magma').copy();cmap.set_bad('#777777')
            env_image=ax.imshow(c['env'][c['order']],extent=[c['lo']-c['zero'],c['hi']-c['zero'],14.5,-.5],aspect='auto',cmap=cmap,vmin=0,vmax=1,interpolation='nearest')
            display.centroid_lines(ax,c['times'][c['order']]-c['zero'],color='#00CFFF');display.contact_axis(ax)
            ax.set(xlim=limits,xlabel='Time (ms)',title=mode);ax.title.set(color=color,weight='bold')
            if j:ax.tick_params(labelleft=False)
            cursor=ax.axvline(c['lo']-c['zero'],color='white',lw=1);cursors.append((cursor,c['lo']-c['zero']))
            ax=fig.add_subplot(gs[row,j+3]);im=ax.imshow(run['movie'][c['start']],origin='lower',extent=[0,20,0,20],cmap='inferno',vmin=0,vmax=75,interpolation='nearest')
            images.append((im,run,c['start']));ax.scatter(*run['contact_xy'].T,facecolors='none',edgecolors='cyan',s=13,lw=.65)
            for center in run['centers']:ax.add_patch(Circle(center,radius,fill=False,color='white',lw=1.1))
            ax.set(xlabel='x (mm)',ylabel='y (mm)' if j==0 else '',xticks=[0,10,20],yticks=[0,10,20])
            if j:ax.tick_params(labelleft=False)
            title=ax.set_title('',color=color,weight='bold',fontsize=11);titles.append((title,mode,c['lo']-c['zero']))
        cb=fig.colorbar(env_image,cax=fig.add_subplot(gs[row,2]),ticks=[0,1]);cb.ax.set_title(r'$\widetilde{E}$')
        cb=fig.colorbar(im,cax=fig.add_subplot(gs[row,5]),ticks=[0,75]);cb.set_label('E cells / mm² / 2 ms')
    def update(frame):
        for im,run,start in images:im.set_data(run['movie'][start+frame])
        for cursor,offset in cursors:cursor.set_xdata([offset+2*frame]*2)
        for title,mode,offset in titles:title.set_text(f'{mode}  t = {offset+2*frame:+.1f} ms')
        return [x[0] for x in images+cursors+titles]
    animation=FuncAnimation(fig,update,frames=125,interval=50,blit=False)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);animation.save(path,writer=PillowWriter(fps=20),dpi=100)
    update(45);fig.savefig(path.with_suffix('.png'),dpi=120,bbox_inches='tight');plt.close(fig)
    readme=path.parent/'README.md'
    with readme.open('a') as stream:stream.write('\n### '+path.name+'\n两个新模拟工作点各取两个合格事件，显示固定15触点顺序的包络和活动E细胞空间分布。每个事件按自身最早质心标时，播放真实250ms窗口。\n**关注点**：工作点按本次J排序；不同事件不代表同时发生。\n')
