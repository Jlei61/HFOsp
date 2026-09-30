"""Actual finite-window spatial evidence for the onset-side continuation.

No equilibrium, periodic branch, stability or bifurcation symbols are drawn.
"""
from common import model, np, read, write
from onset_state_continuation import DEST
from refractory_spatial_resolution import mapping, projections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import argparse


def plot(labels, name):
    s=model(40);coarse=model(20);parent,_=mapping(coarse,s)
    P,count=projections(s,coarse,parent)[20]
    fields=np.load(DEST/'fields.npz');conditions=read(DEST/'conditions.json')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(len(labels),3,figsize=(9.1,3.0*len(labels)+.4),squeeze=False,
                          layout='constrained')
    images=[];metadata=[]
    for i,label in enumerate(labels):
        folder=DEST/label;audit=read(folder/'independent_audit.json')
        assert audit['status']=='AUDIT_PASS'
        jobs=read(folder/'jobs.json');assert jobs['status']=='COMPLETE'
        last=jobs['completed_blocks'][-1];z=np.load(folder/f'block{last:02d}.npz')
        F=z['field_E_hz'].astype(float);Z=fields[conditions[label]['field']]
        maps=[P@Z,F.mean(0),(F>50).mean(0)]
        images=[]
        for j,(data,cmap,lim) in enumerate(zip(maps,['viridis','magma','inferno'],[(0,1),(0,500),(0,1)])):
            ax=axes[i,j]
            im=ax.imshow(data.reshape(20,20),origin='lower',extent=[0,20,0,20],
                         interpolation='nearest',cmap=cmap,vmin=lim[0],vmax=lim[1])
            images.append(im)
            for k,center in enumerate(s.geo['centers_mm']):
                ax.add_patch(Circle(center,1.5,fill=False,color='#25d7dc',lw=1.1))
                ax.text(center[0],center[1]+1.8,'AB'[k],color='#25d7dc',ha='center',va='bottom',fontsize=9)
            ax.set_xticks([0,10,20]);ax.set_yticks([0,10,20])
            ax.set_xlabel('x (mm)')
            if j==0:
                prefix=('Self-limited start\n' if '_from_lower' in label else
                        'Sustained start\n' if '_from_upper' in label else '')
                ax.set_ylabel(prefix+f'$D={audit["D"]:.3f}$\ny (mm)')
            ax.text(-.13,1.02,chr(65+3*i+j),transform=ax.transAxes,weight='bold',fontsize=13)
        metadata.append(dict(label=label,D=audit['D'],source=str(folder/f'block{last:02d}.npz'),
            window_ms=audit['windows'][-1]['window_ms'],readout=audit['windows'][-1],
            original_high_onset_elapsed_ms=audit['original_high_onset_elapsed_ms']))
    for j,(im,label) in enumerate(zip(images,['Resource Z','Mean E rate (Hz / neuron)','Fraction of time above 50 Hz'])):
        cb=fig.colorbar(im,ax=axes[:,j],orientation='horizontal',fraction=.045,pad=.045)
        cb.set_label(label)
    out=DEST/'figures';out.mkdir(exist_ok=True)
    for ext in ['png','pdf','svg']:fig.savefig(out/f'{name}.{ext}',dpi=220)
    plt.close(fig)
    write(out/f'{name}.json',dict(rows=metadata,
        meaning='Final5s finite-window spatial means and duty; mean is not an equilibrium or period mean. Fixed fullZ, dynamicM, constantmean external input, no future count innovations.',
        core_centers_mm=s.geo['centers_mm'].tolist(),model_promoted=False))
    entry=f'### {name}.png / .pdf / .svg\n\n'
    entry+='每行对应一个固定完整 Z 空间场的确定性条件漂移，M 动态；三列依次为 Z 空间场、最后 5 秒平均 E 放电率，以及局部放电率超过 50 Hz 的时间比例。A/B 圈来自同一连接图的核中心，所有行共用色标；平均率不是平衡点或周期均值。\n\n'
    entry+='**关注点**：比较间歇参与和持续招募的空间范围；这张图不标注尚未认证的分岔点。候选图已生成，人工图形验收仍待用户检查。\n\n'
    path=out/'README.md'
    if not path.exists() or f'### {name}.' not in path.read_text():
        with path.open('a') as stream:stream.write(entry)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--labels',nargs='+',default=['lower_endpoint','upper_endpoint'])
    p.add_argument('--name',default='fig_onset_endpoint_spatial_states');a=p.parse_args();plot(a.labels,a.name)
