"""Plot the paired local resource test, without representing it as a branch."""
from common import OUT, model, np, read, write
from refractory_spatial_resolution import mapping, projections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

DEST = OUT/'core_a_resource_bifurcation_20260923'


def main():
    s=model(40); coarse=model(20); parent,_=mapping(coarse,s)
    P,count=projections(s,coarse,parent)[20]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,3,figsize=(10,7),layout='constrained')
    metadata=[]
    for i,label in enumerate(['reference','coreA_depleted']):
        folder=DEST/label
        local=read(folder/'local_state_audit.json')
        audit=read(folder/'independent_audit.json')
        assert local['status']==audit['status']=='AUDIT_PASS'
        jobs=read(folder/'jobs.json');assert jobs['status']=='COMPLETE'
        source=folder/f'block{jobs["completed_blocks"][-1]:02d}.npz'
        z=np.load(source);F=z['field_E_hz'].astype(float)
        assert len(F)==5000
        maps=[P@z['Z'],F.mean(0),(F>50).mean(0)]
        images=[]
        for j,(data,cmap,limits) in enumerate(zip(maps,['viridis','magma','inferno'],[(0,1),(0,500),(0,1)])):
            ax=axes[i,j]
            im=ax.imshow(data.reshape(20,20),origin='lower',extent=[0,20,0,20],
                         interpolation='nearest',cmap=cmap,vmin=limits[0],vmax=limits[1])
            images.append(im)
            for k,center in enumerate(s.geo['centers_mm']):
                ax.add_patch(Circle(center,1.5,fill=False,color='#25d7dc',lw=1.2))
                ax.text(center[0],center[1]+1.8,'AB'[k],color='#25d7dc',ha='center',fontsize=10)
            ax.set_xticks([0,10,20]);ax.set_yticks([0,10,20]);ax.set_xlabel('x (mm)')
            if j==0:ax.set_ylabel(f'$D_A={local["coordinates"]["D_A"]:.3f}$\ny (mm)')
            else:ax.set_yticklabels([])
            ax.text(-.12,1.03,chr(65+3*i+j),transform=ax.transAxes,weight='bold',fontsize=15)
        metadata.append(dict(label=label,source=str(source),local_readout=local,
                             global_readout=audit['windows'][-1],
                             original_global_onset_ms=audit['original_high_onset_elapsed_ms']))
    for j,label in enumerate(['Resource Z','Mean E rate (Hz / neuron)','Fraction of time above 50 Hz']):
        cb=fig.colorbar(images[j],ax=axes[:,j],orientation='horizontal',fraction=.045,pad=.045)
        cb.set_label(label)
    out=DEST/'figures';out.mkdir(exist_ok=True)
    name='fig_core_a_resource_spatial_control'
    for ext in ['png','pdf','svg']:fig.savefig(out/f'{name}.{ext}',dpi=200)
    plt.close(fig)
    write(out/f'{name}.json',dict(rows=metadata,core_centers_mm=s.geo['centers_mm'].tolist(),
        definition='Same complete initial state; only Core A Z differs. All Z held during each trajectory, all M dynamic. Final5s of10s paired continuation. Spatial means/duty are not equilibrium, period means or bifurcation branches.',
        model_promoted=False,human_visual_acceptance='PENDING'))
    (out/'README.md').write_text(f'### {name}.png / .pdf / .svg\n\n'
        '两行只改变 Core A 的 Z，Core B 与外围 Z 固定在同一 native9s 水平，整个二维网络与所有 M 保持动态。三列依次为资源场、10秒续接中最后5秒的平均 E 放电率和局部率超过50Hz的时间比例；核 A/B 圈沿用原连接图几何。它是有限时间的核内资源因果对照，不能解释为平衡点或周期分支；模型与原 SNN 的完整对应仍未验收。\n\n'
        '**关注点**：Core A 是否先持续活动、Core B 和外围是否仍间歇参与；不将局部持续等同于全局 onset。PNG/PDF 候选待用户人工检查。\n')


if __name__=='__main__':main()
