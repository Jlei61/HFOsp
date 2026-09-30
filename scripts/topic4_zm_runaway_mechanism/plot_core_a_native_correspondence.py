"""Same local resource intervention in native SNN and the current rate model."""
from common import OUT,model,np,read,write
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

NATIVE=OUT/'core_a_native_clamp_20260924'
RATE=OUT/'core_a_resource_bifurcation_20260923'


def main():
    nr=read(NATIVE/'result.json');assert nr['status']=='AUDIT_PASS'
    f=np.load(NATIVE/'fields.npz');conditions=read(RATE/'contract.json')['coordinates']
    centers=f['centers_mm'];s=model(40)
    assert np.array_equal(centers,s.geo['centers_mm'])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(8,7.6),layout='constrained')
    meta=[]
    for j,(label,fieldname) in enumerate([('reference','reference9000'),('coreA_depleted','coreA10370_background9000')]):
        ra=read(RATE/label/'local_state_audit.json');assert ra['status']=='AUDIT_PASS'
        z=np.load(RATE/label/'block01.npz')
        arrays=[f[label+'_field_Hz'][-1000:],z['field_E_hz'][-1000:]]
        for i,x in enumerate(arrays):
            assert len(x)==1000 and np.isfinite(x).all()
            ax=axes[i,j]
            im=ax.imshow(x.mean(0).reshape(20,20),origin='lower',extent=[0,20,0,20],
                         vmin=0,vmax=500,cmap='magma',interpolation='nearest')
            for k,center in enumerate(centers):
                ax.add_patch(Circle(center,1.5,fill=False,color='#25d7dc',lw=1.2))
                ax.text(center[0],center[1]+1.8,'AB'[k],ha='center',color='#25d7dc')
            ax.set_xticks([0,10,20]);ax.set_yticks([0,10,20]);ax.set_xlabel('x (mm)')
            if j==0:ax.set_ylabel(['SNN\ny (mm)','Rate model\ny (mm)'][i])
            else:ax.set_yticklabels([])
            ax.text(-.14,1.02,chr(65+2*i+j),transform=ax.transAxes,weight='bold',fontsize=16)
            if i==0:ax.text(.5,1.06,f'$D_A={conditions[fieldname]["D_A"]:.3f}$',transform=ax.transAxes,ha='center')
        meta.append(dict(condition=label,coordinates=conditions[fieldname],
             native_source=str(NATIVE/'fields.npz'),rate_source=str(RATE/label/'block01.npz')))
    cb=fig.colorbar(im,ax=axes,orientation='horizontal',fraction=.045,pad=.04)
    cb.set_label('Mean E rate (Hz / neuron)')
    dest=NATIVE/'figures';dest.mkdir(exist_ok=True);name='fig_core_a_native_rate_resource_control'
    for ext in ['png','pdf','svg']:fig.savefig(dest/f'{name}.{ext}',dpi=200)
    plt.close(fig)
    write(dest/f'{name}.json',dict(rows=meta,
        native_window_ms=[11500,12500],rate_elapsed_window_ms=[9000,10000],
        definition='Both rows average1000ms; matching intervention and original spatial geometry, different within-model histories and external input conditions. EntireZheld,allMdynamic. Native original noisy9s state; rate its own settled state and constantmeaninput. Not a branch or complete model validation.',
        model_promoted=False,human_visual_acceptance='PENDING'))
    (dest/'README.md').write_text(f'### {name}.png / .pdf / .svg\n\n'
        '上排为原 SNN，下排为当前 rate 模型；左列参考核内资源，右列只耗减 Core A。各图均展示末1秒平均 E 率，保持原空间几何、0–500Hz色标；两模型内各自的配对保留相同快/M历史与外部输入，只改核A Z。跨模型的初态与外驱不同，单次局部干预一致不能等同完整动力学验收，也不是分岔图。\n\n'
        '**关注点**：核A局部高活动和向核B/外围扩展的空间范围是否一致；图形候选待人工检查。\n')


if __name__=='__main__':main()
