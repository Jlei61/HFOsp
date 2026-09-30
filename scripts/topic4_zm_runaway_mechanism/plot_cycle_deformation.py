"""Phase-removed branch geometry, with neuron-count normalization.

This diagnostic must not be relabelled as a Floquet instability mode.
"""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def main():
    s=model();folder=OUT/'periodic/rate_turn_G4097'
    rows=[read(folder/f'eval_{i:03d}_spatial_tangent.json') for i in [0,1]]
    counts=np.bincount(s.geo['group_cell'][s.E],weights=s.sizes[s.E],minlength=400)
    counts/=counts.sum()
    maps=[]
    for i in [0,1]:
        z=np.load(folder/f'eval_{i:03d}_spatial_tangent.npz')
        maps.append(np.sqrt(z['field']/counts))
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(9.4,3.2),gridspec_kw={'width_ratios':[1,1,1.15]})
    fig.subplots_adjust(left=.065,right=.88,bottom=.19,top=.85,wspace=.5)
    limit=float(np.max(maps))
    for k in [0,1]:
        ax=axes[k]
        im=ax.imshow(maps[k].reshape(20,20),origin='lower',extent=[0,20,0,20],
                     cmap='magma',vmin=0,vmax=limit)
        ax.set_xlabel('x (mm)');ax.set_ylabel('y (mm)');ax.set_xticks([0,10,20]);ax.set_yticks([0,10,20])
        ax.text(0,1.08,f'{"AB"[k]}   T = {rows[k]["T_ms"]:.2f} ms',transform=ax.transAxes)
        for center,label in zip(s.geo['centers_mm'],'AB'):
            ax.add_patch(Circle(center,1.5,fill=False,ec='#20c4cf',lw=1))
            ax.text(center[0],center[1]+2,label,color='#20c4cf',ha='center',fontsize=8)
    ax=axes[2];x=np.arange(3)
    for k,color in enumerate(['#276a87','#bc7418']):
        ax.bar(x+(k-.5)*.34,rows[k]['per_neuron_energy_relative_to_global_A_B_surround'],
               width=.32,color=color,label=f'T = {rows[k]["T_ms"]:.2f} ms')
    ax.set_xticks(x);ax.set_xticklabels(['Core A','Core B','Surround'])
    ax.set_ylabel('Mean squared deformation / neuron\n(relative to global mean)')
    ax.set_ylim(0,7.2)
    ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,fontsize=8)
    ax.axhline(1,color='black',lw=.7,ls=':');ax.text(-.15,1.08,'C',transform=ax.transAxes)
    cax=fig.add_axes([.92,.22,.014,.57]);fig.colorbar(im,cax=cax,label='Relative RMS deformation')
    dest=OUT/'figures';name='fig_periodic_branch_spatial_deformation'
    for ext in ['png','pdf','svg']:fig.savefig(dest/f'{name}.{ext}',dpi=190)
    plt.close(fig)
    write(dest/f'{name}.json',dict(status='DESCRIPTIVE_BRANCH_GEOMETRY',rows=rows,
        map_definition='sqrt(cell E-neuron weighted tangent energy fraction / cell E-neuron fraction)',
        phase='One overall phase shift removed before squaring; no independent shift of individual groups',
        claim='Not a Floquet mode or proof of which region causes onset',human_visual_acceptance='PENDING'))
    p=dest/'README.md';text=p.read_text();heading=f'### {name}.png / .pdf / .svg'
    if heading not in text:
        p.write_text(text+'\n'+heading+'\n周期回折两侧的分支切线去除整体相位后，显示轨道随参数变化时的空间形变。地图按每格E细胞数归一，右侧比较每个区域单位细胞的均方形变，避免将外围细胞多误认为局部响应强。**关注点**：这是周期支形状的诊断，尚不是认证的Floquet临界模；核心B的单位细胞形变较大与外围Z干预足以产生持续活动是不同的证据。\n')
    log('SPATIAL DEFORMATION FIGURE',dest/f'{name}.png')


if __name__=='__main__':main()
