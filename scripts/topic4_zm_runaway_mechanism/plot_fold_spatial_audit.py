"""Spatial interpretation of verified equilibrium folds, separate from onset."""
from native_path import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def main():
    s=model();folder=OUT/'equilibria/native_down_segment_fold_audit'
    rows=read(folder/'summary.json')['rows'];chosen=[rows[i] for i in [4,0,1]]
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,ax=plt.subplots(2,3,figsize=(9.8,6.4),layout='constrained',sharex=True,sharey=True)
    cells=s.geo['group_cell'];counts=np.bincount(cells[s.E],weights=s.sizes[s.E],minlength=s.grid**2)
    metadata=[]
    for col,q in enumerate(chosen):
        i,j=q['bracket'];source=folder/f'turn_{i:04d}_{j:04d}.npz';z=np.load(source)
        assert q['static_type']=='SN_static_conditions_met' and q['characteristic_zero']['temporal_simple_zero']
        rate=np.bincount(cells[s.E],weights=s.sizes[s.E]*z['r'][s.E]*1000,minlength=s.grid**2)/np.maximum(counts,1)
        energy=np.bincount(cells[s.E],weights=z['energy'][s.E],minlength=s.grid**2)
        im=ax[0,col].imshow(rate.reshape(s.grid,s.grid),origin='lower',extent=[0,20,0,20],cmap='magma',vmin=0,vmax=500)
        em=ax[1,col].imshow(energy.reshape(s.grid,s.grid),origin='lower',extent=[0,20,0,20],cmap='viridis',vmin=0,vmax=.35)
        ax[0,col].text(.03,1.05,f'SN{col+1}   $D={q["D"]:.4f}$',transform=ax[0,col].transAxes,color='black')
        for row in range(2):
            a=ax[row,col];a.set_xticks([0,10,20]);a.set_yticks([0,10,20])
            for center,lab in zip(s.geo['centers_mm'],'AB'):
                a.add_patch(Circle(center,1.5,facecolor='none',edgecolor='#20c4cf',lw=1.2))
                a.text(center[0],center[1]+2,lab,color='#20c4cf',ha='center',fontsize=10)
        ax[1,col].set_xlabel('x (mm)')
        metadata.append(dict(source=str(source),D=q['D'],global_E_hz=q['global_E_hz'],
                             mode_energy_A_B_surround=q['mode_energy_A_B_surround']))
    for row in range(2):ax[row,0].set_ylabel('y (mm)')
    fig.colorbar(im,ax=ax[0].tolist(),fraction=.045,pad=.03,label='E rate (Hz)')
    fig.colorbar(em,ax=ax[1].tolist(),fraction=.045,pad=.03,label='Critical-mode energy / cell')
    name='fig_equilibrium_fold_spatial_audit'
    fig.savefig(dest/f'{name}.png',dpi=180);fig.savefig(dest/f'{name}.pdf');plt.close(fig)
    write(dest/f'{name}.json',dict(panels=metadata,
        meaning='Equilibrium folds and their zero-eigenvalue spatial modes; not identified as the onset boundary',
        row1='E-cell-count-weighted equilibrium rate',row2='E-cell-count-weighted squared right null mode, normalized over all E groups',
        spatial_resolution_mm=1.,human_visual_acceptance='PENDING'))
    (dest/'README.md').write_text('### fig_equilibrium_fold_spatial_audit.png / .pdf\n'
        '上排显示三个已通过零特征值、非退化与延迟特征矩阵检查的平衡支 SN；下排显示相应零模在空间上的能量。'
        '这些点位于高活动平衡支，临界模主要分布在核外，并不是已经证明的实际 onset 边界。'
        '**关注点**：比较不同 SN 所涉及的外围位置；不要把多个局部平衡支折叠合并成一次全局发作阈值。\n')


if __name__=='__main__':main()
