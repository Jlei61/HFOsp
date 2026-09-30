"""Show where three independently checked equilibrium folds change state.

These are zero-eigenvalue modes, not simulated activity snapshots or the
critical mode of the still-unclassified periodic onset candidate.
"""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def main():
    folder=OUT/'equilibria/native_low_selected_fold_audit'
    data=read(folder/'result.json');assert data['status']=='COMPLETE'
    s=model();rows=data['rows'];fields=[]
    for row in rows:
        assert row['static_type']=='SN_static_conditions_met'
        assert row['characteristic_zero']['temporal_simple_zero']
        assert all(x['status']=='UNSTABLE_BY_POSITIVE_ROOT' for x in row['adjacent_equilibria'])
        i,j=row['bracket'];z=np.load(folder/f'fold_{i:04d}_{j:04d}.npz')
        energy=z['energy'];assert abs(energy.sum()-1)<1e-12
        fields.append(np.bincount(s.geo['group_cell'],weights=energy,minlength=400).reshape(20,20))
    vmax=np.ceil(max(f.max() for f in fields)*20)/20
    plt.rcParams.update({'font.size':12,'pdf.fonttype':42,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(10.8,3.8))
    fig.subplots_adjust(left=.07,right=.87,bottom=.18,top=.83,wspace=.29)
    for k,(ax,row,field) in enumerate(zip(axes,rows,fields)):
        im=ax.imshow(field,origin='lower',extent=[0,20,0,20],cmap='magma',vmin=0,vmax=vmax)
        ax.set_xticks([0,10,20]);ax.set_yticks([0,10,20]);ax.set_xlabel('x (mm)')
        if k==0:ax.set_ylabel('y (mm)')
        ax.text(-.20,1.10,chr(65+k),transform=ax.transAxes,fontweight='bold',fontsize=16)
        ax.text(.5,1.08,rf'SN$_{k+1}$   $D={row["D"]:.4f}$',transform=ax.transAxes,ha='center')
        for center,label in zip(s.geo['centers_mm'],'AB'):
            ax.add_patch(Circle(center,1.5,fill=False,ec='#20c4cf',lw=1.2))
            ax.text(center[0],center[1]+2,label,color='#20c4cf',ha='center',fontsize=10)
    cax=fig.add_axes([.90,.22,.015,.55])
    fig.colorbar(im,cax=cax,label='E-mode energy fraction',ticks=np.linspace(0,vmax,3))
    name='fig_native_selected_equilibrium_SN_modes';dest=OUT/'figures'
    for ext in ['png','pdf','svg']:fig.savefig(dest/f'{name}.{ext}',dpi=190)
    plt.close(fig)
    write(dest/f'{name}.json',dict(status='DIAGNOSTIC_SELECTED_SN_NOT_ONSET_BIFURCATION',
        source=str(folder/'result.json'),selected_rows=rows,
        observable='Cell sum of original-E-count-weighted squared zero-eigenvector components, normalized over all E groups',
        spatial_statistical_unit='One deterministic equilibrium fold and its zero mode per panel',
        color_limits=[0,float(vmax)],Z='held native spatial path',M='dynamic',
        limitation='Modes, not actual activity snapshots. All three selected folds have unstable adjacent equilibria. No identification with onset, periodic-fold mode, or SNN causal region.',
        human_visual_acceptance='PENDING'))
    readme=dest/'README.md';text=readme.read_text();heading=f'### {name}.png / .pdf / .svg'
    if heading not in text:
        readme.write_text(text+'\n'+heading+'\n同一原生Z空间路径上，从低率平衡支选取并独立验证的三个SN；各图显示零特征模在二维空间的E细胞数加权能量分布，共用色标。前两个涉及核B及其邻近组织，第三个更多分布于外围；三者邻接平衡点均已证实不稳定。'
            '**关注点**：这些是不同空间平衡解的局部转折，不是三次发作，也不是实际放电快照；自限周期的临界模和onset关系仍单独验证。\n')
    print(dest/f'{name}.png',flush=True)


if __name__=='__main__':main()
