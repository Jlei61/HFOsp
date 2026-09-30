"""Show the actual state and spatial mode of the certified equilibrium SN."""
from common import OUT,np,read,write,model
from refractory_spatial_resolution import mapping,projections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def main():
    source=OUT/'core_a_bifurcation_type_20260924/fold_certificate';cert=read(source/'result.json')
    assert cert['status']=='GENERIC_EQUILIBRIUM_SADDLE_NODE_CERTIFIED'
    s=model(40);coarse=model(20);parent,_=mapping(coarse,s);P,count=projections(s,coarse,parent)[20]
    data=np.load(source/'critical_point.npz');rate=P@data['r']*1000;mode=P@data['v_rate'];mode/=abs(mode).max()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(8.4,3.8),layout='constrained')
    for j,(field,cmap,limits,label) in enumerate([(rate,'magma',(0,500),'E rate (Hz / neuron)'),(mode,'RdBu_r',(-1,1),'Critical E mode (relative)')]):
        ax=axes[j];im=ax.imshow(field.reshape(20,20),origin='lower',extent=[0,20,0,20],cmap=cmap,vmin=limits[0],vmax=limits[1],interpolation='nearest')
        ax.set_xlabel('x (mm)');ax.set_ylabel('y (mm)' if j==0 else '');ax.set_xticks([0,10,20]);ax.set_yticks([0,10,20])
        for k,center in enumerate(s.geo['centers_mm']):
            ax.add_patch(Circle(center,1.5,fill=False,color='#22ccd2',lw=1.3));ax.text(center[0],center[1]+1.8,'AB'[k],color='#22ccd2',ha='center',fontsize=11)
        ax.text(-.14,1.03,'AB'[j],transform=ax.transAxes,fontsize=16,fontweight='bold')
        cb=fig.colorbar(im,ax=ax,shrink=.87,pad=.025);cb.set_label(label);cb.set_ticks([0,250,500] if j==0 else [-1,0,1])
    dest=source/'figures';dest.mkdir(exist_ok=True);name='fig_certified_SN_state_and_mode'
    for ext in ['png','pdf','svg']:fig.savefig(dest/f'{name}.{ext}',dpi=220)
    plt.close(fig)
    write(dest/f'{name}.json',dict(source=str(source/'critical_point.npz'),D_A=cert['D_A'],Z_A=cert['Z_A'],
        fields=dict(equilibrium_E_rate_hz=rate.tolist(),critical_E_mode_relative=mode.tolist()),
        meaning='Full-network equilibrium at the certified SN and its zero-eigenvalue E-rate mode. This is NOT the observed trajectory state or its onset boundary. No stable-equilibrium label is given.',
        source_group_grid=40,display_grid=20,mode_normalization='Cell-count-weighted E mode in each1mmcell, divided by its maximum absolute value; sign convention is arbitrary.',human_visual_acceptance='PENDING'))
    (dest/'README.md').write_text('### fig_certified_SN_state_and_mode.png / .pdf / .svg\n\n'
        '左图是已严格核实的鞍结点处完整二维模型的平衡放电场，右图是零特征值对应的空间 E 放电扰动方向。只改变 Core A 的资源，临界 D_A=0.35315764（Z_A=0.64684236）；外围和 Core B 的 Z 固定，M 属于动态系统并在平衡点满足自身平衡方程。这个平衡点已有其他振荡失稳，主要临界变化在核 A 外侧招募带，尚不能称作真实轨迹进入 onset 的边界。\n\n'
        '**关注点**：核内高活动平衡图与外围临界模态的区别；右图是归一化特征向量，不是额外一次仿真的放电率。候选 PNG/PDF 待用户人工检查。\n')


if __name__=='__main__':main()
