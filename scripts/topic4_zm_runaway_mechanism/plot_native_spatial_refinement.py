"""Show the primary same-Z coarse/fine contrast without implying a branch."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import argparse


def main(include_time=False):
    folder=OUT/'spatial_refinement'
    source=folder/'result.json'
    if not source.exists():source=folder/'partial_audit.json'
    audit=read(source)
    selected=[]
    for detail in ['coarse','lifted']:
        row=next(r for r in audit['rows'] if r['D']==.219 and r['Z_detail']==detail)
        selected.append(row)
    if include_time:
        time_audit=read(folder/'time_refinement_result.json')
        row=time_audit['rows'][1]
        selected.append(dict(D=.219,grid=40,Z_detail='lifted',source=row['source'],
            canonical_common400=row['canonical_common400']))
    nrows=len(selected)
    plt.rcParams.update({'font.size':12,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(nrows,2,figsize=(7.8,7.8 if nrows==2 else 10.8),sharex=True,sharey=True)
    fig.subplots_adjust(left=.19,right=.96,bottom=.22 if nrows==2 else .16,top=.90 if nrows==2 else .94,wspace=.28,hspace=.22)
    s=model();images=[]
    for i,row in enumerate(selected):
        z=np.load(row['source']);n=row['grid']
        field=z['field_E_hz'][-4000:].reshape(400,10,n*n).mean(1)
        values=[field.mean(0),(field>=50).mean(0)]
        for j,ax in enumerate(axes[i]):
            im=ax.imshow(values[j].reshape(n,n),origin='lower',extent=[0,20,0,20],
                cmap='magma' if j==0 else 'viridis',vmin=0,vmax=500 if j==0 else 1)
            ax.set_xticks([0,10,20]);ax.set_yticks([0,10,20])
            if i==nrows-1:ax.set_xlabel('x (mm)')
            if j==0:
                ax.set_ylabel('y (mm)')
                label='1.0 mm grid' if i==0 else '0.5 mm grid'
                if include_time:label+='; '+('$\\Delta t=0.025$ ms' if i==2 else '$\\Delta t=0.05$ ms')
                ax.text(-.52,.5,label,
                    transform=ax.transAxes,rotation=90,ha='center',va='center')
            if i==0:ax.text(.5,1.08,['Mean E rate','Time above 50 Hz'][j],
                transform=ax.transAxes,ha='center')
            ax.text(-.16,1.04,'ABCDEF'[i*2+j],transform=ax.transAxes,
                fontweight='bold',fontsize=16)
            for center,label in zip(s.geo['centers_mm'],'AB'):
                ax.add_patch(Circle(center,1.5,fill=False,ec='#20c4cf',lw=1.1))
                ax.text(center[0],center[1]+2,label,color='#20c4cf',ha='center',fontsize=9)
            if i==0:images.append(im)
    for j,im in enumerate(images):
        pos=axes[-1,j].get_position()
        cax=fig.add_axes([pos.x0,.10 if nrows==2 else .073,pos.width,.015 if nrows==2 else .010])
        fig.colorbar(im,cax=cax,label='E rate (Hz)' if j==0 else 'Time fraction',
            ticks=[0,250,500] if j==0 else [0,.5,1],orientation='horizontal')
    out=OUT/'figures';name='fig_native_Z_spatial_time_resolution_D0219' if include_time else 'fig_native_Z_spatial_resolution_D0219'
    for ext in ['png','pdf','svg']:fig.savefig(out/f'{name}.{ext}',dpi=190)
    plt.close(fig)
    write(out/f'{name}.json',dict(source=str(source),D=.219,Z_mean=.781,
        rows=[dict(source=r['source'],grid=r['grid'],Z_detail=r['Z_detail'],
            category=r['canonical_common400']['category']) for r in selected],
        Z='held identical physical field',M='dynamic',window_ms=[8000,12000],
        same_physical_graph=True,same_lifted_full_initial_history=True,
        time_refinement_source=str(folder/'time_refinement_result.json') if include_time else None,
        panels='Mean E rate and fraction of 10-ms bins at least50Hz; NOT snapshots',
        scope='Finite-window spatial-resolution sensitivity, not a bifurcation or asymptotic certificate.',
        human_visual_acceptance='PENDING'))
    readme=out/'README.md';text=readme.read_text();heading=f'### {name}.png / .pdf / .svg'
    if heading not in text:
        description=(
            '固定同一原生路径Z场（D=0.219，平均Z=0.781），比较同一连接图在1mm和0.5mm网格上的12秒续接；完整初态、M和延迟历史按原细胞精确对应，Z固定而M动态。'
            '左列为末4秒平均E率，右列为每格在10ms分箱中达到50Hz的时间比例。')
        if include_time:
            description+='第三排加入细网格时间步减半结果：原持续分类变为未解决，有长活动段自行结束。**关注点**：细网格结果仍有时间步/有限窗敏感性，不能将第二排当作已收敛的持续吸引态；各图均非瞬时快照或分岔认证。\n'
        else:
            description+='**关注点**：粗网格自限、细网格局部持续的差别，在统一回400格统计后仍存在；这是空间分辨率敏感性结果，尚非临界点的网格收敛，也不是SNN发作分岔认证。\n'
        readme.write_text(text+'\n'+heading+'\n'+description)
    print(out/f'{name}.png')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--include-time',action='store_true')
    main(parser.parse_args().include_time)
