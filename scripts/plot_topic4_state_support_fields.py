"""Display actual applied I-input loading fields; no new physical simulation."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.topic4_core_field_rev9 import array_sha256

OUT=Path('/data/hfosp/topic4_sef_hfo/geometry_threshold_refinement_20260909')
BASE=Path('/data/hfosp/topic4_sef_hfo/core_extent_long_propagation_20260909/duration_90000/units/baseline/847101/workers/trajectory.npz')
pos=np.random.default_rng(2511).uniform(0.,20.,size=(40000,2))
with np.load(BASE) as z:
    assert np.array_equal(pos[:32000].astype(np.float32),z['positions_E'])
    fields=[('原固定作用域',z['I_target_indices'].copy(),z['I_loading'].copy())]
    contact_xy=z['contact_xy_mm'].copy();names=z['contact_names'].astype(str)
for cid,title in [('AB25_state_extent','双核 2.5 mm 的 I 作用域'),('A4_state_extent','左核 4 mm 的 I 作用域')]:
    c=json.loads((OUT/'candidates'/f'{cid}.json').read_text())
    d=np.linalg.norm(pos[:,None]-np.asarray(c['centers_mm'])[None],axis=2)
    near=d.argmin(1);inside=(d<=np.asarray(c['radii_mm'])[None]).any(1)
    a,b=[np.flatnonzero((np.arange(40000)>=32000)&inside&(near==k)) for k in range(2)]
    n=min(len(a),len(b));indices=np.r_[a,b]
    loading=np.r_[np.full(len(a),n/len(a)),np.full(len(b),-n/len(b))]
    for seed in [847101,847102]:
        applied=json.loads((OUT/'duration_90000/units'/cid/str(seed)/'applied_geometry.json').read_text())
        assert array_sha256(indices)==applied['applied_I_indices_sha256']
        assert array_sha256(loading)==applied['applied_I_loading_sha256']
    fields.append((title,indices,loading))

plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'pdf.fonttype':42})
fig,axs=plt.subplots(1,3,figsize=(14,5.9),layout='constrained')
base=fields[0][2][fields[0][2]>0].sum()
for ax,(title,idx,loading) in zip(axs,fields):
    ax.scatter(pos[32000:,0],pos[32000:,1],s=2,color='#d9d9d9',alpha=.25,rasterized=True)
    im=ax.scatter(pos[idx,0],pos[idx,1],c=loading,s=6,cmap='coolwarm',vmin=-1,vmax=1,rasterized=True)
    for prefix,color in [('SCL','#216e8a'),('ICL','#7b642e')]:
        use=np.flatnonzero(np.char.startswith(names,prefix))
        ax.plot(contact_xy[use,0],contact_xy[use,1],color=color,lw=1,zorder=4)
        ax.scatter(contact_xy[use,0],contact_xy[use,1],s=18,facecolors='white',edgecolors=color,zorder=5)
        for i in use:ax.annotate(names[i],contact_xy[i],xytext=(0,5),textcoords='offset points',ha='center',fontsize=5.5,color=color)
    ratio=loading[loading>0].sum()/base
    ax.set(xlim=(0,20),ylim=(0,20),aspect='equal',xlabel='x (mm)',ylabel='y (mm)',
           title=f'{title}\n左/右 I 细胞数：{(loading>0).sum()} / {(loading<0).sum()}')
    ax.text(.5,-.20,f'单侧重新分配幅度 ×{ratio:.3f}\n正、负加载之和 = {loading.sum():.1f}',transform=ax.transAxes,ha='center')
fig.colorbar(im,ax=axs,shrink=.7,label='I 外部输入加载系数；实际相对变化 = q(t) × 系数')
fig.suptitle('抑制输入调制：实际空间范围与重新分配幅度\n灰点为其他 I 细胞；色彩表示输入调制系数，不表示发放或患者传播模式',fontsize=12)
folder=OUT/'geometry_review/figures';folder.mkdir(exist_ok=True,parents=True)
for ext in ['png','pdf']:fig.savefig(folder/f'I_state_support_fields.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)
