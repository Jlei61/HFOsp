"""Matched held/released Z controls; broad-activity readouts are not bifurcation labels."""
from native_path import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def sustained_start(x,length):
    runs=np.diff(np.r_[0,np.asarray(x,dtype=int),0]);a=np.flatnonzero(runs==1);b=np.flatnonzero(runs==-1)
    return next((int(i) for i,j in zip(a,b) if j-i>=length),None)


def main(figure_family='rate'):
    s=model();rows=[];dest=OUT/'figures';dest.mkdir(exist_ok=True)
    for family,stem in [('native','endpoint_D0.2196000_dt0.05'),('rate','endpoint_D0.1429804_dt0.05_rate')]:
        heldpath=OUT/'runs'/stem/'trajectory.npz';freepath=OUT/'runs'/(stem+'_dynamicZ')/'trajectory.npz'
        if not heldpath.exists() or not freepath.exists():continue
        held=np.load(heldpath);free=np.load(freepath)
        hc=read(heldpath.parent/'contract.json');fc=read(freepath.parent/'contract.json')
        assert hc['initial']==fc['initial'] and hc['dt_ms']==fc['dt_ms'] and hc['M']==fc['M']=='dynamic'
        assert np.array_equal(held['Z_source'],free['Z_source'])
        contracts=dict(held=hc,released=fc);readouts={}
        for name,z in [('held',held),('released',free)]:
            rate=z['global_E_hz'].reshape(-1,10).mean(1)
            field=z['field_E_hz'].reshape(-1,10,s.grid**2).mean(1)
            occupancy=(field>=50)@(z['cell_counts']/z['cell_counts'].sum())
            k=sustained_start(rate>=200,20)
            kb=sustained_start((rate>=200)&(occupancy>=.75),20)
            t=None if k is None else k*10
            Zs=np.vstack([z['Z_source'],z['Z_every50ms']]);times=np.arange(len(Zs))*50
            reg=np.array([s.regional_rates(row) for row in Zs])/1000
            D=1-Zs[:,s.E]@s.mean_weights
            readouts[name]=dict(first_200Hz_for_200ms_start_ms=t,
                first_broad_75percent_and_200Hz_for_200ms_start_ms=None if kb is None else kb*10,
                D_at_200Hz_entry=None if t is None else float(np.interp(t,times,D)),
                Z_A_B_surround_at_200Hz_entry=None if t is None else [float(np.interp(t,times,reg[:,j])) for j in range(3)],
                tail4s_mean_hz=float(rate[-400:].mean()),tail4s_mean_occupation=float(occupancy[-400:].mean()),
                D_initial=float(D[0]),D_final=float(D[-1]))
        row=dict(family=family,held=str(heldpath),released=str(freepath),contracts=contracts,readouts=readouts,
                 interpretation='Paired slow-feedback intervention; operational high-rate entry is not a located bifurcation.')
        rows.append(row)
        if family!=figure_family:continue
        entry=readouts['released']['first_200Hz_for_200ms_start_ms']
        if entry is None:continue
        # Include a real held-Z burst before entry, then compare matched clocks.
        smooth=np.convolve(held['global_E_hz'],np.ones(50)/50,mode='same')
        lo=max(25,entry-400);hi=max(lo+1,entry-75)
        burst=lo+int(np.argmax(smooth[lo:hi]))
        centers=[burst,entry+25,entry+225]
        plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
        fig,axs=plt.subplots(2,3,figsize=(9.0,6.4),layout='constrained',sharex=True,sharey=True)
        maps=[]
        for i,(name,z) in enumerate([('Z held',held),('Z dynamic',free)]):
            for j,t in enumerate(centers):
                a=axs[i,j];left=int(t-25);right=left+50
                field=z['field_E_hz'][left:right].mean(0).reshape(s.grid,s.grid)
                im=a.imshow(field,origin='lower',extent=[0,20,0,20],cmap='magma',vmin=0,vmax=500)
                a.text(.0,1.04,f'{"ABCDEF"[i*3+j]}   {t/1000:.3f} s',transform=a.transAxes)
                for c,label in zip(s.geo['centers_mm'],'AB'):
                    a.add_patch(Circle(c,1.5,fill=False,ec='#20c4cf',lw=1.2))
                    a.text(c[0],c[1]+2,label,color='#20c4cf',ha='center',fontsize=10)
                a.set_xticks([0,10,20]);a.set_yticks([0,10,20])
                if i==1:a.set_xlabel('x (mm)')
                if j==0:a.set_ylabel(name+'\ny (mm)')
                maps.append(dict(panel='ABCDEF'[i*3+j],condition=name,window_ms=[left,right],
                                 global_window_mean_hz=float(z['global_E_hz'][left:right].mean())))
        fig.colorbar(im,ax=axs.ravel().tolist(),fraction=.03,pad=.03,label='E rate (Hz)')
        name='fig_Z_release_spatial_control' if family=='rate' else 'fig_native_Z_release_spatial_control'
        for ext in ['png','pdf']:fig.savefig(dest/f'{name}.{ext}',dpi=190)
        plt.close(fig);row['spatial_panels']=maps;write(dest/f'{name}.json',row)
        readme=dest/'README.md';body=readme.read_text() if readme.exists() else ''
        heading='### '+name+'.png / .pdf'
        if heading not in body:
            introduction=('原生SNN空间Z路径上全局Z=.7804的同一完整初态，上排固定Z，下排释放Z；两排都是同一个二维rate模型，M均动态。'
                          if family=='native' else
                          '从同一个二维 rate 模型的短爆发周期完整状态出发，上排固定 Z，下排释放 Z，两者的 M 均动态；列对应完全相同的续接时间。')
            readme.write_text(body+'\n'+heading+'\n'
                +introduction+
                '每幅为50 ms平均空间放电率，色标统一为0–500 Hz，时间以本次共同初态为零。'
                '**关注点**：Z反馈是否使活动覆盖外围和两核；这里的高率进入时刻是操作性读出，不是已经定位的分岔点。\n')
    write(OUT/'release_Z_audit.json',dict(status='COMPLETE',rows=rows))
    for row in rows:log(row['family'],row['readouts'])


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--figure-family',choices=['rate','native'],default='rate')
    main(parser.parse_args().figure_family)
