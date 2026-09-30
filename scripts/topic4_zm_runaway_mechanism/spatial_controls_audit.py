"""Matched-clock spatial readout for the regional Z interventions."""
from native_path import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def main():
    s=model();path=attach_rate_entry_path(s);base=path['fields'][0]
    names=['endpoint_D0.1429804_dt0.05_rate','rate_spatial_Z_cores','rate_spatial_Z_surround','rate_spatial_Z_both']
    labels=['Baseline','Core Z changed','Surround Z changed','Both changed']
    records={r['label']:r for r in read(OUT/'canonical_readouts.json')['rows']}
    allz=[np.load(OUT/'runs'/x/'trajectory.npz') for x in names]
    smooth=np.convolve(allz[0]['global_E_hz'],np.ones(50)/50,mode='same')
    peak=10000+int(np.argmax(smooth[10000:11000]))
    quiet=peak+70+int(np.argmin(smooth[peak+70:peak+230]));centers=[peak,quiet]
    rows=[];plt.rcParams.update({'font.size':12,'pdf.fonttype':42})
    fig,axs=plt.subplots(3,4,figsize=(12.5,9.0),layout='constrained',sharex=True,sharey=True)
    for j,(name,label,z) in enumerate(zip(names,labels,allz)):
        fieldz=z['Z_source'];delta=s.cell_field(base-fieldz).reshape(20,20)/1000
        ax=axs[0,j];imz=ax.imshow(delta,origin='lower',extent=[0,20,0,20],vmin=-.012,vmax=.012,cmap='RdBu_r')
        ax.text(.5,1.12,label,transform=ax.transAxes,ha='center')
        maps=[]
        for i,t in enumerate(centers,1):
            win=[t-25,t+25];field=z['field_E_hz'][win[0]:win[1]].mean(0).reshape(20,20)
            imr=axs[i,j].imshow(field,origin='lower',extent=[0,20,0,20],vmin=0,vmax=500,cmap='magma')
            maps.append(dict(window_ms=win,global_mean_hz=float(z['global_E_hz'][win[0]:win[1]].mean())))
        for i in range(3):
            ax=axs[i,j];ax.set_xticks([0,10,20]);ax.set_yticks([0,10,20])
            ax.text(-.11,1.065,chr(65+i*4+j),transform=ax.transAxes,fontweight='bold')
            for c,cl in zip(s.geo['centers_mm'],'AB'):
                ax.add_patch(Circle(c,1.5,fill=False,ec='#16c7d1',lw=1.2))
                ax.text(c[0],c[1]+2.0,cl,color='#16c7d1',ha='center',fontsize=9)
            if i==2:ax.set_xlabel('x (mm)')
            if j==0:ax.set_ylabel(('Z intervention' if i==0 else f'{centers[i-1]/1000:.3f} s')+'\ny (mm)')
        r=records[name]
        rows.append(dict(label=name,category=r['category'],D=float(1-fieldz[s.E]@s.mean_weights),
            Z_A_B_surround=(np.array(s.regional_rates(fieldz))/1000).tolist(),
            tail4s=r['tail'],matched_spatial_windows=maps,source=str(OUT/'runs'/name/'trajectory.npz')))
    fig.colorbar(imz,ax=axs[0,:].tolist(),fraction=.025,pad=.02,label=r'$\Delta D_i$',ticks=[-.012,0,.012])
    fig.colorbar(imr,ax=axs[1:,:].ravel().tolist(),fraction=.025,pad=.02,label='E rate (Hz)',ticks=[0,250,500])
    dest=OUT/'figures';name='fig_regional_Z_intervention'
    for ext in ['png','pdf']:fig.savefig(dest/f'{name}.{ext}',dpi=190)
    plt.close(fig)
    result=dict(status='COMPLETE',rows=rows,initial=str(OUT/'initial_states/rate_seed_N1024_dt0.05.npz'),
        M='dynamic in all four conditions',Z='held at the specified spatial field',
        finite_time_conclusion='Surround-only depletion is sufficient for sustained local activity over the assessed 4-s tail; core-only depletion does not remove all complete self-terminating events.',
        limits='These are local persistent states, not yet the final broadly saturated runaway; regional intervention is not a bifurcation classification.',
        windows_selection='baseline 50-ms average maximum within 10--11 s, then minimum 70--230 ms later; same clocks in all conditions')
    write(OUT/'spatial_Z_controls_audit.json',result);write(dest/f'{name}.json',result)
    readme=dest/'README.md';text=readme.read_text();heading=f'### {name}.png / .pdf'
    if heading not in text:
        readme.write_text(text+'\n'+heading+'\n'
            '四列从同一完整周期状态及延迟历史出发，依次保持基线Z、只改变双核Z、只改变外围Z、同时改变两者；M始终动态。'
            '上排显示相对基线的逐空间格耗减变化，后两排是按基线选择的爆发与事件间低活动时钟，四列使用完全相同的50 ms窗口。'
            '**关注点**：外围资源变化是否足以使事件间低活动消失；此图检验有限时间的区域因果作用，不单独给出分岔类型或全局饱和onset。\n')
    log('REGIONAL Z AUDIT',[(r['label'],r['category'],r['tail4s']['mean_rate_hz']) for r in rows])


if __name__=='__main__':main()
