"""Native/rate resources and activity at their respective operational entry."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def main():
    s = model(); source = OUT/'shared_variance_network_sensitivity'
    audit = read(source/'entry_resource_coordinates.json')
    assert audit['status'] == 'DESCRIPTIVE_AND_ORIGINAL_CRITERIA_AUDIT_COMPLETE'
    native = np.load(audit['native_reference_checkpoint'])
    zr = np.load(source/'units_history_delay_covariance/trajectory.npz')
    native_readout = np.load(BASE/'native_reference/seed9108401_readouts.npz')
    j = np.searchsorted(zr['state_time_ms'],audit['rate_entry_ms'])-1
    chosen_rate_time = float(zr['state_time_ms'][j])
    cells = s.geo['group_cell'][s.E]; weights = s.sizes[s.E]
    cell_counts = np.bincount(cells,weights=weights,minlength=400)
    zn = np.ones(s.P); zn[s.E] = s.project(native['slow__z'][:32000])[s.E]
    data = [('Native SNN',9870.,zn,native_readout['t'],native_readout['rate_cells']),
            ('Rate model',chosen_rate_time,zr['Z'][j],zr['time_ms'],zr['field_E_hz'])]
    plt.rcParams.update({'font.size':12,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes = plt.subplots(2,2,figsize=(7.2,7.0),sharex=True,sharey=True)
    fig.subplots_adjust(left=.12,right=.81,bottom=.1,top=.91,wspace=.18,hspace=.25)
    rows = []
    for c,(label,t,z,ts,field) in enumerate(data):
        D = np.bincount(cells,weights=(1-z[s.E])*weights,minlength=400)/cell_counts
        sel = (ts>=t-25)&(ts<t+25); assert sel.sum()==50
        activity = field[sel].mean(0)
        for r,(values,cmap,maximum) in enumerate([(D,'viridis',1.),(activity,'magma',500.)]):
            ax = axes[r,c]
            im = ax.imshow(values.reshape(20,20),origin='lower',extent=[0,20,0,20],vmin=0,vmax=maximum,cmap=cmap)
            ax.set_xticks([0,10,20]); ax.set_yticks([0,10,20])
            if c==0:
                ax.set_ylabel('y (mm)'); ax.text(-.23,1.04,'AB'[r],fontweight='bold',fontsize=17,transform=ax.transAxes)
            if r==1: ax.set_xlabel('x (mm)')
            if r==0: ax.text(.5,1.08,f'{label}\n{t/1000:.3f} s',ha='center',transform=ax.transAxes)
            for center,name in zip(s.geo['centers_mm'],'AB'):
                ax.add_patch(Circle(center,1.75,fill=False,ec='#20c4cf',lw=1.1))
                ax.text(center[0],center[1]+2.2,name,ha='center',color='#20c4cf',fontsize=10)
            if c==1:
                cb=fig.add_axes([.86,.575 if r==0 else .145,.022,.285])
                bar=fig.colorbar(im,cax=cb,ticks=[0,.5,1] if r==0 else [0,250,500])
                bar.set_label(r'$D(\mathbf{x})=1-Z_E(\mathbf{x})$' if r==0 else 'E rate (Hz)')
        rows.append(dict(label=label,time_ms=t,field_window_ms=[t-25,t+25],
                         global_D=float(np.average(D,weights=cell_counts)),
                         field_mean_E_Hz=float(np.average(activity,weights=cell_counts))))
    name='fig_native_rate_entry_resource_spatial'; folder=OUT/'figures'
    for ext in ['png','pdf','svg']: fig.savefig(folder/f'{name}.{ext}',dpi=190)
    plt.close(fig)
    write(folder/f'{name}.json',dict(source=str(source/'entry_resource_coordinates.json'),panels=rows,
          human_visual_acceptance='PENDING',agent_PNG_PDF_check='PENDING',
          selection='Native measured9.870s checkpoint; last stored rate resource sample before its operational entry. No time rescaling or coordinate interpolation.',
          scope=audit['scope'],Z_and_M='Dynamic in both trajectories; distinct histories and state-dependent output noise.'))
    p=folder/'README.md'; text=p.read_text();heading=f'### {name}.png / .pdf / .svg'
    assert heading not in text
    p.write_text(text+'\n'+heading+'\n上排比较原生SNN与修正候选各自进入持续高率附近的资源耗减空间场，下排为同一时刻的50ms空间活动。原生使用9.870秒真实检查点，候选使用进入前最后一个8.930秒资源采样，两个模型Z、M均动态。**关注点**：进入时的资源与空间活动是否接近；这是按进入阶段对齐的描述图，不能替换同一时钟D轨迹验收，也不能证明共同分岔。\n')
    log('PLOT',folder/f'{name}.png')


if __name__=='__main__': main()
