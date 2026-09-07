"""Explicitly incomplete original-layout previews, using only completed real results."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import build_full_fig5_two_bases as full
import numpy as np


def main():
    original=full.producer();plt=original.plt
    dest=full.ROOT/'results/paper-ready-figure/fig5_full_two_bases_preview/figures'
    dest.mkdir(parents=True,exist_ok=True)
    for cid in full.recovery.IDS:
        current=full.recovery.DATA/cid/'trajectory.json'
        if not current.exists():current=full.recovery.first.OUTPUT/cid/'trajectory.json'
        r=full.read(current);a=original.load_npz(r['arrays']['path'])
        p=full.read(full.recovery.DATA/cid/'protocol.json');onset=p['states_ms']['pre_onset']+250
        stages=dict(onset_ms=onset,display_ms=[0,min(float(a['time_ms'][-1]),onset+2500)],returned_interictal_ms=[],pre_onset_ms=[onset-250,onset-50])
        adapted=dict(a,transition_lfp_trace=a['lfp'],transition_lfp_dt_ms=1.,transition_spatial_frame_time_ms=a['time_ms'],transition_rate_E_hz_20ms=a['rates_hz'][:,0])
        summary=full.read(full.DATA/cid/'map_summary.json')
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
        fig=plt.figure(figsize=(14.2,10.4),facecolor='white')
        fig.text(.06,.985,'From recurrent interictal events to tonic runaway',fontsize=16,fontweight='bold',va='top')
        name='Base 1: threshold gain 0.7, GABA decay 18 ms' if cid==full.recovery.IDS[0] else 'Base 2: threshold gain 1.0, GABA decay 24 ms'
        fig.text(.06,.957,name+'  |  INCOMPLETE PREVIEW',fontsize=10,color='#884400')
        original.title(fig,(.065,.924),'A','Spontaneous transition in the two-core SNN')
        original.plot_a(fig.add_axes([.065,.65,.485,.237]),adapted,stages)
        original.title(fig,(.065,.604),'B','Population activity and inhibitory state')
        axes=[fig.add_axes([.065,y,.485,.058]) for y in [.509,.439,.369]]
        original.plot_b(axes,adapted,stages,p['job']['config']['eta_m'])
        axes[0].set_ylim(0,510);axes[1].set_ylim(0,1.04)
        for letter,title,y,message in [('C','Spatial recruitment along the same trajectory',.924,'Same-trajectory spatial replay pending'),('D','Paired stimulation-site susceptibility',.604,'Paired perturbations pending')]:
            original.title(fig,(.62,y),letter,title)
            ax=fig.add_axes([.62,y-.23,.34,.18]);ax.set_facecolor('#f6f6f6');ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_color('#cccccc')
            ax.text(.5,.5,message+'\nNo historical results substituted',ha='center',va='center',transform=ax.transAxes,color='#777777',fontsize=10)
        original.title(fig,(.065,.293),'E','Recruitment as inhibition weakens')
        ax=fig.add_axes([.065,.075,.485,.176])
        for prep,ls in [('low','-'),('high','--')]:
            rows=sorted([r for r in summary['rows'] if r['m_gain_scale']==1 and r['preparation']==prep],key=lambda r:r['s'])
            ax.plot([r['s'] for r in rows],[r['population_hz'] for r in rows],ls+'o',ms=3,label=f'{prep.capitalize()} initial rates')
        ax.axhline(300,color='.6',lw=.6,ls=':');ax.set(xlabel='Core disinhibition, s = 1 − Zcore',ylabel='Terminal E rate (Hz)',xlim=(0,.9),ylim=(0,510));ax.legend(fontsize=8,frameon=False);original.style(ax)
        original.title(fig,(.62,.293),'F','Adaptation and initial-state dependence')
        fax=fig.add_axes([.62,.095,.34,.157]);original.plot_f(fax,summary);fax.set_xticks([0,.2,.4,.6,.8])
        stem=dest/('fig5-preview-'+cid)
        fig.savefig(stem.with_suffix('.png'),dpi=180);fig.savefig(stem.with_suffix('.pdf'));plt.close(fig)
        full.write(stem.with_suffix('.metadata.json'),dict(status='INCOMPLETE_PREVIEW',completed_panels=['A','B','E','F'],pending_panels=['C','D'],trajectory_source=str(current),trajectory_status=r['trajectory'],map_source=str(full.DATA/cid/'map_summary.json'),author_accepted=False))
    (dest/'README.md').write_text('\n\n'.join(f'### fig5-preview-{cid}.png\n原版布局的未完成预览，A/B 取已保存的本基底轨迹，E/F 取已完成的 80 条简化模型扫描；C/D 尚未完成并明确留空。E/F 为有限时间响应，不是平衡支证明。\n**关注点**：这不是最终完整 Fig. 5，不能用预览代替 C/D 的计算与核验。' for cid in full.recovery.IDS)+'\n')


if __name__=='__main__':main()
