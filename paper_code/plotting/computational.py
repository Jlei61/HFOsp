"""Export helpers and supplementary plots; paper panel painters are in fig1/2/4/5."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def save(fig, directory, name, description):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    fig.savefig(directory/(name+'.png'),dpi=150,bbox_inches='tight')
    fig.savefig(directory/(name+'.pdf'),bbox_inches='tight')
    readme=directory/'README.md'
    text=readme.read_text() if readme.exists() else '# 本次计算生成的图\n'
    title='### '+name+'.png'
    if title not in text:readme.write_text(text+'\n'+title+'\n'+description+'\n**关注点**：以本次输入、参数和计算规模解释结果；不替换论文定稿。\n')
    return fig


def shared_field_null(result):
    fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
    if result['status']=='OK':
        ax.hist(result['channel_null'],bins=25,color='gray')
        ax.axvline(result['observed_shared_field_r'],color='red',label='observed')
        ax.legend()
    else:
        ax.text(.5,.5,result['reason'],ha='center',transform=ax.transAxes)
    ax.set(xlabel='TA/TB shared-field Pearson r',ylabel='Permutations',
           title='E1146 · shared-field comparison')
    return fig


def energy_alignment(energy):
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained');t=energy['time_sec']
    scores=energy['scores']
    for name,values in scores.items():
        a=np.asarray(values)
        if a.shape==t.shape and name in ('signed_r','signed_projection_z'):axs[0].plot(t,a,label=name)
    axs[0].set(xlabel='Simulation time (s)',ylabel='Signed field score',title='Supplement: energy vs fixed interictal field');axs[0].legend(fontsize=7)
    axs[1].plot(t,np.nanmean(energy['robust_z'],axis=0));axs[1].set(xlabel='Simulation time (s)',ylabel='Mean energy robust z',title='Supplement: mean energy trajectory')
    return fig
