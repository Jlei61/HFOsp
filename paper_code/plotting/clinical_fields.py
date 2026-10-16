"""Supply newly computed Fig5E/F values to the original paper painters."""
import numpy as np
import matplotlib.pyplot as plt
from plotting.paper.onset_field import _draw_field,_normalize_minmax,_display_geometry
from plotting.paper.fig5_statistic import draw_stat


def unavailable(reason):
    fig,ax=plt.subplots(figsize=(6,3));ax.axis('off')
    ax.text(.5,.5,reason,ha='center',va='center',wrap=True,transform=ax.transAxes)
    return fig


def model_and_patient(model,comparison,geometry):
    if comparison['status']!='OK':return unavailable(comparison['reason'])
    scorer=geometry['scorer'];fz=_display_geometry(geometry)
    selected=comparison['selected'];mask=geometry['contact_mask'];support=np.asarray(scorer['support'])
    fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
    for i,(value,title) in enumerate([(model['robust_z'],'Model onset'),(selected['robust_z'],f"E1146 · SZ{selected['seizure_number']}")]):
        mapped=_draw_field(axes[i],fz,_normalize_minmax(np.asarray(value)[mask]),support,cmap='Blues',
             colorbar_values=np.array([0.,1.]),title=title,title_color='black',show_y=i==0)
    fig.colorbar(mapped,ax=axes,fraction=.025,pad=.025,label='Normalized z field')
    return fig


def statistic(result):
    if result['status']!='OK':return unavailable(result['reason'])
    fig=plt.figure(figsize=(6,5.8));ax=draw_stat(fig,[.19,.20,.78,.75],result)
    if result['n_patients']==1:ax.set_xlabel('E1146 (1 patient)',fontsize=18,labelpad=7)
    return fig
