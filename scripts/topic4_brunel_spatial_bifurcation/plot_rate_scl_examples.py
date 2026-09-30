"""Actual single rate events with SCL participation, under the frozen observer."""
import plot_rate_field as p
from rate_field import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    p.F=RATE_OUT/'scl_event_revision/figures';p.F.mkdir(parents=True,exist_ok=True)
    records,contract=p.load_cases(True);s=RateField();geo=dict(s.geo)
    geo['contact_xy']=np.load(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz')['contact_xy']
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(19,8));g=fig.add_gridspec(2,3,width_ratios=[1.05,1.7,1.05],left=.06,right=.985,top=.88,bottom=.12,wspace=.27,hspace=.48)
    for row,q in enumerate(records[1:3]):
        a=q['anchor_ms'];ax=fig.add_subplot(g[row,0]);p.wave(ax,q,xlim=((a-100)/1000,(a+190)/1000),label=True)
        ax.set_ylabel('Rate (Hz / cell)');ax.set_title(f'{q["letter"]}: J={q["J_EE_core"]:g}; SCL {q["selected_SCL_count"]}/4',loc='left',fontsize=12)
        sub=g[row,1].subgridspec(1,4,wspace=.12)
        for j,off in enumerate([-20,20,70,120]):im=p.spatial(fig.add_subplot(sub[j]),q,off,geo,j==0)
        ax=fig.add_subplot(g[row,2]);im2=p.contact(ax,q,contract,True);ax.tick_params(axis='y',labelsize=9)
        ax.set_title('Single-event contact envelope',fontsize=11)
    fig.suptitle('SCL-participating events from the spatial rate model',fontsize=16)
    fig.legend(handles=[Line2D([0],[0],color=p.COL[k],label=n) for k,n in enumerate(['Core A','Core B','All E'])],
        loc='upper left',bbox_to_anchor=(.06,.945),ncol=3,frameon=False,fontsize=10)
    cb=fig.colorbar(im,cax=fig.add_axes([.41,.035,.24,.012]),orientation='horizontal',ticks=[0,10,50,100,250,500]);cb.set_label('E rate (Hz / cell)',fontsize=10)
    cb=fig.colorbar(im2,cax=fig.add_axes([.79,.035,.19,.012]),orientation='horizontal');cb.set_label('Fixed-reference normalized envelope',fontsize=9)
    p.save(fig,'rate_events_with_scl')
    with (p.F/'README.md').open('a') as f:f.write('\n### rate_events_with_scl.png\n从J=0.942和0.95各选取一个有SCL参与的实际rate事件；分别有3和4个SCL触点通过原检测阈值。波形、四幅快照及触点包络来自相同事件，相对时间以该事件最早参与触点的质心对齐。**关注点**：这是按SCL参与条件选出的单次示例，不是平均模板；合格事件中至少一个SCL参与的比例分别为10/20和1/4。\n')


if __name__=='__main__':main()
