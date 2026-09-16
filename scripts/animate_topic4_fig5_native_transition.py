#!/usr/bin/env python3
"""Direct E/I spike-count movie through the native high-state transition."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image
from plot_topic4_fig5_native_dense_v11 import NATIVE,FIG,RATE_CMAP,native_map


def main():
    a=np.load(NATIVE/'native_fields.npz');meta=np.load(NATIVE/'spatial_analysis.npz')
    fig=plt.figure(figsize=(9.2,4.6));gs=fig.add_gridspec(1,5,width_ratios=[1,.045,.17,1,.045],
        left=.075,right=.91,top=.86,bottom=.14,wspace=.30)
    artists=[]
    for k,(key,label,limit) in enumerate([('E_count_1ms','E rate (Hz)',500),('I_count_1ms','I rate (Hz)',800)]):
        col=0 if k==0 else 3
        ax=fig.add_subplot(gs[col]);im=native_map(ax,np.zeros(1600),meta,Normalize(0,limit),RATE_CMAP,
            ylabel=k==0,core_labels=k==0)
        cb=fig.colorbar(im,cax=fig.add_subplot(gs[col+1]),ticks=[0,limit/2,limit]);cb.set_label(label,fontsize=11)
        artists.append(im)
    title=fig.suptitle('',fontsize=15,y=.94)
    frames=[];starts=np.arange(9000,11141,20)
    for lo in starts:
        hi=lo+40
        for k,key in enumerate(['E_count_1ms','I_count_1ms']):
            nc=a['cell_n_E'] if k==0 else a['cell_n_I']
            values=np.divide(a[key][lo:hi].sum(0),nc*.04,out=np.full(1600,np.nan),where=nc>0)
            artists[k].set_data(np.ma.masked_invalid(values.reshape(40,40)))
        title.set_text(f'{lo/1000:.2f}–{hi/1000:.2f} s')
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB'))
    target=FIG/'native_EI_transition.gif'
    frames[0].save(target,save_all=True,append_images=frames[1:],duration=100,loop=0,optimize=False)
    plt.close(fig)
    print(str(target))


if __name__=='__main__':main()
