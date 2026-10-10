import numpy as np
import pickle as pik
import os
import scipy.signal as sig
from scipy.ndimage import gaussian_filter
from scipy.signal import butter,filtfilt,get_window

data=np.load('./zhangkexuan_pickSigs.npz',allow_pickle=True)
annot=pik.load(open('./zhangkexuan_annot_v4.pik','rb'))

print(data['sigs'])
print(annot)
print(np.sum(annot==1).astype('int'))
print(np.sum(annot==2).astype('int'))
print(np.sum(annot==3).astype('int'))
sigs=data['sigs']

hfo=sigs[annot==1]
spike=sigs[annot==2]
coupled=sigs[annot==3]

import matplotlib.pyplot as plt

fs=1000

for tmpsig in [hfo,spike,coupled]:


    specs=[]
    for ts in tmpsig:
        f, t, hfo_spec = sig.spectrogram(ts, fs=fs,
                                         window=get_window('hann', int(0.18 * fs)),
                                         nperseg=int(0.18* fs), nfft=int(0.18 *fs),
                                         noverlap=int(0.16 * fs),
                                         mode='magnitude')
        # hfo_spec = np.log(hfo_spec + 1e-20)
        # hfo_spec = np.log(hfo_spec)
        # hfo_spec = gaussian_filter(hfo_spec, sigma=1.5)
        # spec_x, spec_y = np.meshgrid(t, f[self.spec_range[0]:self.spec_range[1]])
        # self.canvas.axes[2].pcolor(spec_x, spec_y, hfo_spec[self.spec_range[0]:self.spec_range[1]], cmap='jet')
        f_lim = f[(f >= 0) & (f <= 240)]
        hfo_spec = hfo_spec[(f >= 0) & (f <= 240)]
        hfo_spec = gaussian_filter(hfo_spec, sigma=1.5)
        # plt.pcolormesh(t, f_lim, hfo_spec, cmap='coolwarm')
        specs.append(hfo_spec)

    hfo_spec=np.mean(specs,axis=0)
    # hfo_spec = np.log(hfo_spec)
    hfo_spec_pre=np.log(hfo_spec)
    base_spec=hfo_spec[:,t<=0.15]
    hfo_spec=hfo_spec/np.mean(base_spec,axis=1,keepdims=True)-1#/np.mean(base_spec,axis=1,keepdims=True)
    plt.figure('whole',figsize=(3,6))
    ax1=plt.subplot(3,1,1)
    plt.plot(np.arange(len(tmpsig[0]))/fs,tmpsig.T,linewidth=0.5,alpha=0.5,c='k')
    plt.plot(np.arange(len(tmpsig[0]))/fs,np.mean(tmpsig,axis=0),linewidth=1,c='gold')
    plt.title('signal')
    # plt.ylabel('Sig')
    plt.subplot(3,1,2,sharex=ax1)
    plt.pcolormesh(t, f_lim, hfo_spec_pre, cmap='coolwarm')  # ,vmin=-16,vmax=-14)
    plt.ylabel('Freq/Hz')
    plt.title('raw Spec')
    # plt.colorbar()
    plt.subplot(3,1,3,sharex=ax1)
    plt.pcolormesh(t, f_lim, hfo_spec, cmap='coolwarm')#,vmin=-16,vmax=-14)
    plt.ylabel('Freq/Hz')
    plt.xlabel('Time/s')
    plt.title('normalized Spec')
    plt.tight_layout()
    # plt.colorbar()


    plt.figure('sig')
    plt.plot(tmpsig.T,linewidth=0.5,alpha=0.5,c='k')
    plt.plot(np.mean(tmpsig,axis=0),linewidth=1,c='gold')
    plt.figure('mean spec raw')
    plt.pcolormesh(t, f_lim, hfo_spec_pre, cmap='coolwarm')#,vmin=-16,vmax=-14)
    plt.colorbar()
    plt.figure('mean spec norm')
    plt.pcolormesh(t, f_lim, hfo_spec, cmap='coolwarm')#,vmin=-16,vmax=-14)
    plt.colorbar()

    plt.show()


    # def filter_data(data, sfreq, freqband):
    #     nyq = sfreq / 2
    #     b, a = butter(5, [freqband[0] / nyq, freqband[1] / nyq], btype='bandpass')
    #     return filtfilt(b, a, data, axis=-1)
    #
    #
    # self.canvas.axes[1].plot(np.arange(len(self.current_hfo_signal)) / self.sfreq,
    #                          filter_data(self.current_hfo_signal, self.sfreq,
    #                                      [self.filter_bank[0], self.filter_bank[1]]))

    # f, t, hfo_spec = sig.spectrogram(self.current_hfo_signal, fs=self.sfreq,
    #                                  window=get_window('hann', int(0.2 * self.sfreq)),
    #                                  nperseg=int(0.2 * self.sfreq), nfft=int(0.2 * self.sfreq),
    #                                  noverlap=int(0.17 * self.sfreq),
    #                                  mode='magnitude')
    # hfo_spec = np.log(hfo_spec + 1e-20)
    # hfo_spec = gaussian_filter(hfo_spec, sigma=1.5)
    # # spec_x, spec_y = np.meshgrid(t, f[self.spec_range[0]:self.spec_range[1]])
    # # self.canvas.axes[2].pcolor(spec_x, spec_y, hfo_spec[self.spec_range[0]:self.spec_range[1]], cmap='jet')
    # f_lim = f[(f >= 0) & (f <= 240)]
    # hfo_spec = hfo_spec[(f >= 0) & (f <= 240)]
    # self.canvas.axes[2].pcolor(t, f_lim, hfo_spec, cmap='coolwarm')


