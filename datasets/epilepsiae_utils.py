import numpy as np
import struct
import os
from datetime import datetime
from scipy.signal import firwin,filtfilt,iirnotch,fftconvolve,hilbert

eeg_chns=['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2','FZ','CZ','PZ','F7','F8','T3','T4','T5','T6','T1','T2','EOG','EOG1','EOG2','ECG','EMG','EMG1','EMG2','HP1','HP2','HP3','PHO']#+['FT9','FT10']+#['HP1','HP2','HP3']

def band_pass(data,fs,freqs):
    # b,a=butter(5,[freqs[0]/(fs/2),freqs[1]/(fs/2)],btype='bandpass')
    b=firwin(201,[freqs[0]/(fs/2),freqs[1]/(fs/2)],pass_zero=False)
    # return filtfilt(b,a,data)
    return filtfilt(b,1,data)
    # return fftconvolve(data,b[None,:],mode='same')

def notch_filt(data,fs,freqs):
    filtdata=data.copy()
    for f in freqs:
        b,a=iirnotch(f/(fs/2),30)
        filtdata=filtfilt(b,a,filtdata)
    return filtdata

def hilbert_enve(data,fs,freqs):
    if freqs[1]-freqs[0]<=20:
        band_data=band_pass(data,fs,freqs)
        hil_data=hilbert(band_data,axis=-1)
        return np.abs(hil_data)
    freqs_list=np.arange(freqs[0],freqs[1],20)
    freqs_list=np.append(freqs_list,freqs[-1])
    freqs_array=np.array([freqs_list[:-1],freqs_list[1:]]).T
    data_enves=[]
    for fr in freqs_array:
        band_data=band_pass(data,fs,fr)
        hil_data=hilbert(band_data,axis=-1)
        data_enves.append(np.abs(hil_data))
    return np.sum(data_enves,axis=0)

def abs_enve(data,fs,freqs,twin):
    high_data=band_pass(data,fs,freqs)
    high_abs=np.abs(high_data)
    abs_enve=fftconvolve(high_abs,np.ones((1,int(twin*fs)))/int(twin*fs),mode='same')

    return abs_enve




class epilepsiae_block:
    def __init__(self,datafile,headfile):
        self.datafile=datafile
        self.headfile=headfile
        self.headInfo=self.read_headData(headfile)
        self.fs=self.headInfo['sample_freq']
        self.chn_names=self.headInfo['elec_names']
        self.dataHandle=self.open_dataFile(datafile)

    def read_headData(self,filename):
        head_dict={}
        with open(filename,'r') as fh:
            headData=fh.readlines()
            for line in headData:
                ind=line.strip().split('=')[0]
                cont=line.strip().split('=')[1]
                head_dict[ind]=cont
        head_dict['start_stamp']=datetime.strptime(head_dict['start_ts'],"%Y-%m-%d %H:%M:%S.%f").timestamp()
        head_dict['num_samples']=int(head_dict['num_samples'])
        head_dict['sample_freq']=float(head_dict['sample_freq'])
        head_dict['conversion_factor']=float(head_dict['conversion_factor'])
        head_dict['num_channels']=int(head_dict['num_channels'])
        head_dict['elec_names']=head_dict['elec_names'][1:-1].split(',')
        head_dict['duration_in_sec']=float(head_dict['duration_in_sec'])
        head_dict['sample_bytes']=int(head_dict['sample_bytes'])

        return head_dict

    def open_dataFile(self,filename):
        fh=open(filename,'rb')
        return fh


    def fetch_data(self,begin_t,end_t):
        begin_bytes=int(round(begin_t*self.headInfo['sample_freq'])*self.headInfo['sample_bytes']*self.headInfo['num_channels'])
        end_bytes=int(round(end_t*self.headInfo['sample_freq'])*self.headInfo['sample_bytes']*self.headInfo['num_channels'])
        if end_bytes>os.path.getsize(self.datafile):
            end_bytes=os.path.getsize(self.datafile)
        self.dataHandle.seek(begin_bytes)
        fetched_bytes=self.dataHandle.read(end_bytes-begin_bytes)
        fetched_data=struct.unpack('<{}h'.format(int((end_bytes-begin_bytes)/self.headInfo['sample_bytes'])),fetched_bytes)
        fetched_eeg=np.reshape(fetched_data,[-1,self.headInfo['num_channels']]).T
        fetched_eeg=-1*self.headInfo['conversion_factor']*fetched_eeg

        return fetched_eeg

    def __del__(self):
        self.dataHandle.close()


