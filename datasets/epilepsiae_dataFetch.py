import numpy as np
import os
from scipy.signal import welch,iirnotch,filtfilt,fftconvolve
import matplotlib.pyplot as plt
import struct
import time
# import datetime
from datetime import datetime
import matplotlib.pyplot as plt

from scipy.signal import butter,iirnotch,filtfilt,hilbert,firwin
# from epilepsiae_detectHFOs import band_filt_cu,notch_filt_cu,return_hilEnve_norm_cu
# import cupy as cp
# import cusignal as cu
# from sub_dropChns import sub_drop_chns
sub_drop_chns = []

def band_pass(data,fs,freqs):
    # b,a=butter(5,[freqs[0]/(fs/2),freqs[1]/(fs/2)],btype='bandpass')
    b=firwin(201,[freqs[0]/(fs/2),freqs[1]/(fs/2)],pass_zero=False)
    # return filtfilt(b,a,data)
    # return filtfilt(b,1,data)
    return fftconvolve(data,b[None,:],mode='same')

def notch_filt(data,fs,freqs):
    filtdata=data.copy()
    for f in freqs:
        # b,a=iirnotch(f/(fs/2),30)
        b=firwin(801,[(freqs[0]-2)/(fs/2),(freqs[1]+2)/(fs/2)],pass_zero=True)
        # filtdata=filtfilt(b,a,filtdata)
        filtdata=fftconvolve(data,b[None,:],mode='same')
    return filtdata

def hilbert_enve(data,fs,freqs):
    freqs_list=np.arange(freqs[0],freqs[1],20)
    freqs_list=np.append(freqs_list,freqs[-1])
    freqs_array=np.array([freqs_list[:-1],freqs_list[1:]]).T
    data_enves=[]
    for fr in freqs_array:
        band_data=band_pass(data,fs,fr)
        hil_data=hilbert(band_data,axis=-1)
        data_enves.append(np.abs(hil_data))
    return np.sum(data_enves,axis=0)


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



#'916','620','384','818':
# /home/niking314/Documents/3_Data/epilepsiae/all_data_lns/818/all_recs/81802102_0087.data,20



subname='916'
filenum=20
avgRef=True

whole_subData_dir='H:/interilca_inter_results/all_data_lns'
# head_file='/home/niking314/Documents/3_Data/epilepsiae/1096/109600102_0000.head'
# head_file='/home/niking314/Documents/3_Data/epilepsiae/all_data_lns/958/all_recs/95800102_0010.head'
# head_file='/media/niking314/Backup Plus/epilepsiae/inv/pat_95802/adm_958102/rec_95800102/95800102_0003.head'
# head_file='/home/niking314/Documents/3_Data/epilepsiae/all_data_lns/1096/all_recs/109600102_0072.head'
# head_file='/media/niking314/Backup Plus/epilepsiae/inv2/pat_107302/adm_1073102/rec_107300102/107300102_0003.head'

# data_file='/home/niking314/Documents/3_Data/epilepsiae/1096/109600102_0000.data'
# data_file='/home/niking314/Documents/3_Data/epilepsiae/all_data_lns/1096/all_recs/109600102_0072.data'
# data_file='/media/niking314/Backup Plus/epilepsiae/inv/pat_95802/adm_958102/rec_95800102/95800102_0003.data'
# data_file='/media/niking314/Backup Plus/epilepsiae/inv2/pat_107302/adm_1073102/rec_107300102/107300102_0003.data'


def get_headData_file(wholeDir,sub,filenum):
    subdir=os.path.join(wholeDir,sub,'all_recs')
    all_data_files=[]
    for filename in os.listdir(subdir):
        if filename.split('.')[-1]=='data':
            all_data_files.append(filename)
    headFile=os.path.join(wholeDir,sub,'all_recs',all_data_files[filenum].split('.')[0]+'.head')
    dataFile=os.path.join(wholeDir,sub,'all_recs',all_data_files[filenum])
    return headFile,dataFile

head_file,data_file=get_headData_file(whole_subData_dir,subname,filenum)
print(data_file)


testdata=epilepsiae_block(data_file,head_file)

chns=testdata.chn_names
fs=testdata.fs
data_dur=testdata.headInfo['duration_in_sec']

# eeg_chns=['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2','FZ','CZ','PZ','F7','F8','T3','T4','T5','T6','T1','T2','EOG','EOG1','EOG2','ECG','EMG','EMG1','EMG2']#+['FT9','FT10']+['HP1','HP2','HP3']#+['FL2','FL3','M1','M2']
from epilepsiae_utils import eeg_chns
ecog_chns_bool=np.array([False if x in eeg_chns else True for x in chns])
ecog_chns_names=np.array(chns)[ecog_chns_bool]


twin=100
for t in np.arange(0,data_dur,twin):
    tmp_data=testdata.fetch_data(t,t+twin)
    ecog_data0=tmp_data[ecog_chns_bool]
    show_chns=ecog_chns_names

    #drop chns ##################################################
    if avgRef:
        keep_index=np.array([x not in sub_drop_chns[subname] for x in ecog_chns_names])
        ecog_data0=ecog_data0[keep_index]
        show_chns=ecog_chns_names[keep_index]
        ecog_data0=ecog_data0-np.mean(ecog_data0,axis=0)
    #####################################################3

    # ecog_data0=ecog_data
    # ecog_data=ecog_data[:-1]-ecog_data[1:]
    ecog_data=notch_filt(ecog_data0,fs,np.arange(50,251,50))
    ecog_enve=hilbert_enve(ecog_data,fs,[80,250])
    print(t)
    gap=10*ecog_data.std()
    gap_enve=10*ecog_enve[:,int(10*fs):int(90*fs)].std()
    plt.figure('1')
    for i in range(ecog_data.shape[0]):
        plt.plot(np.arange(ecog_data.shape[1])/fs,ecog_data[i]+i*gap,linewidth=0.7)
    plt.yticks(np.arange(ecog_data.shape[0])*gap,show_chns)


    plt.figure('2')
    for i in range(ecog_data.shape[0]):
        plt.plot(np.arange(ecog_enve.shape[1])/fs,ecog_enve[i]+i*gap_enve,linewidth=0.7)

    plt.yticks(np.arange(ecog_enve.shape[0])*gap_enve,show_chns)

    # plt.figure('3')
    # plt.plot(np.arange(ecog_enve.shape[1])/fs,np.sum(ecog_enve,axis=0),linewidth=0.7)


    # ecog_data=cp.asarray(ecog_data0)
    # ecog_data=notch_filt_cu(ecog_data,fs,np.arange(50,251,50))
    # ecog_enve=return_hilEnve_norm_cu(ecog_data,fs,[80,250])
    # ecog_enve=cp.asnumpy(ecog_enve)
    # gap_enve=10*ecog_enve[:,int(10*fs):int(90*fs)].std()
    # plt.figure('3')
    # for i in range(ecog_data.shape[0]):
    #     plt.plot(np.arange(ecog_enve.shape[1])/fs,ecog_enve[i]+i*gap_enve,linewidth=0.7)
    #
    # plt.yticks(np.arange(ecog_enve.shape[0])*gap_enve,ecog_chns_names)

    plt.show()

