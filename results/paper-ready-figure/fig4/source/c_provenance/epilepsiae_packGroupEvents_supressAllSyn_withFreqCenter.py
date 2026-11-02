import mne
import numpy as np
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import os
import shutil
import time
from pathlib import Path
# from subs_elecs_info import subs_elecs_info
import gc
# from preprocessing_utils import *
from highEvents_yuquan0910_utils import *
import cupy as cp
import cusignal
from hfo_net import show_events_timeCourse_ext,pick_states_withThresh_usingChnNum,extend_timeRanges,pick_noOverlap_timeRanges,get_packedEventsTimes_overThresh,get_packedEvents_bool,refine_packedEvents_byAllBool
# from cuda_utils import *
from scipy.signal import spectrogram
from scipy.ndimage import gaussian_filter
# from p16_subs_info import subs_drop_info
# from p16_cuda_24h_bipolar import bipolar_rerefAndDrop_eeg
from sub_dropChns import sub_drop_chns
from epilepsiae_detectHFOs import avg_rerefAndDrop_eeg,notch_filt_cu
from epilepsiae_utils import epilepsiae_block

mne.set_log_level('ERROR')

segment_time=200 #s
# resample_to=800 #hz
highpass_freqband=[80,250]
# drop_chns=np.array(['A8'])
# drop_chns=np.array([])
pickChn_thresh=0.5#a.u., pick channels over thresh

extL=30e-3 #second, extend detected event win egdes
packWinLen=500e-3 #second, paked time win length
chnsThr=0.5 #a.u., pack when more-than-Thresh chans have events

specWinLen=0.05 #second, spectrogram time win length
specFR=[50,300]

# eeg_chns=['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2','FZ','CZ','PZ','F7','F8','T3','T4','T5','T6','T1','T2','EOG1','EOG2','ECG','EMG']
from epilepsiae_utils import eeg_chns


#1096-0002

def split_chnName(chStr):
    split_re=re.search(r"([A-Z]'?)(\d+)",chStr)
    chPre=split_re.group(1)
    chNum=split_re.group(2)
    return chPre,chNum


def plot_perSeg_specCenter(segData,segTime,chNames,fs,packTimes):
    batch_data = segData
    # batch_data = scipy.signal.resample_poly(batch_data, 2, int(round(2 * fs / resample_to)), axis=-1)
    # batch_data = batch_data - np.mean(batch_data, axis=0)
    batch_data = notch_filt(batch_data, fs, np.arange(50, 251, 50))
    batch_high = band_filt(batch_data, fs, highpass_freqband)
    batch_t = segTime[0] + np.arange(batch_high.shape[1]) / fs
    # print('batchHigh shape',batch_high.shape)

    # extract timeWin signals, concatenate
    inSeg_timeWins = packTimes[(packTimes[:, 0] >= segTime[0]) & (packTimes[:, 1] <= segTime[-1])]
    timeWin_boolVec = np.zeros(len(batch_t))
    tWinLen_list = []
    for tw in inSeg_timeWins:
        twBool = (batch_t >= tw[0]) & (batch_t <= tw[1])
        tWinLen_list.append(len(np.where(twBool)[0]))
        timeWin_boolVec[twBool] = 1

    split_contiRaw=batch_data[:,timeWin_boolVec>0.5]
    split_contiHigh = batch_high[:, timeWin_boolVec > 0.5]
    split_border_t = np.cumsum(tWinLen_list) / fs

    spec_times = None
    spec_freqs = None
    all_specs = []
    for chi in range(split_contiHigh.shape[0]):
        # tmp_data = split_contiHigh[chi]
        tmp_f, tmp_t, tmp_spec = spectrogram(split_contiHigh[chi], fs, window='hamming', nperseg=int(specWinLen * fs),
                                             noverlap=int(0.8*specWinLen * fs), nfft=int(specWinLen* fs), mode='magnitude')
        # tmp_norm_spec=(tmp_spec-np.mean(tmp_spec,axis=1,keepdims=True))/np.std(tmp_spec,axis=1,keepdims=True)
        # tmp_norm_spec=gaussian_filter(tmp_norm_spec,sigma=1.5)
        tmp_norm_spec = tmp_spec
        # tmp_norm_spec=(tmp_norm_spec-np.mean(tmp_norm_spec,axis=1,keepdims=True))/np.std(tmp_norm_spec,axis=1,keepdims=True)
        # tmp_norm_spec=(tmp_norm_spec)/np.std(tmp_norm_spec,axis=1,keepdims=True)
        tmp_norm_spec = gaussian_filter(tmp_norm_spec, sigma=1.5)
        hg_fIndex = (tmp_f > specFR[0]) & (tmp_f < specFR[1])
        tmp_norm_spec = tmp_norm_spec[hg_fIndex, :]
        tmp_f = tmp_f[hg_fIndex]

        spec_times = tmp_t
        spec_freqs = tmp_f
        all_specs.append(tmp_norm_spec)


    chns_spec_centers = []
    for si, sp in enumerate(all_specs):
        chn_annot = return_specCenter_packed(sp, spec_times, split_border_t)
        chns_spec_centers.append(chn_annot)

    #prepare data for spec center
    all_specs_cat=np.concatenate(all_specs,axis=0)

    plt.figure('spec center')
    gridSpec=plt.GridSpec(1,4)
    # axRaw=plt.subplot(1,2,1)
    axRaw=plt.subplot(gridSpec[0,:1])
    raw_gap=10*split_contiRaw.std()
    for ri in range(split_contiRaw.shape[0]):
        plt.plot(np.arange(split_contiRaw.shape[1])/fs,split_contiRaw[ri]+ri*raw_gap,c='k',linewidth=0.7)
    plt.yticks(np.arange(split_contiRaw.shape[0])*raw_gap,chNames)
    for st in split_border_t:
        plt.axvline(st,ls='--',c='C1')
    plt.xlabel('time/s')
    # plt.subplot(1,2,2,sharex=axRaw)
    plt.subplot(gridSpec[0,1:],sharex=axRaw)
    plt.pcolormesh(spec_times,np.arange(all_specs_cat.shape[0]),all_specs_cat,cmap='jet',vmin=0,vmax=2)
    plt.colorbar()
    # plt.pcolormesh(spec_times,np.arange(all_specs_cat.shape[0]),all_specs_cat,cmap='jet')
    for st in split_border_t:
        plt.axvline(st,ls='--',c='C1')
    for fi in range(len(all_specs)):
        plt.axhline((fi+1)*len(spec_freqs)-1,ls='-',c='g')
    plt.yticks((np.arange(len(chNames))+0.5)*len(spec_freqs),chNames)
    center_array=np.array(chns_spec_centers)
    for ci in range(center_array.shape[1]):
        tmp_centers=center_array[:,ci,:]
        tmp_centers[:,1]+=np.arange(len(chNames))*len(spec_freqs)
        plt.plot(tmp_centers[:,0],tmp_centers[:,1],c='r')
        plt.scatter(tmp_centers[:,0],tmp_centers[:,1],c='k')
    plt.xlabel('time/s')
    plt.show()



def return_seg_splitContiHigh(segData, segTime, fs, packTimes):

    batch_data=segData
    # batch_data=scipy.signal.resample_poly(batch_data,2,int(round(2*fs/fs)),axis=-1)
    # batch_data=batch_data-np.mean(batch_data,axis=0)
    batch_data=notch_filt(batch_data,fs,np.arange(50,251,50))
    batch_high=band_filt(batch_data,fs,highpass_freqband)
    batch_t=segTime[0]+np.arange(batch_high.shape[1])/fs
    # print('batchHigh shape',batch_high.shape)

    # extract timeWin signals, concatenate
    inSeg_timeWins = packTimes[(packTimes[:, 0] >= segTime[0]) & (packTimes[:, 1] <= segTime[-1])]

    timeWin_boolVec = np.zeros(len(batch_t))
    tWinLen_list = []
    for tw in inSeg_timeWins:
        twBool = (batch_t >= tw[0]) & (batch_t <= tw[1])
        tWinLen_list.append(len(np.where(twBool)[0]))
        timeWin_boolVec[twBool] = 1

    if len(timeWin_boolVec)==0:
        return np.array([[]]),np.array([]),fs,np.array([])

    split_contiHigh = batch_high[:, timeWin_boolVec > 0.5]
    split_border_t = np.cumsum(tWinLen_list) / fs

    inSeg_index=np.where((packTimes[:, 0] >= segTime[0]) & (packTimes[:, 1] <= segTime[-1]))[0]



    return split_contiHigh,split_border_t,fs,inSeg_index

def return_specCenter_packed(chnSpec, specTime, splitBorder_t):
    # chn_centers: time, index
    split_times_ext = np.array([0] + splitBorder_t.tolist())
    split_timeWins = np.vstack([split_times_ext[:-1], split_times_ext[1:]]).T
    chn_centers = []
    for ti, tw in enumerate(split_timeWins):
        win_spec = chnSpec[:, (specTime > tw[0]) & (specTime < tw[1])]
        win_spec = win_spec ** 3
        norm_weight = win_spec / np.sum(win_spec)
        win_times = specTime[(specTime > tw[0]) & (specTime < tw[1])]
        time_indexs = np.tile(win_times, [chnSpec.shape[0], 1])
        freq_indexs = np.tile(np.arange(chnSpec.shape[0]), [len(win_times), 1]).T
        center_time = np.sum(norm_weight * time_indexs)
        # center_freq_index = int(np.sum(norm_weight * freq_indexs))
        center_freq_index = np.sum(norm_weight * freq_indexs)
        chn_centers.append((center_time, center_freq_index))

    return chn_centers


def return_massCenterPat(contiHigh, splitBorder_t, fs):
    spec_times = None
    spec_freqs = None
    all_specs = []
    for chi in range(contiHigh.shape[0]):
        tmp_data = contiHigh[chi]
        tmp_f, tmp_t, tmp_spec = spectrogram(contiHigh[chi], fs, window='hamming', nperseg=int(specWinLen * fs),
                                             noverlap=int(0.8*specWinLen* fs), nfft=int(specWinLen* fs), mode='magnitude')
        # tmp_norm_spec=(tmp_spec-np.mean(tmp_spec,axis=1,keepdims=True))/np.std(tmp_spec,axis=1,keepdims=True)
        # tmp_norm_spec=gaussian_filter(tmp_norm_spec,sigma=1.5)
        tmp_norm_spec = tmp_spec
        # tmp_norm_spec=(tmp_norm_spec)/np.std(tmp_norm_spec,axis=1,keepdims=True)
        tmp_norm_spec = gaussian_filter(tmp_norm_spec, sigma=1.5)
        hg_fIndex = (tmp_f > specFR[0]) & (tmp_f < specFR[1])
        tmp_norm_spec = tmp_norm_spec[hg_fIndex, :]
        tmp_f = tmp_f[hg_fIndex]

        spec_times = tmp_t
        spec_freqs = tmp_f
        all_specs.append(tmp_norm_spec)

    # plt.pcolormesh(spec_times,spec_freqs,all_specs[0])
    # for st in splitBorder_t:
    #     plt.axvline(st)
    # plt.show()
    chns_spec_centers = []
    for si, sp in enumerate(all_specs):
        chn_annot = return_specCenter_packed(sp, spec_times, splitBorder_t)
        chns_spec_centers.append(chn_annot)

    center_lagPat = [np.array(x)[:, 0] for x in chns_spec_centers]
    center_lagPat = np.array(center_lagPat)
    center_lagPat_freq=[np.array(x)[:,1] for x in chns_spec_centers]
    center_lagPat_freq=np.array(center_lagPat_freq)
    # center_lagPat_freq=center_lagPat_freq/(len(hg_fIndex)-1)*(specFR[1]-specFR[0])+specFR[0]
    # center_lagPat_freq=center_lagPat_freq/(len(hg_fIndex)-1)*(spec_freqs[-1]-spec_freqs[0])+spec_freqs[0]
    center_lagPat_freq=center_lagPat_freq/(len(spec_freqs)-1)*(spec_freqs[-1]-spec_freqs[0])+spec_freqs[0]
    center_lagPat_rank = np.array([np.argsort(np.argsort(x)) for x in center_lagPat.T]).T

    return center_lagPat,center_lagPat_freq,center_lagPat_rank


def return_per2h_lagPattern(subname,filename):
    file_dir=os.path.dirname(filename)
    base_name=os.path.basename(filename)
    baseName_pre=base_name.split('.')[0]
    # subject_name=filename.split('/')[-2]
    # subject_dropChns=subs_drop_info[subject_name]
    print('file: ',baseName_pre)

    dets_file=os.path.join(file_dir,baseName_pre+'_gpu.npz')

    dets_data=np.load(dets_file,allow_pickle=True)
    file_dets=dets_data['whole_dets']
    file_chnNames=dets_data['chns_names']

    ## load edf data
    edf_data=epilepsiae_block(filename,os.path.join(file_dir,baseName_pre+'.head'))
    fs=edf_data.fs
    fileStartTime=edf_data.headInfo['start_stamp']
    file_tLen = edf_data.headInfo['duration_in_sec']

    chns=edf_data.chn_names
    ecog_chnsBool = np.array([x not in eeg_chns for x in chns])
    ecog_chns = np.array(chns)[ecog_chnsBool]
    dropChns = sub_drop_chns[subname]

    tstarts = np.arange(0, file_tLen, segment_time)
    tranges = np.append(tstarts, file_tLen)
    tranges = np.array([tranges[:-1], tranges[1:]]).T


    refine_counts=None
    for filename in os.listdir(file_dir):
        if filename.split('_')[-1]=='refineGpu.npz':
            refine_dets=np.load(os.path.join(file_dir,filename),allow_pickle=True)
            refine_counts=refine_dets['events_count']
            refine_chnNames=refine_dets['chns_names']
            assert np.all(refine_chnNames==file_chnNames),'chns not matching'

    all_chnCounts=refine_counts
    all_chnCounts_mean=np.mean(all_chnCounts)
    all_chnCounts_std=np.std(all_chnCounts)
    pickChns_index=np.where(all_chnCounts>(all_chnCounts_mean+pickChn_thresh*all_chnCounts_std))[0]

    pickChns_names=file_chnNames[pickChns_index] ### picked chns
    # plt.figure('pick_chns',figsize=(20,10))
    # plt.bar(np.arange(len(all_chnCounts)),all_chnCounts)
    # plt.axhline(all_chnCounts_mean+pickChn_thresh*all_chnCounts_std)
    # for ci,chN in enumerate(file_chnNames):
    #     plt.text(ci,all_chnCounts[ci],chN,va='bottom',ha='center')
    # plt.savefig(os.path.join(file_dir,'pick_chns.png'))
    plt.close('all')


    # chnsThr=(2/len(pickChns_index))
    chnsThr=0.5

    ## pack groupEvent times
    pickChn_highEvents_times=[file_dets[x] for x in pickChns_index]
    # print(pickChn_highEvents_times)
    # print(len(pickChn_highEvents_times))
    extended_packed_timeWins=get_packedEventsTimes_overThresh(pickChn_highEvents_times,fs=500,ext=extL,chns_num=len(pickChns_index),chns_threh=chnsThr,cut_t=packWinLen)
    if len(extended_packed_timeWins)==0 or (0 in file_dets.shape):
        return
    extended_packed_timeWins=refine_packedEvents_byAllBool(file_dets,extended_packed_timeWins,fs=500,thresh=0.7)
    if len(extended_packed_timeWins)==0:
        return
    extended_packed_timeWins=pick_noOverlap_timeRanges(extended_packed_timeWins,2)
    if len(extended_packed_timeWins)==0:
        return

    eventsBool_matrix=get_packedEvents_bool(pickChn_highEvents_times,extended_packed_timeWins,fs=500)

    file_segsPatRaw_list=[]
    file_segsPatFreq_list=[]
    file_segsPatRank_list=[]
    file_segsEventsBool_list=[]
    file_segsWinTimes_list=[]
    for ti,tr in enumerate(tranges):
        print('seg: ',ti)
        ecogData=edf_data.fetch_data(tr[0],tr[1])
        ecogData=ecogData[ecog_chnsBool]
        ecogData = cp.asarray(ecogData)
        avg_ecogData, avg_chns = avg_rerefAndDrop_eeg(ecogData, ecog_chns, dropChns)
        avg_ecogData = notch_filt_cu(avg_ecogData, fs, np.arange(50, 251, 50))
        avg_t=np.arange(avg_ecogData.shape[1])/fs+tr[0]
        avgPickIndex=np.array([x in pickChns_names for x in avg_chns])
        avgPickChns=avg_chns[avgPickIndex]
        avg_ecogData=avg_ecogData[avgPickIndex]
        avg_ecogData=cp.asnumpy(avg_ecogData)

        if tr[1]-tr[0]<5:
            continue

        seg_splitContiHigh,seg_splitBordersT,seg_splitFs,inSeg_index=return_seg_splitContiHigh(avg_ecogData,avg_t,fs,extended_packed_timeWins)

        if seg_splitContiHigh.shape[1]==0:
            continue

        segLagPattern_raw,segLagPattern_freq,segLagPattern_rank=return_massCenterPat(seg_splitContiHigh,seg_splitBordersT,seg_splitFs)
        file_segsPatRaw_list.append(segLagPattern_raw)
        file_segsPatFreq_list.append(segLagPattern_freq)
        file_segsPatRank_list.append(segLagPattern_rank)
        file_segsEventsBool_list.append(eventsBool_matrix[:,inSeg_index])
        file_segsWinTimes_list.append(extended_packed_timeWins[inSeg_index])
        assert segLagPattern_rank.shape[1]==len(inSeg_index),'group events counts not matched'

        # if segI==1:
        # plt.figure('tmp bool matrix')
        # plt.pcolormesh(eventsBool_matrix[:,inSeg_index])
        plot_perSeg_specCenter(avg_ecogData,avg_t,pickChns_names,fs,extended_packed_timeWins)

    if len(file_segsPatRaw_list)==0:
        return

    file_segsPatRaw=np.concatenate(file_segsPatRaw_list,axis=1)
    file_segsPatFreq=np.concatenate(file_segsPatFreq_list,axis=1)
    file_segsPatRank=np.concatenate(file_segsPatRank_list,axis=1)
    file_segsEventsBool=np.concatenate(file_segsEventsBool_list,axis=1)
    file_segsWinTimes=np.concatenate(file_segsWinTimes_list,axis=0)
    # np.savez(os.path.join(file_dir,baseName_pre+'_lagPat_withFreqCent.npz'),lagPatRaw=file_segsPatRaw,lagPatFreq=file_segsPatFreq,lagPatRank=file_segsPatRank,eventsBool=file_segsEventsBool,chnNames=avgPickChns,start_t=fileStartTime)
    # # np.save(os.path.join(file_dir,baseName_pre+'_packedTimes.npy'),extended_packed_timeWins)
    # np.save(os.path.join(file_dir,baseName_pre+'_packedTimes_withFreqCent.npy'),file_segsWinTimes)
    # # # #



if __name__=='__main__':

    pickChn_thresh = 1  # a.u., pick channels over thresh
    packWinLen = 300e-3  # second, paked time win length


    sub_dirs_dict = {'1096': ['/media/niking314/Backup Plus/epilepsiae/inv/pat_109602/adm_1096102/rec_109600102'],
                     '1084': ['/media/niking314/Backup Plus/epilepsiae/inv/pat_108402/adm_1084102/rec_108400102',
                              '/media/niking314/Backup Plus/epilepsiae/inv/pat_108402/adm_1084102/rec_108401102',
                              '/media/niking314/Backup Plus/epilepsiae/inv/pat_108402/adm_1084102/rec_108402102'],
                     '958': ['/media/niking314/Backup Plus/epilepsiae/inv/pat_95802/adm_958102/rec_95800102',
                             '/media/niking314/Backup Plus/epilepsiae/inv/pat_95802/adm_958102/rec_95801102',
                             '/media/niking314/Backup Plus/epilepsiae/inv/pat_95802/adm_958102/rec_95802102'],
                     '922': ['/media/niking314/Backup Plus/epilepsiae/inv/pat_92202/adm_922102/rec_92200102',
                             '/media/niking314/Backup Plus/epilepsiae/inv/pat_92202/adm_922102/rec_92201102'],
                     '590': ['/media/niking314/Backup Plus/epilepsiae/inv_1_part/pat_59002/adm_590102/rec_59000102',
                             '/media/niking314/Backup Plus/epilepsiae/inv_1_part/pat_59002/adm_590102/rec_59001102'],
                     '1150': ['/media/niking314/Backup Plus/epilepsiae/inv2/pat_115002/adm_1150102/rec_115000102'],
                     '442': ['/media/niking314/Backup Plus/epilepsiae/inv2/pat_44202/adm_442102/rec_44200102',
                             '/media/niking314/Backup Plus/epilepsiae/inv2/pat_44202/adm_442102/rec_44201102',
                             '/media/niking314/Backup Plus/epilepsiae/inv2/pat_44202/adm_442102/rec_44202102',
                             '/media/niking314/Backup Plus/epilepsiae/inv2/pat_44202/adm_442102/rec_44203102', ],
                     '1073': ['/media/niking314/Backup Plus/epilepsiae/inv2/pat_107302/adm_1073102/rec_107300102',
                              '/media/niking314/Backup Plus/epilepsiae/inv2/pat_107302/adm_1073102/rec_107301102',
                              '/media/niking314/Backup Plus/epilepsiae/inv2/pat_107302/adm_1073102/rec_107302102',
                              '/media/niking314/Backup Plus/epilepsiae/inv2/pat_107302/adm_1073102/rec_107303102',
                              '/media/niking314/Backup Plus/epilepsiae/inv2/pat_107302/adm_1073102/rec_107304102', ],

                     # 12 patients
                     '253': ['/media/niking314/Backup Plus/epilepsiae/inv_1_part/pat_25302/adm_253102/rec_25300102',
                             '/media/niking314/Backup Plus/epilepsiae/inv_1_part/pat_25302/adm_253102/rec_25301102'],
                     '1146': ['/media/niking314/Backup Plus/epilepsiae/inv/pat_114602/adm_1146102/rec_114600102'],
                     '916': ['/media/niking314/Backup Plus/epilepsiae/inv/pat_91602/adm_916102/rec_91600102',
                             '/media/niking314/Backup Plus/epilepsiae/inv/pat_91602/adm_916102/rec_91601102',
                             '/media/niking314/Backup Plus/epilepsiae/inv/pat_91602/adm_916102/rec_91602102',
                             '/media/niking314/Backup Plus/epilepsiae/inv/pat_91602/adm_916102/rec_91603102'],
                     '620': ['/media/niking314/Backup Plus/epilepsiae/inv_1_part/pat_62002/adm_620102/rec_62000102',
                             '/media/niking314/Backup Plus/epilepsiae/inv_1_part/pat_62002/adm_620102/rec_62001102',
                             '/media/niking314/Backup Plus/epilepsiae/inv_1_part/pat_62002/adm_620102/rec_62002102'],
                     '583': ['/media/niking314/Backup Plus/epilepsiae/inv_1_part/pat_58302/adm_583102/rec_58300102',
                             '/media/niking314/Backup Plus/epilepsiae/inv_1_part/pat_58302/adm_583102/rec_58301102',
                             '/media/niking314/Backup Plus/epilepsiae/inv_1_part/pat_58302/adm_583102/rec_58304102'],# part 256hz
                     '548': ['/media/niking314/Backup Plus/epilepsiae/inv_1_part/pat_54802/adm_548102/rec_54800102',
                             '/media/niking314/Backup Plus/epilepsiae/inv_1_part/pat_54802/adm_548102/rec_54801102',
                             '/media/niking314/Backup Plus/epilepsiae/inv_1_part/pat_54802/adm_548102/rec_54802102'],
                     '384': ['/media/niking314/Backup Plus/epilepsiae/inv_1_part/pat_38402/adm_384102/rec_38400102',
                             '/media/niking314/Backup Plus/epilepsiae/inv_1_part/pat_38402/adm_384102/rec_38401102'],
                     # ,'/media/niking314/Backup Plus/epilepsiae/inv_1_part/pat_38402/adm_384102/rec_38402102'] # part 256hz
                     '139': ['/media/niking314/Backup%20Plus/epilepsiae/inv2/pat_13902/adm_139102/rec_13901102',
                             '/media/niking314/Backup%20Plus/epilepsiae/inv2/pat_13902/adm_139102/rec_13902102'],# part 256hz
                     '1125': ['/media/niking314/Backup Plus/epilepsiae/inv2/pat_112502/adm_1125102/rec_112500102',
                              '/media/niking314/Backup Plus/epilepsiae/inv2/pat_112502/adm_1125102/rec_112501102',
                              '/media/niking314/Backup Plus/epilepsiae/inv2/pat_112502/adm_1125102/rec_112502102',
                              '/media/niking314/Backup Plus/epilepsiae/inv2/pat_112502/adm_1125102/rec_112503102',
                              '/media/niking314/Backup Plus/epilepsiae/inv2/pat_112502/adm_1125102/rec_112504102', ],
                     '1077': ['/media/niking314/Backup Plus/epilepsiae/inv2/pat_107702/adm_1077102/rec_107700102',
                              '/media/niking314/Backup Plus/epilepsiae/inv2/pat_107702/adm_1077102/rec_107701102', ],
                     '818': ['/media/niking314/Backup Plus/epilepsiae/inv2/pat_81802/adm_818102/rec_81800102',
                             '/media/niking314/Backup Plus/epilepsiae/inv2/pat_81802/adm_818102/rec_81801102',
                             '/media/niking314/Backup Plus/epilepsiae/inv2/pat_81802/adm_818102/rec_81802102', ],
                     '635': ['/media/niking314/Backup Plus/epilepsiae/inv2/pat_63502/adm_635102/rec_63500102'],

                     }

    sub_toDirs_dict = {
        '1096': '/home/niking314/Documents/3_Data/epilepsiae/all_data_lns/1096/all_recs',
        '1084': '/home/niking314/Documents/3_Data/epilepsiae/all_data_lns/1084/all_recs',
        '958': '/home/niking314/Documents/3_Data/epilepsiae/all_data_lns/958/all_recs',
        '922':'/home/niking314/Documents/3_Data/epilepsiae/all_data_lns/922/all_recs',
        '590':'/home/niking314/Documents/3_Data/epilepsiae/all_data_lns/590/all_recs',
        '1150':'/home/niking314/Documents/3_Data/epilepsiae/all_data_lns/1150/all_recs',
        '442':'/home/niking314/Documents/3_Data/epilepsiae/all_data_lns/442/all_recs',
        '1073':'/home/niking314/Documents/3_Data/epilepsiae/all_data_lns/1073/all_recs',

        '253':'/home/niking314/Documents/3_Data/epilepsiae/all_data_lns/253/all_recs',
        '1146':'/home/niking314/Documents/3_Data/epilepsiae/all_data_lns/1146/all_recs',
        '916':'/home/niking314/Documents/3_Data/epilepsiae/all_data_lns/916/all_recs',
        '620':'/home/niking314/Documents/3_Data/epilepsiae/all_data_lns/620/all_recs',
        '583':'/home/niking314/Documents/3_Data/epilepsiae/all_data_lns/583/all_recs',
        '548':'/home/niking314/Documents/3_Data/epilepsiae/all_data_lns/548/all_recs',
        '384':'/home/niking314/Documents/3_Data/epilepsiae/all_data_lns/384/all_recs',
        '139':'/home/niking314/Documents/3_Data/epilepsiae/all_data_lns/139/all_recs',
        '1125':'/home/niking314/Documents/3_Data/epilepsiae/all_data_lns/1125/all_recs',
        '1077':'/home/niking314/Documents/3_Data/epilepsiae/all_data_lns/1077/all_recs',
        '818':'/home/niking314/Documents/3_Data/epilepsiae/all_data_lns/818/all_recs',
        '635':'/home/niking314/Documents/3_Data/epilepsiae/all_data_lns/635/all_recs',


    }

    # sub_list = ['1150','442','1125','1077','1073','818','635']
    # sub_list = ['1096']
    # sub_list = ['253', '1146', '916', '620', '583', '548', '384', '139', '1125', '1077', '818', '635']
    # sub_list = ['1077', '818', '635']
    # sub_list=['635']
    # sub_list=['1096','1084','958','922','590','1150','442','1073','253','1146','916','620','583','548','384','139','1125','1077','818','635']
    sub_list=['590']


    sub_pickT_list={'1096':1,'1084':1,'958':1,'922':1,'590':1,'1150':1,'442':1,'1073':1.5,
                    '253':0.2,'1146':1,'916':1,'620':0.5,'583':0.6,'548':1,'384':1,'139':0.5,'1125':1,'1077':0.5,'818':1,'635':0.5}
    # sub_packWL_list={'1096':250e-3}
    sub_packWL_list={'1096':180e-3,'1084':180e-3,'958':250e-3,'922':200e-3,'590':500e-3,'1150':300e-3,'442':300e-3,'1073':110e-3,
                     '253': 300e-3, '1146': 250e-3, '916': 150e-3, '620': 220e-3, '583': 220e-3, '548': 300e-3, '384': 400e-3, '139': 200e-3, '1125': 150e-3,'1077': 200e-3, '818': 250e-3, '635': 200e-3 }

    for subname in sub_list:
        pickChn_thresh=sub_pickT_list[subname]
        packWinLen=sub_packWL_list[subname]

        i = 0
        for filename in os.listdir(sub_toDirs_dict[subname]):
            if filename.split('.')[-1]=='data':
                fileNum=int(filename.split('.')[0].split('_')[-1])
                # if i<245:
                #     i+=1
                #     continue
                # if fileNum<245:
                #     continue
                # if fileNum not in np.arange(98,102):
                #     continue
                tmp_ts=time.time()
                return_per2h_lagPattern(subname,os.path.join(sub_toDirs_dict[subname],filename))
                print(subname)
                print('# %d done in %.2f '%(i,time.time()-tmp_ts))
                i+=1

