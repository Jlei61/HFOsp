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
from hfo_net import show_events_timeCourse_ext,pick_states_withThresh_usingChnNum,extend_timeRanges,pick_noOverlap_timeRanges,get_packedEventsTimes_overThresh,get_packedEvents_bool
# from cuda_utils import *
from scipy.signal import spectrogram
from scipy.ndimage import gaussian_filter
from p16_subs_info import subs_drop_info
from p16_cuda_24h_bipolar import bipolar_rerefAndDrop_eeg

mne.set_log_level('ERROR')

segment_time=200 #s
resample_to=800 #hz
highpass_freqband=[80,250]
# drop_chns=np.array(['A8'])
drop_chns=np.array([])
pickChn_thresh=1#a.u., pick channels over thresh

extL=30e-3 #second, extend detected event win egdes
packWinLen=500e-3 #second, paked time win length
chnsThr=0.5 #a.u., pack when more-than-Thresh chans have events

specWinLen=0.05 #second, spectrogram time win length
specFR=[50,300]


def split_chnName(chStr):
    split_re=re.search(r"([A-Z]'?)(\d+)",chStr)
    chPre=split_re.group(1)
    chNum=split_re.group(2)
    return chPre,chNum

def bipolar_reref_eeg(data,chn_names):
    chn_splitReulst=list(map(split_chnName,chn_names))
    # print(chn_splitReulst)
    chn_pre_list=[x[0] for x in chn_splitReulst]
    chn_num_list=[x[1] for x in chn_splitReulst]
    chn_pre_set=set(chn_pre_list)
    chn_pre_set=sorted(list(chn_pre_set))
    reref_data_list=[]
    reref_chns_list=[]
    for chnPre in chn_pre_set:
        tmp_chnIndex=np.where(np.array(chn_pre_list)==chnPre)[0]
        tmp_chnNums=np.array([int(chn_num_list[x]) for x in tmp_chnIndex])
        # assert np.all(np.diff(tmp_chnNums)==np.ones(len(tmp_chnNums)-1)),'chnNum not incremental'
        sorted_index=np.argsort(tmp_chnNums)
        tmp_chnIndex=tmp_chnIndex[sorted_index]
        tmp_chnNums=tmp_chnNums[sorted_index]
        tmp_eegData=data[tmp_chnIndex]
        # tmp_reref_data=tmp_eegData.copy()
        reref_data_list.append(tmp_eegData[:-1]-tmp_eegData[1:])
        # reref_chns_list+=[chnPre+str(x)+'-'+chnPre+str(y) for (x,y) in zip(tmp_chnNums[:-1],tmp_chnNums[1:])]
        reref_chns_list+=[chnPre+str(x) for x in tmp_chnNums[:-1]]
    reref_data=np.concatenate(reref_data_list,axis=0)
    reref_chns=np.array(reref_chns_list)
    assert reref_data.shape[0]==len(reref_chns),'data and chnNames not matched'

    return reref_data,reref_chns

def norm_theSpec_toMaxOne(allSpecs, specFreqs,specTime, splitBorder_t):
    # chn_centers: time, index
    split_times_ext = np.array([0] + splitBorder_t.tolist())
    split_timeWins = np.vstack([split_times_ext[:-1], split_times_ext[1:]]).T
    norm_specs=allSpecs.copy()
    # for ti, tw in enumerate(split_timeWins):
    #     win_spec = allSpecs[:, (specTime > tw[0]) & (specTime < tw[1])]
    #     # win_spec=np.log10(win_spec)
    #     # norm_specs[:,(specTime > tw[0]) & (specTime < tw[1])]=(win_spec-win_spec.min())/(win_spec.max()-win_spec.min())
    #     # norm_specs[:,(specTime > tw[0]) & (specTime < tw[1])]=win_spec/win_spec.max()
    #     norm_specs[:,(specTime > tw[0]) & (specTime < tw[1])]=win_spec/np.quantile(win_spec.ravel(),0.99)
    for ti, tw in enumerate(split_timeWins):
        for fi in range(int(round(allSpecs.shape[0]/len(specFreqs)))):
            win_spec = allSpecs[fi*len(specFreqs):(fi+1)*len(specFreqs),:][:, (specTime > tw[0]) & (specTime < tw[1])]
            norm_specs[fi*len(specFreqs):(fi+1)*len(specFreqs),:][:,(specTime > tw[0]) & (specTime < tw[1])]=win_spec/win_spec.max()

    return norm_specs

def plot_perSeg_specCenter(segData,segTime,chNames,fs,packTimes):
    batch_data = segData
    batch_data = scipy.signal.resample_poly(batch_data, 2, int(round(2 * fs / resample_to)), axis=-1)
    # batch_data = batch_data - np.mean(batch_data, axis=0)
    batch_data = notch_filt(batch_data, resample_to, np.arange(50, 251, 50))
    batch_high = band_filt(batch_data, resample_to, highpass_freqband)
    batch_t = segTime[0] + np.arange(batch_high.shape[1]) / resample_to
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
    split_border_t = np.cumsum(tWinLen_list) / resample_to

    # timeWin_boolVec=[]
    # tWinLen_list = []
    # for tw in inSeg_timeWins:
    #     twBool = (batch_t >= tw[0]) & (batch_t <= tw[1])
    #     tWinLen_list.append(len(np.where(twBool)[0]))
    #     # timeWin_boolVec[twBool] = 1
    #     timeWin_boolVec+=list(np.where(twBool)[0])
    # if len(timeWin_boolVec)==0:
    #     return np.array([[]]),np.array([]),resample_to
    #
    # split_contiRaw=batch_data[:,np.array(timeWin_boolVec)]
    # split_contiHigh = batch_high[:, np.array(timeWin_boolVec)]
    # split_border_t = np.cumsum(tWinLen_list) / resample_to



    spec_times = None
    spec_freqs = None
    all_specs = []
    for chi in range(split_contiHigh.shape[0]):
        tmp_data = split_contiHigh[chi]
        tmp_f, tmp_t, tmp_spec = spectrogram(split_contiHigh[chi], resample_to, window='hamming', nperseg=int(specWinLen * resample_to),
                                             noverlap=int(0.8*specWinLen * resample_to), nfft=int(specWinLen* resample_to), mode='magnitude')
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

    # plt.pcolormesh(spec_times,spec_freqs,all_specs[0])
    # for st in splitBorder_t:
    #     plt.axvline(st)
    # plt.show()
    chns_spec_centers = []
    for si, sp in enumerate(all_specs):
        chn_annot = return_specCenter_packed(sp, spec_times, split_border_t)
        chns_spec_centers.append(chn_annot)

    #prepare data for spec center
    all_specs_cat=np.concatenate(all_specs,axis=0)

    normedSpecs_cat = norm_theSpec_toMaxOne(all_specs_cat, spec_freqs, spec_times, split_border_t)

    plt.figure('spec center')
    gridSpec=plt.GridSpec(1,4)
    # axRaw=plt.subplot(gridSpec[0,0])
    # raw_gap=10*split_contiRaw.std()
    # for ri in range(split_contiRaw.shape[0]):
    #     plt.plot(np.arange(split_contiRaw.shape[1])/resample_to,split_contiRaw[ri]+ri*raw_gap,c='k',linewidth=0.7)
    # plt.yticks(np.arange(split_contiRaw.shape[0])*raw_gap,chNames)
    # for st in split_border_t:
    #     plt.axvline(st,ls='--',c='C1')
    # plt.xlabel('time/s')
    axHigh = plt.subplot(gridSpec[0, 0])#,sharex=axRaw)
    high_gap = 10 * split_contiHigh.std()
    for ri in range(split_contiHigh.shape[0]):
        plt.plot(np.arange(split_contiHigh.shape[1]) / resample_to, split_contiHigh[ri] + ri * high_gap, c='k',
                 linewidth=0.7)
    plt.yticks(np.arange(split_contiHigh.shape[0]) * high_gap, chNames)
    for st in split_border_t:
        plt.axvline(st, ls='--', c='C1')
    plt.xlabel('time/s')
    # plt.subplot(1,2,2,sharex=axRaw)
    plt.subplot(gridSpec[0,1:],sharex=axHigh)
    # plt.pcolormesh(spec_times,np.arange(all_specs_cat.shape[0]),all_specs_cat,cmap='jet',vmin=0,vmax=7e-7)
    plt.pcolormesh(spec_times, np.arange(all_specs_cat.shape[0]), normedSpecs_cat, cmap='coolwarm', vmin=0, vmax=1,
                   shading='nearest')
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
    # # preprocess, down sample, avg ref, notch filter, bandfilter
    # batch_data = cp.asarray(segData)
    # batch_data = cusignal.resample_poly(batch_data, 2, int(2 * fs / resample_to), axis=-1)
    # batch_data = batch_data - cp.mean(batch_data, axis=0)
    # batch_data = notch_filt_cu(batch_data, resample_to, np.arange(50, 251, 50))
    # batch_high = band_filt_cu(batch_data, resample_to, highpass_freqband)
    # batch_t = segTime[0] + np.arange(batch_high.shape[1]) / resample_to

    batch_data=segData
    batch_data=scipy.signal.resample_poly(batch_data,2,int(round(2*fs/resample_to)),axis=-1)
    # batch_data=batch_data-np.mean(batch_data,axis=0)
    batch_data=notch_filt(batch_data,resample_to,np.arange(50,251,50))
    batch_high=band_filt(batch_data,resample_to,highpass_freqband)
    batch_t=segTime[0]+np.arange(batch_high.shape[1])/resample_to
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
        return np.array([[]]),np.array([]),resample_to,np.array([])

    split_contiHigh = batch_high[:, timeWin_boolVec > 0.5]
    split_border_t = np.cumsum(tWinLen_list) / resample_to

    inSeg_index=np.where((packTimes[:, 0] >= segTime[0]) & (packTimes[:, 1] <= segTime[-1]))[0]

    # timeWin_boolVec=[]
    # tWinLen_list = []
    # for tw in inSeg_timeWins:
    #     twBool = (batch_t >= tw[0]) & (batch_t <= tw[1])
    #     tWinLen_list.append(len(np.where(twBool)[0]))
    #     # timeWin_boolVec[twBool] = 1
    # #     timeWin_boolVec+=list(np.where(twBool)[0])
    # if len(timeWin_boolVec)==0:
    #     return np.array([[]]),np.array([]),resample_to
    # #
    # split_contiRaw=batch_data[:,np.array(timeWin_boolVec)]
    # split_contiHigh = batch_high[:, np.array(timeWin_boolVec)]
    # split_border_t = np.cumsum(tWinLen_list) / resample_to


    # segLagPattern = return_massCenterPat(split_contiHigh, split_border_t, resample_to)

    return split_contiHigh,split_border_t,resample_to,inSeg_index

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
    center_lagPat_rank = np.array([np.argsort(np.argsort(x)) for x in center_lagPat.T]).T

    return center_lagPat,center_lagPat_rank


def return_per2h_lagPattern(filename):
    file_dir=os.path.dirname(filename)
    base_name=os.path.basename(filename)
    baseName_pre=base_name.split('.')[0]
    subject_name=filename.split('/')[-2]
    subject_dropChns=subs_drop_info[subject_name]
    print('file: ',baseName_pre)

    dets_file=os.path.join(file_dir,baseName_pre+'_gpu.npz')

    dets_data=np.load(dets_file,allow_pickle=True)
    file_dets=dets_data['whole_dets']
    file_chnNames=dets_data['chns_names']

    ## load edf data
    edf_data=mne.io.read_raw_edf(filename,preload=False)
    fs = int(round(edf_data.info['sfreq']))
    fileStartTime=edf_data.info['meas_date']
    fileStartTimeStamp=fileStartTime.timestamp()
    valid_chns_index = return_valid_chan_index(edf_data)
    valid_chns = np.array(edf_data.ch_names)[valid_chns_index]
    valid_chns_st = np.array(list(map(standard_chan_name, valid_chns)))

    # if len(drop_chns)>0:
    #     after_dropChns_stIndex=np.isin(valid_chns_st,drop_chns)
    #     after_dropChns_index=valid_chns_index[after_dropChns_stIndex]
    #     after_dropChns_st=valid_chns_st[after_dropChns_stIndex]
    #
    #     valid_chns_index=after_dropChns_index
    #     valid_chns_st=after_dropChns_st



    # assert np.all(file_chnNames==valid_chns_st),'electrodes not matched'
    time_inter = np.arange(0, edf_data.times[-1], segment_time)
    time_inter = np.append(time_inter, edf_data.times[-1])
    time_ranges = np.array(list(zip(time_inter[:-1], time_inter[1:])))

    ## pick chns
    # files_counts_list=[]
    # for filename in os.listdir(file_dir):
    #     if filename.split('_')[-1]=='gpu.npz':
    #         # npz_files_list.append(filename)
    #         tmp_dets=np.load(os.path.join(file_dir,filename),allow_pickle=True)
    #         tmp_counts=tmp_dets['events_count']
    #         files_counts_list.append(tmp_counts)
    # files_counts_list=np.array(files_counts_list)
    # all_chnCounts=np.sum(files_counts_list,axis=0)
    refine_counts=None
    for filename in os.listdir(file_dir):
        if filename.split('_')[-1]=='refineGpu.npz':
            refine_dets=np.load(os.path.join(file_dir,filename),allow_pickle=True)
            refine_counts=refine_dets['events_count']
            refine_chnNames=refine_dets['chns_names']
            print(file_chnNames)
            print(refine_chnNames)
            assert np.all(refine_chnNames==file_chnNames),'chns not matching'
    all_chnCounts=refine_counts
    all_chnCounts_mean=np.mean(all_chnCounts)
    all_chnCounts_std=np.std(all_chnCounts)
    pickChns_index=np.where(all_chnCounts>(all_chnCounts_mean+pickChn_thresh*all_chnCounts_std))[0]
    print(pickChns_index)
    print(len(pickChns_index))
    pickChns_names=file_chnNames[pickChns_index] ### picked chns
    plt.bar(np.arange(len(all_chnCounts)),all_chnCounts)
    plt.axhline(all_chnCounts_mean+pickChn_thresh*all_chnCounts_std)
    for ci,chN in enumerate(file_chnNames):
        plt.text(ci,all_chnCounts[ci],chN,va='bottom',ha='center')
    # plt.show()
    plt.savefig(os.path.join(file_dir,'pick_chns.png'))
    plt.close('all')


    # chnsThr=(2/len(pickChns_index))
    chnsThr=0.5

    ## pack groupEvent times
    pickChn_highEvents_times=[file_dets[x] for x in pickChns_index]
    # print(pickChn_highEvents_times)
    # print(len(pickChn_highEvents_times))
    extended_packed_timeWins=get_packedEventsTimes_overThresh(pickChn_highEvents_times,fs=500,ext=extL,chns_num=len(pickChns_index),chns_threh=chnsThr,cut_t=packWinLen)
    extended_packed_timeWins=pick_noOverlap_timeRanges(extended_packed_timeWins,2)
    if len(extended_packed_timeWins)==0:
        # return
        # file_segsPatRaw=np.array([[]])
        # file_segsPatRank=np.array([[]])
        # file_segsEventsBool=np.array([[]])
        # file_segsWinTimes=np.array([[]])
        # np.savez(os.path.join(file_dir, baseName_pre + '_lagPat.npz'), lagPatRaw=file_segsPatRaw,
        #          lagPatRank=file_segsPatRank, eventsBool=file_segsEventsBool, chnNames=file_chnNames,
        #          start_t=fileStartTimeStamp)
        # # np.save(os.path.join(file_dir,baseName_pre+'_packedTimes.npy'),extended_packed_timeWins)
        # np.save(os.path.join(file_dir, baseName_pre + '_packedTimes.npy'), file_segsWinTimes)
        return

    eventsBool_matrix=get_packedEvents_bool(pickChn_highEvents_times,extended_packed_timeWins,fs=500)
    # plt.figure('tmp bool matrix')
    # plt.pcolormesh(eventsBool_matrix)


    # cumu_index_vec,packed_timeWins,packed_fs=show_events_timeCourse_ext(pickChn_highEvents_times,fs=500,ext=30e-3)
    # overThresh_packed_timeWins,_=pick_states_withThresh_usingChnNum(cumu_index_vec,packed_timeWins,packed_fs,chns_num=len(pickChns_index),pick_thresh=0.5)
    # extended_packed_timeWins=np.array(overThresh_packed_timeWins)
    # extended_packed_timeWins=extend_timeRanges(overThresh_packed_timeWins,ext_t=50e-3)
    # extended_packed_timeWins=pick_noOverlap_timeRanges(extended_packed_timeWins,less_than=2) ### picked packed timeWins, nX2 array
    # print(extended_packed_timeWins.shape)
    # print(extended_packed_timeWins)
    # print(extended_packed_timeWins[(extended_packed_timeWins[:,0]>0)&(extended_packed_timeWins[:,1]<200)])

    ## construct continuous data (pick chns X timeWins), per dataSegments
    ## compute spectrograms per chan, compute mass center t

    # origin_pickChns_index=valid_chns_index[pickChns_index]
    file_segsPatRaw_list=[]
    file_segsPatRank_list=[]
    file_segsEventsBool_list=[]
    file_segsWinTimes_list=[]
    for segI,tr in enumerate(time_ranges):
        print('seg: ',segI)
        segStart,segEnd=edf_data.time_as_index(tr)
        segData,segTime=edf_data[valid_chns_index,segStart:segEnd]
        # segData,bipolarChns=bipolar_reref_eeg(segData,valid_chns_st)
        segData,bipolarChns=bipolar_rerefAndDrop_eeg(segData,valid_chns_st,subject_dropChns)
        bipolarPickIndex=np.array([x in pickChns_names for x in bipolarChns])
        bipolarPickChns=bipolarChns[bipolarPickIndex]
        segData=segData[bipolarPickIndex]

        if (segTime[-1]-segTime[0])<5:
            continue
        seg_splitContiHigh,seg_splitBordersT,seg_splitFs,inSeg_index=return_seg_splitContiHigh(segData,segTime,fs,extended_packed_timeWins)
        # print('seg_splitContiHigh shape',seg_splitContiHigh.shape)
        if seg_splitContiHigh.shape[1]==0:
            continue
        # print(seg_splitContiHigh)
        # plt.plot(seg_splitContiHigh[0])
        # plt.show()
        segLagPattern_raw,segLagPattern_rank=return_massCenterPat(seg_splitContiHigh,seg_splitBordersT,seg_splitFs)
        file_segsPatRaw_list.append(segLagPattern_raw)
        file_segsPatRank_list.append(segLagPattern_rank)
        file_segsEventsBool_list.append(eventsBool_matrix[:,inSeg_index])
        file_segsWinTimes_list.append(extended_packed_timeWins[inSeg_index])
        assert segLagPattern_rank.shape[1]==len(inSeg_index),'group events counts not matched'

        # if segI==1:
        # plt.figure('tmp bool matrix')
        # plt.pcolormesh(eventsBool_matrix[:,inSeg_index])
        # plot_perSeg_specCenter(segData,segTime,pickChns_names,fs,extended_packed_timeWins)

    if len(file_segsPatRaw_list)==0:
        # return
        # file_segsPatRaw=np.array([[]])
        # file_segsPatRank=np.array([[]])
        # file_segsEventsBool=np.array([[]])
        # file_segsWinTimes=np.array([[]])
        # np.savez(os.path.join(file_dir, baseName_pre + '_lagPat.npz'), lagPatRaw=file_segsPatRaw,
        #          lagPatRank=file_segsPatRank, eventsBool=file_segsEventsBool, chnNames=bipolarPickChns,
        #          start_t=fileStartTimeStamp)
        # # np.save(os.path.join(file_dir,baseName_pre+'_packedTimes.npy'),extended_packed_timeWins)
        # np.save(os.path.join(file_dir, baseName_pre + '_packedTimes.npy'), file_segsWinTimes)
        return

    file_segsPatRaw=np.concatenate(file_segsPatRaw_list,axis=1)
    file_segsPatRank=np.concatenate(file_segsPatRank_list,axis=1)
    file_segsEventsBool=np.concatenate(file_segsEventsBool_list,axis=1)
    file_segsWinTimes=np.concatenate(file_segsWinTimes_list,axis=0)
    # np.savez(os.path.join(file_dir,baseName_pre+'_lagPat.npz'),lagPatRaw=file_segsPatRaw,lagPatRank=file_segsPatRank,eventsBool=file_segsEventsBool,chnNames=bipolarPickChns,start_t=fileStartTimeStamp)
    # # np.save(os.path.join(file_dir,baseName_pre+'_packedTimes.npy'),extended_packed_timeWins)
    # np.save(os.path.join(file_dir,baseName_pre+'_packedTimes.npy'),file_segsWinTimes)


    # plt.pcolormesh(file_segsPatRank)
    # plt.yticks(np.arange(len(pickChns_names))+0.5,pickChns_names)
    # plt.show()
    ## compute time lag & lagPattern



if __name__=='__main__':
    # edf_filename='/home/niking314/Documents/3_Data/yuquan_24h/FA134AX5.edf'
    # m_start=time.time()
    # return_per2h_lagPattern(edf_filename)
    # print('2h lagPat cost:',time.time()-m_start)
    # edf_filesDir='/home/niking314/Documents/3_Data/yuquan_24h/pengzihang_24hResults'



    # edf_filesDir='/home/niking314/Documents/3_Data/yuquan_24h_wholeData/zhangjiaqi'
    # i=0
    # for filename in os.listdir(edf_filesDir):
    #     if filename.split('.')[-1]=='edf':
    #         tmp_ts=time.time()
    #         return_per2h_lagPattern(os.path.join(edf_filesDir,filename))
    #         print('# %d done in %.2f '%(i,time.time()-tmp_ts))
    #         i+=1

    pickChn_thresh = 1  # a.u., pick channels over thresh
    packWinLen = 300e-3  # second, paked time win length

    data_dir='/home/niking314/Documents/3_Data/yuquan_24h_wholeData'
    # sub_list=['zhangkexuan','pengzihang','chengshuai','huangwanling','liyouran','songzishuo','zhangbichen','zhangjiaqi','zhaochenxi','zhaojinrui','zhourongxuan']
    # sub_list=['chengshuai','huangwanling','liyouran','songzishuo','zhangbichen','zhangjiaqi','zhaochenxi','zhaojinrui','zhourongxuan']
    sub_list=['pengzihang']
    sub_pickT_list={'zhangkexuan':0.5,'pengzihang':1,'chengshuai':1,'huangwanling':3,'liyouran':1,'songzishuo':1,'zhangbichen':0.5,'zhangjiaqi':0.7,'zhaochenxi':0.5,\
                    'zhaojinrui':1,'zhourongxuan':1,'sunyuanxin':1}
    sub_packWL_list={'zhangkexuan':300e-3,'pengzihang':230e-3,'chengshuai':500e-3,'huangwanling':300e-3,'liyouran':250e-3,'songzishuo':300e-3,'zhangbichen':300e-3,\
                     'zhangjiaqi':250e-3,'zhaochenxi':300e-3,'zhaojinrui':300e-3,'zhourongxuan':200e-3,'sunyuanxin':400e-3}
    # sub_list=['zhourongxuan']
    for subname in sub_list:
        pickChn_thresh=sub_pickT_list[subname]
        packWinLen=sub_packWL_list[subname]
        i=0
        for filename in os.listdir(os.path.join(data_dir,subname)):
            if filename.split('.')[-1]=='edf':
                tmp_ts=time.time()
                return_per2h_lagPattern(os.path.join(data_dir,subname,filename))
                print(subname)
                print('# %d done in %.2f '%(i,time.time()-tmp_ts))
                i+=1

