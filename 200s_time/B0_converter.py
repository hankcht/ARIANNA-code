import os
import json
import math
import bisect
import datetime
import time
import numpy as np
import argparse
from icecream import ic
import matplotlib
import matplotlib.pyplot as plt
import NuRadioReco
from NuRadioReco.utilities import units
from NuRadioReco.utilities.io_utilities import read_pickle
from NuRadioReco.modules.io import NuRadioRecoio
from NuRadioReco.modules import channelSignalReconstructor
from NuRadioReco.modules import channelBandPassFilter
from NuRadioReco.modules import correlationDirectionFitter
from NuRadioReco.modules import triggerTimeAdjuster
from NuRadioReco.modules import channelLengthAdjuster
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.detector import generic_detector

from A0_Utilities import getMaxChi, getMaxSNR
import templateCrossCorr as txc

channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelSignalReconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()
correlationDirectionFitter = NuRadioReco.modules.correlationDirectionFitter.correlationDirectionFitter()
correlationDirectionFitter.begin(debug=False)

# Need blackout times for high-rate noise regions
def inBlackoutTime(time, blackoutTimes):
    # This check removes data that have bad datetime format. No events should be recorded before 2013 season when the first stations were installed
    if datetime.datetime.fromtimestamp(time) > datetime.datetime(2019, 3, 31):
        return True
    # This check removes events happening during periods of high noise
    for blackouts in blackoutTimes:
        if blackouts[0] < time and time < blackouts[1]:
            return True
    return False

blackoutFile = open('/pub/tangch3/ARIANNA/DeepLearning/BlackoutCuts.json')
blackoutData = json.load(blackoutFile)
blackoutFile.close()

blackoutTimes = []

for iB, tStart in enumerate(blackoutData['BlackoutCutStarts']):
    tEnd = blackoutData['BlackoutCutEnds'][iB]
    blackoutTimes.append([tStart, tEnd])

def getVrms(nurFile, save_chans, station_id, det, check_forced=False, max_check=1000):
    template = NuRadioRecoio.NuRadioRecoio(nurFile)

    Vrms_sum = 0
    num_avg = 0

    for i, evt in enumerate(template.get_events()):
        station = evt.get_station(station_id)
        stationtime = station.get_station_time().unix
        if inBlackoutTime(stationtime, blackoutTimes):
            continue

        channelSignalReconstructor.run(evt, station, det)
        for ChId, channel in enumerate(station.iter_channels(use_channels=save_chans)):
            Vrms_sum += channel[chp.noise_rms]
            num_avg += 1
            
        if num_avg >= max_check:
            break

    return Vrms_sum / num_avg

def converter(nurFile, savename, save_chans, station_id = 1, det=None, BW=[80*units.MHz, 500*units.MHz], normalize=True):
    
    template = NuRadioRecoio.NuRadioRecoio(nurFile)

    # load events
    timeCutTimes, ampCutTimes, deepLearnCutTimes, allCutTimes = np.load(f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/eventsPassingNoiseCuts/timesPassedCuts_FilteredStation{station_id}_TimeCut_1perDay_Amp0.95%.npy', allow_pickle=True)

    # # Normalizing will save traces with values of sigma, rather than voltage
    # if normalize:
    #     Vrms = getVrms(nurFile, save_chans, station_id, det)
    #     print(f'normalizing to {Vrms} vrms')

    #Load 100s/200s template
    templates_RCR = f'../../../../../dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/templates/reflectedCR_template_{amp_type}eries.pkl'
    print(f'amp type: {amp_type}')
    templates_RCR = read_pickle(templates_RCR)
    for key in templates_RCR:
        temp = templates_RCR[key]
    templates_RCR = temp

    PassingCut_SNRs = [] # Circled events
    PassingCut_RCR_Chi = []
    PassingCut_Zen = []
    PassingCut_Azi = []
    PassingCut_Traces = []
    PassingCut_Times = []

    All_SNRs = []
    All_RCR_Chi = []
    forcedMask = []

    All_traces = []
    Unix_time = []

    for i, evt in enumerate(template.get_events()):
        # if i == 1:
        #     break

        #If in a blackout region, skip event
        station = evt.get_station(station_id)
        stationtime = station.get_station_time().unix # This gets station time in unix

        if inBlackoutTime(stationtime, blackoutTimes):
            continue
 
        Unix_time.append(stationtime)
        forcedMask.append(station.has_triggered())

        traces = []
        channelBandPassFilter.run(evt, station, det, passband=[1*units.Hz, 1000*units.MHz], filter_type='butter', order=10)
        for ChId, channel in enumerate(station.iter_channels(use_channels=save_chans)):
            y = channel.get_trace()
            traces.append(y)
        

        # All_SNRs.append(getMaxSNR(traces, noiseRMS=noiseRMS))
        # All_RCR_Chi.append(getMaxChi(traces, 2*units.GHz, templates_RCR, 2*units.GHz))
        All_traces.append(traces)
        
   

        if datetime.datetime.fromtimestamp(stationtime) > datetime.datetime(2019, 1, 1):
            continue
        for goodTime in allCutTimes:
            if datetime.datetime.fromtimestamp(stationtime) == goodTime:
                # correlationDirectionFitter.run(evt, station, det, n_index=1.35, ZenLim=[0*units.deg, 180*units.deg])
                try:
                    correlationDirectionFitter.run(evt, station, det, n_index=1.35)
                except LookupError:
                    print(f'Error for date {datetime.datetime.fromtimestamp(stationtime)}, skipping')
                    continue
                zen = station[stnp.zenith]
                azi = station[stnp.azimuth]

                print(f'found event on good day, plotting')
                # PassingCut_SNRs.append(All_SNRs[-1])
                # PassingCut_RCR_Chi.append(All_RCR_Chi[-1])
                # PassingCut_Zen.append(np.rad2deg(zen)) 
                # PassingCut_Azi.append(np.rad2deg(azi))
                # PassingCut_Traces.append(traces)
                PassingCut_Times.append(stationtime)


    # print(f'number of passed (circled) events is {len(PassingCut_Times)} events')
    # print(f'number of total data events is {len(All_SNRs)} events')
    print(f'total number of UNIX time: {len(Unix_time)}')
    print(f'total number of passing UNIX time: {len(PassingCut_Times)}')

    # print(len(PassingCut_Traces))
    # np.save(f'/pub/tangch3/ARIANNA/DeepLearning/all_data/Station_SNR/{amp_type}/Stn{station_id}_SNR.npy', All_SNRs)
    # np.save(f'/pub/tangch3/ARIANNA/DeepLearning/all_data/Station_Chi/{amp_type}/Stn{station_id}_Chi.npy', All_RCR_Chi)
    # np.save(f'/pub/tangch3/ARIANNA/DeepLearning/all_data/indv_traces/{savename}_Traces.npy', All_traces)
    print('saving unix times')
    np.save(f'/pub/tangch3/ARIANNA/DeepLearning/all_data/indv_time/{savename}_UNIX.npy', Unix_time)
    # np.save(f'/pub/tangch3/ARIANNA/DeepLearning/circled_data/Station_UNIX/{amp_type}/Stn{station_id}_UNIX.npy', PassingCut_Times)


    return All_traces, Unix_time, PassingCut_Times


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Analysis on particular station')
    parser.add_argument('station', type=int, default=17, help='Station to run on')
    parser.add_argument('--single_file', type=str, default=None, help='Single file to run on')
    args = parser.parse_args()
    station_id = args.station
    single_file = args.single_file

    print(f'------> Station {station_id}')

    if station_id in [14, 17, 19, 30]:
        amp_type = '200s'
        noiseRMS = 22.53 * units.mV
    elif station_id in [13, 15, 18]:
        amp_type = '100s'
        noiseRMS = 20 * units.mV

    station_path = f"/dfs8/sbarwick_lab/ariannaproject/station_nur/station_{station_id}/"

    detector = generic_detector.GenericDetector(json_filename=f'/pub/tangch3/ARIANNA/DeepLearning/station_configs/station{station_id}.json', assume_inf=False, antenna_by_depth=False, default_station=station_id)

    if single_file is None:
        DataFiles = []
        for filename in os.listdir(station_path):
            if filename.endswith('_statDatPak.root.nur'):
                continue
            elif not filename.endswith('.nur'):
                continue
            else:
                if os.path.getsize(os.path.join(station_path, filename)) == 0:
                    print(f'File {filename} is empty, skipping')
                    continue
                DataFiles.append(os.path.join(station_path, filename))
        savename = f'Stn{station_id}'
    else:
        DataFiles = [single_file]
        savename = f'stn{station_id}_Data_{single_file.split("/")[-1].replace(".root.nur", "")}'
    saveChannels = [0, 1, 2, 3]

    All_traces, Unix_time, PassingCut_Times = converter(DataFiles, savename, saveChannels, station_id = station_id, det=detector)
    


    





