import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from icecream import ic
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
import os
import json
import templateCrossCorr as txc
from matplotlib.markers import MarkerStyle
import math
import bisect
import json
import datetime

channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelSignalReconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()
correlationDirectionFitter = NuRadioReco.modules.correlationDirectionFitter.correlationDirectionFitter()
correlationDirectionFitter.begin(debug=False)


def getMaxChi(traces, sampling_rate, template_trace, template_sampling_rate, parallelChannels=[[0, 2], [1, 3]]):
    #Parallel channels should be index corresponding to the channel in traces

    maxCorr = []
    for parChans in parallelChannels:
        parCorr = 0
        for chan in parChans:
            xCorr = txc.get_xcorr_for_channel(traces[chan], template_trace, sampling_rate, template_sampling_rate)
            parCorr += np.abs(xCorr)
        maxCorr.append(parCorr / len(parChans))

    return max(maxCorr)

def getMaxSNR(traces, noiseRMS=20 * units.mV):

    SNRs = []
    for trace in traces:
        p2p = (np.max(trace) + np.abs(np.min(trace))) * units.V
        SNRs.append(p2p / (2*noiseRMS))

    return max(SNRs)

def loadTemplate(type='RCR', amp='100s'):
    if type == 'RCR':
        if amp == '200s':
                templates_RCR = '../../../../../dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/templates/reflectedCR_template_200series.pkl'
                templates_RCR = read_pickle(templates_RCR)
                for key in templates_RCR:
                    temp = templates_RCR[key]
                templates_RCR = temp
                return templates_RCR
        elif amp == '100s':
                templates_RCR = '../../../../../dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/templates/reflectedCR_template_100series.pkl'
                templates_RCR = read_pickle(templates_RCR)
                for key in templates_RCR:
                    temp = templates_RCR[key]
                templates_RCR = temp
                return templates_RCR

    print(f'{type} {amp} not implemented')
    quit()


def plotSimSNRChi(templates_RCR, noiseRMS, amp='100s', type='RCR'):

    path = '../../../../../dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/'
    simulation_date = '2.9.24' #We have new BL simulation

    RCR_files = []
    if type == 'RCR':
        path += f'simulatedRCRs/{amp}_{simulation_date}/'
        for filename in os.listdir(path):
            if filename.startswith(f'FilteredSimRCR'):
                RCR_files.append(os.path.join(path, filename))
            if filename.startswith(f'SimWeights'):
                RCR_weights_file = os.path.join(path, filename)
    elif type == 'Backlobe':
        path += f'simulatedBacklobes/{amp}_{simulation_date}/'
        for filename in os.listdir(path):
            if filename.startswith(f'Backlobe'):
                RCR_files.append(os.path.join(path, filename))
            if filename.startswith(f'SimWeights'):
                RCR_weights_file = os.path.join(path, filename)

    ic(RCR_files, RCR_weights_file)

    for file in RCR_files:
        RCR_sim = np.load(file)
    RCR_weights = np.load(RCR_weights_file)    

    sim_SNRs = []
    sim_Chi = []
    sim_weights = []
    for iR, RCR in enumerate(RCR_sim):
            
        traces = []
        for trace in RCR:
            traces.append(trace * units.V)
        sim_SNRs.append(getMaxSNR(traces, noiseRMS=noiseRMS))
        sim_Chi.append(getMaxChi(traces, 2*units.GHz, templates_RCR, 2*units.GHz))

        sim_weights.append(RCR_weights[iR])
        
    SNRbins = np.logspace(0.477, 2, num=80)
    maxCorrBins = np.arange(0, 1.0001, 0.01)

    sim_weights = np.array(sim_weights)
    sim_SNRs = np.array(sim_SNRs)
    sim_Chi = np.array(sim_Chi)

    sort_order = sim_weights.argsort()
    sim_SNRs = sim_SNRs[sort_order]
    sim_Chi = sim_Chi[sort_order]
    sim_weights = sim_weights[sort_order]

    if type == 'RCR':
        cmap = 'seismic'
    else:
        cmap = 'PiYG'
    # plt.scatter(sim_SNRs, sim_Chi, c=sim_weights, label=f'Simulated {type}', cmap=cmap, alpha=0.9, norm=matplotlib.colors.LogNorm())
    plt.scatter(sim_SNRs, sim_Chi, c=sim_weights, cmap=cmap, alpha=0.9, norm=matplotlib.colors.LogNorm())

    return (sim_Chi, sim_SNRs, sim_weights)

#curve is a list of tuples (x, y) with x sorted 
#be sure that all x values of blobs are in the curve
def get_curve_y(curve_x, curve_y, snr): 

    right_index = bisect.bisect_left(curve_x, snr)
    if curve_x[right_index - 1] == snr:
        return curve_y[right_index - 1]
    left_x = curve_x[right_index - 1]
    right_x = curve_x[right_index]
    t = snr - left_x
    t /= (right_x - left_x)
    predicted_y = (1-t)*(curve_y[right_index - 1])+t*(curve_y[right_index])
    # print(left_x, right_x, curve_y[right_index - 1], curve_y[right_index], snr, predicted_y)
    return predicted_y

#Need blackout times for high-rate noise regions
def inBlackoutTime(time, blackoutTimes):
    #This check removes data that have bad datetime format. No events should be recorded before 2013 season when the first stations were installed
    if datetime.datetime.fromtimestamp(time) > datetime.datetime(2019, 3, 31):
        return True
    #This check removes events happening during periods of high noise
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

# Originally I only had the Chi, SNR of the data events on the Chi-SNR graph. Now that I have ALL the raw numpy traces, 
# I need to get the indices of good ones above the curve
# Ideally, I would just load all of the data from after the time cut, then make the graphs accordingly, but right now I load the data from 
# two different places. 

def converter(nurFile, folder, type, save_chans, station_id = 1, det=None, plot=False, 
              filter=False, BW=[80*units.MHz, 500*units.MHz], normalize=True, saveTimes=False, timeAdjust=True):
    count = 0
    part = 0
    max_events = 500000
    ary = np.zeros((max_events, 4, 256))
    if saveTimes:
        art = np.zeros(max_events)
    template = NuRadioRecoio.NuRadioRecoio(nurFile)

    timeCutTimes, ampCutTimes, deepLearnCutTimes, allCutTimes = np.load(f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/eventsPassingNoiseCuts/timesPassedCuts_FilteredStation{station_id}_TimeCut_1perDay_Amp0.95%.npy', allow_pickle=True)

    #100s Noise
    noiseRMS = 20 * units.mV

    #Normalizing will save traces with values of sigma, rather than voltage
    if normalize:
        Vrms = getVrms(nurFile, save_chans, station_id, det)
        print(f'normalizing to {Vrms} vrms')

    #Load 100s template
    templates_RCR = '../../../../../dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/templates/reflectedCR_template_100series.pkl'
    templates_RCR = read_pickle(templates_RCR)
    for key in templates_RCR:
        temp = templates_RCR[key]
    templates_RCR = temp

    PassingCut_SNRs = []
    PassingCut_RCR_Chi = []
    PassingCut_Zen = []
    PassingCut_Azi = []
    All_SNRs = []
    All_RCR_Chi = []
    forcedMask = []

    PassingCut_Traces = []
    Unix_time = []

    for i, evt in enumerate(template.get_events()):
        #If in a blackout region, skip event
        station = evt.get_station(station_id)
        stationtime = station.get_station_time().unix #This gets station time in unix

        if inBlackoutTime(stationtime, blackoutTimes):
            continue

        Unix_time.append(stationtime)
        forcedMask.append(station.has_triggered())

        #Checking if event on Chris' golden day
        # if i % 1000 == 0:
        #     print(f'{i} events processed...')
        traces = []
        channelBandPassFilter.run(evt, station, det, passband=[1*units.Hz, 1000*units.MHz], filter_type='butter', order=10)
        for ChId, channel in enumerate(station.iter_channels(use_channels=save_chans)):
            y = channel.get_trace()
            traces.append(y)
        All_SNRs.append(getMaxSNR(traces, noiseRMS=noiseRMS))
        All_RCR_Chi.append(getMaxChi(traces, 2*units.GHz, templates_RCR, 2*units.GHz))

        
        if datetime.datetime.fromtimestamp(stationtime) > datetime.datetime(2019, 1, 1):
            continue
        for goodTime in allCutTimes:
            if datetime.datetime.fromtimestamp(stationtime) == goodTime:
                correlationDirectionFitter.run(evt, station, det, n_index=1.35, ZenLim=[0*units.deg, 180*units.deg])
                zen = station[stnp.zenith]
                azi = station[stnp.azimuth]


                # print(f'found event on good day, plotting')
                PassingCut_SNRs.append(All_SNRs[-1])
                PassingCut_RCR_Chi.append(All_RCR_Chi[-1])
                PassingCut_Zen.append(np.rad2deg(zen)) 
                PassingCut_Azi.append(np.rad2deg(azi))
                PassingCut_Traces.append(traces)

    print(f'number of passed (circled) events is {len(PassingCut_SNRs)} events')
    print(f'number of total data events is {len(All_SNRs)} events')
    print(f'total number of UNIX time: {len(Unix_time)}')

    return All_SNRs, All_RCR_Chi


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Analysis on particular station')
    parser.add_argument('station', type=int, default=19, help='Station to run on')
    args = parser.parse_args()
    station_id = args.station

    if station_id in [14, 17, 19, 30]:
        amp_type = '200s'
        noiseRMS = 22.53 * units.mV
    elif station_id in [13, 15, 18]:
        amp_type = '100s'
        noiseRMS = 20 * units.mV
    templates_RCR = loadTemplate(type='RCR', amp=amp_type)

    # print(f'templates RCR has {templates_RCR}')

    plot_folder = f'/pub/tangch3/ARIANNA/DeepLearning/plots/ChiSNR/Station_{station_id}' 
    Path(plot_folder).mkdir(parents=True, exist_ok=True)


    # data = np.load(f'../../../../../dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/Station{station_id}_SNR_Chi.npy', allow_pickle=True)

    # All_SNRs = data[0]
    # All_RCR_Chi = data[1]
    # All_Azi = data[2]
    # All_Zen = data[3]
    # PassingCut_SNRs = data[4]
    # PassingCut_RCR_Chi = data[5]
    # PassingCut_Azi = data[6]
    # PassingCut_Zen = data[7]

    # print(f'total events: {len(data[4])}')

    #Before we find the passed events, we 1) load data and 2) Implement a time cut

    station_path = f"/dfs8/sbarwick_lab/ariannaproject/station_nur/station_{station_id}/"

    detector = generic_detector.GenericDetector(json_filename=f'/pub/tangch3/ARIANNA/DeepLearning/station_configs/station{station_id}.json', assume_inf=False, antenna_by_depth=False, default_station=station_id)

    DataFiles = []
    for filename in os.listdir(station_path):
        if filename.endswith('_statDatPak.root.nur'):
            continue
        else:
            DataFiles.append(os.path.join(station_path, filename))

        saveChannels = [0, 1, 2, 3]

    All_data_SNRs, All_data_Chi = converter(DataFiles, "2ndpass", f'FilteredStation{station_id}_Data', saveChannels, station_id = station_id, det=detector, filter=True, saveTimes=True, plot=False)

    # SNRbins = np.logspace(0.477, 2, num=80)
    # maxCorrBins = np.arange(0, 1.0001, 0.01)

    # sim_chi, sim_SNRs, sim_weights = plotSimSNRChi(templates_RCR, noiseRMS)
    # print(f'len of sim_chi is {len(sim_chi)}')
    # sim_chi = np.array(sim_chi)

    # # Here we define the BL curve cut for each station
    # curve_x = np.linspace(3, 150, len(sim_chi)) #sim_chi same length as sim_SNRs 

    # if station_id == 13:
    #     #Curve for station 13
        
    #     def find_curve(curve_x):
    #         curve_y = []
            
    #         x1, y1 = 4.2, 0.48
    #         x2, y2 = 8, 0.6
    #         x3, y3 = 30, 0.63
    #         x4, y4 = 40, 0.75

    #         for x in curve_x:
    #             if x <= x1:
    #                 curve_y.append(y1)
    #             elif x1 < x <= x2:
    #                 y = (y2 - y1)/(x2 - x1)*(x - x1) + y1
    #                 curve_y.append(y)
    #             elif x2 < x <= x3:
    #                 y = (y3 - y2)/(x3 - x2)*(x - x2) + y2  
    #                 curve_y.append(y)
    #             elif x3 < x <= x4:
    #                 y = (y4 - y3)/(x4 - x3)*(x - x3) + y3  
    #                 curve_y.append(y)
    #             else:
    #                 curve_y.append(y4)
            
    #         return curve_y
        
    #     curve_y = find_curve(curve_x)

    # elif station_id == 14:

    #     #Curve for station 14
        
    #     def find_curve(curve_x):
    #         curve_y = []
            
    #         x1, y1 = 4.2, 0.48
    #         x2, y2 = 10, 0.6
    #         x3, y3 = 20, 0.675
    #         x4, y4 = 40, 0.625

    #         for x in curve_x:
    #             if x <= x1:
    #                 curve_y.append(y1)
    #             elif x1 < x <= x2:
    #                 y = (y2 - y1)/(x2 - x1)*(x - x1) + y1
    #                 curve_y.append(y)
    #             elif x2 < x <= x3:
    #                 y = (y3 - y2)/(x3 - x2)*(x - x2) + y2  
    #                 curve_y.append(y)
    #             else:
    #                 curve_y.append(y3)
            
    #         return curve_y
        
    #     curve_y = find_curve(curve_x)

    # elif station_id == 15: 

    #     #Curve for station 15
        
    #     def find_curve(curve_x):
    #         curve_y = []
            
    #         x1, y1 = 5, 0.5
    #         x2, y2 = 7, 0.55
    #         x3, y3 = 40, 0.7

    #         for x in curve_x:
    #             if x <= x1:
    #                 curve_y.append(y1)
    #             elif x1 < x <= x2:
    #                 y = (y2 - y1)/(x2 - x1)*(x - x1) + y1
    #                 curve_y.append(y)
    #             elif x2 < x <= x3:
    #                 y = (y3 - y2)/(x3 - x2)*(x - x2) + y2  
    #                 curve_y.append(y)
    #             else:
    #                 curve_y.append(y3)
            
    #         return curve_y
        
    #     curve_y = find_curve(curve_x)

    # elif station_id == 17: 

    #     #Curve for station 17
        
    #     def find_curve(curve_x):
    #         curve_y = []
            
    #         x1, y1 = 3.8, 0.6
    #         x2, y2 = 9, 0.7

    #         for x in curve_x:
    #             if x <= x1:
    #                 curve_y.append(y1)
    #             elif x1 < x <= x2:
    #                 y = (y2 - y1)/(x2 - x1)*(x - x1) + y1
    #                 curve_y.append(y)
    #             else:
    #                 curve_y.append(y2)
            
    #         return curve_y
        
    #     curve_y = find_curve(curve_x)

    # elif station_id == 18: 

    #     #Curve for station 18
        
    #     def find_curve(curve_x):
    #         curve_y = []
            
    #         x1, y1 = 4.5, 0.5
    #         x2, y2 = 7, 0.57
    #         x3, y3 = 40, 0.63

    #         for x in curve_x:
    #             if x <= x1:
    #                 curve_y.append(y1)
    #             elif x1 < x <= x2:
    #                 y = (y2 - y1)/(x2 - x1)*(x - x1) + y1
    #                 curve_y.append(y)
    #             elif x2 < x <= x3:
    #                 y = (y3 - y2)/(x3 - x2)*(x - x2) + y2
    #                 curve_y.append(y)
    #             else:
    #                 curve_y.append(y3)
            
    #         return curve_y
        
    #     curve_y = find_curve(curve_x)

    # elif station_id == 19:

    #     #Curve for station 19
        
    #     def find_curve(curve_x):
    #         curve_y = []
            
    #         x1, y1 = 4, 0.5
    #         x2, y2 = 14, 0.615
    

    #         for x in curve_x:
    #             if x <= x1:
    #                 curve_y.append(y1)
    #             elif x1 < x <= x2:
    #                 y = (y2 - y1)/(x2 - x1)*(x - x1) + y1
    #                 curve_y.append(y)
    #             else:
    #                 curve_y.append(y2)
            
    #         return curve_y
        
    #     curve_y = find_curve(curve_x)


    # elif station_id == 30: 

    #     # #Curve for station 30
        
    #     def find_curve(curve_x):
    #         curve_y = []
            
    #         x1, y1 = 4.5, 0.53
    #         x2, y2 = 20, 0.63

    #         for x in curve_x:
    #             if x <= x1:
    #                 curve_y.append(y1)
    #             elif x1 < x <= x2:
    #                 y = (y2 - y1)/(x2 - x1)*(x - x1) + y1
    #                 curve_y.append(y)
    #             else:
    #                 curve_y.append(y2)
            
    #         return curve_y
        
    #     curve_y = find_curve(curve_x)

    # curve_y = np.array(curve_y)

    # #We can now get events above the curve that we defined for the cut
    # #returns a list of points where the y value of the blob is greater than the y value of the curve at the blob's x
    # filtered_sim = list(filter(lambda p: p[1] >= get_curve_y(curve_x, curve_y, p[0]), zip(sim_SNRs, sim_chi, sim_weights)))
    # filtered_sim_x, filtered_sim_y, filtered_weights = list(zip(*filtered_sim))

    # filtered_data = list(filter(lambda p: p[1] >= get_curve_y(curve_x, curve_y, p[0]), zip(All_data_SNRs, All_data_Chi, range(len(All_data_SNRs)))))
    # filtered_data_x, filtered_data_y, filtered_data_index = list(zip(*filtered_data)) 

    # #Calculate RCR efficiency (We actually don't need this, but good for sanity check)
    # RCR_efficiency = sum(filtered_weights)/sum(sim_weights)
    # RCR_efficiency = round(RCR_efficiency *100, 4)
    # print(f'{RCR_efficiency}')      

    # #Define function that plots curve:
    # def plot_orange_curve(type = 'filtered_data_x'):
    #     if type == 'filtered_data_x':
    #         plt.plot(curve_x, curve_y, color = 'orange')
    #         plt.xlim((3,100))
    #         plt.xscale('log')
    #         plt.ylim((0,1))
    #         # plt.figtext(0.48, 0.75, f'RCR efficiency: {RCR_efficiency}%')
    #         plt.figtext(0.48, 0.7, f'Above Cut: {len(filtered_data_x)} events')
    #     # elif type == 'BL_cut_station_data_SNRs':  
    #     #     plt.plot(curve_x, curve_y, color = 'orange')
    #     #     plt.xlim((3,100))
    #     #     plt.xscale('log')
    #     #     plt.ylim((0,1))
    #     #     plt.figtext(0.48, 0.75, f'RCR efficiency: {RCR_efficiency}%')
    #     #     plt.figtext(0.48, 0.7, f'Above Cut: {len(BL_cut_station_data_SNRs)} events')

    # #plot the good sim cuts
    # # # plot_orange_curve()
    # # plt.scatter(filtered_sim_x, filtered_sim_y, c=filtered_weights, cmap='seismic', alpha=0.9, norm=matplotlib.colors.LogNorm())
    # # plt.xlim((3, 100))
    # # plt.ylim((0, 1))
    # # plt.xlabel('SNR')
    # # plt.ylabel('Avg Chi Highest Parallel Channels')
    # # # plt.legend()
    # # plt.xscale('log')
    # # plt.tick_params(axis='x', which='minor', bottom=True)
    # # plt.grid(visible=True, which='both', axis='both') 
    # # plt.title(f'Station {station_id}')
    # # print(f'Saving {plot_folder}/BLSimcut_stn{station_id}.png')
    # # plt.savefig(f'{plot_folder}/BLSimcut_stn{station_id}.png')
    # # plt.clf()

    # # #plot the good data cuts
    # # # Note: In response to the change of total number of events, the bin values, and color of the whole graph, will be different 
    # # plot_orange_curve()
    # # plt.hist2d(filtered_data_x, filtered_data_y, bins=[SNRbins, maxCorrBins], norm=matplotlib.colors.LogNorm())
    # # plt.colorbar()
    # # plt.xlim((3, 100))
    # # plt.ylim((0, 1))
    # # plt.xlabel('SNR')
    # # plt.ylabel('Avg Chi Highest Parallel Channels')
    # # # plt.legend()
    # # plt.xscale('log')
    # # plt.tick_params(axis='x', which='minor', bottom=True)
    # # plt.grid(visible=True, which='both', axis='both') 
    # # plt.title(f'Station {station_id}')
    # # print(f'Saving {plot_folder}/BLdatacut_stn{station_id}.png')
    # # plt.savefig(f'{plot_folder}/BLdatacut_stn{station_id}.png')
    # # plt.clf()
    
    # #Plot of all events in Chi-SNR space
    # plot_orange_curve()
    # plt.hist2d(All_data_SNRs, All_data_Chi, bins=[SNRbins, maxCorrBins], norm=matplotlib.colors.LogNorm())
    # plt.colorbar()
    # plt.xlim((3, 100))
    # plt.ylim((0, 1))
    # plt.xlabel('SNR')
    # plt.ylabel('Avg Chi Highest Parallel Channels')
    # # plt.legend()
    # plt.xscale('log')
    # plt.tick_params(axis='x', which='minor', bottom=True)
    # plt.grid(visible=True, which='both', axis='both') 
    # plt.title(f'Station {station_id}')
    # print(f'Saving {plot_folder}/BL_curve_stn{station_id}.png')
    # plt.savefig(f'{plot_folder}/BL_curve_stn{station_id}.png')

    # #Plot of sim overlayed on top of all events
    # plotSimSNRChi(templates_RCR, noiseRMS)
    # plt.scatter([], [], color='red', label='Simulated Air Showers')
    # plt.legend()
    # print(f'Saving {plot_folder}/BL_curve_WithSim_stn{station_id}.png')
    # plt.savefig(f'{plot_folder}/BL_curve_WithSim_stn{station_id}.png')
    # plt.clf()


    # # get index of station data above BL cut    
    # # Now we have the indices of data, we can find their traces
    # # load data

    # data_path = f'../../../../../dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/events_prepared_for_model_training/'

    # all_data_files = [] 

    # for filename in os.listdir(data_path):
    #     if filename.startswith(f'FilteredStation{station_id}'):
    #         print(f'appending {filename}')
    #         all_data_files.append(os.path.join(data_path, filename))

    # print(f'size of data: {len(all_data_files)}') 

    # shapes = [np.load(file).shape for file in all_data_files]
    # total_rows = sum(shape[0] for shape in shapes)
    # first_shape = shapes[0][1:]
    # print(f'first shape is: {first_shape}')
    # all_station_data = np.empty((total_rows, *first_shape), dtype=np.float32)

    # start_idx = 0

    # for i, file in enumerate(all_data_files):
    #     data = np.load(file)
    #     num_rows = data.shape[0]

    #     all_station_data[start_idx:start_idx + num_rows] = data
   
    #     start_idx += num_rows   

    # all_data_SNRs = np.array(All_data_SNRs)
    # all_data_Chi = np.array(All_data_Chi)

    # all_filtered_data = list(filter(lambda p: p[1] >= get_curve_y(curve_x, curve_y, p[0]), zip(all_data_SNRs, all_data_Chi, range(len(all_data_SNRs)))))
    # all_filtered_data_x, all_filtered_data_y, all_filtered_data_index = list(zip(*all_filtered_data)) 
    # all_filtered_data_index = np.array(all_filtered_data_index)

    # ic(all_data_SNRs.shape)
    # ic(all_data_Chi.shape)
    # ic(all_station_data.shape)
    # # ic(all_filtered_data_index)

    # BL_cut_station_data_SNRs = all_data_SNRs[all_filtered_data_index]
    # BL_cut_station_data_Chi = all_data_Chi[all_filtered_data_index]
    # BL_cut_station_data = all_station_data[all_filtered_data_index]

    # # time data, same length as cut station data
    # BL_cut_station_times = []

    # print(f'saving: /pub/tangch3/ARIANNA/DeepLearning/AboveCurve_npfiles/Stn{station_id}.npy')
    # np.save(f'/pub/tangch3/ARIANNA/DeepLearning/AboveCurve_npfiles/Stn{station_id}.npy', [BL_cut_station_data, BL_cut_station_times])
    # print(f'total number of old passed events for stn{station_id} is {len(filtered_data_x)} ')
    # print(f'total number of passed events for stn{station_id} is {len(BL_cut_station_data)}')

    # # #ex loading
    # # data = np.load(...)
    # # BL_cut_station_data, BL_cut_station_times = data

    # print('Done!')



