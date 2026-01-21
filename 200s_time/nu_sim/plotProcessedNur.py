import os
import argparse
import numpy as np
import NuRadioReco.modules.io.eventReader
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.detector import detector
from astropy.time import Time
import datetime
import pickle
from NuRadioReco.utilities import units
import itertools
from NuRadioReco.modules.io import NuRadioRecoio
import NuRadioReco.modules.channelBandPassFilter
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import channelParameters as chp
import NuRadioReco.modules.io.eventWriter

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.colors

color = itertools.cycle(('black', 'blue', 'green', 'orange'))

endAnalysisTime = Time('2019-02-04').unix

parser = argparse.ArgumentParser(description='Run template matching analysis on on data')
parser.add_argument('--station', type=int, default=14, help='Station data is being ran on, default 14')
parser.add_argument('--num_traces', type=int, default=0, help='Number of traces to save, default 0')
parser.add_argument('-files', '--list', type=str, nargs='+', default='', help='File to run on', required=True)

args = parser.parse_args()
filesToRead = args.list
station_id = args.station
num_traces = args.num_traces

det = detector.Detector()
det.update(datetime.datetime(2015, 12, 12))
parallelChannels = det.get_parallel_channels(station_id)
eventReader = NuRadioReco.modules.io.eventReader.eventReader()

channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelBandPassFilter.begin()

nu_maxCorr = []
cr_maxCorr = []
nu_forcedMask = []
cr_forcedMask = []

noise_rms = 20 * units.mV  # RMS used by Chris

tracesPlotted = 0
for file in filesToRead:
    eventReader.begin(file)
    print(f'running file {file}')
    

    for evt in eventReader.run():
        station = evt.get_station(station_id)
        stationTime = station.get_station_time().unix
        if stationTime > endAnalysisTime:
            print(f'skipping event, stationTime > endAnalysisTime')
            continue

        channelBandPassFilter.run(evt, station, det, passband=[1*units.Hz, np.inf*units.MHz])	#Remove DC bias

        # for parChans in parallelChannels:
        #     print(f'parallel channels are: {parChans}')


        for channel in station.iter_channels(): # use_channels=parChans
            
            # Save traces for forced triggers only (no SNR/Chi)
            if tracesPlotted < num_traces:
                os.makedirs(f'/pub/tangch3/ARIANNA/DeepLearning/true_therm_noise/StationDataAnalysis/plots/traces/station_{station_id}/', exist_ok=True)
                plt.plot(channel.get_times(), channel.get_trace())
                plt.xlabel('ns')
                plt.title(f'Stn {station_id} {station.get_station_time().fits}')
                plt.savefig(f'/pub/tangch3/ARIANNA/DeepLearning/true_therm_noise/StationDataAnalysis/plots/traces/station_{station_id}/station{station_id}_trace_{station.get_station_time().fits}.png')
                print(f'saving to /pub/tangch3/ARIANNA/DeepLearning/true_therm_noise/StationDataAnalysis/plots/traces/station_{station_id}/station{station_id}_trace_{station.get_station_time().fits}.png')
                plt.clf()

                plt.plot(channel.get_frequencies()/units.MHz, np.abs(channel.get_frequency_spectrum()))
                plt.xlabel('Freq (MHz)')
                plt.xlim([0, 500])
                plt.title(f'Stn {station_id} {station.get_station_time().fits}')
                plt.savefig(f'/pub/tangch3/ARIANNA/DeepLearning/true_therm_noise/StationDataAnalysis/plots/traces/station_{station_id}/station{station_id}_freqs_{station.get_station_time().fits}.png')
                print(f'saving to /pub/tangch3/ARIANNA/DeepLearning/true_therm_noise/StationDataAnalysis/plots/traces/station_{station_id}/station{station_id}_freqs_{station.get_station_time().fits}.png')
                plt.clf()

                tracesPlotted += 1



