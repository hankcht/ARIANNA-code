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

        nu_xCorr = 0
        cr_xCorr = 0

        # for parChans in parallelChannels:
        #     print(f'parallel channels are: {parChans}')
        #     nu_avgCorr = []
        #     cr_avgCorr = []

        for channel in station.iter_channels(): # use_channels=parChans
            if channel.has_parameter(chp.nu_xcorrelations):
                print(f'Calculating chi values')
                nu_avgCorr.append(np.abs(channel.get_parameter(chp.nu_xcorrelations)))
                cr_avgCorr.append(np.abs(channel.get_parameter(chp.cr_xcorrelations)))
            else:
                nu_avgCorr.append(0)
                cr_avgCorr.append(0)
            
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
                

            nu_avgCorr = np.mean(np.abs(nu_avgCorr))
            cr_avgCorr = np.mean(np.abs(cr_avgCorr))

            if nu_avgCorr > nu_xCorr:
                nu_xCorr = nu_avgCorr
            if cr_avgCorr > cr_xCorr:
                cr_xCorr = cr_avgCorr

        # Store correlation values and forced triggers
        nu_maxCorr.append(nu_xCorr)
        nu_forcedMask.append(station.has_triggered())
        cr_maxCorr.append(cr_xCorr)
        cr_forcedMask.append(station.has_triggered())

nu_maxCorr = np.array(nu_maxCorr)
cr_maxCorr = np.array(cr_maxCorr)
nu_forcedMask = np.array(nu_forcedMask)
cr_forcedMask = np.array(cr_forcedMask)

# Correlation scatter plot (no Chi)
plt.scatter(nu_maxCorr[np.logical_not(nu_forcedMask)], cr_maxCorr[np.logical_not(nu_forcedMask)], facecolors='none', edgecolor='black', label='Non-forced')
plt.scatter(nu_maxCorr[nu_forcedMask], cr_maxCorr[cr_forcedMask], label=f'Forced Events {len(nu_maxCorr[nu_forcedMask])}')
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.xlabel('Nu Corr')
plt.ylabel('CR Corr')
plt.legend()
plt.title(f'Station {station_id} Nu vs CR Correlations')
plt.savefig(f'/pub/tangch3/ARIANNA/DeepLearning/true_therm_noise/StationDataAnalysis/plots/station{station_id}_NuCR_corr.png')
plt.clf()

# Save processed data
data_dump = {
    f'station_{station_id}': {
        'station_id': station_id,
        'nu_xCorr': nu_maxCorr,
        'cr_xCorr': cr_maxCorr,
        'nu_trigMask': nu_forcedMask,
        'cr_trigMask': cr_forcedMask
    }
}

with open(f'/pub/tangch3/ARIANNA/DeepLearning/true_therm_noise/StationDataAnalysis/processedPkl/data_station_{station_id}.pkl', 'wb') as output:
    pickle.dump(data_dump, output)
