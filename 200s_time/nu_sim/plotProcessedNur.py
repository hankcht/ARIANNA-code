import os
import argparse
import numpy as np
import NuRadioReco.modules.io.eventReader
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.detector import detector
from astropy.time import Time
import datetime
import pickle
from NuRadioReco.utilities import units, fft
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
station_events = []
for file in filesToRead:
    eventReader.begin(file)
    print(f'running file {file}')
    
    file_events = []

    for evt in eventReader.run():
        station = evt.get_station(station_id)
        stationTime = station.get_station_time().unix
        if stationTime > endAnalysisTime:
            print(f'skipping event, stationTime > endAnalysisTime')
            continue

        channelBandPassFilter.run(evt, station, det, passband=[1*units.Hz, np.inf*units.MHz])	#Remove DC bias

        event_traces = []

        for channel in station.iter_channels(): 

            trace = channel.get_trace()
            event_traces.append(trace)
            
            # # Save traces for forced triggers only (no SNR/Chi)
            # if tracesPlotted < num_traces:
            #     os.makedirs(f'/pub/tangch3/ARIANNA/DeepLearning/true_therm_noise/StationDataAnalysis/plots/traces/station_{station_id}/', exist_ok=True)
            #     plt.plot(channel.get_times(), trace)
            #     plt.xlabel('ns')
            #     plt.title(f'Stn {station_id} {station.get_station_time().fits}')
            #     plt.savefig(f'/pub/tangch3/ARIANNA/DeepLearning/true_therm_noise/StationDataAnalysis/plots/traces/station_{station_id}/station{station_id}_trace_{station.get_station_time().fits}.png')
            #     print(f'saving to /pub/tangch3/ARIANNA/DeepLearning/true_therm_noise/StationDataAnalysis/plots/traces/station_{station_id}/station{station_id}_trace_{station.get_station_time().fits}.png')
            #     plt.clf()

            #     plt.plot(channel.get_frequencies()/units.MHz, np.abs(channel.get_frequency_spectrum()))
            #     plt.xlabel('Freq (MHz)')
            #     plt.xlim([0, 500])
            #     plt.title(f'Stn {station_id} {station.get_station_time().fits}')
            #     plt.savefig(f'/pub/tangch3/ARIANNA/DeepLearning/true_therm_noise/StationDataAnalysis/plots/traces/station_{station_id}/station{station_id}_freqs_{station.get_station_time().fits}.png')
            #     print(f'saving to /pub/tangch3/ARIANNA/DeepLearning/true_therm_noise/StationDataAnalysis/plots/traces/station_{station_id}/station{station_id}_freqs_{station.get_station_time().fits}.png')
            #     plt.clf()

            #     tracesPlotted += 1

        file_events.append(event_traces)

    station_events.append(file_events)
    if len(station_events) > 20:
        break
event_traces = np.array(event_traces)
file_events = np.array(file_events)
station_events = np.array(station_events)
print(f'size of event {event_traces.shape}')
print(f'size of file {file_events.shape}')
print(f'size of station {station_events.shape}')

def pT(traces, title, saveLoc, sampling_rate=2, show=False, average_fft_per_channel=[]):
    # Sampling rate should be in GHz
    print(f'printing')
    # Important Clarification: In our actual experiment, we receive one data point per 0.5ns, so our duration of 128ns gives 256 data points
    # it is different from here where I organize one data point to one ns and make the total time 256ns (these two are mathematically identical)
    # x = np.linspace(1, int(256 / sampling_rate), num=256)

    trace_len = len(traces[0])  
    x = np.linspace(1, trace_len / sampling_rate, num=trace_len)

    x_freq = np.fft.rfftfreq(len(x), d=(1 / sampling_rate * units.GHz)) / units.MHz

    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(6, 5), sharex=False)
    fmax = 0
    vmax = 0

    for chID, trace in enumerate(traces):
        trace = trace.reshape(len(trace)) 
        freqtrace = np.abs(fft.time2freq(trace, sampling_rate * units.GHz))

        # Plot time-domain trace
        axs[chID][0].plot(x, trace)
        
        # Plot frequency-domain trace and average FFT if provided
        if len(average_fft_per_channel) > 0:
            axs[chID][1].plot(x_freq, average_fft_per_channel[chID], color='gray', linestyle='--')
        axs[chID][1].plot(x_freq, freqtrace)

        # Update fmax and vmax for axis limits
        fmax = max(fmax, max(freqtrace))
        vmax = max(vmax, max(trace))

        # Add grid to each subplot
        axs[chID][0].grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)  # Time-domain grid
        axs[chID][1].grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)  # Frequency-domain grid

    # Set axis labels
    axs[3][0].set_xlabel('time [ns]', fontsize=12)
    axs[3][1].set_xlabel('Frequency [MHz]', fontsize=12)

    for chID, trace in enumerate(traces):
        axs[chID][0].set_ylabel(f'ch{chID}', labelpad=10, rotation=0, fontsize=10)
        axs[chID][0].set_xlim(-3, 260 / sampling_rate)
        axs[chID][1].set_xlim(-3, 1000)
        axs[chID][0].tick_params(labelsize=10)
        axs[chID][1].tick_params(labelsize=10)

        # Set y-axis limits
        axs[chID][0].set_ylim(-vmax * 1.1, vmax * 1.1)
        axs[chID][1].set_ylim(-0.05, fmax * 1.1)

    axs[0][0].tick_params(labelsize=10)
    axs[0][1].tick_params(labelsize=10)
    axs[0][0].set_ylabel(f'ch{0}', labelpad=10, rotation=0, fontsize=10)

    # Final x and y axis limits
    axs[chID][0].set_xlim(-3, 260 / sampling_rate)
    axs[chID][1].set_xlim(-3, 1000)

    # Add a common y-axis label for the entire figure
    fig.text(0.05, 0.5, 'Voltage [V]', ha='right', va='center', rotation='vertical', fontsize=12)

    plt.suptitle(title)

    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(left=0.175) 

    if show:
        plt.show()
    else:
        print(f'saving to {saveLoc}')
        plt.savefig(saveLoc, format='png')

    plt.clf()
    plt.close(fig)

    return

events_to_plot = [0, 1, 2,]   # event indices
file_idx = 0    

plot_dir = f'/pub/tangch3/ARIANNA/DeepLearning/true_therm_noise/StationDataAnalysis/plots/thermal_forced_trigger/station_{station_id}'
os.makedirs(plot_dir, exist_ok=True)


for evt_idx in range(10):
    traces = station_events[file_idx][evt_idx]   # shape: (n_channels, n_samples)

    save_path = os.path.join(
        plot_dir,
        f'station{station_id}_event{evt_idx}_pT.png'
    )

    pT(
        traces=traces,
        title=f'Station {station_id} - Event {evt_idx}',
        saveLoc=save_path,
        sampling_rate=2,   # GHz (matches your assumption)
        show=False
    )

np.save(f'/pub/tangch3/ARIANNA/DeepLearning/true_therm_noise/StationDataAnalysis/plots/thermal_forced_trigger/station_{station_id}/trace_subset.npy', station_events)
    
