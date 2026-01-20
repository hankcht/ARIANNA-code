from NuRadioReco.modules.io import NuRadioRecoio
from NuRadioReco.detector import detector
from NuRadioReco.utilities import units
import argparse
import numpy as np
import time
import NuRadioReco.modules.channelBandPassFilter

from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import channelParameters as chp
from astropy.time import Time
import datetime

import matplotlib.pyplot as plt

import json
from NuRadioReco.utilities.io_utilities import read_pickle
import templateCrossCorr as txc

import NuRadioReco.modules.io.eventWriter
import os

def inBlackoutTime(time, blackoutTimes):
    for blackouts in blackoutTimes:
        if blackouts[0] < time and time < blackouts[1]:
            return True
    return False

blackoutFile = open('/pub/tangch3/ARIANNA/DeepLearning/code/200s_time/nu_sim/BlackoutCuts.json')
blackoutData = json.load(blackoutFile)
blackoutFile.close()

blackoutTimes = []

for iB, tStart in enumerate(blackoutData['BlackoutCutStarts']):
    tEnd = blackoutData['BlackoutCutEnds'][iB]
    blackoutTimes.append([tStart, tEnd])


parser = argparse.ArgumentParser(description='Run template matching analysis on on data')
parser.add_argument('files', type=str, default='', help='File to run on')
parser.add_argument('--station', type=int, default=14, help='Station data is being ran on, default 14')
parser.add_argument('--output', type=str, default='', help='File to save event writer to')

args = parser.parse_args()
filesToRead = args.files
station_id = args.station
output_filename = args.output



if output_filename == '':
    filename = os.path.basename(filesToRead)    
    os.makedirs(f'/pub/tangch3/ARIANNA/DeepLearning/true_therm_noise/StationDataAnalysis/processedNur/station_{station_id}/', exist_ok=True)
    output_filename = f'/pub/tangch3/ARIANNA/DeepLearning/true_therm_noise/StationDataAnalysis/processedNur/station_{station_id}/'+filename
print(f'saving file to {output_filename}')



fin = NuRadioRecoio.NuRadioRecoio(filesToRead)

eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
eventWriter.begin(output_filename)
os.makedirs(os.path.dirname(output_filename), exist_ok=True)
mode = {'Channels':True, 'ElectricFields':False, 'SimChannels':False, 'SimElectricFields':False}

det = detector.Detector()
det.update(datetime.datetime(2015, 12, 12))
parallelChannels = det.get_parallel_channels(station_id)
noiseRMS = det.get_noise_RMS(station_id, channel_id=0, stage='raw')


channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelBandPassFilter.begin()



forcedNoiseRMS = []
#noiseRMS = 22.53 * units.mV

for iE, evt in enumerate(fin.get_events()):

    station = evt.get_station(station_id)
    channelBandPassFilter.run(evt, station, det, passband=[1*units.Hz, np.inf*units.MHz])	#Remove DC bias


    if not station.has_triggered():
        noises = []
        for parChans in parallelChannels:
            for channel in station.iter_channels(use_channels=parChans):
                cTrace = channel.get_trace()
                channelNoiseRMS = np.sqrt(np.mean(np.square(cTrace)))
                noises.append(channelNoiseRMS)

        forcedNoiseRMS.append(np.mean(noises))

        eventWriter.run(evt)
        

print(f'Noise for station {station_id} is {np.mean(forcedNoiseRMS)}')
quit()