import os
import configparser
import templateCrossCorr as txc
import numpy as np
from NuRadioReco.utilities import units
from NuRadioReco.utilities.io_utilities import read_pickle
from icecream import ic
from A0_Utilities import getMaxChi, getMaxAllChi

def loadSingleTemplate(series):
    # Series should be 200 or 100
    # Loads the first version of a template made for an average energy/zenith
    templates_RCR = f'StationDataAnalysis/templates/reflectedCR_template_{series}series.pkl' # should be exact same as A2.5
    templates_RCR = read_pickle(templates_RCR)
    for key in templates_RCR:
        temp = templates_RCR[key]
    templates_RCR = temp

    return templates_RCR

def loadMultipleTemplates(series, date='3.29.25', addSingle=False, bad=False):
    # Dates 
    #       - 2016 : found backlobe events from 2016
    #       - 3.29.25 : noiseless, 100s and 200s, pruned by-hand for 'good' templates

    # 10.1.24 has issues with the templates, so use 9.16.24
    # Series should be 200 or 100
    # Loads all the templates made for an average energy/zenith
    template_series_RCR = {}
    if not date == '2016':
        template_series_RCR_location = f'DeepLearning/templates/RCR/{date}/' 
        for filename in os.listdir(template_series_RCR_location):
            if filename.startswith(f'{series}s'):
                temp = np.load(os.path.join(template_series_RCR_location, filename))
                template_series_RCR.append(temp)
    else:
        templates_2016_location = f'StationDataAnalysis/templates/confirmed2016Templates/'
        for filename in os.listdir(templates_2016_location):
            temp = np.load(os.path.join(templates_2016_location, filename))
            # Only use the channel of highest amplitude
            max_temp = [0]
            for t in temp:
                if max(np.abs(t)) > max(max_temp):
                    max_temp = t
            key = filename.split('_')[1] # since the filenames are Event2016_1449861609.0_Chi0.68_SNR20.33.npy, we key the UNIX time
            template_series_RCR[key] = max_temp

    if addSingle:
        template_series_RCR.append(loadSingleTemplate(series))

    return template_series_RCR

if __name__ == "__main__":

    config = configparser.ConfigParser()
    config.read('HRAStationDataAnalysis/config.ini')
    date = config['PARAMETERS']['date']


    # data_folder = f'HRAStationDataAnalysis/StationData/nurFiles/{date}/'
    load_path = f'/pub/tangch3/ARIANNA/DeepLearning/new_chi_data/4.4.25/Station14/'

    stations_100s = [13, 15, 18]
    stations_200s = [14, 17, 19, 30]
    stations = {100: stations_100s, 200: stations_200s}

    templates_2016 = loadMultipleTemplates(200, date='2016')
    for file in os.listdir(load_path):
        if file.startswith(f'station14_all_Traces'):
            traces_array = np.load(load_path+file, allow_pickle=True)

            chi_2016 = np.zeros((len(traces_array)))
            chi_RCR = np.zeros((len(traces_array)))
            chi_RCR_bad = np.zeros((len(traces_array)))

            for iT, traces in traces_array:

                chi_2016[iT] = getMaxAllChi(traces, 2*units.GHz, templates_2016, 2*units.GHz)
                # chi_RCR[iT] = getMaxAllChi(traces, 2*units.GHz, template_series, 2*units.GHz)

            print(chi_2016)
            print(len(chi_2016))
    check_chi_2016 = np.load('/pub/tangch3/ARIANNA/DeepLearning/new_chi_data/4.4.25/Station14/station14_all_Chi2016.npy')
    print(check_chi_2016)
    print(len(check_chi_2016))


    # for series in stations.keys():
    #     templates_2016 = loadMultipleTemplates(series, date='2016')    # selection of 2016 events that are presumed to all be backlobes
    #     template_series = loadMultipleTemplates(series)                         # selection of 'good' RCR simulated events for templates
    #     for station_id in stations[series]:
    #         for file in os.listdir(load_path):
    #             if file.startswith(f'station{station_id}_Traces'):
    #                 traces_array = np.load(load_path+file, allow_pickle=True)

    #                 chi_2016 = np.zeros((len(traces_array)))
    #                 chi_RCR = np.zeros((len(traces_array)))
    #                 chi_RCR_bad = np.zeros((len(traces_array)))

    #                 for iT, traces in traces_array:

    #                     chi_2016[iT] = getMaxAllChi(traces, 2*units.GHz, templates_2016, 2*units.GHz)
    #                     chi_RCR[iT] = getMaxAllChi(traces, 2*units.GHz, template_series, 2*units.GHz)

    #                 print(chi_2016)
    #                 print(len(chi_2016))

                    # Save the chi values
                    # np.save(data_folder+file.replace('Traces', 'Chi_2016'), chi_2016)
                    # print(f'Saved {file.replace("Traces", "Chi_2016")}')
                    # np.save(data_folder+file.replace('Traces', 'Chi_RCR'), chi_RCR)
                    # print(f'Saved {file.replace("Traces", "Chi_RCR")}')