import os
import math
import bisect
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from icecream import ic

import NuRadioReco
from NuRadioReco.utilities import units
from NuRadioReco.utilities.io_utilities import read_pickle

import templateCrossCorr as txc
from A0_Utilities import getMaxChi, getMaxSNR, load_data

# curve is a list of tuples (x, y) with x sorted 
# be sure that all x values of blobs are in the curve 
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

def loadTemplate(type='RCR', amp='200s'):
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

def plotSimSNRChi(templates_RCR, noiseRMS, amp, type):

    path = '../../../../../dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/'
    
    RCR_files = []
    if type == 'RCR':
        if amp == '200s':
            simulation_date = '10.30.24' # simulations have weird chi and snr
            print(f'200s simulation date is {simulation_date}')
        elif amp == '100s':
            simulation_date = '10.30.24'
            print(f'100s simulation date is {simulation_date}')

        path += f'simulatedRCRs/{amp}_{simulation_date}/'
        for filename in os.listdir(path):
            if filename.startswith(('SimRCR','FilteredSimRCR')):
                RCR_files.append(os.path.join(path, filename))
                print(filename)
            if filename.startswith(f'SimWeights'):
                RCR_weights_file = os.path.join(path, filename)
    elif type == 'Backlobe':
        simulation_date = '10.25.24'
        path += f'simulatedBacklobes/{amp}_{simulation_date}/'
        for filename in os.listdir(path):
            if filename.startswith(f'Backlobe'):
                RCR_files.append(os.path.join(path, filename))
            if filename.startswith(f'SimWeights'):
                RCR_weights_file = os.path.join(path, filename)

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
    RCR_sim = RCR_sim[sort_order]
    sim_SNRs = sim_SNRs[sort_order]
    sim_Chi = sim_Chi[sort_order]
    sim_weights = sim_weights[sort_order]

    if type == 'RCR':
        cmap = 'seismic'
    else:
        cmap = 'PiYG'

    # plt.scatter(sim_SNRs, sim_Chi, c=sim_weights, cmap=cmap, alpha=0.9, norm=matplotlib.colors.LogNorm())

    return RCR_sim, sim_Chi, sim_SNRs, sim_weights, simulation_date

def plotalldata(plot_folder):
    plt.hist2d(All_data_SNR, All_data_Chi, bins=[SNRbins, maxCorrBins], norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    plt.xlim((3, 100))
    plt.ylim((0, 1))
    plt.xlabel('SNR')
    plt.ylabel('Avg Chi Highest Parallel Channels')
    # plt.legend()
    plt.xscale('log')
    plt.tick_params(axis='x', which='minor', bottom=True)
    plt.grid(visible=True, which='both', axis='both') 
    plt.title(f'Station {station_id}')
    print(f'Saving {plot_folder}/All_stn{station_id}.png')
    plt.savefig(f'{plot_folder}/All_stn{station_id}.png')
    plt.clf()
    
def plotabovecurvedata(plot_folder):
    plt.hist2d(above_curve_data_SNR, above_curve_data_Chi, bins=[SNRbins, maxCorrBins], norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    plt.xlim((3, 100))
    plt.ylim((0, 1))
    plt.xlabel('SNR')
    plt.ylabel('Avg Chi Highest Parallel Channels')
    # plt.legend()
    plt.xscale('log')
    plt.tick_params(axis='x', which='minor', bottom=True)
    plt.grid(visible=True, which='both', axis='both') 
    plt.title(f'Station {station_id}')
    print(f'Saving {plot_folder}/Above_curve_stn{station_id}.png')
    plt.savefig(f'{plot_folder}/2Above_curve_stn{station_id}.png')
    plt.clf()

def plotalldata_withsim(plot_folder):
    plt.hist2d(All_data_SNR, All_data_Chi, bins=[SNRbins, maxCorrBins], norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    plt.scatter(sim_SNRs, sim_chi, c=sim_weights, cmap=cmap, alpha=0.9, norm=matplotlib.colors.LogNorm())
    plt.xlim((3, 100))
    plt.ylim((0, 1))
    plt.xlabel('SNR')
    plt.ylabel('Avg Chi Highest Parallel Channels')
    # plt.legend()
    plt.xscale('log')
    plt.tick_params(axis='x', which='minor', bottom=True)
    plt.grid(visible=True, which='both', axis='both') 
    # plt.figtext(0.48, 0.75, f'RCR efficiency: {RCR_efficiency}%')
    plt.title(f'Station {station_id} - sim date: {simulation_date}')
    print(f'Saving {plot_folder}/All_withSim{RorB}_stn{station_id}.png')
    plt.savefig(f'{plot_folder}/All_withSim{RorB}_stn{station_id}.png')
    plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Analysis on particular station')
    parser.add_argument('station', type=int, default=17, help='Station to run on')
    parser.add_argument('--RCRorBL', type=str, default='RCR', help='using RCR or BL sim')
    args = parser.parse_args()
    station_id = args.station
    RorB = args.RCRorBL

    print(f'Entering ---> Station {station_id}')

    if station_id in [14, 17, 19, 30]:
        amp_type = '200s'
        noiseRMS = 22.53 * units.mV
    elif station_id in [13, 15, 18]:
        amp_type = '100s'
        noiseRMS = 20 * units.mV

    if RorB == 'RCR':
        cmap = 'seismic'
    else:
        cmap = 'PiYG'

    templates_RCR = loadTemplate(type='RCR', amp=amp_type)
    RCR_sim, sim_chi, sim_SNRs, sim_weights, simulation_date = plotSimSNRChi(templates_RCR, noiseRMS, amp=amp_type, type = RorB)
    sim_chi = np.array(sim_chi)

    curve_x = np.linspace(3, 150, len(sim_chi)) #sim_chi same length as sim_SNRs 

    # define BL curves
    def find_curve_13(curve_x):
        curve_y = []
        x1, y1 = 4.2, 0.48
        x2, y2 = 8, 0.6
        x3, y3 = 30, 0.63
        x4, y4 = 40, 0.75

        for x in curve_x:
            if x <= x1:
                curve_y.append(y1)
            elif x1 < x <= x2:
                y = (y2 - y1) / (x2 - x1) * (x - x1) + y1
                curve_y.append(y)
            elif x2 < x <= x3:
                y = (y3 - y2) / (x3 - x2) * (x - x2) + y2  
                curve_y.append(y)
            elif x3 < x <= x4:
                y = (y4 - y3) / (x4 - x3) * (x - x3) + y3  
                curve_y.append(y)
            else:
                curve_y.append(y4)
        
        return curve_y
        
    def find_curve_14(curve_x):
        
        curve_y = []
        x1, y1 = 4.2, 0.48
        x2, y2 = 10, 0.6
        x3, y3 = 20, 0.675
        x4, y4 = 40, 0.625

        for x in curve_x:
            if x <= x1:
                curve_y.append(y1)
            elif x1 < x <= x2:
                y = (y2 - y1) / (x2 - x1) * (x - x1) + y1
                curve_y.append(y)
            elif x2 < x <= x3:
                y = (y3 - y2) / (x3 - x2) * (x - x2) + y2  
                curve_y.append(y)
            else:
                curve_y.append(y3)
        
        return curve_y

    def find_curve_15(curve_x):
        curve_y = []
        x1, y1 = 5, 0.5
        x2, y2 = 7, 0.55
        x3, y3 = 40, 0.7

        for x in curve_x:
            if x <= x1:
                curve_y.append(y1)
            elif x1 < x <= x2:
                y = (y2 - y1) / (x2 - x1) * (x - x1) + y1
                curve_y.append(y)
            elif x2 < x <= x3:
                y = (y3 - y2) / (x3 - x2) * (x - x2) + y2  
                curve_y.append(y)
            else:
                curve_y.append(y3)
        
        return curve_y

    def find_curve_17(curve_x):
        curve_y = []
        x1, y1 = 3.8, 0.6
        x2, y2 = 9, 0.7

        for x in curve_x:
            if x <= x1:
                curve_y.append(y1)
            elif x1 < x <= x2:
                y = (y2 - y1) / (x2 - x1) * (x - x1) + y1
                curve_y.append(y)
            else:
                curve_y.append(y2)

        return curve_y

    def find_curve_18(curve_x):
        curve_y = []
        x1, y1 = 4.5, 0.5
        x2, y2 = 7, 0.57
        x3, y3 = 40, 0.63

        for x in curve_x:
            if x <= x1:
                curve_y.append(y1)
            elif x1 < x <= x2:
                y = (y2 - y1) / (x2 - x1) * (x - x1) + y1
                curve_y.append(y)
            elif x2 < x <= x3:
                y = (y3 - y2) / (x3 - x2) * (x - x2) + y2
                curve_y.append(y)
            else:
                curve_y.append(y3)
        
        return curve_y

    def find_curve_19(curve_x):
        curve_y = []
        x1, y1 = 4, 0.5
        x2, y2 = 14, 0.615

        for x in curve_x:
            if x <= x1:
                curve_y.append(y1)
            elif x1 < x <= x2:
                y = (y2 - y1) / (x2 - x1) * (x - x1) + y1
                curve_y.append(y)
            else:
                curve_y.append(y2)

        return curve_y

    def find_curve_30(curve_x):
        curve_y = []
        x1, y1 = 4.5, 0.53
        x2, y2 = 20, 0.63

        for x in curve_x:
            if x <= x1:
                curve_y.append(y1)
            elif x1 < x <= x2:
                y = (y2 - y1) / (x2 - x1) * (x - x1) + y1
                curve_y.append(y)
            else:
                curve_y.append(y2)

        return curve_y

    curve_functions = {
        13: find_curve_13,
        14: find_curve_14,
        15: find_curve_15,
        17: find_curve_17,
        18: find_curve_18,
        19: find_curve_19,
        30: find_curve_30
    }

    find_curve_func = curve_functions.get(args.station)
    print(curve_functions.get(args.station))
    curve_y = find_curve_func(curve_x)
    curve_y = np.array(curve_y)

    All_data_SNR, All_data_Chi, All_data_Traces, All_data_UNIX = load_data('All_data', amp_type, station_id)
    above_curve_data_SNR, above_curve_data_Chi, above_curve_data_Traces, above_curve_data_UNIX = load_data('AboveCurve_data', amp_type, station_id)

    # Now I want data above the BL curve we defined above
    # returns a list of points where the y value of the blob is greater than the y value of the curve at the blob's x
    # Above_curve_sim = list(filter(lambda p: p[1] >= get_curve_y(curve_x, curve_y, p[0]), zip(sim_SNRs, sim_chi, sim_weights)))
    # Above_curve_sim_x, Above_curve_sim_y, Above_curve_weights = list(zip(*Above_curve_sim))

    Above_curve_data = list(filter(lambda p: p[1] >= get_curve_y(curve_x, curve_y, p[0]), zip(All_data_SNR, All_data_Chi, range(len(All_data_SNR)))))
    Above_curve_data_x, Above_curve_data_y, Above_curve_data_index = list(zip(*Above_curve_data)) 

    # #Calculate RCR efficiency (We actually don't need this, but good for sanity check)
    # RCR_efficiency = sum(Above_curve_weights)/sum(sim_weights)
    # RCR_efficiency = round(RCR_efficiency *100, 4)
    # print(f'{RCR_efficiency}') 

    # plot_folder = f'/pub/tangch3/ARIANNA/DeepLearning/plots/ChiSNR/Station_{station_id}' 
    plot_folder = f'/pub/tangch3/ARIANNA/DeepLearning/plots/ChiSNR/Station_{station_id}'
    # Path(plot_folder).mkdir(parents=True, exist_ok=True)

    def plot_BL_curve():
        plt.plot(curve_x, curve_y, color = 'orange')
        plt.xlim((3,100))
        plt.xscale('log')
        plt.ylim((0,1))
        plt.figtext(0.48, 0.7, f'Above Cut: {len(Above_curve_data_x)} events')

    print(f'creating plots at {plot_folder}')

    SNRbins = np.logspace(0.477, 2, num=80)
    maxCorrBins = np.arange(0, 1.0001, 0.01)

    plot_BL_curve()
    plotalldata(plot_folder)

    plot_BL_curve()
    plotabovecurvedata(plot_folder)

    plot_BL_curve()
    plotalldata_withsim(plot_folder)

    print('Plotting Done!')

    def saveabovecurve_info():
        above_curve_folder = '/pub/tangch3/ARIANNA/DeepLearning/AboveCurve_data'
        np.save(f'{above_curve_folder}/Station_SNR/{amp_type}/Stn{station_id}_SNR.npy', Above_curve_data_x)
        np.save(f'{above_curve_folder}/Station_Chi/{amp_type}/Stn{station_id}_Chi.npy', Above_curve_data_y)

        above_curve_data_Traces = [All_data_Traces[i] for i in Above_curve_data_index]
        np.save(f'{above_curve_folder}/Station_Traces/{amp_type}/Stn{station_id}_Traces.npy', above_curve_data_Traces)

        above_curve_data_UNIX = [All_data_UNIX[i] for i in Above_curve_data_index]
        np.save(f'{above_curve_folder}/Station_UNIX/{amp_type}/Stn{station_id}_UNIX.npy', above_curve_data_UNIX)

        print('Above Curve files SAVED')

    # for ts in BL_cut_station_time_unix:
    #     formatted_time = datetime.datetime.fromtimestamp(ts).strftime("%m-%d-%Y, %H:%M:%S")
    #     BL_cut_station_time_calendar.append(formatted_time)


    print(f'Station {station_id} Done!')

