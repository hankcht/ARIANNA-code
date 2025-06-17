import os
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from PIL import Image
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, Flatten, Dense
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import templateCrossCorr as txc
import NuRadioReco
from NuRadioReco.utilities import units, fft


    
def getMaxChi(traces, sampling_rate, template_trace, template_sampling_rate, parallelChannels=[[0, 2], [1, 3]]):
    # Parallel channels should be index corresponding to the channel in traces

    maxCorr = []
    for parChans in parallelChannels:
        parCorr = 0
        for chan in parChans:
            xCorr = txc.get_xcorr_for_channel(traces[chan], template_trace, sampling_rate, template_sampling_rate)
            parCorr += np.abs(xCorr)
        maxCorr.append(parCorr / len(parChans))

    return max(maxCorr)

def getMaxSNR(traces, noiseRMS=22.53 * units.mV):

    SNRs = []
    for trace in traces:
        p2p = (np.max(trace) + np.abs(np.min(trace))) * units.V
        SNRs.append(p2p / (2*noiseRMS))
    
    if max(SNRs)==0:
        print(f'zero error')
        SNRs = []
        for trace in traces:
            print(f'trace {trace}')
            p2p = (np.max(trace) + np.abs(np.min(trace))) * units.V
            SNRs.append(p2p / (2*noiseRMS))

    return max(SNRs)

def load_data(type, amp_type, station_id):

    data_folder = f'/pub/tangch3/ARIANNA/DeepLearning/{type}'

    if type == 'All_data':
        print(f'using {type} files')
        All_data_SNR = np.load(f'{data_folder}/Station_SNR/{amp_type}/Stn{station_id}_SNR.npy')
        All_data_Chi = np.load(f'{data_folder}/Station_Chi/{amp_type}/Stn{station_id}_Chi.npy')

        temporary_Traces = []
        for filename in os.listdir(f'{data_folder}/Station_Traces/{amp_type}/'):
            if filename.startswith(f'Stn{station_id}'):
                print(filename)
                temporary_Traces.append(np.load(f'{data_folder}/Station_Traces/{amp_type}/{filename}'))
        
        All_data_Traces = []
        for file in temporary_Traces:
            All_data_Traces.extend(file)
        
        All_data_UNIX = np.load(f'{data_folder}/Station_UNIX/{amp_type}/Stn{station_id}_UNIX.npy')

        return All_data_SNR, All_data_Chi, All_data_Traces, All_data_UNIX
    

    if type == 'AboveCurve_data':
        print(f'using {type}')        
        above_curve_data_SNR = np.load(f'{data_folder}/Station_SNR/{amp_type}/Stn{station_id}_SNR.npy')
        above_curve_data_Chi = np.load(f'{data_folder}/Station_Chi/{amp_type}/Stn{station_id}_Chi.npy')

        temporary_Traces = []
        for filename in os.listdir(f'{data_folder}/Station_Traces/{amp_type}/'):
            if filename.startswith(f'Stn{station_id}'):
                print(filename)
                temporary_Traces.append(np.load(f'{data_folder}/Station_Traces/{amp_type}/{filename}'))
        
        above_curve_data_Traces = []
        for file in temporary_Traces:
            above_curve_data_Traces.extend(file)

        above_curve_data_UNIX = np.load(f'{data_folder}/Station_UNIX/{amp_type}/Stn{station_id}_UNIX.npy')

        return above_curve_data_SNR, above_curve_data_Chi, above_curve_data_Traces, above_curve_data_UNIX
        

    if type == 'Circled_data':
        print(f'using {type}')
        amp_type = '200s'
        Circled_data_SNR = np.load(f'{data_folder}/Station_SNR/{amp_type}/Stn{station_id}_SNR.npy')
        Circled_data_Chi = np.load(f'{data_folder}/Station_Chi/{amp_type}/Stn{station_id}_Chi.npy')

        temporary_Traces = []
        for filename in os.listdir(f'{data_folder}/Station_Traces/{amp_type}/'):
            if filename.startswith(f'Stn{station_id}'):
                print(filename)
                temporary_Traces.append(np.load(f'{data_folder}/Station_Traces/{amp_type}/{filename}'))
        
        Circled_data_Traces = []
        for file in temporary_Traces:
            Circled_data_Traces.extend(file)

        Circled_data_UNIX = np.load(f'{data_folder}/Station_UNIX/{amp_type}/Stn{station_id}_UNIX.npy')

        return Circled_data_SNR, Circled_data_Chi, Circled_data_Traces, Circled_data_UNIX

def load_sim(path, RCR_path, backlobe_path, amp):
    RCR_files = []
    print(f'path {path + RCR_path}')
    for filename in os.listdir(path + RCR_path):
        if filename.startswith(f'FilteredSimRCR_{amp}_'):
            RCR_files.append(path + RCR_path +  filename)
    rcr = np.empty((0, 4, 256))
    for file in RCR_files:
        print(f'RCR file {file}')
        RCR_data = np.load(file)[0:, 0:4]
        print(f'RCR data shape {RCR_data.shape} and RCR shape {rcr.shape}')
        rcr = np.concatenate((rcr, RCR_data))
    
    Backlobes_files = []
    for filename in os.listdir(path + backlobe_path):
        if filename.startswith(f'Backlobe_{amp}_'):
            Backlobes_files.append(path + backlobe_path + filename)
    Backlobe = np.empty((0, 4, 256))
    for file in Backlobes_files:
        print(f'Backlobe file {file}')
        Backlobe_data = np.load(file)[0:, 0:4]
        Backlobe = np.concatenate((Backlobe, Backlobe_data))
    
    # # prints out every byte in this RCR file, was printing only zeros
    # with open('/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/simulatedRCRs/200s_10.30.24/SimRCR_200s_forcedTrue_5214events_part0.npy', mode="rb") as f:
    #     data = f.read()
    #     for c in data:
    #         print(c, end = " ")

    print(f'loaded sim RCR {len(rcr)} Backlobe {len(Backlobe)}')

    return rcr, Backlobe

def pT(traces, title, saveLoc, sampling_rate=2, show=False, average_fft_per_channel=[]):
    # Sampling rate should be in GHz
    print(f'printing')
    # Important Clarification: In our actual experiment, we receive one data point per 0.5ns, so our duration of 128ns gives 256 data points
    # it is different from here where I organize one data point to one ns and make the total time 256ns (these two are mathematically identical)
    x = np.linspace(1, int(256 / sampling_rate), num=256)
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

def combine_plots(plot_path, station_id, type):
    
    plot_part_1 = Image.open(f'{plot_path}/All_part_1_{type}NO.png')
    plot_part_2 = Image.open(f'{plot_path}/All_part_2_{type}NO.png')

    # We might take different partitions, so the general form applies
    combined_width = plot_part_1.width + plot_part_2.width
    combined_height = max(plot_part_1.height, plot_part_2.height)


    if station_id == 17:
        plot_part_3 = Image.open(f'{plot_path}/All_part_3_{type}NO.png')
        combined_width += plot_part_3.width
        combined_height = max(plot_part_1.height, plot_part_2.height, plot_part_3.height)       


    combined = Image.new('RGB', (combined_width, combined_height))

    combined.paste(plot_part_1, (0, 0))
    combined.paste(plot_part_2, (plot_part_1.width, 0))

    if station_id == 17:
        combined.paste(plot_part_3, (plot_part_1.width + plot_part_2.width, 0))

    combined.save(f'{plot_path}/All_{type}NO_plot.png')

def RunTrainedModel(events, model_path):

    # window size 10 on first layer
    # FIRST TRY data_data_2025-01-10_22-31_RCR_Backlobe_model_2Layer.h5
    # BEST      data_data_2025-01-28_22-52_RCR_Backlobe_model_2Layer.h5
    # CURRENT /pub/tangch3/ARIANNA/DeepLearning/models/200s_time/data_data_2025-01-30_16-38_RCR_Backlobe_model_2Layer.h5
    
    # window size 30 on first layer
    # 
    # CURRENT DeepLearning/models/200s_time/data_data_2025-01-30_21-38_RCR_Backlobe_model_2Layer.h5

    model = keras.models.load_model(f'{model_path}200s_time/data_data_2025-01-30_16-38_RCR_Backlobe_model_2Layer.h5')
    prob_events = model.predict(events)
    

    return prob_events


def load_new_chi(load_path, station_numbers, file_types, thresholds, single_files=None):
    """
    Returns:
        dict: A dictionary where keys are constructed filenames (e.g., 'Stn14_Chi2016_ge0p60')
              and values are the loaded NumPy arrays.
    """
    loaded_data = {}
    print(f"Loading data from: {load_path}\n")

    if single_files:
        for single_file in single_files:
            file_path = os.path.join(load_path, single_file)
            dict_key = os.path.splitext(single_file)[0] # Remove .npy extension for key
            try:
                loaded_data[dict_key] = np.load(file_path)
                print(f"Loaded {single_file}. Shape: {loaded_data[dict_key].shape}")
            except FileNotFoundError:
                print(f"Warning: {single_file} not found at {file_path}")
            except Exception as e:
                print(f"Error loading {single_file}: {e}")

    # Load files based on the station, file_type, and threshold patterns
    for stn_num in station_numbers:
        for file_type in file_types:
            for threshold in thresholds:
                filename = f"Stn{stn_num}_{file_type}_ge0p{threshold}.npy"
                file_path = os.path.join(load_path, filename)
                dict_key = f"Stn{stn_num}_{file_type}_ge0p{threshold}"

                try:
                    loaded_data[dict_key] = np.load(file_path)
                    print(f"Loaded {filename}. Shape: {loaded_data[dict_key].shape}")
                except FileNotFoundError:
                    print(f"Warning: {filename} not found at {file_path}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    
    print("\n--- Data loading complete ---")
    # access using loaded_data['Stn{stn_num}_{file_type}_ge0p{threshold}']
    return loaded_data

if __name__ == "__main__":

    # # profiling method:
    # start = time.time()

    # end = time.time()
    # print(f"task: {end - start}s")
    # start = end

    # # to delete files:
    # directory = '/pub/tangch3/ARIANNA/DeepLearning/logs'
    # for i in range(154):
    #     files_to_delete = glob.glob(os.path.join(directory, f'Stn17_{i}.out'))

    #     for file in files_to_delete:
    #         os.remove(file)
    #         print(f'Deleted :{file}')

    # load_path = '/pub/tangch3/ARIANNA/DeepLearning/new_chi_data/4.4.25/'
    # station_numbers = [13, 15, 18, 14, 17, 19, 30]
    # file_types = ['Chi2016', 'ChiRCR', 'SNR', 'Times', 'Traces']
    # thresholds = ['60', '65', '70']
    # new_chi_dict = load_new_chi(load_path, station_numbers, file_types, thresholds)

    # plot_folder = f'/pub/tangch3/ARIANNA/DeepLearning/plots/ChiSNR/4.4.25/' 
    # Path(plot_folder).mkdir(parents=True, exist_ok=True)

    SNRbins = np.logspace(0.477, 2, num=80)
    maxCorrBins = np.arange(0, 1.0001, 0.01)


    # for station_id in station_numbers:
    #     for threshold in thresholds:
    #         snr_key = f"Stn{station_id}_SNR_ge0p{threshold}"
    #         chir_key = f"Stn{station_id}_Chi2016_ge0p{threshold}"

    #         current_snrs = None
    #         current_rcr_chi = None

    #         if snr_key in new_chi_dict:
    #             current_snrs = new_chi_dict[snr_key]
    #         else:
    #             print(f"Warning: {snr_key} not found. Skipping for station {station_id}, threshold {threshold}.")
    #             continue 

    #         if chir_key in new_chi_dict:
    #             current_rcr_chi = new_chi_dict[chir_key]
    #         else:
    #             print(f"Warning: {chir_key} not found. Skipping for station {station_id}, threshold {threshold}.")
    #             continue 

    #         if current_snrs.shape != current_rcr_chi.shape:
    #             print(f"Error: Mismatched shapes for {snr_key} ({current_snrs.shape}) and {chir_key} ({current_rcr_chi.shape}). Skipping.")
    #             continue

    #         plt.figure(figsize=(10, 8)) 
            
    #         plt.hist2d(current_snrs, current_rcr_chi, bins=[SNRbins, maxCorrBins],
    #                 norm=matplotlib.colors.LogNorm(), cmap='viridis')
    #         plt.colorbar(label='Count (log scale)')
    #         plt.xlim((3, 100))
    #         plt.ylim((0, 1))
    #         plt.xlabel('SNR')
    #         plt.ylabel('Avg Chi Highest Parallel Channels')
    #         plt.xscale('log') 
    #         plt.tick_params(axis='x', which='minor', bottom=True) 
    #         plt.grid(visible=True, which='both', axis='both', linestyle=':', alpha=0.7) 
    #         plt.title(f'Station {station_id} - Threshold: {threshold} (Events: {len(current_snrs):,})')

    #         output_filename = f'ChiSNR_Stn{station_id}_ge0p{threshold}.png'
    #         save_path = os.path.join(plot_folder, output_filename)
    #         print(f'Saving {save_path}')
    #         plt.savefig(save_path, bbox_inches='tight')
    #         plt.close() 

    # print("\nAll station and threshold plots generated and saved.")

    # station_ids_for_plotting = [13,15,18]
    # plot_output_folder = f'/pub/tangch3/ARIANNA/DeepLearning/plots/ChiSNR/'
    # for current_station_id in station_ids_for_plotting:
    #     # Call your load_data function. It returns 4 values, but we only need the first two here.
    #     # We use _ for the values we don't need (Traces and UNIX).
    #     all_data_snr, all_data_chi, _, _ = load_data('All_data', '100s', current_station_id)
    #     print(f'total num of events is {len(all_data_chi)}')

    #     # Check if the required data was successfully loaded and is not empty
    #     if all_data_snr is not None and all_data_chi is not None and len(all_data_snr) > 0:
    #         # Ensure SNR and Chi arrays have the same number of events for plotting
    #         if len(all_data_snr) != len(all_data_chi):
    #             print(f"Warning: Mismatched event counts for Station {current_station_id}. SNR: {len(all_data_snr)}, Chi: {len(all_data_chi)}. Skipping plot.")
    #             continue # Skip this station if data lengths don't match

    #         # Create a new figure for each plot
    #         plt.figure(figsize=(10, 8))

    #         # Generate the 2D histogram
    #         plt.hist2d(all_data_snr, all_data_chi, bins=[SNRbins, maxCorrBins],
    #                    norm=matplotlib.colors.LogNorm(), cmap='viridis')

    #         # Add a color bar to show the count scale
    #         plt.colorbar(label='Count (log scale)')

    #         # Set plot limits (matching your previous plot settings)
    #         plt.xlim((3, 100))
    #         plt.ylim((0, 1))

    #         # Add labels and grid
    #         plt.xlabel('SNR')
    #         plt.ylabel('Avg Chi Highest Parallel Channels')
    #         plt.xscale('log') # Set x-axis to logarithmic scale
    #         plt.tick_params(axis='x', which='minor', bottom=True) # Show minor ticks on log scale
    #         plt.grid(visible=True, which='both', axis='both', linestyle=':', alpha=0.7)

    #         # Set the plot title, including the station ID and number of events
    #         plt.title(f'Station {current_station_id} - SNR vs. Chi (Events: {len(all_data_snr):,})')

    #         # Define the filename and save the plot
    #         output_file_name = f'ChiSNR_Stn{current_station_id}_100s.png'
    #         full_save_path = os.path.join(plot_output_folder, output_file_name)
    #         print(f'Saving {full_save_path}')
    #         plt.savefig(full_save_path, bbox_inches='tight') # Use bbox_inches='tight' to prevent labels from being cut off
    #         plt.close() # Close the current figure to free up memory

    #     else:
    #         print(f"No valid data available for Station {current_station_id} with Amp Type 200s. Skipping plot generation.")

    data_directory = '/pub/tangch3/ARIANNA/DeepLearning/new_chi_data/4.4.25/'
    plot_output_folder = '/pub/tangch3/ARIANNA/DeepLearning/plots/ChiSNR/'
    os.makedirs(plot_output_folder, exist_ok=True)


    # Specific base file mappings for Station 14
    station_data_files = {
        14: {
            'snr': 'Station14_SNR_base.npy',
            'chi': 'Station14_Chi2016_base.npy' # Using ChiRCR as the default Chi.
                                            # Change to 'Station14_Chi2016_base.npy' if needed.
        }
    }

    # List of station IDs to process (only Station 14 as per your request)
    station_ids_for_plotting = [14]

    def load_specified_base_data(station_id, data_files_map, data_dir):
        """
        Loads SNR and Chi data from specific base .npy files for a given station.
        """
        if station_id not in data_files_map:
            print(f"Error: No base file definitions found for Station {station_id} in the map.")
            return None, None

        snr_filename = data_files_map[station_id]['snr']
        chi_filename = data_files_map[station_id]['chi']

        snr_filepath = os.path.join(data_dir, snr_filename)
        chi_filepath = os.path.join(data_dir, chi_filename)

        all_data_snr = None
        all_data_chi = None

        try:
            # Load SNR data
            if os.path.exists(snr_filepath):
                all_data_snr = np.load(snr_filepath)
                print(f"Loaded SNR data from: {snr_filepath} (Shape: {all_data_snr.shape})")
            else:
                print(f"Error: SNR base file not found: {snr_filepath}")
                return None, None

            # Load Chi data
            if os.path.exists(chi_filepath):
                all_data_chi = np.load(chi_filepath)
                print(f"Loaded Chi data from: {chi_filepath} (Shape: {all_data_chi.shape})")
            else:
                print(f"Error: Chi base file not found: {chi_filepath}")
                return None, None

        except Exception as e:
            print(f"An error occurred while loading base files for Station {station_id}: {e}")
            return None, None

        return all_data_snr, all_data_chi

    # --- Main Plotting Logic ---
    for current_station_id in station_ids_for_plotting:
        print(f"\nProcessing Station {current_station_id} using specified base files...")

        # Load the SNR and Chi data from the defined base files
        all_data_snr, all_data_chi = load_specified_base_data(current_station_id, station_data_files, data_directory)

        # Proceed only if both datasets are successfully loaded and not empty
        if all_data_snr is not None and all_data_chi is not None and len(all_data_snr) > 0:
            # Crucial check: Ensure both arrays have the same number of events
            if len(all_data_snr) != len(all_data_chi):
                print(f"Warning: Mismatched event counts for Station {current_station_id} base files. "
                    f"SNR events: {len(all_data_snr)}, Chi events: {len(all_data_chi)}. Skipping plot.")
                continue # Skip this station if data lengths don't match

            print(f'Total number of events for Station {current_station_id} is {len(all_data_snr):,}')

            # Create a new figure for each plot to ensure clean separation
            plt.figure(figsize=(10, 8))

            # Generate the 2D histogram (density plot)
            plt.hist2d(all_data_snr, all_data_chi, bins=[SNRbins, maxCorrBins],
                    norm=matplotlib.colors.LogNorm(), cmap='viridis')

            # Add a color bar to indicate the count scale
            plt.colorbar(label='Event Count (log scale)')

            # Set the plot limits for clarity
            plt.xlim((3, 100))
            plt.ylim((0, 1))

            # Add axis labels, logarithmic scale for SNR, and a grid
            plt.xlabel('SNR')
            plt.ylabel('Avg Chi Highest Parallel Channels')
            plt.xscale('log') # Use a logarithmic scale for the SNR axis
            plt.tick_params(axis='x', which='minor', bottom=True) # Show minor ticks on log scale
            plt.grid(visible=True, which='both', axis='both', linestyle=':', alpha=0.7)

            # Set the plot title with station ID and event count
            plt.title(f'Station {current_station_id} - SNR vs. Chi (Base Files - Events: {len(all_data_snr):,})')

            # Define the output filename and save the plot
            # The filename will reflect that it's from base files
            chi_type_for_filename = station_data_files[current_station_id]['chi'].replace('.npy', '').split('_')[-2] # Extracts 'ChiRCR' or 'Chi2016'
            output_file_name = f'ChiSNR_Stn{current_station_id}_BaseFiles_{chi_type_for_filename}.png'
            full_save_path = os.path.join(plot_output_folder, output_file_name)
            print(f'Saving plot to: {full_save_path}')
            plt.savefig(full_save_path, bbox_inches='tight') # Ensures labels aren't cut off
            plt.close() # Close the figure to free up memory

        else:
            print(f"No valid data loaded for Station {current_station_id} from base files. Skipping plot generation.")

    print("\nAll requested base file plots are complete!")

        


        
