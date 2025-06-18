import numpy as np
import glob
import os
import argparse
import datetime
from icecream import ic
import matplotlib.pyplot as plt
import matplotlib.colors

# --- Need Max Memory to Run this File ---

def load_parameter_data(base_folder, date_str, station_id, param_name):
    """
    Loads and concatenates parameter data for a given station, date, and parameter name.
    Handles potential .squeeze() issues if data is scalar or nearly scalar after loading parts.
    """
    file_pattern = os.path.join(base_folder, f'{date_str}_Station{station_id}_{param_name}*')
    files = sorted(glob.glob(file_pattern))
    if not files:
        ic(f"Warning: No files found for Station {station_id}, {param_name}, Date {date_str} with pattern {file_pattern}")
        return None

    try:
        data_list = [np.load(f) for f in files]
        # Handle cases where loaded arrays might be 0-dimensional after squeeze in saving
        # or if only one file with one event is loaded.
        
        # Ensure all parts are at least 1D before trying to concatenate more complex structures like Traces
        processed_list = []
        for arr_part in data_list:
            squeezed_part = arr_part.squeeze()
            if squeezed_part.ndim == 0: # Was scalar
                processed_list.append(np.array([squeezed_part.item()]))
            elif param_name == "Traces" and squeezed_part.ndim == 2 and arr_part.shape[0]==1: # Single trace was (1,4,256) squeezed to (4,256)
                 processed_list.append(arr_part) # Keep original shape for single trace
            elif param_name == "Traces" and squeezed_part.ndim == 3 and arr_part.shape[0]==1 : # Single trace already (1,4,256)
                 processed_list.append(arr_part)
            else:
                processed_list.append(squeezed_part if param_name != "Traces" else arr_part)


        if not processed_list:
            return None

        # Concatenate. Special care for Traces if they are already (N, 4, 256)
        if param_name == "Traces":
            # Check if all parts have the expected subsequent dimensions (4, 256)
            if all(p.ndim >=3 and p.shape[1:3] == (4,256) for p in processed_list):
                 concatenated_data = np.concatenate(processed_list, axis=0)
            elif all(p.ndim == 2 and p.shape == (4,256) for p in processed_list): # list of (4,256) from single events
                 concatenated_data = np.stack(processed_list, axis=0)
            else:
                # Try a general concatenate, might fail or produce wrong shape if mixed.
                ic(f"Warning: Traces for {param_name} have inconsistent shapes. Attempting simple concatenation.")
                concatenated_data = np.concatenate([p.reshape(-1, 4, 256) if p.size % (4*256) == 0 and p.size > 0 else np.array([]).reshape(0,4,256) for p in processed_list if p.size > 0], axis=0)

        else: # For 1D parameters like Times, SNR, Chi*
            concatenated_data = np.concatenate(processed_list, axis=0)
        
        return concatenated_data.squeeze() # Final squeeze for 1D arrays if they ended up (N,1)

    except Exception as e:
        ic(f"Error loading or concatenating data for {param_name} at Station {station_id}, Date {date_str}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Filter station data based on Chi2016 thresholds and save selected parameters.")
    parser.add_argument("station_id", type=int, help="Station ID to process.")
    parser.add_argument("date", type=str, help="Date string (e.g., YYYYMMDD) for the data.")
    
    args = parser.parse_args()
    station_id = args.station_id
    date_str = args.date

    ic.enable()
    ic(f"Processing Station {station_id} for Date {date_str}")

    # --- Configuration for Paths ---
    base_input_folder = os.path.join('/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/', date_str)
    base_output_folder = f"/pub/tangch3/ARIANNA/DeepLearning/new_chi_data/4.4.25/Station{station_id}"   # Modify as needed
    
    os.makedirs(base_output_folder, exist_ok=True)

    # --- Parameters to Load and Save ---
    params_to_process = ['Traces', 'SNR', 'Chi2016', 'ChiRCR', 'Times']
    loaded_data_raw = {}

    ic("Loading data...")
    for param in params_to_process:
        data = load_parameter_data(base_input_folder, date_str, station_id, param)
        if data is None or data.size == 0:
            ic(f"Failed to load or data is empty for {param}. Exiting.")
            return
        loaded_data_raw[param] = data
        ic(f"  Loaded {param}: shape {data.shape}")

    # --- Basic Consistency Check (Number of Events) ---
    num_events_initial = -1
    for param, data in loaded_data_raw.items():
        if data is None: # Should have exited above if critical data is None
            ic(f"Critical error: {param} data is None after load attempt. Exiting.")
            return
        current_param_event_count = data.shape[0]
        if num_events_initial == -1:
            num_events_initial = current_param_event_count
        elif num_events_initial != current_param_event_count:
            ic(f"Error: Mismatch in number of events between parameters.")
            ic(f"  Times has {loaded_data_raw.get('Times', np.array([])).shape[0]} events.")
            ic(f"  {param} has {current_param_event_count} events.")
            ic(f"Please check data consistency for Station {station_id}, Date {date_str}. Exiting.")
            return
            
    if num_events_initial == 0:
        ic("No events found in the loaded data. Exiting.")
        return
    ic(f"Successfully loaded data for {num_events_initial} initial events.")

    # --- Initial Time Filtering (Consistent with C00_eventSearchCuts.py) ---
    ic("Applying initial time filters...")
    times_arr = loaded_data_raw['Times']
    
    zerotime_mask = (times_arr != 0)
    min_datetime_threshold = datetime.datetime(2013, 1, 1).timestamp()
    pretime_mask = (times_arr >= min_datetime_threshold)
    
    initial_valid_mask = zerotime_mask & pretime_mask
    
    num_after_initial_filter = np.sum(initial_valid_mask)
    if num_after_initial_filter == 0:
        ic("No events remaining after initial time filtering. Exiting.")
        return
    ic(f"{num_after_initial_filter} events remaining after initial time filters.")

    filtered_data_for_chi_cuts = {}
    for param in params_to_process:
        filtered_data_for_chi_cuts[param] = loaded_data_raw[param][initial_valid_mask]
        ic(f"  {param} after initial filters: shape {filtered_data_for_chi_cuts[param].shape}")

    # --- Save all filtered data ---
    for param in params_to_process:
        all_output_filename = f'station{station_id}_all_{param}.npy'
        all_output_filepath = os.path.join(base_output_folder, all_output_filename)
        np.save(all_output_filepath, filtered_data_for_chi_cuts[param])
        ic(f"  Successfully saved: {all_output_filepath}")



    # --- Chi2016 Thresholding and Saving ---
    chi2016_thresholds = [0.7, 0.65, 0.6]
    base_chi2016_values = filtered_data_for_chi_cuts['Chi2016']

    for chi_thresh in chi2016_thresholds:
        ic(f"\nApplying Chi2016 threshold: >= {chi_thresh}")
        
        # Ensure base_chi2016_values is not empty and is 1D for comparison
        if base_chi2016_values.size == 0 :
            ic(f"  No Chi2016 values to apply threshold {chi_thresh} to (array is empty). Skipping.")
            continue
        
        current_chi_mask = (base_chi2016_values >= chi_thresh)
        num_passed_events = np.sum(current_chi_mask)
        
        ic(f"  {num_passed_events} events passed Chi2016 >= {chi_thresh}")

        if num_passed_events > 0:
            output_data_dict = {}
            for param in params_to_process:
                output_data_dict[param] = filtered_data_for_chi_cuts[param][current_chi_mask]
            
            # Construct filename
            # Replacing '.' in threshold with 'p' for cleaner filenames (e.g., 0.7 -> 0p70)
            thresh_str = f"{chi_thresh:.2f}".replace('.', 'p') 
            output_filename = f"St{station_id}_{date_str}_Chi2016_ge{thresh_str}_{num_passed_events}evts_SelectedData.npy"
            output_filepath = os.path.join(base_output_folder, output_filename)
            
            try:
                np.save(output_filepath, output_data_dict, allow_pickle=True)
                ic(f"  Successfully saved: {output_filepath}")
                for param, data in output_data_dict.items():
                    ic(f"    Saved {param} shape: {data.shape}")
            except Exception as e:
                ic(f"  Error saving file {output_filepath}: {e}")
        else:
            ic(f"  No events to save for Chi2016 >= {chi_thresh}")

    ic(f"\nProcessing complete for Station {station_id}, Date {date_str}.")

def plot_new_chi_data(param, All_SNRs, All_Chi, station_id, plot_folder, extraname=""):
    # --- Plot of all events in Chi-SNR space ---
    plt.hist2d(All_SNRs, All_Chi, bins=[SNRbins, maxCorrBins], norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    plt.xlim((3, 100))
    plt.ylim((0, 1))
    plt.xlabel('SNR')
    plt.ylabel('Avg Chi Highest Parallel Channels')
    # plt.legend()
    plt.xscale('log')
    plt.tick_params(axis='x', which='minor', bottom=True)
    plt.grid(visible=True, which='both', axis='both') 
    plt.title(f'Station {station_id} - SNR vs. Chi (Events: {len(All_SNRs):,})')
    print(f'Saving {plot_folder}/{extraname}SNR-Chi{param}_All_Stn{station_id}.png')
    plt.savefig(f'{plot_folder}/{extraname}SNR-Chi{param}_All_Stn{station_id}.png')
    plt.close()

    return

if __name__ == '__main__':
    # main()


    SNRbins = np.logspace(0.477, 2, num=80)
    maxCorrBins = np.arange(0, 1.0001, 0.01)
    
    stations = [13,15,18,14,17,19,30]
    parameters = ['2016', 'RCR'] # to plot both Chi2016 and ChiRCR4

    for station_id in stations:

        data_directory = f'/pub/tangch3/ARIANNA/DeepLearning/new_chi_data/4.4.25/Station{station_id}'
        plot_output_folder = '/pub/tangch3/ARIANNA/DeepLearning/plots/ChiSNR'
        os.makedirs(plot_output_folder, exist_ok=True)

        All_SNRs = np.load(f'{data_directory}/station{station_id}_all_SNR.npy')
        for param in parameters:
            All_Chi = np.load(f'{data_directory}/station{station_id}_all_Chi{param}.npy')

            ic(f'number of all data is {len(All_SNRs)} and {len(All_Chi)}')

            plot_new_chi_data(param, All_SNRs, All_Chi, station_id, plot_output_folder)