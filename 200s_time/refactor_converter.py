"""
Provided by Ryan Rice-Smith Sep 26 2025

Example script demonstrating how to load event parameters for a single station.
This script uses the unified loader methods from C_utils to extract parameter arrays.

ASSUMPTIONS ABOUT FILE STRUCTURE:
- Data files are in: /dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/{date}/
- Cuts files are in: /dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_cuts/{date_cuts}/
- File naming follows pattern: {date}_Station{station_id}_{parameter}*_{num}evts_*.npy

MODIFY the file path templates in load_station_parameter() if your structure differs.
"""

import numpy as np
import os
import re
from C_utils import load_station_events


def build_cuts_path_custom(date_cuts, date_str, station_id, cuts_root):
    """
    Custom function to build cuts file path for the specific directory structure.
    
    Args:
        date_cuts (str): Date string for cuts (e.g., '9.18.25')
        date_str (str): Date string for data (e.g., '9.1.25') 
        station_id (int): Station ID
        cuts_root (str): Root directory for cuts files
    
    Returns:
        str: Full path to cuts file
    """
    return os.path.join(cuts_root, date_cuts, f'{date_str}_Station{station_id}_Cuts.npy')


def load_station_parameter(parameter_name, date_str, station_id, date_cuts, 
                          station_data_root, cuts_root):
    """
    Load a specific parameter for all events from a single station after applying cuts.
    
    Args:
        parameter_name (str): Name of parameter to load ('Traces', 'Times', 'SNR', etc.)
        date_str (str): Date string for data files (e.g., '9.1.25')
        station_id (int): Station ID number (e.g., 13, 14, 15, 17, 18, 19, 30)
        date_cuts (str): Date string for cuts files (e.g., '9.18.25')
        station_data_root (str): Root path to station data directory
        cuts_root (str): Root path to cuts directory
    
    Returns:
        np.array: Array of parameter values for all events that pass cuts
                 Returns empty array if no data found or parameter not available
    """
    
    # --- FILE PATH TEMPLATES (MODIFY THESE FOR DIFFERENT FILE STRUCTURES) ---
    
    # Template for time files - adjust pattern if your files have different naming
    time_files_template = os.path.join(
        station_data_root, "{date}", 
        "{date}_Station{station_id}_Times*.npy"
    )
    
    # Template for event ID files - usually follows same pattern as time files
    event_id_files_template = time_files_template.replace("_Times", "_EventIDs")
    
    # Template for parameter files - adjust if your parameter files have different naming
    parameter_files_template = os.path.join(
        station_data_root, "{date}", 
        "{date}_Station{station_id}_{parameter_name}*_*evts_*.npy"
    )
    
    # Regex to extract event count from parameter filenames
    # Modify if your files use different naming convention (e.g., "_100events_" instead of "_100evts_")
    filename_event_count_regex = re.compile(r"_(\d+)evts_")
    
    # Path to cuts file - using custom cuts directory structure
    cuts_file_path = build_cuts_path_custom(
        date_cuts=date_cuts, 
        date_str=date_str, 
        station_id=station_id,
        cuts_root=cuts_root
    )
    
    print(f"Loading parameter '{parameter_name}' for Station {station_id} on date {date_str}")
    print(f"Using cuts from: {cuts_file_path}")
    print(f"Data directory: {station_data_root}")
    print(f"Cuts directory: {cuts_root}")
    
    # Use the unified loader to get the parameter data
    loader_result = load_station_events(
        date_str=date_str,
        station_id=station_id,
        time_files_template=time_files_template,
        event_id_files_template=event_id_files_template,
        external_cuts_file_path=cuts_file_path,
        apply_external_cuts=True,  # Set to False if you don't want to apply cuts
        parameter_name=parameter_name,
        parameter_files_template=parameter_files_template,
        filename_event_count_regex=filename_event_count_regex,
    )
    
    # Extract the parameter values aligned with final (post-cuts) events
    if parameter_name in loader_result.get('parameters', {}):
        parameter_values = loader_result['parameters'][parameter_name]['by_final_idx']
        parameter_array = np.array(parameter_values)
        print(f"Successfully loaded {len(parameter_array)} {parameter_name} values after cuts")
        return parameter_array
    else:
        print(f"Warning: Parameter '{parameter_name}' not found or no data available")
        return np.array([])


def example_usage(station_id):
    """
    Example showing how to use the parameter loader with specific directory paths.
    """
    
    # --- HARD-CODED CONFIGURATION (MODIFY THESE VALUES AS NEEDED) ---
    
    # Fixed paths and dates as specified
    date = '9.1.25'           # Data date
    date_cuts = '9.18.25'     # Cuts date
    station_id = station_id         # CHANGE THIS to desired station (13, 14, 15, 17, 18, 19, 30)
    
    # Specific directory paths as provided
    data_root = '/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/'
    cuts_root = '/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_cuts/'
    
    print(f"=== Loading Parameters for Station {station_id} ===")
    print(f"Date: {date}, Cuts Date: {date_cuts}")
    print(f"Data Root: {data_root}")
    print(f"Cuts Root: {cuts_root}")
    print()
    
    # --- LOAD INDIVIDUAL PARAMETERS ---
    
    # Load Traces parameter
    traces_array = load_station_parameter('Traces', date, station_id, date_cuts, data_root, cuts_root)
    if len(traces_array) > 0:
        print(f"Traces shape: {traces_array.shape}")
        print(f"Traces data type: {traces_array.dtype}")
    print()
    
    # Load Times parameter  
    times_array = load_station_parameter('Times', date, station_id, date_cuts, data_root, cuts_root)
    if len(times_array) > 0:
        print(f"Times shape: {times_array.shape}")
        print(f"Times data type: {times_array.dtype}")
        print(f"Time range: {np.min(times_array)} to {np.max(times_array)}")
    print()
    
    # LOAD ALL AVAILABLE PARAMETERS ---
    all_parameters = ['Traces', 'SNR', 'ChiRCR', 'Chi2016', 'ChiBad', 'Zen', 'Azi', 'Times']
    parameter_dict = {}
    for param_name in all_parameters:
        param_array = load_station_parameter(param_name, date, station_id, date_cuts, data_root, cuts_root)
        if len(param_array) > 0:
            parameter_dict[param_name] = param_array
            print(f"{param_name}: {len(param_array)} values loaded")
        else:
            print(f"{param_name}: No data found")
    
    print(f"\nLoaded {len(parameter_dict)} parameters successfully")
    
    return traces_array, times_array, parameter_dict



if __name__ == '__main__':
    
    station_ids = [13,15,18,14,17,19,30]


    for station_id in station_ids:
        traces, times, parameter_dict = example_usage(station_id)

        import pickle
        with open(f'/pub/tangch3/ARIANNA/DeepLearning/refactor/station_data/all_data/10.17.25/all_data_stn{station_id}.pkl', 'wb') as f:
            pickle.dump(parameter_dict, f)

        with open(f'/pub/tangch3/ARIANNA/DeepLearning/refactor/station_data/all_data/10.17.25/all_data_stn{station_id}.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)

        # from A0_Utilities import pT    
        # for i in [1,10,15,30]:
        #     pT(traces[i], "test plot trace after 9.18.25 noise cut", f"/pub/tangch3/ARIANNA/DeepLearning/refactor/other/10.7.25_plot_9.18.25_noise_cut_trace{i}.png")

        import matplotlib.pyplot as plt
        import matplotlib
        plot_folder = f"/pub/tangch3/ARIANNA/DeepLearning/refactor/other/"
        chi_types = ['Chi2016', 'ChiRCR']
        SNRbins = np.logspace(0.477, 2, num=80)
        maxCorrBins = np.arange(0, 1.0001, 0.01)

        for param in chi_types:
            plt.hist2d(loaded_dict['SNR'], loaded_dict[param], bins=[SNRbins, maxCorrBins], norm=matplotlib.colors.LogNorm())
            plt.colorbar()
            plt.xlim((3, 100))
            plt.ylim((0, 1))
            plt.xlabel('SNR')
            plt.ylabel('Avg Chi Highest Parallel Channels')
            # plt.legend()
            plt.xscale('log')
            plt.tick_params(axis='x', which='minor', bottom=True)
            plt.grid(visible=True, which='both', axis='both') 
            plt.title(f'Station {station_id} {len(loaded_dict['SNR'])} events')
            print(f'Saving {plot_folder}10.7.25_All_stn{station_id}_{param}.png')
            plt.savefig(f'{plot_folder}10.7.25_All_stn{station_id}_{param}.png')
            plt.clf()

        
        