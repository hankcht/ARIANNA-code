import numpy as np
import glob
import os
import argparse
import datetime
from icecream import ic

def load_parameter_data(base_folder, date_str, station_id, param_name, partition_number):
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
        if partition_number == 0:
            data_list = [np.load(f) for i, f in enumerate(files) if i <= 30]
        elif partition_number == 1:
            data_list = [np.load(f) for i, f in enumerate(files) if 30 <= i <= 60]
        elif partition_number == 2:
            data_list = [np.load(f) for i, f in enumerate(files) if 60 <= i]
        
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
    base_output_folder = os.path.join(f'/pub/tangch3/ARIANNA/DeepLearning/new_chi_data/', date_str, f"Station{station_id}")   # Modify as needed
    
    os.makedirs(base_output_folder, exist_ok=True)

    # --- Parameters to Load and Save ---
    params_to_process = ['Traces', 'SNR', 'Chi2016', 'ChiRCR', 'Times'] # 
    loaded_data_raw = {}
    partition_number = 0

    ic("Loading data...") 
    for param in params_to_process:
        data = load_parameter_data(base_input_folder, date_str, station_id, param, partition_number)
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
    
    # --- Chi2016 Thresholding and Saving ---
    chi2016_thresholds = [0.7, 0.65, 0.6]
    base_chi2016_values = filtered_data_for_chi_cuts['Chi2016']
    base_chircr_values = filtered_data_for_chi_cuts['ChiRCR']
    base_snr_values = filtered_data_for_chi_cuts['SNR']

    # Modified: Include partition_number in the filename
    chi2016_base_filename = os.path.join(base_output_folder, f"Station{station_id}_Part{partition_number}_Chi2016_base.npy")
    chircr_base_filename = os.path.join(base_output_folder, f"Station{station_id}_Part{partition_number}_ChiRCR_base.npy")
    snr_base_filename = os.path.join(base_output_folder, f"Station{station_id}_Part{partition_number}_SNR_base.npy")
    
    np.save(chi2016_base_filename, base_chi2016_values)
    ic(f"Saved base Chi2016 values to: {chi2016_base_filename}")

    # Correction: You were saving ChiRCR to the Chi2016 filename, fixed this.
    np.save(chircr_base_filename, base_chircr_values)
    ic(f"Saved base ChiRCR values to: {chircr_base_filename}")

    np.save(snr_base_filename, base_snr_values)
    ic(f"Saved base SNR values to: {snr_base_filename}")

    # for chi_thresh in chi2016_thresholds:
    #     ic(f"\nApplying Chi2016 threshold: >= {chi_thresh}")
        
    #     # Ensure base_chi2016_values is not empty and is 1D for comparison
    #     if base_chi2016_values.size == 0 :
    #         ic(f"  No Chi2016 values to apply threshold {chi_thresh} to (array is empty). Skipping.")
    #         continue
        
    #     current_chi_mask = (base_chi2016_values >= chi_thresh)
    #     num_passed_events = np.sum(current_chi_mask)
        
    #     ic(f"  {num_passed_events} events passed Chi2016 >= {chi_thresh}")

    #     if num_passed_events > 0:
    #         output_data_dict = {}
    #         for param in params_to_process:
    #             output_data_dict[param] = filtered_data_for_chi_cuts[param][current_chi_mask]
            
    #         # Construct filename
    #         # Replacing '.' in threshold with 'p' for cleaner filenames (e.g., 0.7 -> 0p70)
    #         thresh_str = f"{chi_thresh:.2f}".replace('.', 'p') 
    #         output_filename = f"St{station_id}_{date_str}_Chi2016_ge{thresh_str}_{num_passed_events}evts_SelectedData_part{partition_number}.npy"
    #         output_filepath = os.path.join(base_output_folder, output_filename)
            
    #         try:
    #             np.save(output_filepath, output_data_dict, allow_pickle=True)
    #             ic(f"  Successfully saved: {output_filepath}")
    #             for param, data in output_data_dict.items():
    #                 ic(f"    Saved {param} shape: {data.shape}")
    #         except Exception as e:
    #             ic(f"  Error saving file {output_filepath}: {e}")
    #     else:
    #         ic(f"  No events to save for Chi2016 >= {chi_thresh}")

    ic(f"\nProcessing complete for Station {station_id}, Date {date_str}.")

if __name__ == '__main__':
    main()

#     station_id = 18
#     load_path = f'/pub/tangch3/ARIANNA/DeepLearning/new_chi_data/4.4.25/Station{station_id}/'

#     import re
#     file_names = [
# "St18_4.4.25_Chi2016_ge0p60_254evts_SelectedData_part1.npy", "St18_4.4.25_Chi2016_ge0p65_99evts_SelectedData_part1.npy", "St18_4.4.25_Chi2016_ge0p60_675evts_SelectedData_part0.npy", "St18_4.4.25_Chi2016_ge0p70_13evts_SelectedData_part1.npy", "St18_4.4.25_Chi2016_ge0p65_233evts_SelectedData_part0.npy", "St18_4.4.25_Chi2016_ge0p70_36evts_SelectedData_part0.npy"
#     ]

#     # Define the correct load path based on your clarification.
#     load_path = f'/pub/tangch3/ARIANNA/DeepLearning/new_chi_data/4.4.25/Station{station_id}/'

#     # Create a dictionary to store files grouped by their ge0p number.
#     grouped_files = {}

#     # Iterate over the filenames and extract the ge0p value for each file.
#     for filename in file_names:
#         # Use regular expression to extract the ge0p number.
#         match = re.search(r'ge0p(\d+)', filename)
#         if match:
#             ge0p = 'ge0p' + match.group(1)
#             if ge0p not in grouped_files:
#                 grouped_files[ge0p] = []
#             grouped_files[ge0p].append(filename)

#     print("Processing and saving data...")

#     # Iterate over the grouped files and process each ge0p number separately.
#     for ge0p, files in grouped_files.items():
#         # Initialize lists to store data for each parameter.
#         all_traces = []
#         all_snr = []
#         all_chi2016 = []
#         all_chiRCR = []
#         all_times = []

#         # Sort files by part number to ensure consistent concatenation if order matters
#         # (though concatenate should handle it based on event content).
#         files.sort(key=lambda x: int(re.search(r'part(\d+)', x).group(1)))

#         # Iterate over the files for the current ge0p number.
#         for filename in files:
#             # Construct the full file path.
#             full_path = os.path.join(load_path, filename)

#             try:
#                 # Load the data from the file.
#                 loaded_data = np.load(full_path, allow_pickle=True).item()

#                 # Extract the data for each parameter and append to the lists.
#                 all_traces.append(loaded_data['Traces'])
#                 all_snr.append(loaded_data['SNR'])
#                 all_chi2016.append(loaded_data['Chi2016'])
#                 all_chiRCR.append(loaded_data['ChiRCR'])
#                 all_times.append(loaded_data['Times'])
#             except FileNotFoundError:
#                 print(f"Error: File not found at {full_path}. Skipping this file.")
#                 continue
#             except KeyError as e:
#                 print(f"Error: Missing key '{e}' in {filename}. Skipping this file.")
#                 continue

#         # Only attempt concatenation if data was loaded successfully.
#         if all_traces: # Check if the list is not empty
#             # Concatenate the lists of arrays into single NumPy arrays.
#             traces_combined = np.concatenate(all_traces, axis=0)
#             snr_combined = np.concatenate(all_snr, axis=0)
#             chi2016_combined = np.concatenate(all_chi2016, axis=0)
#             chiRCR_combined = np.concatenate(all_chiRCR, axis=0)
#             times_combined = np.concatenate(all_times, axis=0)

#             # Save the combined data for each parameter.
#             np.save(f'/pub/tangch3/ARIANNA/DeepLearning/new_chi_data/4.4.25/Stn{station_id}_Traces_{ge0p}.npy', traces_combined)
#             np.save(f'/pub/tangch3/ARIANNA/DeepLearning/new_chi_data/4.4.25/Stn{station_id}_SNR_{ge0p}.npy', snr_combined)
#             np.save(f'/pub/tangch3/ARIANNA/DeepLearning/new_chi_data/4.4.25/Stn{station_id}_Chi2016_{ge0p}.npy', chi2016_combined)
#             np.save(f'/pub/tangch3/ARIANNA/DeepLearning/new_chi_data/4.4.25/Stn{station_id}_ChiRCR_{ge0p}.npy', chiRCR_combined)
#             np.save(f'/pub/tangch3/ARIANNA/DeepLearning/new_chi_data/4.4.25/Stn{station_id}_Times_{ge0p}.npy', times_combined)

#             print(f"\nSuccessfully saved data for {ge0p}:")
#             print(f"  Traces_{ge0p}.npy with shape: {traces_combined.shape}")
#             print(f"  SNR_{ge0p}.npy with shape: {snr_combined.shape}")
#             print(f"  Chi2016_{ge0p}.npy with shape: {chi2016_combined.shape}")
#             print(f"  ChiRCR_{ge0p}.npy with shape: {chiRCR_combined.shape}")
#             print(f"  Times_{ge0p}.npy with shape: {times_combined.shape}")
#         else:
#             print(f"\nNo data loaded for {ge0p}. Skipping saving.")



#     # You can now work with each NumPy array
#     # For example, print the first 5 SNR values:
#     print(f"First 5 SNR values: {snr_combined[:5]}")



