import numpy as np
import glob
import os
import argparse
import datetime
from icecream import ic

def load_parameter_data(base_folder, date_str, station_id, param_name, partition_number, total_partitions=3):
    """
    Loads, concatenates ALL parameter data for a given station, date, and parameter name,
    then returns a specific non-overlapping partition of that data.
    """
    file_pattern = os.path.join(base_folder, f'{date_str}_Station{station_id}_{param_name}*')
    files = sorted(glob.glob(file_pattern))

    if not files:
        ic(f"Warning: No files found for Station {station_id}, {param_name}, Date {date_str} with pattern {file_pattern}")
        return None

    try:
        # Step 1: Load all data from all relevant files first
        all_data_list = []
        for f in files:
            try:
                all_data_list.append(np.load(f))
            except Exception as e:
                ic(f"Error loading individual file {f}: {e}")
                continue # Skip this file and try to load others

        if not all_data_list:
            ic(f"No valid data loaded from any files for {param_name}.")
            return None

        # Step 2: Concatenate all loaded data
        processed_all_data_list = []
        for arr_part in all_data_list:
            squeezed_part = arr_part.squeeze()
            if squeezed_part.ndim == 0: # Was scalar
                processed_all_data_list.append(np.array([squeezed_part.item()]))
            elif param_name == "Traces" and squeezed_part.ndim == 2 and arr_part.shape[0]==1:
                processed_all_data_list.append(arr_part)
            elif param_name == "Traces" and squeezed_part.ndim == 3 and arr_part.shape[0]==1:
                processed_all_data_list.append(arr_part)
            else:
                processed_all_data_list.append(squeezed_part if param_name != "Traces" else arr_part)
        
        if not processed_all_data_list:
            return None # No data after processing, even if files were found

        # Concatenate all data from all files based on parameter type
        if param_name == "Traces":
            # Assuming Traces are (N, 4, 256) or (4, 256) for single events.
            # Convert single event (4,256) to (1,4,256) for stacking
            processed_all_data_list = [p if p.ndim == 3 else p[np.newaxis, :, :] for p in processed_all_data_list]
            if not all(p.shape[1:] == (4,256) for p in processed_all_data_list):
                 ic(f"Warning: Traces for {param_name} have inconsistent shapes beyond (N,4,256). Attempting general concatenation.")
            concatenated_all_data = np.concatenate(processed_all_data_list, axis=0)
        else: # For 1D parameters like Times, SNR, Chi*
            concatenated_all_data = np.concatenate(processed_all_data_list, axis=0)
        
        # Ensure it's squeezed appropriately at the end of full concatenation
        full_data_squeezed = concatenated_all_data.squeeze()

        # Step 3: Apply partitioning based on event count
        total_events = full_data_squeezed.shape[0] # Assumes first dimension is event count
        
        # Handle cases where total_events might be 0 after squeeze or for empty files
        if total_events == 0:
            ic(f"No events found after concatenating all files for {param_name}.")
            return None

        # Calculate start and end indices for the current partition
        # Ensure even distribution, with last partition getting any remainder
        start_idx = (total_events * partition_number) // total_partitions
        end_idx = (total_events * (partition_number + 1)) // total_partitions
        
        # For the very last partition, make sure it goes to the end
        if partition_number == total_partitions - 1:
            end_idx = total_events

        # Extract the specific partition
        partition_data = full_data_squeezed[start_idx:end_idx]
        
        if partition_data.shape[0] == 0:
            ic(f"Warning: Partition {partition_number} for {param_name} is empty. Check partition logic or data size.")

        return partition_data

    except Exception as e:
        ic(f"Error loading or concatenating data for {param_name} at Station {station_id}, Date {date_str}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Filter station data based on Chi2016 thresholds and save selected parameters.")
    parser.add_argument("station_id", type=int, help="Station ID to process.")
    parser.add_argument("date", type=str, help="Date string (e.g., YYYYMMDD) for the data.")
    # Add an argument for partition number
    parser.add_argument("--partition", type=int, default=0,
                        help="Partition number to process (0-indexed).")
    parser.add_argument("--total_partitions", type=int, default=3,
                        help="Total number of partitions to divide the data into.")
    
    args = parser.parse_args()
    station_id = args.station_id
    date_str = args.date
    partition_number = args.partition
    total_partitions = args.total_partitions

    ic.enable()
    ic(f"Processing Station {station_id} for Date {date_str}, Partition {partition_number}/{total_partitions}")

    # --- Configuration for Paths ---
    base_input_folder = os.path.join('/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/', date_str)
    base_output_folder = os.path.join(f'/pub/tangch3/ARIANNA/DeepLearning/new_chi_data/', date_str, f"Station{station_id}") # Modify as needed
    
    os.makedirs(base_output_folder, exist_ok=True)

    # --- Parameters to Load and Save ---
    params_to_process = ['Traces', 'SNR', 'Chi2016', 'ChiRCR', 'Times']
    loaded_data_raw = {}

    ic("Loading data for specified partition...") 
    for param in params_to_process:
        # Pass total_partitions to the load function
        data = load_parameter_data(base_input_folder, date_str, station_id, param, partition_number, total_partitions)
        if data is None or data.size == 0:
            ic(f"Failed to load or data is empty for {param} for partition {partition_number}. Exiting.")
            return
        loaded_data_raw[param] = data
        ic(f" Loaded {param} for partition {partition_number}: shape {data.shape}")

    # --- Basic Consistency Check (Number of Events) ---
    num_events_initial = -1
    for param, data in loaded_data_raw.items():
        if data is None:
            ic(f"Critical error: {param} data is None after load attempt. Exiting.")
            return
        current_param_event_count = data.shape[0]
        if num_events_initial == -1:
            num_events_initial = current_param_event_count
        elif num_events_initial != current_param_event_count:
            ic(f"Error: Mismatch in number of events between parameters for partition {partition_number}.")
            ic(f" Times has {loaded_data_raw.get('Times', np.array([])).shape[0]} events.")
            ic(f" {param} has {current_param_event_count} events.")
            ic(f"Please check data consistency for Station {station_id}, Date {date_str}. Exiting.")
            return
            
    if num_events_initial == 0:
        ic(f"No events found in the loaded data for partition {partition_number}. Exiting.")
        return
    ic(f"Successfully loaded data for {num_events_initial} events in partition {partition_number}.")

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
        ic(f" {param} after initial filters: shape {filtered_data_for_chi_cuts[param].shape}")
    
    # --- Chi2016 Thresholding and Saving ---
    # chi2016_thresholds = [0.7, 0.65, 0.6] # Commented out as this section was also commented out
    base_chi2016_values = filtered_data_for_chi_cuts['Chi2016']
    base_chircr_values = filtered_data_for_chi_cuts['ChiRCR']
    base_snr_values = filtered_data_for_chi_cuts['SNR']

    # Modified: Include partition_number in the filename
    chi2016_base_filename = os.path.join(base_output_folder, f"Station{station_id}_Part{partition_number}_Chi2016_base.npy")
    chircr_base_filename = os.path.join(base_output_folder, f"Station{station_id}_Part{partition_number}_ChiRCR_base.npy")
    snr_base_filename = os.path.join(base_output_folder, f"Station{station_id}_Part{partition_number}_SNR_base.npy")
    
    np.save(chi2016_base_filename, base_chi2016_values)
    ic(f"Saved base Chi2016 values to: {chi2016_base_filename}")

    np.save(chircr_base_filename, base_chircr_values)
    ic(f"Saved base ChiRCR values to: {chircr_base_filename}")

    np.save(snr_base_filename, base_snr_values)
    ic(f"Saved base SNR values to: {snr_base_filename}")

    ic(f"\nProcessing complete for Station {station_id}, Date {date_str}, Partition {partition_number}.")

import re

def concatenate_npy_by_station(input_directory, output_directory):
    """
    Concatenates .npy files by station and type (e.g., Chi2016, ChiRCR, SNR).

    Args:
        input_directory (str): The path to the directory containing the .npy files.
        output_directory (str): The path where the concatenated .npy files will be saved.
    """
    station_files = {}

    # Regex to parse the filename:
    # Captures: Station Number, Part Number, Data Type (e.g., Chi2016)
    pattern = re.compile(r'Station(\d+)_Part(\d+)_(Chi2016|ChiRCR|SNR)_base\.npy')

    for filename in os.listdir(input_directory):
        match = pattern.match(filename)
        if match:
            station_num = match.group(1)
            part_num = int(match.group(2))
            data_type = match.group(3)
            full_path = os.path.join(input_directory, filename)

            if station_num not in station_files:
                station_files[station_num] = {'Chi2016': {}, 'ChiRCR': {}, 'SNR': {}}

            station_files[station_num][data_type][part_num] = full_path

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    for station_num, data_types in station_files.items():
        print(f"--- Processing Station {station_num} ---")
        for data_type, parts in data_types.items():
            if not parts:
                continue

            # Sort parts by their number (e.g., Part0, Part1, Part2)
            sorted_parts = sorted(parts.items())
            arrays_to_concatenate = []

            print(f"  Concatenating {data_type} files:")
            for part_num, filepath in sorted_parts:
                print(f"    Loading: {os.path.basename(filepath)}")
                try:
                    arrays_to_concatenate.append(np.load(filepath))
                except Exception as e:
                    print(f"      Error loading {filepath}: {e}")
                    continue

            if arrays_to_concatenate:
                try:
                    # Concatenate along axis 0. Adjust this if your data requires a different axis.
                    concatenated_array = np.concatenate(arrays_to_concatenate, axis=0)
                    # The output filename now reflects the concatenation of all parts
                    output_filename = f"Station{station_num}_{data_type}_base.npy" 
                    output_path = os.path.join(output_directory, output_filename)
                    np.save(output_path, concatenated_array)
                    print(f"  Successfully saved concatenated {data_type} data for Station {station_num} to: {output_path}")
                except ValueError as e:
                    print(f"  Could not concatenate {data_type} arrays for Station {station_num}. Error: {e}")
                    print("  This often happens if arrays have incompatible shapes for concatenation along the specified axis.")
            else:
                print(f"  No {data_type} files loaded for concatenation for Station {station_num}.")


if __name__ == "__main__":
    main()
    

    # input_dir = '/pub/tangch3/ARIANNA/DeepLearning/new_chi_data/4.4.25/Station14/'
    # output_dir = '/pub/tangch3/ARIANNA/DeepLearning/new_chi_data/4.4.25/'

    # # Run the concatenation process
    # concatenate_npy_by_station(input_dir, output_dir)

    # print("\n--- Concatenation process complete ---")

    # # Load the specific concatenated file and print its length
    # target_file_path = os.path.join(output_dir, 'Station14_Chi2016_base.npy')
    
    # if os.path.exists(target_file_path):
    #     try:
    #         loaded_array = np.load(target_file_path)
    #         print(f"Length of '{os.path.basename(target_file_path)}': {len(loaded_array)}")
    #         print(f"Shape of '{os.path.basename(target_file_path)}': {loaded_array.shape}")
    #     except Exception as e:
    #         print(f"Error loading or getting length of '{os.path.basename(target_file_path)}': {e}")
    # else:
    #     print(f"File '{os.path.basename(target_file_path)}' not found in '{output_dir}'.")


    
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



