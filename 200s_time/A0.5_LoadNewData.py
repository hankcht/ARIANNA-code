import numpy as np
import os
import glob
import re
import argparse # Import the argparse module

def load_arrianna_data(station_id, date_str, load_base_path, chi_threshold=None):
    """
    Loads ARIANNA data from .npy files for a given station, date, and optional Chi2016 threshold.

    Args:
        station_id (int): The ID of the station (e.g., 14).
        date_str (str): The date string as it appears in the file paths (e.g., '4.4.25').
        load_base_path (str): The base directory where the Station and date folders are located.
        chi_threshold (str, optional): A specific Chi2016 threshold to load (e.g., '0p60', '0p65').
                                       If None, all available thresholds will be loaded.

    Returns:
        tuple: A tuple containing (traces_data, snr_data, chi2016_data, chiRCR_data, times_data)
               as concatenated NumPy arrays. Returns empty lists if no data is loaded.
    """
    traces_data = []
    snr_data = []
    chi2016_data = []
    chiRCR_data = []
    times_data = []

    target_directory = os.path.join(load_base_path, date_str, f'Station{station_id}')
    print(f"Attempting to find files in directory: {target_directory}")

    # Construct the search pattern based on whether a specific threshold is provided
    if chi_threshold:
        search_pattern = os.path.join(target_directory, f'St{station_id}_{date_str}_Chi2016_ge{chi_threshold}_*evts_SelectedData_part*.npy')
        print(f"Using specific search pattern for threshold '{chi_threshold}': {search_pattern}")
    else:
        search_pattern = os.path.join(target_directory, f'St{station_id}_{date_str}_Chi2016_ge*_evts_SelectedData_part*.npy')
        print(f"Using general search pattern (all thresholds): {search_pattern}")

    all_partition_files = glob.glob(search_pattern)

    if not all_partition_files:
        print(f"No partition files found for Station {station_id}, Date {date_str}")
        if chi_threshold:
            print(f" (for Chi2016 threshold '{chi_threshold}').")
        else:
            print(" (across all Chi2016 thresholds).")

        print("Double-check your parameters and file paths.")
        try:
            print("\nContents of target directory:")
            for item in os.listdir(target_directory):
                print(f"- {item}")
        except FileNotFoundError:
            print(f"Error: Target directory '{target_directory}' does not exist.")
        except Exception as e:
            print(f"Could not list directory contents: {e}")
        return [], [], [], [], [] # Return empty lists if no files found

    print(f"Found {len(all_partition_files)} files matching the pattern.")

    def natural_sort_key(s):
        match = re.search(r'Chi2016_ge(\d+p\d+)_.*_part(\d+)', os.path.basename(s))
        if match:
            threshold_str = match.group(1).replace('p', '.')
            threshold_val = float(threshold_str)
            part_num = int(match.group(2))
            return (part_num, threshold_val)
        return (0, 0) # Fallback

    all_partition_files.sort(key=natural_sort_key)
    print("Files will be loaded in this order:")
    for f in all_partition_files:
        print(f"  {os.path.basename(f)}")

    for file_path in all_partition_files:
        try:
            loaded_dict = np.load(file_path, allow_pickle=True)

            traces_data.append(loaded_dict['Traces'])
            snr_data.append(loaded_dict['SNR'])
            chi2016_data.append(loaded_dict['Chi2016'])
            chiRCR_data.append(loaded_dict['ChiRCR'])
            times_data.append(loaded_dict['Times'])

            print(f"Successfully loaded file: {os.path.basename(file_path)}")

        except Exception as e:
            print(f"An error occurred while loading {os.path.basename(file_path)}: {e}")
            break # Stop if a file cannot be loaded

    if traces_data:
        traces_data = np.concatenate(traces_data, axis=0)
        snr_data = np.concatenate(snr_data, axis=0)
        chi2016_data = np.concatenate(chi2016_data, axis=0)
        chiRCR_data = np.concatenate(chiRCR_data, axis=0)
        times_data = np.concatenate(times_data, axis=0)

        print(f"\nConcatenated Data Shapes:")
        print(f"Traces shape: {traces_data.shape}")
        print(f"SNR shape: {snr_data.shape}")
        print(f"Chi2016 shape: {chi2016_data.shape}")
        print(f"ChiRCR shape: {chiRCR_data.shape}")
        print(f"Times shape: {times_data.shape}")

        print(f"\nFirst 5 SNR values: {snr_data[:5]}")
        return traces_data, snr_data, chi2016_data, chiRCR_data, times_data
    else:
        print("No data was loaded from any partition, so no concatenation performed.")
        return [], [], [], [], [] # Return empty lists if no data concatenated

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load ARIANNA data for a given station, date, and optional Chi2016 threshold."
    )
    parser.add_argument(
        '--station', type=int, default=14,
        help='The station ID (e.g., 14).'
    )
    parser.add_argument(
        '--chi_threshold', type=str, default=None,
        help='Optional: A specific Chi2016 threshold to load (e.g., "0p60", "0p65", "0p70"). '
             'If not provided, all available thresholds for the given station/date will be loaded.'
    )

    args = parser.parse_args()
    date = '4.4.25'

    load_path = '/pub/tangch3/ARIANNA/DeepLearning/new_chi_data/'
    traces, snr, chi2016, chiRCR, times = load_arrianna_data(
        args.station,
        date,
        load_path,
        args.chi_threshold
    )

    print(snr[:10])
    # You can now work with the loaded data (traces, snr, etc.)
    # For example:
    if len(traces) > 0:
        print(f"\nTotal number of events loaded: {traces.shape[0]}")
    else:
        print("\nNo data loaded. Exiting.")