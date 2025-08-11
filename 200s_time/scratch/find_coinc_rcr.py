import numpy as np
import sys, os

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from A0_Utilities import pT, load_520_data

station_data_folder = '/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/5.20.25/'


amp = '200s'
station_id = [13, 17]
all_Backlobe = []
all_Backlobe_UNIX = [] 
all_Backlobe_SNR = []
all_Backlobe_Chi2016 = []
for id in station_id:
    # snr, chi, trace, unix = load_data('All_data', amp_type = amp, station_id=id)
    unix = np.load(f'/pub/tangch3/ARIANNA/DeepLearning/new_chi_data/4.4.25/Station{id}/station{id}_all_Times.npy')
    trace = np.load(f'/pub/tangch3/ARIANNA/DeepLearning/new_chi_data/4.4.25/Station{id}/station{id}_all_Traces.npy')
    snr = np.load(f'/pub/tangch3/ARIANNA/DeepLearning/new_chi_data/4.4.25/Station{id}/station{id}_all_SNR.npy')
    chi2016 = np.load(f'/pub/tangch3/ARIANNA/DeepLearning/new_chi_data/4.4.25/Station{id}/station{id}_all_Chi2016.npy')
    all_Backlobe.extend(trace)
    all_Backlobe_UNIX.extend(unix)
    all_Backlobe_SNR.extend(snr)
    all_Backlobe_Chi2016.extend(chi2016)
print(len(all_Backlobe_UNIX))


from datetime import datetime, timezone
target_unix = 1487300991

if target_unix in all_Backlobe_UNIX:
    idx = all_Backlobe_UNIX.index(target_unix)
    trace = all_Backlobe[idx]
    snr = all_Backlobe_SNR[idx]
    chi = all_Backlobe_Chi2016[idx]
    thetime = all_Backlobe_UNIX[idx]
    utc_time = datetime.fromtimestamp(thetime, tz=timezone.utc)

    print(f"\n Event found at index {idx}")
    print(f"  UTC Time:    {utc_time}")
    print(f"  UNIX Time:   {thetime}")
    print(f"  SNR:         {snr}")
    print(f"  Chi2016:     {chi}")
    print(f"  Trace shape: {trace.shape}")



for id in station_id:
    unix, count = load_520_data(id, 'Times', station_data_folder)
    trace, num = load_520_data(id, 'Traces', station_data_folder)
    all_Backlobe_UNIX.extend(unix.tolist())
    all_Backlobe.extend(trace.tolist())
all_Backlobe_UNIX = np.array(all_Backlobe_UNIX)
all_Backlobe = np.array(all_Backlobe)

from datetime import timezone, datetime

target_unix_time = 1487272191 # Feb 16, 2017 at 19:09:51 UTC
similarity_window_seconds = 5

exact_match_indices = []
exact_match_count = 0

print(f"\nSearching for EXACT matches to UNIX time: {target_unix_time}")
print(f"Total events in all_Backlobe_UNIX: {all_Backlobe_UNIX.shape}")

# Use all_Backlobe_UNIX consistently
for idx, unix_time in enumerate(all_Backlobe_UNIX):
    if unix_time == target_unix_time:
        exact_match_count += 1
        # Use timezone.utc for precise conversion if Unix time is UTC
        std_time = datetime.fromtimestamp(unix_time, tz=timezone.utc)
        print(f"  Exact match found: Event {exact_match_count} at index {idx} with time {std_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        exact_match_indices.append(idx)

print(f"\n--- Summary of Exact Matches ---")
print(f"Total exact matches found: {exact_match_count}")
print(f"Indices of exact matching events: {exact_match_indices}")

for index in exact_match_indices:
    pT(all_Backlobe[index], 'test plot confirmed RCR', f'/pub/tangch3/ARIANNA/DeepLearning/78_test_plot_confirmed_RCR_{amp}_{index}.png')

# --- Search for SIMILAR matches ---
similar_match_indices = []
similar_match_count = 0

print(f"\nSearching for SIMILAR matches (within +/- {similarity_window_seconds} seconds) to UNIX time: {target_unix_time}")

# Using all_Backlobe_UNIX for the search
for idx, unix_time in enumerate(all_Backlobe_UNIX):
    # Check if the UNIX time falls within the defined window
    if target_unix_time - similarity_window_seconds <= unix_time <= target_unix_time + similarity_window_seconds:
        similar_match_count += 1
        # Use timezone.utc for precise conversion if Unix time is UTC
        std_time = datetime.fromtimestamp(unix_time, tz=timezone.utc)
        print(f"  Similar match found: Event {similar_match_count} at index {idx} with time {std_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        similar_match_indices.append(idx)

print(f"\n--- Summary of Similar Matches ---")
print(f"Total similar matches found: {similar_match_count}")


        
