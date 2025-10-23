# Refactoring Summary: Data Loading for R01_1D_CNN_train_and_run.py

## Date: October 23, 2025

## Purpose
Refactored R01 to load backlobe data directly from the combined pickle file saved by `refactor_converter.py`, ensuring all stations' data is loaded consistently and using `traces2016` for the training backlobe set.

## Changes Made

### 1. A0_Utilities.py - New Function Added

**Function:** `load_combined_backlobe_data(combined_pkl_path)`

**Location:** Added after the `load_data()` function (around line 220)

**Purpose:** 
- Loads the combined backlobe data pickle file created by `refactor_converter.py`
- This file contains data from ALL stations (13, 14, 15, 17, 18, 19, 30) that has passed chi and bin cuts

**Returns:**
- `snr2016`: SNR values for 2016 template
- `snrRCR`: SNR values for RCR template  
- `chi2016`: Chi values for 2016 template
- `chiRCR`: Chi values for RCR template
- `traces2016`: Trace data matched to 2016 template (numpy array)
- `tracesRCR`: Trace data matched to RCR template (numpy array)
- `unix2016`: Unix timestamps for 2016 template
- `unixRCR`: Unix timestamps for RCR template

**Default File Path:**
```
/dfs8/sbarwick_lab/ariannaproject/tangch3/station_data/above_curve_data/5000evt_10.17.25/above_curve_combined.pkl
```

### 2. R01_1D_CNN_train_and_run.py - Modified Function

**Function:** `load_and_prep_data_for_training(config)`

**Changes:**
1. **Import statement updated** (line 13):
   ```python
   from A0_Utilities import load_sim_rcr, load_data, pT, load_config, load_combined_backlobe_data
   ```

2. **Data loading refactored**:
   - **REMOVED:** Loop through individual stations calling `load_data()` for each
   - **REMOVED:** Manual dictionary construction and extension of lists
   - **ADDED:** Single call to `load_combined_backlobe_data()` to load all station data at once

3. **Key improvements**:
   - Simpler, cleaner code
   - Loads data exactly as saved by `refactor_converter.py`
   - Ensures all stations are included automatically
   - **Explicitly uses `traces2016` for `training_backlobe`** (see line 79 comment)
   - No longer depends on `config['station_ids']`

## Data Flow

### Before:
```
R01 → load_data() for each station → manually combine into dictionary → extract traces2016
```

### After:
```
R01 → load_combined_backlobe_data() → directly get all stations' traces2016
```

## Key Variables in Returned Dictionary

The `load_and_prep_data_for_training()` function returns a dictionary with:

- `training_rcr`: Random subset of RCR simulations for training
- `training_backlobe`: Random subset of **traces2016** for training ← KEY CHANGE
- `sim_rcr_all`: All RCR simulation data
- `data_backlobe_traces2016`: All backlobe traces matched to 2016 template
- `data_backlobe_tracesRCR`: All backlobe traces matched to RCR template
- `data_backlobe_unix2016_all`: Unix timestamps for 2016
- `data_backlobe_chi2016_all`: Chi values for 2016
- `rcr_non_training_indices`: Indices not used in training (for validation)
- `bl_non_training_indices`: Indices not used in training (for validation)

## Verification Points

✅ `training_backlobe` is explicitly set from `backlobe_traces_2016` (line 79)
✅ `backlobe_traces_2016` is loaded from `traces2016` key in combined pickle (line 68)
✅ All stations' data is included automatically from the combined file
✅ Data structure matches what `refactor_converter.py` saves
✅ No changes needed to downstream functions - return dictionary structure maintained

## Testing Recommendations

1. Run R01 with these changes and verify:
   - Training data loads successfully
   - Shape of `training_backlobe` matches expected dimensions
   - Number of events matches what was saved by `refactor_converter.py`
   
2. Check console output for:
   ```
   Loading combined backlobe data from: [path]
   Loaded combined data:
     SNR2016: [N] events
     Traces2016 shape: [shape]
   ```

3. Verify training proceeds normally with expected accuracy/loss curves

## Notes

- The combined pickle file path is currently hardcoded in `load_and_prep_data_for_training()`
- Consider adding this path to `config.yaml` for easier configuration
- The refactoring maintains backward compatibility with the rest of R01's workflow
- All existing functionality for model training, evaluation, and plotting remains unchanged
