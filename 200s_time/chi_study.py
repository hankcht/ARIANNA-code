import os
from glob import glob


from refactor_checks import load_all_coincidence_traces, load_2016_backlobe_templates

# template_dir = "/pub/tangch3/ARIANNA/DeepLearning/refactor/confirmed_2016_templates/"
# template_paths = sorted(glob(os.path.join(template_dir, "filtered_Event2016_Stn*.npy"))) # using filtered confirmed BL
# all_2016_backlobes_200, dict_2016_200 = load_2016_backlobe_templates(template_paths, amp_type='200s')
# all_2016_backlobes_100, dict_2016_100 = load_2016_backlobe_templates(template_paths, amp_type='100s')


pkl_path = '/pub/tangch3/ARIANNA/DeepLearning/refactor/coincidence_events/filtered_coinc.pkl'

new_coinc_dict, new_coinc_traces, new_metadata = load_all_coincidence_traces(pkl_path, trace_key='Filtered_Traces') 

# Chi study criteria are:
# 1. ChiRCR and Chi2016 both >= 0.55
# 2. if Chi2016 - ChiRCR <= -0.09 → RCR-like
# 3. if Chi2016 - ChiRCR >= 0.15 → Backlobe-like

# takes Chi2016 and ChiRCR of each event and calculate the difference
# passing condition is: difference larger than - 0.1, less than

# Initialize counters
count_rcr_like = 0
count_backlobe_like = 0
count_unclassified = 0
count_below_threshold = 0

# Loop through events and apply Chi study criteria
for index in range(len(new_metadata)):
    
    chi2016 = new_metadata[index]['Chi2016']
    chircr = new_metadata[index]['ChiRCR']
    difference = chi2016 - chircr

    label = None
    if chi2016 >= 0.5 and chircr >= 0.5:
        if difference <= -0.09:
            label = 'RCR-like'
            count_rcr_like += 1
        elif difference >= 0.15:
            label = 'Backlobe-like'
            count_backlobe_like += 1
        else:
            label = 'High Chi but Not Classified'
            count_unclassified += 1
    else:
        label = 'Below Chi Threshold'
        count_below_threshold += 1

    # Log result for this event
    print(f'Index {index} | Chi2016: {chi2016:.3f} | ChiRCR: {chircr:.3f} | Δ: {difference:.3f} | Label: {label}')

    # Highlight master event
    if index == 1297 or index == 1298:
        print(f'>>> AWARE Master Event (Index {index}) | Difference: {difference:.3f} | Label: {label}')

# Print summary
print("\n--- Summary ---")
print(f'Backlobe-like:               {count_backlobe_like}')
print(f'RCR-like:                    {count_rcr_like}')
print(f'High Chi but Not Classified: {count_unclassified}')
print(f'Below Chi Threshold:         {count_below_threshold}')
