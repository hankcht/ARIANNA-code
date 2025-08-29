import os
from glob import glob


from refactor_checks import load_all_coincidence_traces, load_2016_backlobe_templates

# template_dir = "/pub/tangch3/ARIANNA/DeepLearning/refactor/confirmed_2016_templates/"
# template_paths = sorted(glob(os.path.join(template_dir, "filtered_Event2016_Stn*.npy"))) # using filtered confirmed BL
# all_2016_backlobes_200, dict_2016_200 = load_2016_backlobe_templates(template_paths, amp_type='200s')
# all_2016_backlobes_100, dict_2016_100 = load_2016_backlobe_templates(template_paths, amp_type='100s')


pkl_path = '/pub/tangch3/ARIANNA/DeepLearning/refactor/coincidence_events/filtered_coinc.pkl'

new_coinc_dict, new_coinc_traces, new_metadata = load_all_coincidence_traces(pkl_path, trace_key='Filtered_Traces') 

# takes Chi2016 and ChiRCR of each event and calculate the difference
# passing condition is  

for index in range(len(new_metadata)):
    
    chi2016 = new_metadata[index]['Chi2016']
    chircr = new_metadata[index]['ChiRCR']
    difference = chi2016 - chircr

    print(f'Difference: 2016 {chi2016:.3f} - RCR {chircr:.3f} = {difference:.3f}')
    if index == 1297 or index == 1298:
        print(f'AWARE Master 578 DIFFERENCE IS {difference:.3f}')
