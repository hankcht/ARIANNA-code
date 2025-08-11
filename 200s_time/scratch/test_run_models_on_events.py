import sys, os, keras
import numpy as np
import matplotlib.pyplot as plt

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from A0_Utilities import load_data, load_sim_rcr, pT
station_id = [14,17,19,30]
amp_type = '200s'
for id in station_id:
    '''load old rcr'''
    # path = f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/'                                                                                                     #Set which amplifier to run on
    # RCR_path = f'simulatedRCRs/{amp_type}_2.9.24/'
    # backlobe_path = f'simulatedBacklobes/{amp_type}_2.9.24/'
    # rcr, sim_Backlobe = load_sim(path, RCR_path, backlobe_path, amp_type)
    '''load new rcr'''
    sim_folder = f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedRCRs/{amp_type}/5.28.25/'
    sim_rcr = load_sim_rcr(sim_folder, noise_enabled=True, filter_enabled=True, amp=amp_type)

    '''load old above curve'''
    # All_data_SNR, All_data_Chi, All_Traces, All_data_UNIX = load_data('AboveCurve_data', amp_type, id)
    '''load new above curve'''
    Above_curve_data_SNR, Above_curve_data_Chi2016, Above_curve_data_ChiRCR, all_Traces2016, all_TracesRCR, Above_curve_data_UNIX = load_data('new_chi_above_curve', amp_type, id)
    
    '''select 200 events for each station'''
    All_Traces = np.array(all_TracesRCR)
    # num_total_events = All_Traces.shape[0]
    # selected_indices = np.random.choice(num_total_events, 200, replace=False)
    # random_200_events = All_Traces[selected_indices]

    '''predict with old model'''
    # network_output = RunTrainedModel(random_200_events, '/pub/tangch3/ARIANNA/DeepLearning/models/')
    # prob_RCR = RunTrainedModel(sim_rcr, '/pub/tangch3/ARIANNA/DeepLearning/models/')
    '''predict with new model'''
    model = keras.models.load_model(f'/pub/tangch3/ARIANNA/DeepLearning/models/200s_time/new_chi_2025-06-20_16-06_RCR_Backlobe_model_2Layer.h5')
    network_output = model.predict(All_Traces)
    prob_RCR = model.predict(sim_rcr)

    plt.figure(figsize=(10, 6))
    plt.hist(network_output, bins=50, range=(0, 1), edgecolor='black', alpha=0.7)
    plt.hist(prob_RCR, bins=20, range=(0,1), histtype='step', color='red', linestyle='solid', label=f'RCR {len(prob_RCR)}', density=False)
    plt.title(f'Distribution of Network Output for Station {id} {len(network_output)} events')
    plt.xlabel('Network Output (Probability)')
    plt.ylabel('Number of Events')
    plt.yscale('log')
    plt.grid(axis='y', alpha=0.75)
    plt.axvline(x=0.9, color='red', linestyle='--', label='Threshold = 0.9')
    plt.legend()
    plt.show() 
    plot_output_dir = '/pub/tangch3/ARIANNA/DeepLearning/'
    os.makedirs(plot_output_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_output_dir, f'new_mod_on_RCRtemplate_network_output_distribution_stn{id}.png'))
    plt.clf() # Clear the current figure

    '''plot RCR like events'''
    threshold = 0.9
    high_output_indices = np.where(network_output > threshold)[0]
    events_above_threshold_traces = All_Traces[high_output_indices]

    print(f"\nTotal events: {len(network_output)}")
    print(f"Events with network output > {threshold}: {len(high_output_indices)}")
    print(f"Shape of traces for events above threshold: {events_above_threshold_traces.shape}")

    plot_output_directory = '/pub/tangch3/ARIANNA/DeepLearning/potential_RCR_plots/'
    os.makedirs(plot_output_directory, exist_ok=True)

    for event_data, original_index in zip(events_above_threshold_traces, high_output_indices):
        plot_filename = os.path.join(plot_output_directory, f'7.2_potential_RCR_event_original_idx_{original_index}.png')
        pT(event_data, f'Potential RCR (Original Event Index: {original_index})', plot_filename)
        print(f"Plotting and saving event with original index {original_index} to {plot_filename}")