from A0_Utilities import combine_plots

# no station 13 because 13 does not require this feature
for id in [14,15,17,18,19,30]:
    snr_plot_path = f'/pub/tangch3/ARIANNA/DeepLearning/plots/SNR_NetworkOutput/Station_{id}'
    combine_plots(snr_plot_path, id, type='SNR')
    print('SNR plots combined')
    chi_plot_path = f'/pub/tangch3/ARIANNA/DeepLearning/plots/Chi_NetworkOutput/Station_{id}'
    combine_plots(chi_plot_path, id, type='Chi')
    print('Chi plots combined')