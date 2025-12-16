import os
import subprocess
import sys

def RunMultipleJobs(commandtorun, jobName='Batchjob', runDirectory='/pub/tangch3/ARIANNA/DeepLearning/', python_env='default'):
    cmd = f'{commandtorun}'
    print(f'running {cmd}')

    header = "#!/bin/bash\n"
    header += f"#SBATCH --job-name={jobName}         ##Name of the job.\n"
    header += "#SBATCH -A sbarwick_lab                 ##Account to charge to\n"
    header += "#SBATCH --partition=standard            ##Partition/queue name\n" 
    header += "#SBATCH --time=0-24:00:00               ##Max runtime D-HH:MM:SS, 3 days free maximum\n"
    header += "#SBATCH --nodes=1                       ##Nodes to be used\n"
    
    header += "#SBATCH --ntasks=10                    ##Number of processes to be launched (CPUs)\n" 
    header += "#SBATCH --mem-per-cpu=6G                ##Requesting 6GB memory per CPU\n"
    # If keep --mem, it will override --mem-per-cpu * --ntasks if it's a lower value.
    # It's better to explicitly calculate and rely on mem-per-cpu * ntasks for clarity.
    # header += "#SBATCH --mem=18G\n" 
    
    header += "#SBATCH --output={}\n".format(os.path.join(runDirectory, 'logs', f'{jobName}.out'))
    header += "#SBATCH --error={}\n".format(os.path.join(runDirectory, 'logs', f'{jobName}.err'))
    header += "#SBATCH --mail-type=FAIL,END\n"
    header += "#SBATCH --mail-user=tangch3@uci.edu\n"


    header += "export PYTHONPATH=$NuM:$PYTHONPATH\n"
    header += "export PYTHONPATH=$Nu:$PYTHONPATH\n"
    header += "export PYTHONPATH=$Radio:$PYTHONPATH\n"
   
    header += "module purge\n"
    if python_env == 'hgq2':
        header += "module load python/3.10.2\n"
        header += "source /pub/tangch3/ARIANNA/DeepLearning/code/hgq2_env/bin/activate\n"
    else:
        header += "module load python/3.8.0\n"



    slurm_name = os.path.join(runDirectory, 'sh', os.path.basename(jobName)) + ".sh"
    print(f'running cmd {cmd}')
    with open(slurm_name, 'w') as fout:
        fout.write(header)
        fout.write(cmd + "\n")
    fout.close()

    slurm_name = 'sbatch ' + slurm_name
    print(f'running {slurm_name}')
    errLoc = os.path.join(runDirectory, 'logs', f'{jobName}.err')
    print(f'Logs at {errLoc}')
    process = subprocess.Popen(slurm_name.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    # process = .Process(os.getpid())
    # print(f"Memory used: {process.memory_info().rss / 1024 ** 2} MB")

    return


def main(multi_run):
    if multi_run:
        # --- Run multiple stations ---
        stations = [13, 15, 18, 14, 17, 19, 30]
        for station_id in stations:
            if station_id in [14, 17, 19, 30]:
                amp = '200s'
                print(f'amp: {amp}')
            elif station_id in [13, 15, 18]:
                amp = '100s'
                print(f'amp: {amp}')

            cmd = f'python /pub/tangch3/ARIANNA/DeepLearning/code/200s_time/B1_BLcurve.py {station_id}'
            # cmd = f'python /pub/tangch3/ARIANNA/DeepLearning/code/200s_time/A2_RealRunCNN.py confirmed_BL'
            RunMultipleJobs(cmd, jobName='BLcurve')
    else:
        cmd = 'python /pub/tangch3/ARIANNA/DeepLearning/code/200s_time/refactor_checks.py'
        # cmd = f'python /pub/tangch3/ARIANNA/DeepLearning/code/200s_time/A2_RealRunCNN.py confirmed_BL'
        RunMultipleJobs(cmd, jobName='workbench')


if __name__ == "__main__":
    # multi_run = True
    # main(multi_run=multi_run)
    # cmd = 'python /pub/tangch3/ARIANNA/DeepLearning/code/200s_time/refactor_converter.py'
    # RunMultipleJobs(cmd, jobName='converter')

    # cmd = f"python /pub/tangch3/ARIANNA/DeepLearning/code/200s_time/HGQ_test_smalljet.py"
    # RunMultipleJobs(cmd, jobName='HGQ2_small_jet', python_env='hgq2')
    epochs = [1795] # 30, 50, 80, 100, 300, 500, 850, 1000, 1200, 1500, 1800, 2000
    for epoch in epochs: 
        cmd = f"python /pub/tangch3/ARIANNA/DeepLearning/code/200s_time/HGQ_1D_CNN.py --epochs {epoch}"
        RunMultipleJobs(cmd, jobName=f'HGQ2_{epoch}', python_env='hgq2')


    # # Define available model types
    # model_types = ['1d_cnn', 'parallel', 'strided', 'astrid_2d'] # 'parallel_strided', need debug
    
    # # Run refactored model with multiple learning rate settings and model types
    # learning_rates = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    
    # # Iterate over both model types and learning rates
    # for model_type in model_types:
    #     for lr in learning_rates:
    #         cmd = f'python /pub/tangch3/ARIANNA/DeepLearning/code/200s_time/R01_1D_CNN_train_and_run.py --learning_rate {lr} --model_type {model_type}'
    #         RunMultipleJobs(cmd, jobName=f'{model_type}_lr_{lr}')
    
    # Original single runs (commented out for reference)
    # for lr in learning_rates:
    #     cmd = f'python /pub/tangch3/ARIANNA/DeepLearning/code/200s_time/refactor_train_and_run.py --learning_rate {lr}'
    #     RunMultipleJobs(cmd, jobName=f'train_and_run_lr_{lr}') # check if the above curve is filtered!
    # for lr in learning_rates: # check prefix in config to match model type
    #     cmd = f'python /pub/tangch3/ARIANNA/DeepLearning/code/200s_time/R01_1D_CNN_train_and_run.py --learning_rate {lr}'
    #     RunMultipleJobs(cmd, jobName=f'train_and_run_lr_{lr}') # check if the above curve is filtered!

    # cmd = 'python /pub/tangch3/ARIANNA/DeepLearning/code/200s_time/model_vizualizer.py'
    # RunMultipleJobs(cmd, jobName=f'model_vizualizer')

    # cmd = 'python /pub/tangch3/ARIANNA/DeepLearning/code/200s_time/refactor_train_and_run_all_amp.py' 
    # RunMultipleJobs(cmd, jobName='train_and_run') # check if the above curve is filtered!
    # cmd = 'python /pub/tangch3/ARIANNA/DeepLearning/code/200s_time/refactor_checks.py'
    # RunMultipleJobs(cmd, jobName='checks')
    # cmd = 'python /pub/tangch3/ARIANNA/DeepLearning/code/200s_time/test_1d_CNN.py'
    # RunMultipleJobs(cmd, jobName='train_and_run')
    # cmd = 'python /pub/tangch3/ARIANNA/DeepLearning/code/200s_time/PCA.py'
    # RunMultipleJobs(cmd, jobName='test')
    # cmd = 'python /pub/tangch3/ARIANNA/DeepLearning/code/200s_time/A0_Utilities.py'
    # RunMultipleJobs(cmd, jobName='test')
    # cmd = 'python /pub/tangch3/ARIANNA/DeepLearning/code/200s_time/chi_study.py'
    # RunMultipleJobs(cmd, jobName='test')
    # cmd = 'python /pub/tangch3/ARIANNA/DeepLearning/code/200s_time/scratch/apply_bandpass_on_coinc.py'
    # RunMultipleJobs(cmd, jobName='bandpass')

# Set parameters
single_file = False # If True, we run each nur file individually (See DO4B1)
stations_to_run_on_pt1 = [14,17,19,30,13,15,18]
stations_to_run_on_pt2 = [14,17,19,30,15,18]

# TODO: PROCESS
'''always remember to first input a new model in A0'''

# Can Train if needed
# cmd = f'python /pub/tangch3/ARIANNA/DeepLearning/code/200s_time/simpleCutForDL2.py 14 4.4.25'
#'python /pub/tangch3/ARIANNA/DeepLearning/code/200s_time/A0_Utilities.py'
#
    #'python /pub/tangch3/ARIANNA/DeepLearning/code/200s_time/A1.5_StudyCNN.py data_data'
    #  

# RunMultipleJobs(cmd , jobName=f'genericBatchJob')

# Part One: sim and data network output analysis
# 
# cmd = f'python /pub/tangch3/ARIANNA/DeepLearning/code/200s_time/A2_RealRunCNN.py sim'
# RunMultipleJobs(cmd , jobName=f'genericBatchJob')

 
# confirmed_stations = [14,15,17,18,30]
# for station_id in confirmed_stations:
#     if station_id in [14,17,19,30]:
#         amp = '200s'   
#         print(f'amp: {amp}') 
#     elif station_id in [13,15,18]:
#         amp = '100s'
#         print(f'amp: {amp}') 

#     cmd = f'python /pub/tangch3/ARIANNA/DeepLearning/code/200s_time/A2_RealRunCNN.py data --station {station_id}'
#     # cmd = f'python /pub/tangch3/ARIANNA/DeepLearning/code/200s_time/A2_RealRunCNN.py confirmed_BL'       
#     RunMultipleJobs(cmd , jobName=f'genericBatchJob')

# Part Two: analysis on Chi and SNR
'Input partition numbers, stn 17 requires up to 3'
paths = [f'/pub/tangch3/ARIANNA/DeepLearning/code/200s_time/A3_SNR_NetworkOutput.py data --station',
         f'/pub/tangch3/ARIANNA/DeepLearning/code/200s_time/A4_Chi_NetworkOutput.py data --station'
]

# start with stations_to_run_on_pt1
# for station_id in stations_to_run_on_pt1:
#     # stations_to_run_on_pt2, maybe add argparse later
#     if station_id in [14,17,19,30]:
#         amp = '200s'   
#         print(f'amp: {amp}') 
#     elif station_id in [13,15,18]:
#         amp = '100s'
#         print(f'amp: {amp}') 

#     for path in paths:
#         cmd = f'python {path} {station_id}'      
#         RunMultipleJobs(cmd , jobName=f'genericBatchJob')

# # then for partition 3
# for path in paths:
#     cmd = f'python {path} 17'      
#     RunMultipleJobs(cmd , jobName=f'genericBatchJob')

# 'Then combine the SNR/Chi NetworkOutput plots'




# OLD ------------------------------------------------------------

# for station_id in stations_to_run_on:
#     if station_id in [14,17,19,30]:
#         amp = '200s'   
#         print(f'amp: {amp}') 
#     elif station_id in [13,15,18]:
#         amp = '100s'
#         print(f'amp: {amp}') 
#     else:
#         print('wrong station number')
#         break

#     station_path = f"/dfs8/sbarwick_lab/ariannaproject/station_nur/station_{station_id}/"
#     i = 0
#     if single_file:
#         for file in os.listdir(station_path):
#             if file.endswith('_statDatPak.root.nur'):
#                 continue    
#             else:
#                 filename = os.path.join(station_path, file)
#                 cmd = f'python /pub/tangch3/ARIANNA/DeepLearning/code/200s_time/B0_converter.py {station_id} --single_file {filename}'
#                 RunMultipleJobs(cmd, jobName=f'Stn{station_id}_{i}')
#                 i += 1
#     else:
#         cmd = f'python /pub/tangch3/ARIANNA/DeepLearning/code/200s_time/A4_Chi_NetworkOutput.py data --station {station_id}' # Here single_file = False in B1 argument
#         # /pub/tangch3/ARIANNA/DeepLearning/code/200s_time/B1_BLcurve.py {station_id} --RCRorBL RCR
#         RunMultipleJobs(cmd , jobName=f'genericBatchJob')



# if __name__ == "__main__":

#     if_sim = 'sim_sim'

#     cmd = f'python /pub/tangch3/ARIANNA/DeepLearning/code/200s_time/A1_TrainAndRunCNN.py {if_sim}'
#     RunMultipleJobs(cmd , jobName=f'genericBatchJob')

#     # runs = ['sim_sim','data_data']
#     # for i in runs:
#     #     if_sim = i

#     #     cmd = f'python DeepLearning/code/200s_time/TrainAndRunCNN.py {if_sim}'
#     #     RunMultipleJobs(cmd , jobName=f'genericBatchJob')


