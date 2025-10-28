import os
import subprocess

def RunMultipleJobs(commandtorun, jobName='Batchjob', runDirectory='/pub/tangch3/ARIANNA/DeepLearning/'):
    cmd = f'{commandtorun}'
    print(f'running {cmd}')

    header = "#!/bin/bash\n"
    header += f"#SBATCH --job-name={jobName}         ##Name of the job.\n"
    header += "#SBATCH -A sbarwick_lab                 ##Account to charge to\n"
    header += "#SBATCH --partition=standard            ##Partition/queue name\n" 
    header += "#SBATCH --time=0-01:00:00               ##Max runtime D-HH:MM:SS, 3 days free maximum\n"
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
   
    header += "module load python/3.8.0\n"
    header += "cd $pub/FROG\n"



    slurm_name = os.path.join(runDirectory, 'sh', os.path.basename(jobName)) + ".sh"
    print(f'running cmd {cmd}')
    with open(slurm_name, 'w') as fout:
        fout.write(header)
        fout.write(cmd)
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

    # Define available model types
    model_types = ['1d_cnn', 'parallel', 'strided', 'astrid_2d'] # 'parallel_strided', need debug
    
    # Run refactored model with multiple learning rate settings and model types
    learning_rates = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    
    # Iterate over both model types and learning rates
    for model_type in model_types:
        for lr in learning_rates:
            cmd = f'python /pub/tangch3/ARIANNA/DeepLearning/code/200s_time/model_runners/R01_1D_CNN_train_and_run.py --learning_rate {lr} --model_type {model_type}'
            RunMultipleJobs(cmd, jobName=f'{model_type}_lr_{lr}') # check A0 to input config path
    