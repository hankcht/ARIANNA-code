import os
import subprocess

# This function is copied from your SlurmJob.py
def RunMultipleJobs(commandtorun, jobName='Batchjob', runDirectory='/pub/tangch3/ARIANNA/DeepLearning/'):
    cmd = f'{commandtorun}'
    print(f'running {cmd}')

    header = "#!/bin/bash\n"
    header += f"#SBATCH --job-name={jobName}         ##Name of the job.\n"
    header += "#SBATCH -A sbarwick_lab                 ##Account to charge to\n"
    header += "#SBATCH --partition=standard            ##Partition/queue name\n" 
    header += "#SBATCH --time=0-01:00:00               ##Max runtime D-HH:MM:SS, 1 hour\n"
    header += "#SBATCH --nodes=1                       ##Nodes to be used\n"
    header += "#SBATCH --ntasks=10                    ##Number of processes to be launched (CPUs)\n" 
    header += "#SBATCH --mem-per-cpu=6G                ##Requesting 6GB memory per CPU\n"
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
    return


if __name__ == "__main__":

    # Define available model types from your original script
    model_types = [
        '1d_cnn',
        '1d_cnn_freq',
        'parallel',
        'parallel_freq',
        'strided',
        'strided_freq',
        'astrid_2d',
        'astrid_2d_freq'
    ]
    
    # --- Parameters to sweep for DANN ---
    learning_rates = [1e-4, 1e-5]
    lambda_weights = [0.1, 0.5, 1.0] # Hyperparameter for domain loss
    
    # Iterate over model types, learning rates, and lambda weights
    for model_type in model_types:
        for lr in learning_rates:
            for lw in lambda_weights:
                # Note: Changed to R02_DANN_train_and_run.py
                cmd = (
                    f'python /pub/tangch3/ARIANNA/DeepLearning/code/model_runners/R02_DANN_train_and_run.py '
                    f'--learning_rate {lr} '
                    f'--model_type {model_type} '
                    f'--lambda_weight {lw}'
                )
                
                jobName = f'{model_type}_DANN_lr_{lr}_lambda_{lw}'
                RunMultipleJobs(cmd, jobName=jobName)
