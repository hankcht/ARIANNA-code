import os
import subprocess

# This function is copied from your SlurmJob.py
def RunMultipleJobs(commandtorun, jobName='Batchjob', runDirectory='/dfs8/sbarwick_lab/ariannaproject/tangch3/'):
    cmd = f'{commandtorun}'
    print(f'running {cmd}')

    header = "#!/bin/bash\n"
    header += f"#SBATCH --job-name={jobName}         ##Name of the job.\n"
    header += "#SBATCH -A sbarwick_lab                 ##Account to charge to\n"
    header += "#SBATCH --partition=standard            ##Partition/queue name\n" 
    header += "#SBATCH --time=0-01:00:00               ##Max runtime D-HH:MM:SS, 1 hour\n"
    header += "#SBATCH --nodes=1                       ##Nodes to be used\n"
    header += "#SBATCH --ntasks=3                    ##Number of processes to be launched (CPUs)\n" 
    header += "#SBATCH --mem-per-cpu=6G                ##Requesting 6GB memory per CPU\n"
    header += "#SBATCH --output={}\n".format(os.path.join(runDirectory, 'logs', f'{jobName}.out'))
    header += "#SBATCH --error={}\n".format(os.path.join(runDirectory, 'logs', f'{jobName}.err'))
    header += "#SBATCH --mail-type=FAIL\n"
    header += "#SBATCH --mail-user=rricesmi@uci.edu\n"
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

    # Define model types for Autoencoder
    # NOTE: This script currently only provides '1d_autoencoder'
    # You would need to add '1d_autoencoder_freq' to model_builder_autoencoder.py
    # and add it to this list to run it.
    model_types = [
        '1d_autoencoder',
        '1d_autoencoder_freq',
        '1d_autoencoder_tightneck',
        '1d_autoencoder_denoising',
        '1d_autoencoder_mae',
        '1d_autoencoder_dropout'
    ]
    
    # --- Parameters to sweep for Autoencoder ---
    learning_rates = [1e-3, 1e-4, 5e-5, 3e-5, 1e-5, 5e-6, 3e-6, 1e-6]
    
    # Iterate over model types and learning rates
    for model_type in model_types:
        for lr in learning_rates:
            # Note: Changed to R03_Autoencoder_train_and_run.py
            cmd = (
                f'python /dfs8/sbarwick_lab/ariannaproject/tangch3/ARIANNA-code/model_runners/R03_Autoencoder_train_and_run.py '
                f'--learning_rate {lr} '
                f'--model_type {model_type}'
            )
            
            jobName = f'{model_type}_lr_{lr}'
            RunMultipleJobs(cmd, jobName=jobName)
