#!/bin/bash
#SBATCH --job-name=runScript  # Specify job name
#SBATCH --partition=shared   # Specify partition name
#SBATCH --nodes=1              # Specify number of nodes
#SBATCH --mem=24000            # Specify memory to be used for job (MB) #24000 
#SBATCH --time=08:00:00        # Set a limit on the total run time
#SBATCH --mail-type=FAIL       # Notify user by email in case of job failure
#SBATCH --account=bb1170       # Charge resources on this project account
#SBATCH --output=/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/2019_10/my_job.o%j    # File name for standard output
#SBATCH --error=/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/2019_10/my_job.e%j     # File name for standard error output

#start this script with: sbatch start_indiv_bash.sh PathToIndFile File.sh
cd $1
./$2
