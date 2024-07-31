#!/bin/bash

#SBATCH --job-name=GF-HSC
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jpic@umich.edu
#SBATCH --nodes=1
#SBATCH --mem-per-gpu=180g
#SBATCH --time=01:00:00
#SBATCH --account=indikar0
#SBATCH --partition=gpu,spgpu,gpu_mig40
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --output=/home/jpic/HSC-logs/%x-%j.log
#SBATCH --array=1-252

### name this something like: geneformerHSCembeddings.sh
###    good practice is to pair the name of this bash script with the name of the python file it will run
### to run this use the following commands:
### ```
### >>> sbatch geneformerHSCembeddings.sh
### >>> squeue --account=indikar0
### ```

# run the script
python geneformerHSCembeddings.py $SLURM_ARRAY_TASK_ID