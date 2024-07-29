#!/bin/bash

#SBATCH --job-name=GF-HSC
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=oliven@umich.edu
#SBATCH --nodes=1
#SBATCH --mem-per-gpu=180g
#SBATCH --time=10:00:00
#SBATCH --account=indikar0
#SBATCH --partition=gpu,spgpu,gpu_mig40
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --gpus=1
#SBATCH --output=/home/oliven/%x-%j.log

### name this something like: geneformerHSCembeddings.sh
###    good practice is to pair the name of this bash script with the name of the python file it will run
### to run this use the following commands:
### ```
### >>> sbatch geneformerHSCembeddings.sh
### >>> squeue --account=indikar0
### ```

# run the script
python geneformerHSCembeddings.py