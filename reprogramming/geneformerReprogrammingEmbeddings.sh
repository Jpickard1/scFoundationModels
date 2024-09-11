#!/bin/bash

#SBATCH --job-name=GF-REP
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jpic@umich.edu
#SBATCH --nodes=1
#SBATCH --mem-per-gpu=100g
#SBATCH --time=03:00:00
#SBATCH --account=indikar0
#SBATCH --partition=gpu,spgpu,gpu_mig40
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --output=/home/jpic/logs/%x-%j.log
#SBATCH --array=0-40

# ## name this something like: geneformerHSCembeddings.sh
# ##    good practice is to pair the name of this bash script with the name of the python file it will run
# ## to run this use the following commands:
# ## ```
# ## >>> sbatch geneformerHSCembeddings.sh
# ## >>> squeue --account=indikar0
# ## ```

# run the script
python geneformerReprogrammingEmbeddings.py
