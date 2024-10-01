#!/bin/bash

#SBATCH --job-name=GF1k
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jpic@umich.edu
#SBATCH --nodes=1
#SBATCH --mem-per-gpu=100g
#SBATCH --time=12:00:00
#SBATCH --account=indikar0
#SBATCH --partition=gpu,spgpu,gpu_mig40
#SBATCH --gpus=1
#SBATCH --output=/home/jpic/logs/%x-%j.log

# ## name this something like: geneformerHSCembeddings.sh
# ##    good practice is to pair the name of this bash script with the name of the python file it will run
# ## to run this use the following commands:
# ## ```
# ## >>> sbatch geneformerHSCembeddings.sh
# ## >>> squeue --account=indikar0
# ## ```

# run the script
python geneformer-1k-jpic.py
