#!/bin/bash

#SBATCH --array=0-30
#SBATCH --job-name=scGPT
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jpic@umich.edu
#SBATCH --nodes=1
#SBATCH --mem-per-gpu=180g
#SBATCH --time=5:00:00
#SBATCH --account=indikar0
#SBATCH --partition=gpu,spgpu,gpu_mig40
#SBATCH --gpus=1
#SBATCH --output=/home/jpic/scFoundationProject/scFoundationModels/notebooks/datasets/logs/%x-%j.log

conda activate scgpt3
python ts_embedding-scGPT.py $SLURM_ARRAY_TASK_ID
conda deactivate