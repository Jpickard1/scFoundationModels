#!/bin/bash
#SBATCH --job-name=kmeans_clustering
#SBATCH --array=5-100          # Range of cluster numbers to test
#SBATCH --ntasks=1            # Number of tasks per array job
#SBATCH --cpus-per-task=4     # Number of CPU cores per task
#SBATCH --time=06:00:00       # Time limit
#SBATCH --mem=16G              # Memory per node
#SBATCH --output=kmeans_%A_%a.out  # Standard output log
#SBATCH --error=kmeans_%A_%a.err   # Standard error log


# module load anaconda3
# source activate geneformer2

# Set cluster number and run based on SLURM_ARRAY_TASK_ID
N_CLUSTERS=$SLURM_ARRAY_TASK_ID
RUNS=10

# Iterate kmeans
for (( run=1; run<=RUNS; run++ ))
do
    echo "Now clustering kmeans_${N_CLUSTERS}_${run}:"
    python kmeans_clustering_n_9_20.py --n_clusters $N_CLUSTERS --run $run
done
