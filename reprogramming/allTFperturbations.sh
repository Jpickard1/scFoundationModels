#!/bin/bash

#SBATCH --job-name=TFs-All
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jpic@umich.edu
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=100g
#SBATCH --time=1:00:00
#SBATCH --account=indikar0
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --output=/home/jpic/logs/%x-%j.log

### good practice is to pair the name of this bash script with the name of the python file it will run
### to run this use the following commands:
### ```
### >>> sbatch geneformerHSCembeddings.sh
### >>> squeue --account=indikar0
### ```

# run the script
python allTFperturbations.py
