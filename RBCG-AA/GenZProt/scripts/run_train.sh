#!/bin/bash
#SBATCH --job-name=train_model    # create a short name for your job
#SBATCH -p extended-40core
#SBATCH --nodes=1               # node count
#SBATCH --ntasks-per-node=4      # total number of tasks per node
#SBATCH --time=48:00:00          # total run time limit (HH:MM:SS)
#SBATCH --error=/gpfs/scratch/eswong/GenZProt/logs/genzprot_train.%J.err
#SBATCH --output=/gpfs/scratch/eswong/GenZProt/logs/genzprot_train.%J.out
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfs/software/Anaconda/lib

source activate cgvae
python train_model.py