#!/bin/bash
#SBATCH --job-name=inference    # create a short name for your job
#SBATCH -p a100-large
#SBATCH --nodes=1               # node count
#SBATCH --ntasks-per-node=4      # total number of tasks per node
#SBATCH --gres=gpu:1            # number of gpus per node
#SBATCH --time=8:00:00          # total run time limit (HH:MM:SS)
#SBATCH --error=/gpfs/scratch/eswong/GenZProt/logs/genzprot.%J.err
#SBATCH --output=/gpfs/scratch/eswong/GenZProt/logs/genzprot.%J.out

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfs/software/Anaconda/lib
source activate cgvae

MPATH=../ckpt/model_seed_12345
ca_trace_path=../data/eIF4E/ncgs10_recon_traj.pdb
top_path=../data/eIF4E/eIF4E_noH.pdb

python inference.py -load_model_path ${MPATH} -ca_trace_path ${ca_trace_path} -topology_path ${top_path} -logdir ../logs/eIF4E -device 0