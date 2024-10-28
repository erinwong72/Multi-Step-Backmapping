#!/bin/bash
#SBATCH --job-name=bead100 # create a short name for your job
#SBATCH -p extended-96core
#SBATCH --nodes=1               # node count
#SBATCH --ntasks-per-node=4      # total number of tasks per node
#SBATCH --time=7-00:00:00          # total run time limit (HH:MM:SS)
#SBATCH --error=/gpfs/scratch/eswong/CoarseGrainingVAE/logs/sweep_long_FG_ddp.%J.err
#SBATCH --output=/gpfs/scratch/eswong/CoarseGrainingVAE/logs/sweep_long_FG_ddp.%J.out

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfs/software/Anaconda/lib
source activate cgvae

# to AA
#python create_wandb_sweep.py
# get environmental variable
#SWEEP_ID=$(cat sweep_id.txt)
# ncgs=15 1vixixfu ncgs=35 nq96sv84 ncgs=100 gidynicz
SWEEP_ID=8eg4z070


# delete the file
#rm sweep_id.txt

for i in {1..30}; do
    wandb agent ${SWEEP_ID} --entity erinwong510 --project sequential_backmapping --count 1 &
    wait;
done


# to alpha C CG
#python run_ala.py -logdir ./Ca-eIF4E_exp_19_dec6 -dataset Ca-eIF4E -device 0 -n_cgs 19 -batch_size 2 -nsamples 2 -ndata 3003 -nepochs 100 -atom_cutoff 30.0 -cg_cutoff 50.0 -nsplits 2 -beta 0.05 -gamma 50.0 -eta 0.0 -kappa 0.0 -activation swish -dec_nconv 6 -enc_nconv 2 -lr 0.0001 -n_basis 300 -n_rbf 10 -cg_method cgae --graph_eval -n_ensemble 2 -factor 0.3 -patience 15
