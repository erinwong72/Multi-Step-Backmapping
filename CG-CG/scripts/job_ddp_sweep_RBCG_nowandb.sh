#!/bin/bash
#SBATCH --job-name=rerun   # create a short name for your job
#SBATCH -p extended-28core
#SBATCH --nodes=1               # node count
#SBATCH --ntasks-per-node=4     # total number of tasks per node
#SBATCH --time=4:00:00          # total run time limit (HH:MM:SS)
#SBATCH --error=/gpfs/scratch/eswong/CoarseGrainingVAE/logs/sweep_extended_ddp.%J.err
#SBATCH --output=/gpfs/scratch/eswong/CoarseGrainingVAE/logs/sweep_extended_ddp.%J.out

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfs/software/Anaconda/lib
source activate cgvae

# to AA
#python create_wandb_sweep.py
# get environmental variable
#SWEEP_ID=$(cat sweep_id.txt)


# delete the file
#rm sweep_id.txt
for id in 1xgt0cil; do
    python run_ala_ddp_rerun.py --run_id $id --queue "gpu-long" --n_gpus 4 --dataset "RBCG-eIF4E" --new_seed 1234 &
    wait;
done

# to alpha C CG
#python run_ala.py -logdir ./Ca-eIF4E_exp_19_dec6 -dataset Ca-eIF4E -device 0 -n_cgs 19 -batch_size 2 -nsamples 2 -ndata 3003 -nepochs 100 -atom_cutoff 30.0 -cg_cutoff 50.0 -nsplits 2 -beta 0.05 -gamma 50.0 -eta 0.0 -kappa 0.0 -activation swish -dec_nconv 6 -enc_nconv 2 -lr 0.0001 -n_basis 300 -n_rbf 10 -cg_method cgae --graph_eval -n_ensemble 2 -factor 0.3 -patience 15
