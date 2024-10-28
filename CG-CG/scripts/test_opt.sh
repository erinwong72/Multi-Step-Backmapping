#!/bin/bash
#SBATCH --job-name=test    # create a short name for your job
#SBATCH -p a100
#SBATCH --nodes=1               # node count
#SBATCH --ntasks-per-node=8     # total number of tasks per node
#SBATCH --gres=gpu:1              # number of gpus per node
#SBATCH --mem=120G
#SBATCH --time=8:00:00          # total run time limit (HH:MM:SS)
#SBATCH --error=/gpfs/scratch/eswong/CoarseGrainingVAE/logs/test_model.%J.err
#SBATCH --output=/gpfs/scratch/eswong/CoarseGrainingVAE/logs/test_model.%J.out

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfs/software/Anaconda/lib
source activate cgvae

# to AA
#python create_wandb_sweep.py
# get environmental variable
#SWEEP_ID=$(cat sweep_id.txt)
# delete the file
#rm sweep_id.txt
for id in "sbcg5-35sarscov2"; do
    python test_opt.py --run_id $id --dataset SBCG35-sarscov2 #--cg_coords "/gpfs/scratch/eswong/CoarseGrainingVAE/scripts/sbcg5-25sarscov2/sbcg5-25sarscov2_recon.pdb" &
    wait;
done


# to alpha C CG
#python run_ala.py -logdir ./Ca-eIF4E_exp_19_dec6 -dataset Ca-eIF4E -device 0 -n_cgs 19 -batch_size 2 -nsamples 2 -ndata 3003 -nepochs 100 -atom_cutoff 30.0 -cg_cutoff 50.0 -nsplits 2 -beta 0.05 -gamma 50.0 -eta 0.0 -kappa 0.0 -activation swish -dec_nconv 6 -enc_nconv 2 -lr 0.0001 -n_basis 300 -n_rbf 10 -cg_method cgae --graph_eval -n_ensemble 2 -factor 0.3 -patience 15
