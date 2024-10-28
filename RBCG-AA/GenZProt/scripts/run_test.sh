#!/bin/bash
#SBATCH --job-name=inference    # create a short name for your job
#SBATCH -p a100
#SBATCH --nodes=1               # node count
#SBATCH --ntasks-per-node=4      # total number of tasks per node
#SBATCH --gres=gpu:1            # number of gpus per node
#SBATCH --mem=250G               # total memory limit
#SBATCH --time=8:00:00          # total run time limit (HH:MM:SS)
#SBATCH --error=/gpfs/scratch/eswong/GenZProt/logs/genzprot_test.%J.err
#SBATCH --output=/gpfs/scratch/eswong/GenZProt/logs/genzprot_test.%J.out
source activate cgvae
ca_trace_paths=(/gpfs/scratch/eswong/CoarseGrainingVAE/scripts/cbnci1da/cbnci1da_recon.pdb)
MPATH=/gpfs/home/eswong/GenZProt/ckpt
test_path=/gpfs/scratch/eswong/CoarseGrainingVAE/data/RBCG-sarscov2/sarscov2_FG_test_traj_noH.pdb
top_path=/gpfs/home/eswong/GenZProt/data/sarscov2/sarscov2_top_noH_reg.pdb
for ca_trace_path in ${ca_trace_paths[@]}; do
    python test_model_modded.py -load_model_path $MPATH -test_data_path $test_path -ca_trace_path $ca_trace_path -topology_path $top_path &
    wait;
done