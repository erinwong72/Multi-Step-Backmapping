import sys
# sys.path.append("../scripts/")
# sys.path.append("../src/")
import gc
import os
import argparse, signal, subprocess
import functools
import pandas as pd
import CoarseGrainingVAE
from CoarseGrainingVAE.data import *
from CoarseGrainingVAE.cgvae import *
from CoarseGrainingVAE.conv import *
from CoarseGrainingVAE.datasets_fsdp import *
from utils import *
from CoarseGrainingVAE.visualization import xyz_grid_view, rotate_grid, save_rotate_frames
from CoarseGrainingVAE.sidechain import *

# import sleep
from time import sleep
from sampling import *
from run_baseline import retrieve_recon_structures
import builtins
import torch
from torch import nn
from torch.nn import Sequential

import numpy as np
import copy
from torch_scatter import scatter_mean
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import json
import time
from datetime import timedelta
import statsmodels.api as sm
import wandb
import mdtraj
from time import sleep
import yaml

# set random seed
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

optim_dict = {'adam':  torch.optim.Adam, 'sgd':  torch.optim.SGD}


def build_split_dataset(traj, args, config, mapping=None, split=None, cg_coords=None):
    atomic_nums, protein_index = get_atomNum(traj)

    new_mapping, frames, cg_coord = get_cg_and_xyz(traj, args, config, cg_method=config.cg_method, n_cgs=config.n_cgs,
                                                     mapshuffle=config.mapshuffle, mapping=mapping)

    if mapping is None:
        mapping = new_mapping
    if cg_coords is not None:
        cg_coord = mdtraj.load(cg_coords).xyz * 10.0
        print(cg_coord)
    dataset = build_dataset(mapping,
                        frames,
                        config.atom_cutoff,
                        config.cg_cutoff,
                        atomic_nums,
                        traj.top,
                        order=config.edgeorder ,
                        cg_traj=cg_coord, split=split)
    return None, None
    # get n_atoms
    if config.cg_radius_graph:
        dataset.generate_neighbor_list(atom_cutoff=config.atom_cutoff, cg_cutoff=None, device="cpu", undirected=True)
    else:
        dataset.generate_neighbor_list(atom_cutoff=config.atom_cutoff, cg_cutoff= config.cg_cutoff, device="cpu", undirected=True)

    # if auxcutoff is defined, use the aux cutoff
    if config.auxcutoff > 0.0:
        dataset.generate_aux_edges(config.auxcutoff)

    return dataset, mapping

def run_cv(args, run=None):
    run_id = args.run_id
    config = yaml.load(open(f"configs/config_{run_id}.yaml"), Loader=yaml.FullLoader)
    # turn into argparse object
    config = argparse.Namespace(**config)
    # turn config dict into object
    config.cross = args.cross
    config.cg_radius_graph = args.cg_radius_graph
    config.shuffle = args.shuffle
    config.cg_mp = args.cg_mp
    config.det = args.det
    config.graph_eval = args.graph_eval
    config.tqdm_flag = args.tqdm_flag
    config.invariantdec = args.invariantdec
    config.reflectiontest = args.reflectiontest
    working_dir = config.logdir
    n_cgs = config.n_cgs
    n_basis = config.n_basis
    n_rbf = config.n_rbf
    atom_cutoff = config.atom_cutoff
    cg_cutoff = config.cg_cutoff
    enc_nconv = config.enc_nconv
    dec_nconv = config.dec_nconv
    batch_size = config.batch_size
    beta = config.beta
    #ndata = config.ndata
    nsamples = config.nsamples
    nepochs = config.nepochs
    lr = config.lr
    activation = config.activation
    optim = optim_dict[config.optimizer]
    dataset_label = args.dataset
    shuffle_flag = config.shuffle
    cg_mp_flag = config.cg_mp
    nevals = config.nevals
    graph_eval = config.graph_eval
    tqdm_flag = config.tqdm_flag
    n_ensemble = config.n_ensemble
    det = config.det
    gamma = config.gamma
    factor = config.factor
    patience = config.patience
    eta = 0.0
    kappa = 0.0
    config.mapshuffle = 0.0
    threshold = config.threshold
    edgeorder = config.edgeorder
    config.auxcutoff = 0.0
    invariantdec = False
    config.device = 0
    epoch = 29
    cg_coords = args.cg_coords

    failed = False
    min_lr = 5e-8

    if det:
        beta = 0.0
        print("Recontruction Task")
    else:
        print("Sampling Task")

    # download data from mdshare
    #mdshare.fetch('pentapeptide-impl-solv.pdb', working_directory='../data')
    #mdshare.fetch('pentapeptide-*-500ns-impl-solv.xtc', working_directory='../data')
    #mdshare.fetch('alanine-dipeptide-nowater.pdb', working_directory='../data')
   # mdshare.fetch('alanine-dipeptide-*-250ns-nowater.xtc', working_directory='../data')
    device = args.device
    print('device', device)
    
    #traj = md.load('/gpfs/home/eswong/CoarseGrainingVAE/data/Ca-eIF4E/Ca-eIF4E_test_traj_full.pdb')
    split_dir = f"/gpfs/scratch/eswong/CoarseGrainingVAE/scripts/{run_id}"
    args.split_dir = split_dir
    #kf = KFold(n_splits=nsplits, shuffle=True)

    #split_iter = kf.split(list(range(ndata)))

    cv_stats_pd = pd.DataFrame( {'test_all_recon': [],
                    'test_heavy_recon': [],
                    'recon_all_ged': [], 'recon_heavy_ged': [],
                    'recon_all_valid_ratio': [],
                    'recon_heavy_valid_ratio': [],
                    'sample_all_ged': [], 'sample_heavy_ged': [],
                    'sample_all_valid_ratio': [],
                    'sample_heavy_valid_ratio': [],
                    'sample_all_rmsd': [], 'sample_heavy_rmsd':[]}  )

    #for i, (train_index, test_index) in enumerate(split_iter):

        # start timing
    start =  time.time()

        #split_dir = os.path.join(working_dir, 'fold{}'.format(i))
        #if args.rank == 0:
            #create_dir(split_dir)

    if os.path.exists(f'/gpfs/home/eswong/CoarseGrainingVAE/scripts/mapping_{dataset_label}_ncgs{config.n_cgs}.pt'):
        if dataset_label == "RBCG-sarscov2":
            if config.n_cgs == 100:
                mapping = torch.load(f'/gpfs/home/eswong/CoarseGrainingVAE/scripts/mapping_{dataset_label}_ncgs{config.n_cgs}_better.pt')
            elif config.n_cgs == 15:
                mapping = torch.load(f'/gpfs/home/eswong/CoarseGrainingVAE/scripts/mapping_RBCG-sarscov2_ncgs15.pt')
        else:
            mapping = torch.load(f'/gpfs/home/eswong/CoarseGrainingVAE/scripts/mapping_{dataset_label}_ncgs{config.n_cgs}.pt')
    else:
        mapping = None
    with open("/gpfs/home/eswong/CoarseGrainingVAE/scripts/indices_sarscov2.json", 'r') as f:
        indices = json.load(f)
        test_index = indices['test']
        train_index = indices['train']
        val_index = indices['val']
    #test_set = md.load('/gpfs/home/eswong/CoarseGrainingVAE/data/Ca-eIF4E/Ca-eIF4E_test_traj_connect.pdb')
    
    if dataset_label == "Ca-eIF4E":
        test_set = md.load('/gpfs/home/eswong/CoarseGrainingVAE/data/Ca-eIF4E/Ca-eIF4E_test_traj_connect.pdb')
    elif dataset_label == "SBCG-eIF4E":
        test_set = md.load('/gpfs/home/eswong/CoarseGrainingVAE/data/SBCG-eIF4E/sbcg50_test_traj_connect.pdb')
    elif dataset_label == "RBCG-sarscov2":
        #test_set = md.load('/gpfs/home/eswong/CoarseGrainingVAE/data/sarscov2/RBCG/sarscov2_0_0_RBCG.xtc',top='/gpfs/home/eswong/CoarseGrainingVAE/data/sarscov2/RBCG/sarscov2_top_RBCG_connect.pdb')
        mapping = torch.load(f'/gpfs/home/eswong/CoarseGrainingVAE/scripts/mapping_RBCG-sarscov2_ncgs{config.n_cgs}.pt')
        test_set = md.load('/gpfs/scratch/eswong/CoarseGrainingVAE/data/RBCG-sarscov2/sarscov2_FG_noirreg_RBCG_connect.pdb')
    elif "SBCG" in dataset_label and "sarscov2" in dataset_label:
        backmapping_res = int(dataset_label.split('-')[0].split('SBCG')[1])
        test_set = md.load(f'/gpfs/scratch/eswong/CoarseGrainingVAE/data/SBCG-sarscov2/SBCG-sarscov2_test_cg_{backmapping_res}_connect.pdb')
    elif dataset_label == "sarscov2":
        test_set = md.load('/gpfs/scratch/eswong/CoarseGrainingVAE/data/SBCG-sarscov2/sarscov2_FG_test_traj.pdb')
    else:
        traj = load_protein_traj(dataset_label)
        traj = shuffle_traj(traj, random_seed=123)
        atomic_nums, protein_index = get_atomNum(traj)
        n_atoms = atomic_nums.shape[0]
        test_set = traj[test_index]
    #else:
        #test_set = traj[test_index]
    # build_split_dataset(traj[train_index], args=args, config=config, mapping=mapping, split='train')
    # build_split_dataset(traj[val_index], args=args, config=config, mapping=mapping, split='val')
    build_split_dataset(test_set, args=args, config=config, mapping=mapping, split='test')
    return None
    #testset, mapping = build_split_dataset(test_set, args=args, config=config, mapping=mapping, split='test', cg_coords=cg_coords)
    #train_sampler = DistributedSampler(trainset, num_replicas=args.world_size, rank=args.rank)
    #val_sampler = DistributedSampler(valset, num_replicas=args.world_size, rank=args.rank)
    testloader = DataLoader(testset, batch_size=batch_size, collate_fn=CG_collate, shuffle=False, pin_memory=True)
    gc.collect()
    torch.cuda.empty_cache()
    # initialize model
    atom_mu = nn.Sequential(nn.Linear(n_basis, n_basis), nn.ReLU(), nn.Linear(n_basis, n_basis)).to(device)
    atom_sigma = nn.Sequential(nn.Linear(n_basis, n_basis), nn.ReLU(), nn.Linear(n_basis, n_basis)).to(device)

    # # register encoder
    # decoder = EquivariantDecoder(n_atom_basis=n_basis, n_rbf = n_rbf,
    #                               cutoff=atom_cutoff, num_conv = dec_nconv, activation=activation,
    #                               cross_flag=params['cross'])

    if n_cgs == 3:
        breaksym= True
    else:
        breaksym = False

    decoder = EquivariantPsuedoDecoder(n_atom_basis=n_basis, n_rbf = n_rbf,
                                    cutoff=atom_cutoff, num_conv = dec_nconv, activation=activation, breaksym=breaksym).to(device)

    encoder = EquiEncoder(n_conv=enc_nconv, n_atom_basis=n_basis,
                                    n_rbf=n_rbf, cutoff=cg_cutoff, activation=activation,
                                    cg_mp=cg_mp_flag, dir_mp=False).to(device)

    # define prior
    cgPrior = CGprior(n_conv=enc_nconv, n_atom_basis=n_basis,
                                    n_rbf=n_rbf, cutoff=cg_cutoff, activation=activation,
                                        dir_mp=False).to(device)

    model = CGequiVAE(encoder, decoder, atom_mu, atom_sigma, n_cgs, feature_dim=n_basis, prior_net=cgPrior,
                            det=det, equivariant= not invariantdec).to(device)

    #model = DDP(model, device_ids=[device], find_unused_parameters=True)
    # load previously saved model
    state_dict = torch.load(os.path.join(split_dir, 'model_final.pt'))
    # check if model was saved with DDP
    if 'module' in list(state_dict.keys())[0]:
#  if it was from model.module
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace('module.', '')  # Each weight k will be prefix with the word "module"
            new_state_dict[k] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    optimizer = optim(model.parameters(), lr=lr)
 
    testing=True
    if testing:
            print("Starting testing")
            atomic_nums, protein_index = get_atomNum(test_set)
            n_atoms = atomic_nums.shape[0]
            # dump learning trajectory
            #recon_hist = np.concatenate(recon_hist)
            #dump_numpy2xyz(recon_hist, atomic_nums, os.path.join(split_dir, 'recon_hist.xyz'))

            # save sampled geometries
            # trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=CG_collate, shuffle=True)
            # train_true_xyzs, train_recon_xyzs, train_cg_xyzs, train_all_valid_ratio, train_heavy_valid_ratio, train_all_ged, train_heavy_ged= get_all_true_reconstructed_structures(trainloader,
            #                                                                                     device,
            #                                                                                     args,
            #                                                                                     model,
            #                                                                                     atomic_nums,
            #                                                                                     n_cg=n_cgs,
            #                                                                                     tqdm_flag=tqdm_flag, reflection=False)

            # sample geometries
            #train_samples = sample(trainloader, mu, sigma, device, model, atomic_nums, n_cgs)

            test_true_xyzs, test_recon_xyzs, test_cg_xyzs, test_all_valid_ratio, test_heavy_valid_ratio, test_all_ged, test_heavy_ged = get_all_true_reconstructed_structures(testloader,
                                                                                                device,
                                                                                                args,
                                                                                                model,
                                                                                                atomic_nums,
                                                                                                n_cg=n_cgs,
                                                                                                tqdm_flag=tqdm_flag, reflection=False)

            test_recon = test_recon_xyzs.reshape(-1, n_atoms, 3)
            test_true = test_true_xyzs.reshape(-1, n_atoms, 3)
            try:
                # save test_true_xyzs, test_recon_xyzs, test_cg_xyzs
                dump_numpy2xyz(test_true, atomic_nums, os.path.join(split_dir, 'test_true_full.xyz'))
                dump_numpy2xyz(test_recon, atomic_nums, os.path.join(split_dir, 'test_recon_full.xyz'))

                sleep(2)

                # load numpy of coordinates
                if dataset_label == 'Ca-eIF4E':
                    top = "/gpfs/home/eswong/CoarseGrainingVAE/data/Ca-eIF4E/traj_alpha_top.pdb"
                elif dataset_label == 'eIF4E':
                    top = "/gpfs/home/eswong/CoarseGrainingVAE/data/eIF4E/eIF4E.gro"
                elif dataset_label == 'RBCG-sarscov2':
                    top = "/gpfs/home/eswong/CoarseGrainingVAE/data/sarscov2/RBCG/sarscov2_top_RBCG_connect.pdb"
                elif dataset_label == 'SBCG-eIF4E':
                    top = "/gpfs/home/eswong/CoarseGrainingVAE/data/SBCG-eIF4E/SBCG50_top.pdb"
                elif "SBCG" in dataset_label and "sarscov2" in dataset_label:
                    top = f'/gpfs/scratch/eswong/CoarseGrainingVAE/data/SBCG-sarscov2/SBCG-sarscov2_train_cg_{backmapping_res}_connect.pdb'
                elif dataset_label == 'sarscov2':
                    top = "/gpfs/home/eswong/CoarseGrainingVAE/data/sarscov2/FG/sarscov2_cleaned.pdb"
                traj = mdtraj.load(f"{split_dir}/test_recon_full.xyz", top=top)

                # save pdb
                traj.save_pdb(f"{split_dir}/{args.run_id}_recon.pdb")

                traj = mdtraj.load(f"{split_dir}/test_true_full.xyz", top=top)

                # save pdb
                traj.save_pdb(f"{split_dir}/test_true.pdb")
            # save as numpy files too
            # end script here
            except Exception as e:
                print(e)
                pass
            #return None
            # this is just to get KL loss
            # test_loss, mean_kl_test, mean_recon_test, mean_graph_test, xyz_test, xyz_test_recon = loop(testloader, optimizer, device,
            #                                         model, beta, epoch,
            #                                         train=False,
            #                                             gamma=gamma,
            #                                             eta=eta,
            #                                             kappa=kappa,
            #                                             looptext='Ncg {}'.format(n_cgs),
            #                                             tqdm_flag=tqdm_flag
            #                                             )

                # sample geometries
                #test_samples = sample(testloader, mu, sigma, device, model, atomic_nums, n_cgs, atomwise_z=atom_decode_flag)

                #dump_numpy2xyz(train_samples[:nsamples], atomic_nums, os.path.join(split_dir, 'train_samples.xyz'))
                # dump_numpy2xyz(train_true_xyzs[:nsamples], atomic_nums, os.path.join(split_dir, 'train_original.xyz'))
                # dump_numpy2xyz(train_recon_xyzs[:nsamples], atomic_nums, os.path.join(split_dir, 'train_recon.xyz'))
                # dump_numpy2xyz(train_cg_xyzs[:nsamples], np.array([6] * n_cgs), os.path.join(split_dir, 'train_cg.xyz'))

                #dump_numpy2xyz(test_samples[:nsamples], atomic_nums, os.path.join(split_dir, 'test_samples.xyz'))
                # dump_numpy2xyz(test_true_xyzs[:nsamples], atomic_nums, os.path.join(split_dir, 'test_original.xyz'))
                # dump_numpy2xyz(test_recon_xyzs[:nsamples], atomic_nums, os.path.join(split_dir, 'test_recon.xyz'))
                # dump_numpy2xyz(test_cg_xyzs[:nsamples], np.array([6] * n_cgs), os.path.join(split_dir, 'test_cg.xyz'))

                # compute test rmsds
            heavy_filter = atomic_nums != 1.
            test_all_dxyz = (test_recon_xyzs - test_true_xyzs)#.reshape(-1)
            test_heavy_dxyz = (test_recon_xyzs - test_true_xyzs).reshape(-1, n_atoms, 3)[:, heavy_filter, :]#.reshape(-1)
            unaligned_test_all_rmsd = np.sqrt(np.power(test_all_dxyz, 2).sum(-1).mean())
            unaligned_test_heavy_rmsd = np.sqrt(np.power(test_heavy_dxyz, 2).sum(-1).mean())

            # compute train rmsd
            # train_all_dxyz = (train_recon_xyzs - train_true_xyzs)#.reshape(-1)
            # train_heavy_dxyz = (train_recon_xyzs - train_true_xyzs).reshape(-1, n_atoms, 3)[:, heavy_filter, :]#.reshape(-1)
            # unaligned_train_all_rmsd = np.sqrt(np.power(train_all_dxyz, 2).sum(-1).mean())
            # unaligned_train_heavy_rmsd = np.sqrt(np.power(train_heavy_dxyz, 2).sum(-1).mean())

            # dump test rmsd
            np.savetxt(os.path.join(split_dir, 'test_all_rmsd{:.4f}.txt'.format(unaligned_test_all_rmsd)), np.array([unaligned_test_all_rmsd]))
            np.savetxt(os.path.join(split_dir, 'test_heavy_rmsd{:.4f}.txt'.format(unaligned_test_heavy_rmsd)), np.array([unaligned_test_heavy_rmsd]))

            ##### generate rotating movies for visualization #####

            sampleloader = DataLoader(testset, batch_size=1, collate_fn=CG_collate, shuffle=False)
            sample_xyzs, data_xyzs, cg_xyzs, recon_xyzs, all_rmsds, all_heavy_rmsds, \
            sample_valid, sample_allatom_valid, sample_graph_val_ratio_list, \
            sample_graph_allatom_val_ratio_list = sample_ensemble(sampleloader, device, args,
                                                                                    model, atomic_nums,
                                                                                    n_cgs, n_sample=n_ensemble,
                                                                                    graph_eval=True, reflection=False)

            if graph_eval:
                sample_valid = np.array(sample_valid).mean()
                sample_allatom_valid = np.array(sample_allatom_valid).mean()

                if all_rmsds is not None:
                    mean_all_rmsd = np.array(all_rmsds)[:, 0].mean()
                else:
                    mean_all_rmsd = None

                if all_heavy_rmsds is not None:
                    mean_heavy_rmsd = np.array(all_heavy_rmsds)[:, 1].mean()
                else:
                    mean_heavy_rmsd = None

                mean_graph_diff = np.array(sample_graph_val_ratio_list).mean()
                mean_graph_allatom_diff = np.array(sample_graph_allatom_val_ratio_list).mean()

            test_stats = {'test_all_recon': unaligned_test_all_rmsd,
                    'test_heavy_recon': unaligned_test_heavy_rmsd,
                    'recon_all_ged': test_all_ged, 'recon_heavy_ged': test_heavy_ged,
                    'recon_all_valid_ratio': test_all_valid_ratio,
                    'recon_heavy_valid_ratio': test_heavy_valid_ratio,
                    'sample_all_ged': mean_graph_allatom_diff, 'sample_heavy_ged': mean_graph_diff,
                    'sample_all_valid_ratio': sample_allatom_valid,
                    'sample_heavy_valid_ratio': sample_valid,
                    'sample_all_rmsd': mean_all_rmsd, 'sample_heavy_rmsd':mean_heavy_rmsd}

            for key in test_stats:
                print(key, test_stats[key])

            cv_stats_pd = cv_stats_pd.append(test_stats, ignore_index=True)
            cv_stats_pd.to_csv(os.path.join(split_dir, 'cv_stats.csv'),  index=False)

            #save_rotate_frames(sample_xyzs, data_xyzs, cg_xyzs, recon_xyzs, n_cgs, n_ensemble, atomic_nums, split_dir)
    return None

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str)
    parser.add_argument("--auxcutoff", type=float, default=0.0)
    parser.add_argument("--eta", type=float, default=0.01)
    parser.add_argument("--kappa", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--mapshuffle", type=float, default=0.0)
    parser.add_argument("--cgae_reg_weight", type=float, default=0.25)
    parser.add_argument("--dec_type", type=str, default='EquivariantDecoder')
    parser.add_argument("--cross", action='store_true', default=False)
    parser.add_argument("--graph_eval", action='store_true', default=True)
    parser.add_argument("--shuffle", action='store_true', default=True)
    parser.add_argument("--cg_mp", action='store_true', default=False)
    parser.add_argument("--tqdm_flag", action='store_true', default=False)
    parser.add_argument("--det", action='store_true', default=False)
    parser.add_argument("--cg_radius_graph", action='store_true', default=False)
    parser.add_argument("--invariantdec", action='store_true', default=False)
    parser.add_argument("--reflectiontest", action='store_true', default=False)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--cg_coords", type=str)
    args = parser.parse_args()
    args.device = 'cuda'
    run_cv(args)
