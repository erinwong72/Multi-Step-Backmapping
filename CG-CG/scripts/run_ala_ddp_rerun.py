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
import yaml
from sampling import * 
from run_baseline import retrieve_recon_structures
import builtins
import torch
from torch import nn
from torch.nn import Sequential
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# parallel processing
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
# ddp
import submitit

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
# set random seed 
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

optim_dict = {'adam':  torch.optim.Adam, 'sgd':  torch.optim.SGD}

 
def build_split_dataset(traj, args, config, mapping=None):

    atomic_nums, protein_index = get_atomNum(traj)
    new_mapping, frames, cg_coord = get_cg_and_xyz(traj, args, config, cg_method=config.cg_method, n_cgs=config.n_cgs,
                                                     mapshuffle=config.mapshuffle, mapping=mapping)

    if mapping is None:
        mapping = new_mapping

    dataset = build_dataset(mapping,
                        frames, 
                        config.atom_cutoff, 
                        config.cg_cutoff,
                        atomic_nums,
                        traj.top,
                        order=config.edgeorder ,
                        cg_traj=cg_coord)
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
    # load yaml config file into config variable
    run_id = args.run_id
    config = yaml.load(open(f"/gpfs/home/eswong/CoarseGrainingVAE/scripts/configs/config_{run_id}.yaml"), Loader=yaml.FullLoader)
    # turn into argparse object
    config = argparse.Namespace(**config)
    init_dist_gpu(gpu=None, args=args)
    # if args.rank == 0:
    #     run = wandb.init(project='sequential_backmapping', dir="/gpfs/scratch/eswong/CoarseGrainingVAE/logs/")
    #config = args
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
    nsamples = config.nsamples
    nepochs = config.nepochs
    lr = config.lr
    activation = config.activation
    optim = optim_dict["adam"]
    #config.dataset = args.dataset:
    #if args.dataset:
    #config.dataset = args.dataset
    #else:
        #args.dataset = config.dataset
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
    config.device = args.rank
    #run_id = wandb.run.id
    args.dataset_label = dataset_label
    new_seed = False
    if args.new_seed != 123:
        seed = args.new_seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        new_seed = True


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
    # if dataset_label in PROTEINFILES.keys():
    #     print(dataset_label)
    #     # if dataset_label == 'RBCG-sarscov2':
    #     #     fg_label = 'sarscov2'
    #     #     fg_traj = load_protein_traj(fg_label)
    #     #     fg_traj = shuffle_traj(fg_traj, random_seed=seed)
    #     # if 'PED' in dataset_label:
    #     #     fg_label = dataset_label.split('_')[0]
    #     #     fg_traj = load_protein_traj(fg_label)
    #     #     fg_traj = shuffle_traj(fg_traj, random_seed=seed)
    #     traj = load_protein_traj(dataset_label)
    #     traj = shuffle_traj(traj, random_seed=seed)
    #     atomic_nums, protein_index = get_atomNum(traj)
    #     n_atoms = atomic_nums.shape[0]
    # else:
    #     raise ValueError("data label {} not recognized".format(dataset_label))
    # create subdirectory
    if args.rank == 0:
        #working_dir = annotate_job(config.cg_method + '_ndata{}'.format(config.ndata), config.logdir, config.n_cgs, run_id)
        working_dir = run_id
        # replace home in workingdir with scratch

        working_dir = f'/gpfs/scratch/eswong/CoarseGrainingVAE/scripts/{working_dir}_{seed}'
        create_dir(working_dir)
        split_dir = working_dir
    #kf = KFold(n_splits=nsplits, shuffle=True)

    #split_iter = kf.split(list(range(ndata)))
     
    cv_stats_pd = pd.DataFrame( { 'train_recon': [],
                    'test_all_recon': [],
                    'test_heavy_recon': [],
                    'train_KL': [], 'test_KL': [], 
                    'train_graph': [],  'test_graph': [],
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
        mapping = torch.load(f'/gpfs/home/eswong/CoarseGrainingVAE/scripts/mapping_{dataset_label}_ncgs{config.n_cgs}.pt')
    else:
        mapping = None
    if new_seed:
        # with open("/gpfs/home/eswong/CoarseGrainingVAE/scripts/indices.json", 'r') as f:
        #     indices = json.load(f)
        #     test_index = indices['test']
        #     train_index = indices['train']
        #     val_index = indices['val']
        traj = load_protein_traj(dataset_label)
        fg_label = 'eIF4E'
        fg_traj = load_protein_traj(fg_label)
        fg_traj = shuffle_traj(fg_traj, random_seed=seed)
        n_frames = len(traj)
        # get indices for training and testing
        train_index, val_index = train_test_split(list(range(n_frames)), test_size=0.2, random_state=seed)
        val_index, test_index = train_test_split(val_index, test_size=0.5, random_state=seed)
        train_data = traj[train_index]
        val_data = traj[val_index]
        test_data = traj[test_index]
        test_data.save(f"/gpfs/scratch/eswong/CoarseGrainingVAE/data/{dataset_label}_seed_{seed}test_traj_new_indices.pdb")
        with open(f"/gpfs/home/eswong/CoarseGrainingVAE/scripts/indices_{dataset_label}.json", 'w') as f:
            json.dump({'train': train_index, 'val': val_index, 'test': test_index}, f)
        if "RBCG" in dataset_label:
              test_fg_data = fg_traj[test_index]
              test_fg_data.save(f"/gpfs/scratch/eswong/CoarseGrainingVAE/data/{fg_label}_seed_{seed}_FG_test_traj_new_indices.pdb")
    else:
        # build validation set for early stopping 
        if dataset_label == 'RBCG-eIF4E':
            train_data = md.load("/gpfs/home/eswong/CoarseGrainingVAE/data/Ca-eIF4E/Ca-eIF4E_train_traj_connect.pdb")
            val_data = md.load("/gpfs/home/eswong/CoarseGrainingVAE/data/Ca-eIF4E/Ca-eIF4E_val_traj_connect.pdb")
        elif dataset_label == 'SBCG-eIF4E':
            train_data = md.load("/gpfs/home/eswong/CoarseGrainingVAE/data/SBCG-eIF4E/sbcg50_train_traj_connect.pdb")
            val_data = md.load("/gpfs/home/eswong/CoarseGrainingVAE/data/SBCG-eIF4E/sbcg50_val_traj_connect.pdb")
        elif dataset_label == 'SBCG100-sarscov2':
            train_data = md.load("/gpfs/scratch/eswong/CoarseGrainingVAE/data/SBCG-sarscov2/SBCG-sarscov2_train_cg_100_connect.pdb")
            val_data = md.load("/gpfs/scratch/eswong/CoarseGrainingVAE/data/SBCG-sarscov2/SBCG-sarscov2_val_cg_100_connect.pdb")
        elif dataset_label == 'SBCG35-sarscov2':
            train_data = md.load("/gpfs/scratch/eswong/CoarseGrainingVAE/data/SBCG-sarscov2/SBCG-sarscov2_train_cg_35_connect.pdb")
            val_data = md.load("/gpfs/scratch/eswong/CoarseGrainingVAE/data/SBCG-sarscov2/SBCG-sarscov2_val_cg_35_connect.pdb")
        elif dataset_label == 'SBCG25-sarscov2':
            train_data = md.load("/gpfs/scratch/eswong/CoarseGrainingVAE/data/SBCG-sarscov2/SBCG-sarscov2_train_cg_25_connect.pdb")
            val_data = md.load("/gpfs/scratch/eswong/CoarseGrainingVAE/data/SBCG-sarscov2/SBCG-sarscov2_val_cg_25_connect.pdb")
        elif dataset_label == 'SBCG15-sarscov2':
            train_data = md.load("/gpfs/scratch/eswong/CoarseGrainingVAE/data/SBCG-sarscov2/SBCG-sarscov2_train_cg_15_connect.pdb")
            val_data = md.load("/gpfs/scratch/eswong/CoarseGrainingVAE/data/SBCG-sarscov2/SBCG-sarscov2_val_cg_15_connect.pdb")
    # elif dataset_label == 'RBCG-sarscov2':
    #     train_data = md.load("/gpfs/scratch/eswong/CoarseGrainingVAE/data/RBCG-sarscov2/RBCG-sarscov2_train_traj.xtc", top='/gpfs/home/eswong/CoarseGrainingVAE/data/sarscov2/RBCG/sarscov2_top_RBCG_connect.pdb')
    #     val_data = md.load("/gpfs/scratch/eswong/CoarseGrainingVAE/data/RBCG-sarscov2/RBCG-sarscov2_val_traj_connect.pdb")
    
    # else:
    #     train_data = traj[train_index]
    #     #train_data.save(f"/gpfs/scratch/eswong/CoarseGrainingVAE/data/{dataset_label}_train_traj_new_indices.pdb")
    #     val_data = traj[val_index]
    #     val_data.save(f"/gpfs/scratch/eswong/CoarseGrainingVAE/data/{dataset_label}_val_traj_new_indices.pdb")
    #     test_data = traj[test_index]
        #test_data.save(f"/gpfs/scratch/eswong/CoarseGrainingVAE/data/{dataset_label}_test_traj_new_indices.pdb")
        # if dataset_label == 'RBCG-sarscov2' or "RBCG" in dataset_label:
        #      test_fg_data = fg_traj[test_index]
        #      test_fg_data.save(f"/gpfs/scratch/eswong/CoarseGrainingVAE/data/{fg_label}_FG_test_traj_new_indices.pdb")
    #return None
    atomic_nums, protein_index = get_atomNum(train_data)
    n_atoms = atomic_nums.shape[0]
    #     n_atoms = atomic_nums.shape[0]
    trainset, mapping = build_split_dataset(train_data, args=args, config=config, mapping=mapping)
    
    true_n_cgs = len(list(set(mapping.tolist())))

    if true_n_cgs < n_cgs:
        print('True n cgs', true_n_cgs, 'desired n cgs', n_cgs)
        # reorder the mapping
        # get distinct entries
        uniques = list(set(mapping.tolist()))
        for iuniq, val in enumerate(uniques):
            mapping[mapping == val] = iuniq
        true_n_cgs = len(list(set(mapping.tolist())))
        
        #while true_n_cgs < n_cgs:
            #trainset, mapping = build_split_dataset(traj[train_index], args=args, config=config, mapping=mapping)
        n_cgs = true_n_cgs

    valset, mapping = build_split_dataset(val_data, args=args, config=config, mapping=mapping)
    
    train_sampler = DistributedSampler(trainset, num_replicas=args.world_size, rank=args.rank)
    val_sampler = DistributedSampler(valset, num_replicas=args.world_size, rank=args.rank)
    trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=CG_collate, pin_memory=True, sampler=train_sampler)
    valloader = DataLoader(valset, batch_size=batch_size, collate_fn=CG_collate, sampler=val_sampler, pin_memory=True)
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

    model = DDP(model, device_ids=[device], find_unused_parameters=True)
    
    optimizer = optim(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=2, 
                                                            factor=factor, verbose=True, 
                                                            threshold=threshold,  min_lr=min_lr)
    early_stopping = EarlyStopping(patience=patience)
    
    model.train()
    # if args.rank == 0:
    #     wandb.watch(model, log='all')

    gc.collect()
    torch.cuda.empty_cache()
    
    recon_hist = []
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

    print(mapping)
    #print(mapping.shape)
    # dump model hyperparams 
    # with open(os.path.join(split_dir, 'modelparams.json'), "w") as outfile: 
    #     params['mapping'] = mapping.numpy().tolist()
    #     json.dump(params, outfile, indent=4)

    # dump mapping, save tensor
    #torch.save(mapping, os.path.join(split_dir, 'mapping.pt'))
    # intialize training log 
    train_log = pd.DataFrame({'epoch': [], 'lr': [], 'train_loss': [], 'val_loss': [], 'train_recon': [], 'val_recon': [],
                'train_KL': [], 'val_KL': [], 'train_graph': [], 'val_graph': []})

    torch.cuda.empty_cache()
    try:
        for epoch in range(nepochs):
            gc.collect()
            torch.cuda.empty_cache()
            # train
            trainloader.sampler.set_epoch(epoch)
            valloader.sampler.set_epoch(epoch)
            train_loss, mean_kl_train, mean_recon_train, mean_graph_train, xyz_train, xyz_train_recon = loop(trainloader, optimizer, device,
                                                    model, beta, epoch,
                                                    train=True,
                                                        gamma=gamma,
                                                        eta=eta,
                                                        kappa=kappa,
                                                        looptext='Ncg {} train'.format(n_cgs),
                                                        tqdm_flag=tqdm_flag)
            val_loss, mean_kl_val, mean_recon_val, mean_graph_val, xyz_val, xyz_val_recon = loop(valloader, optimizer, device,
                                                    model, beta, epoch, 
                                                    train=False,
                                                        gamma=gamma,
                                                        kappa=kappa,
                                                        looptext='Ncg {} test'.format(n_cgs),
                                                        tqdm_flag=tqdm_flag)

            stats = {'epoch': epoch, 'lr': optimizer.param_groups[0]['lr'], 
                    'train_loss': train_loss, 'val_loss': val_loss, 
                    'train_recon': mean_recon_train, 'val_recon': mean_recon_val,
                    'train_KL': mean_kl_train, 'val_KL': mean_kl_val, 
                    'train_graph': mean_graph_train, 'val_graph': mean_graph_val}
            #dist.barrier()
            #if args.rank == 0:
                #run.log(stats)
            #dist.barrier()
            #if args.rank == 0:
            train_log = train_log.append(stats, ignore_index=True)
            #dist.barrier()
            # smoothen the validation curve 
            smooth = sm.nonparametric.lowess(train_log['val_loss'].values,  # y
                                            train_log['epoch'].values, # x
                                            frac=0.2)
            smoothed_valloss = smooth[-1, 1]

            scheduler.step(smoothed_valloss)
            recon_hist.append(xyz_train_recon.detach().cpu().numpy().reshape(-1, n_atoms, 3))

            if optimizer.param_groups[0]['lr'] <= min_lr * 1.5:
                print('converged')
                break

            early_stopping(smoothed_valloss)
            if early_stopping.early_stop:
                break

            # check NaN
            if np.isnan(mean_recon_val):
                print("NaN encoutered, exiting...")
                failed = True
                break 

            # dump training curve
            if args.rank == 0:
                train_log.to_csv(os.path.join(split_dir, 'train_log.csv'),  index=False)
                if (epoch > 50) and (epoch % 10) == 0:
                    torch.save(model.module.state_dict(), os.path.join(split_dir, 'model_epoch{}.pt'.format(epoch)))
            #dist.barrier()
            #print("past barrier")
        if args.rank == 0:
            # detach the model from the gpu
            model = model.module.to('cpu')
            torch.save(model.state_dict(), os.path.join(split_dir, 'model_final.pt')) 
    except Exception as e:
        if isinstance(e, KeyboardInterrupt):
            if args.rank == 0:
                print('Training early-stopped by hyperband')
                # cause all the other nodes to exit
                dist.destroy_process_group()
                # make sure the other runs on the gpus are killed
                os.kill(os.getpid(), signal.SIGTERM)
            else:
                model = model.module.to('cpu')
                torch.save(model.state_dict(), os.path.join(split_dir, 'model_final.pt'))
                dist.destroy_process_group()
                os.kill(os.getpid(), signal.SIGTERM)
            
        elif isinstance(e, RuntimeError) and "out of memory" in str(e):
            print("WARNING: out of memory, skipping fold")
            failed = True
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            # raise error on the other gpus
            if args.rank != 0:
                model = model.module.to('cpu')
                torch.save(model.state_dict(), os.path.join(split_dir, 'model_final.pt'))
                dist.destroy_process_group()
                os.kill(os.getpid(), signal.SIGTERM)
            else:
                raise e
        else:
            raise e
    
    finally:
        if args.rank == 0:
            #run.finish()
            # save the final model
            model = model.module.to('cpu')
            torch.save(model.state_dict(), os.path.join(split_dir, 'model_final.pt'))
            
            dist.destroy_process_group()
            os.kill(os.getpid(), signal.SIGTERM)
    #if args.rank == 0:
        #run.finish()
        #dist.destroy_process_group()
    return None

class SLURM_Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):

        init_dist_node(self.args)
        print("initialized nodes!")
        
        run_cv(self.args)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str)
    parser.add_argument("--queue", type=str)
    parser.add_argument("--n_gpus", type=int)
    parser.add_argument("--logdir", type=str)
    parser.add_argument("--n_cgs", type=int)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--dataset", type=str, default='dipeptide')
    parser.add_argument("--n_basis", type=int, default=512)
    parser.add_argument("--n_rbf", type=int, default=10)
    parser.add_argument("--activation", type=str, default='swish')
    parser.add_argument("--cg_method", type=str, default='minimal')
    parser.add_argument("--atom_cutoff", type=float, default=4.0)
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--cg_cutoff", type=float, default=4.0)
    parser.add_argument("--enc_nconv", type=int, default=4)
    parser.add_argument("--dec_nconv", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--nepochs", type=int, default=2)
    parser.add_argument("--n_data", type=int, default=200)
    parser.add_argument("--nsamples", type=int, default=200)
    parser.add_argument("--n_ensemble", type=int, default=16)
    parser.add_argument("--nevals", type=int, default=36)
    parser.add_argument("--edgeorder", type=int, default=2)
    parser.add_argument("--auxcutoff", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--eta", type=float, default=0.01)
    parser.add_argument("--kappa", type=float, default=0.01)
    parser.add_argument("--threshold", type=float, default=1e-3)
    parser.add_argument("--nsplits", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--factor", type=float, default=0.6)
    parser.add_argument("--mapshuffle", type=float, default=0.0)
    parser.add_argument("--cgae_reg_weight", type=float, default=0.25)
    parser.add_argument("--dec_type", type=str, default='EquivariantDecoder')
    parser.add_argument("--cross", action='store_true', default=False)
    parser.add_argument("--graph_eval", action='store_true', default=True)
    parser.add_argument("--shuffle", action='store_true', default=False)
    parser.add_argument("--cg_mp", action='store_true', default=False)
    parser.add_argument("--tqdm_flag", action='store_true', default=False)
    parser.add_argument("--det", action='store_true', default=False)
    parser.add_argument("--cg_radius_graph", action='store_true', default=False)
    parser.add_argument("--invariantdec", action='store_true', default=False)
    parser.add_argument("--reflectiontest", action='store_true', default=False)
    parser.add_argument("--new_seed", type=int, default=123)
    args = parser.parse_args()
    args.gpus_per_node = args.n_gpus#int(os.environ["SLURM_GPUS_ON_NODE"])
    args.port = random.randint(49152,65535)
    #wandb.require("service")
    if (args.queue == "a100") or (args.queue == "gpu"):
        timeout_min = 60*8
        if args.queue == "a100":
            mem = '250G'
        else:
            mem = '120G'
    elif (args.queue == "a100-long") or (args.queue == "gpu-long"):
        timeout_min = 60*48
        if args.queue == "a100-long":
            mem = '250G'
        else:
            mem = '120G'
    elif args.queue == ("v100"):
        timeout_min = 60*24
        mem = '120G'
    elif args.queue == ("p100"):
        timeout_min = 60*24
        mem = '60G'
    executor = submitit.AutoExecutor(folder="/gpfs/scratch/eswong/CoarseGrainingVAE/logs/slurmy")
    executor.update_parameters(
        timeout_min=timeout_min,
        slurm_partition=args.queue,
        slurm_gres=f'gpu:{args.gpus_per_node}',
        mem= mem,
        slurm_ntasks_per_node = args.gpus_per_node,
        nodes=1
    )
    trainer = SLURM_Trainer(args)
    job = executor.submit(trainer)
    print(f"Submitted job_id: {job.job_id}")
    # wait until job is finished
    job.results()
    # kill job
    job.cancel()