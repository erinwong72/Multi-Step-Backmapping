import os 
import glob
import subprocess
import sys
import argparse 
import random
import copy
import json
import time
from datetime import timedelta

import numpy as np
from tqdm import tqdm 
import pandas as pd
import statsmodels.api as sm

from GenZProt.data import CGDataset, CG_collate
from GenZProt.cgvae import *
from GenZProt.genzprot import *
from GenZProt.e3nn_enc import e3nnEncoder, e3nnPrior, e3nnEncoder_plpro
from GenZProt.conv import * 
from GenZProt.datasets_orig import *
from utils import * 
from utils_ic import *
from sampling import * 
import submitit
import torch
from torch import nn
from torch.nn import Sequential 
from torch_scatter import scatter_mean
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import gc
import warnings
import signal
from pathlib import Path
warnings.filterwarnings("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def init_dist_node(args):

    # requeue job on SLURM preemption
    signal.signal(signal.SIGUSR1, handle_sigusr1)
    signal.signal(signal.SIGTERM, handle_sigterm)

    # find a common host name on all nodes
    cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
    stdout = subprocess.check_output(cmd.split())
    host_name = stdout.decode().splitlines()[0]
    args.dist_url = f'tcp://{host_name}:{args.port}'

    # distributed parameters
    args.rank = int(os.getenv('SLURM_NODEID')) * args.gpus_per_node
    args.world_size = int(os.getenv('SLURM_NNODES')) * args.gpus_per_node
    print(f'rank = {args.rank}, world_size = {args.world_size}')

        
def init_dist_gpu(gpu, args):
    job_env = submitit.JobEnvironment()
    args.output_dir = Path(("/gpfs/scratch/eswong/GenZProt/logs").replace("%j", str(job_env.job_id)))
    args.gpu = job_env.local_rank
    args.rank = job_env.global_rank

    dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.rank, timeout=timedelta(seconds=1000000))
    device = torch.device('cuda:{}'.format(args.rank))
    torch.cuda.set_device(device)
    args.device = device
    dist.barrier()

    #args.main = (args.rank == 0)
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()

def handle_sigterm(signum, frame):
    pass

optim_dict = {'adam':  torch.optim.Adam, 'sgd':  torch.optim.SGD}

def build_split_dataset(traj, params, mapping=None, n_cgs=None, prot_idx=None):

    if n_cgs == None:
        n_cgs = params['n_cgs']

    atomic_nums, protein_index = get_atomNum(traj)
    new_mapping, frames, cg_coord = get_cg_and_xyz(traj, params=params, cg_method=params['cg_method'], n_cgs=n_cgs,
                                                     mapshuffle=params['mapshuffle'], mapping=mapping)

    if mapping is None:
        mapping = new_mapping

    dataset = build_ic_peptide_dataset(mapping,
                                        frames, 
                                        params['atom_cutoff'], 
                                        params['cg_cutoff'],
                                        atomic_nums,
                                        traj.top,
                                        order=params['edgeorder'] ,
                                        cg_traj=cg_coord, prot_idx=prot_idx)


    return dataset, mapping
    

def run_cv(args):
    init_dist_gpu(gpu=None, args=args)
    print("got here")
    json_file = "/gpfs/home/eswong/GenZProt/ckpt/modelparams_sarscov2_larger.json"
    dataset = "sarscov2"

    print("loading json")
    with open(json_file, 'rt') as f:
        params = json.load(f)
    
    params['dataset'] = dataset
    params['logdir'] = annotate_job(params['seed'], params['logdir'])

    working_dir = params['logdir']
    params['device'] = args.device
    device  = params['device']

    batch_size  = params['batch_size']
    nepochs = params['nepochs']
    lr = params['lr']
    optim = optim_dict[params['optimizer']]
    factor = params['factor']
    patience = params['patience']
    threshold = params['threshold']

    beta  = params['beta']
    gamma = params['gamma']
    delta = params['delta']
    eta = params['eta']
    zeta = params['zeta']

    n_basis  = params['n_basis']
    n_rbf  = params['n_rbf']
    atom_cutoff = params['atom_cutoff']
    cg_cutoff = params['cg_cutoff']

    # for model
    enc_nconv  = params['enc_nconv']
    dec_nconv  = params['dec_nconv']
    enc_type = params['enc_type']
    dec_type = params['dec_type']

    # unused
    activation = params['activation']
    #shuffle_flag = params['shuffle']
    #nevals = params['nevals']
    params['tqdm_flag'] = args.tqdm_flag
    tqdm_flag = params['tqdm_flag']
    params['det'] = args.det
    det = params['det']
    #mapshuffle = params['mapshuffle']
    savemodel = True
    params['invariantdec'] = args.invariantdec
    invariantdec = params['invariantdec']
    n_cgs  = params['n_cgs']
    params['rank'] = args.rank
    params['world_size'] = args.world_size
    params['device'] = args.device
    params['port'] = args.port
    params['dist_url'] = args.dist_url
    params['gpus_per_node'] = args.gpus_per_node
    params['mapshuffle'] = args.mapshuffle
    # set random seed 
    seed = params['seed']
    os.environ['PYTHONHASHSEED']=str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    min_lr = 1e-8
    if det: 
        beta = 0.0
        print("Recontruction Task")
    else:
        print("Sampling Task")

    # print('cuda: ', torch.cuda.is_available())
    # print('using cuda: ', torch.cuda.current_device())
    # device = torch.cuda.current_device()
    used_saved_dataset = True
    # Load PED files
    if params['dataset'] == 'all':
        with open('../data/train_id.txt','r') as fh:
            train_label_list = fh.readlines()
            train_label_list = [label.strip('\n') for label in train_label_list]

        with open('../data/val_id.txt','r') as fh:
            val_label_list = fh.readlines()
            val_label_list = [label.strip('\n') for label in val_label_list]
        train_label_list.extend(['sarscov2_0_1_noH', 'sarscov2_0_0_noH', 'sarscov2_0_2_train_noH'])
        val_label_list.extend(['sarscov2_0_2_val1_noH', 'sarscov2_0_2_val2_noH'])
    elif params['dataset'] == 'sarscov2':
        train_label_list = ['sarscov2_0_1_noH', 'sarscov2_0_0_noH', 'sarscov2_0_2_train_noH', 'sarscov2_0_3_train']#'sarscov2_0_4_train', 
                           # 'sarscov2_0_5', 'sarscov2_0_6_train']
        val_label_list = ['sarscov2_0_2_val1_noH', 'sarscov2_0_2_val2_noH', 'sarscov2_0_4_val'] #'sarscov2_0_6_val', 
                          #'sarscov2_0_7_val']
    elif params['dataset'] == 'ped':
        with open('../data/train_id.txt','r') as fh:
            train_label_list = fh.readlines()
            train_label_list = [label.strip('\n') for label in train_label_list]

        with open('../data/val_id.txt','r') as fh:
            val_label_list = fh.readlines()
            val_label_list = [label.strip('\n') for label in val_label_list]
    print("num training data entries", len(train_label_list))

    train_n_cg_list, train_traj_list, info_dict = create_info_dict(train_label_list, PROTEINFILES=PROTEINFILES)
    val_n_cg_list, val_traj_list, val_info_dict = create_info_dict(val_label_list, PROTEINFILES=PROTEINFILES)

    val_info_dict = {k+len(train_label_list): val_info_dict[k] for k in val_info_dict.keys()}
    info_dict.update(val_info_dict)

    # create subdirectory 
    # replace home with scratch
    working_dir = working_dir.replace('/gpfs/home/', '/gpfs/scratch/')
    create_dir(working_dir)     

    # start timing 
    start =  time.time()
    if not used_saved_dataset:
        trainset_list, valset_list = [], []
        success_list = []
        print("TRAINING DATA")
        for i, traj in enumerate(train_traj_list):
            print("start generating dataset-------", train_label_list[i])
            n_cgs = train_n_cg_list[i]
            trainset, mapping = build_split_dataset(traj, params, mapping=None, n_cgs=n_cgs, prot_idx=0)
            print("created dataset-------", train_label_list[i])
            success_list.append(train_label_list[i])
            trainset_list.append(trainset)
        
        print("TEST DATA")
        for i, traj in enumerate(val_traj_list):
            print("start generating dataset-------", val_label_list[i])
            n_cgs = val_n_cg_list[i]
            valset, mapping = build_split_dataset(traj, params, mapping=None, n_cgs=n_cgs, prot_idx=0)
            print("created dataset-------", val_label_list[i])
            success_list.append(val_label_list[i])
            valset_list.append(valset)
    
        trainset = torch.utils.data.ConcatDataset(trainset_list)
        valset = torch.utils.data.ConcatDataset(valset_list)
    trainset = torch.load('/gpfs/scratch/eswong/GenZProt/data/sarscov2/trainset.pt')
    valset = torch.load('/gpfs/scratch/eswong/GenZProt/data/sarscov2/valset.pt')
    if params['rank'] == 0:
        torch.save(trainset, os.path.join(working_dir, 'trainset.pt'))
        torch.save(valset, os.path.join(working_dir, 'valset.pt'))
    train_sampler = DistributedSampler(trainset, num_replicas=params['world_size'], rank=params['rank'])
    val_sampler = DistributedSampler(valset, num_replicas=params['world_size'], rank=params['rank'])
    trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=CG_collate, sampler=train_sampler, pin_memory=True)
    valloader = DataLoader(valset, batch_size=batch_size, collate_fn=CG_collate, sampler=val_sampler, pin_memory=True)
    gc.collect()
    torch.cuda.empty_cache()
    # initialize model 
    if n_cgs == 3:
        breaksym= True 
    else:
        breaksym = False

    # Z-matrix generation or xyz generation
    if dec_type == 'ic_dec': ic_flag = True
    else: ic_flag = False

    if ic_flag:
        decoder = ZmatInternalDecoder(n_atom_basis=n_basis, n_rbf = n_rbf, cutoff=cg_cutoff, num_conv = dec_nconv, activation=activation)
        if params['rank'] ==0:
            print("using invariant decoder")
    else:
        decoder = EquivariantPsuedoDecoder(n_atom_basis=n_basis, n_rbf = n_rbf, cutoff=cg_cutoff, num_conv = dec_nconv, activation=activation, breaksym=breaksym)
        if params['rank'] ==0:
            print("using CGVAE decoder")

    if enc_type == 'equiv_enc':
        if params['dataset'] == 'sarscov2' or params['dataset'] == 'all' or params['dataset'] == 'ped':
            encoder = e3nnEncoder_plpro(device=device, n_atom_basis=n_basis, use_second_order_repr=False, num_conv_layers=enc_nconv,
                        cross_max_distance=cg_cutoff+5, atom_max_radius=atom_cutoff+5, cg_max_radius=cg_cutoff+5).to(device)
        else:
            encoder = e3nnEncoder(device=device, n_atom_basis=n_basis, use_second_order_repr=False, num_conv_layers=enc_nconv,
        cross_max_distance=cg_cutoff+5, atom_max_radius=atom_cutoff+5, cg_max_radius=cg_cutoff+5).to(device)
        cgPrior = e3nnPrior(device=device, n_atom_basis=n_basis, use_second_order_repr=False, num_conv_layers=enc_nconv,
        cg_max_radius=cg_cutoff+5).to(device)
        if params['rank'] ==0:
            print("using equivariant encoder")
    else:
        encoder = EquiEncoder(n_conv=enc_nconv, n_atom_basis=n_basis, n_rbf=n_rbf, activation=activation, cutoff=atom_cutoff).to(device)
        cgPrior = CGprior(n_conv=enc_nconv, n_atom_basis=n_basis, n_rbf=n_rbf, activation=activation, cutoff=cg_cutoff).to(device)
        if params['rank'] ==0:
            print("using invariant encoder")

    atom_mu = nn.Sequential(nn.Linear(n_basis, n_basis), nn.ReLU(), nn.Linear(n_basis, n_basis)).to(device)
    atom_sigma = nn.Sequential(nn.Linear(n_basis, n_basis), nn.ReLU(), nn.Linear(n_basis, n_basis)).to(device)

    if ic_flag:
        model = GenZProt(encoder, decoder, atom_mu, atom_sigma, n_cgs, feature_dim=n_basis, prior_net=cgPrior, det=det, equivariant= not invariantdec).to(device)
    else:
        model = CGequiVAE(encoder, decoder, atom_mu, atom_sigma, n_cgs, feature_dim=n_basis, prior_net=cgPrior, det=det, equivariant= not invariantdec).to(device)

    #model.load_state_dict(torch.load('/gpfs/scratch/eswong/GenZProt/ckpt/ped_10-18_12345/model_100.pt'))
    model = DDP(model, device_ids=[device], find_unused_parameters=True)
    optimizer = optim(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=5, 
                                                            factor=factor, verbose=True, 
                                                            threshold=threshold,  min_lr=min_lr, cooldown=1)
    early_stopping = EarlyStopping(patience=25)
    
    model.train()

    # with open(os.path.join(working_dir, 'modelparams.json'), "w") as outfile: 
    #     json.dump(params, outfile, indent=4)

    # intialize training log 
    train_log = pd.DataFrame({'epoch': [], 'lr': [], 'train_loss': [], 'val_loss': [], 'train_recon': [], 'val_recon': [], 'train_xyz': [], 'val_xyz': [],
                'train_KL': [], 'val_KL': [], 'train_graph': [], 'val_graph': [], 'train_nbr': [], 'val_nbr': [], 'train_inter': [], 'val_inter': []})

    for epoch in range(nepochs):
        # train
        train_loss, mean_kl_train, mean_recon_train, mean_graph_train, mean_nbr_train, mean_inter_train, mean_xyz_train = loop(trainloader, optimizer, device,
                                                    model, beta, gamma, delta, eta, zeta, epoch, 
                                                    train=True,
                                                    looptext='epoch {} train'.format(epoch),
                                                    tqdm_flag=tqdm_flag,
                                                    ic_flag=ic_flag, info_dict=info_dict)


        val_loss, mean_kl_val, mean_recon_val, mean_graph_val, mean_nbr_val, mean_inter_val, mean_xyz_val = loop(valloader, optimizer, device,
                                                    model, beta, gamma, delta, eta, zeta, epoch, 
                                                    train=False,
                                                    looptext='epoch {} train'.format(epoch),
                                                    tqdm_flag=tqdm_flag,
                                                    ic_flag=ic_flag, info_dict=info_dict)

        stats = {'epoch': epoch, 'lr': optimizer.param_groups[0]['lr'], 
                'train_loss': train_loss, 'val_loss': val_loss, 
                'train_recon': mean_recon_train, 'val_recon': mean_recon_val,
                'train_xyz': mean_xyz_train, 'val_xyz': mean_xyz_val,
                'train_KL': mean_kl_train, 'val_KL': mean_kl_val, 
                'train_graph': mean_graph_train, 'val_graph': mean_graph_val,
                'train_nbr': mean_nbr_train, 'val_nbr': mean_nbr_val,
                'train_inter': mean_inter_train, 'val_inter': mean_inter_val}
        dist.barrier()
        train_log = train_log.append(stats, ignore_index=True)
        
        # smoothen the validation curve 
        smooth = sm.nonparametric.lowess(train_log['val_loss'].values, 
                                        train_log['epoch'].values, # x
                                        frac=0.2)
        #try:
        smoothed_valloss = smooth[-1, 1]
        #except:
            #smoothed_valloss = val_loss

        scheduler.step(smoothed_valloss)

        if optimizer.param_groups[0]['lr'] <= min_lr * 1.5:
            if params['rank'] == 0:
                print('converged')
            break

        early_stopping(smoothed_valloss)
        if early_stopping.early_stop:
            break

        # check NaN
        if np.isnan(mean_recon_val):
            if params['rank'] == 0:
                print("NaN encoutered, exiting...")
            break 

        # dump training curve 
        if params['rank'] == 0:
            train_log.to_csv(os.path.join(working_dir, 'train_log.csv'),  index=False, float_format='%.5f')
        dist.barrier()
        if savemodel and epoch%5==0:
            if params['rank'] == 0:
                print("saving model")
                model = model.to('cpu')
                torch.save(model.state_dict(), os.path.join(working_dir, f'model_{epoch}.pt'))
                # resume training
                model = model.to(device)
        dist.barrier()
    if params['rank'] == 0:
        print("saving final model")
        model = model.to('cpu')
        torch.save(model.state_dict(), os.path.join(working_dir, f'model_{epoch}.pt'))
        print("finished training")
        dist.destroy_process_group()
class SLURM_Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):

        init_dist_node(self.args)
        print("initialized nodes!")
        
        run_cv(self.args)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # paths and environment
    parser.add_argument("-load_json", type=str, default=None)
    parser.add_argument("-logdir", type=str)
    #parser.add_argument("-device", type=int)

    # dataset
    parser.add_argument("-dataset", type=str, default='dipeptide')
    parser.add_argument("-cg_method", type=str, default='minimal')

    # training
    parser.add_argument("-seed", type=int, default=12345)
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-optimizer", type=str, default='adam')
    parser.add_argument("-nepochs", type=int, default=2)

    # learning rate
    parser.add_argument("-lr", type=float, default=2e-4)
    parser.add_argument("-threshold", type=float, default=1e-3)
    parser.add_argument("-patience", type=int, default=5)
    parser.add_argument("-factor", type=float, default=0.6)

    # loss
    parser.add_argument("-beta", type=float, default=0.05)
    parser.add_argument("-gamma", type=float, default=1.0)
    parser.add_argument("-delta", type=float, default=1.0)
    parser.add_argument("-eta", type=float, default=1.0)
    parser.add_argument("-zeta", type=float, default=3.0)

    # model
    parser.add_argument("-enc_type", type=str, default='equiv_enc')
    parser.add_argument("-dec_type", type=str, default='ic_dec')

    parser.add_argument("-n_basis", type=int, default=512)
    parser.add_argument("-n_rbf", type=int, default=10)
    parser.add_argument("-atom_cutoff", type=float, default=4.0)
    parser.add_argument("-cg_cutoff", type=float, default=4.0)
    parser.add_argument("-edgeorder", type=int, default=2)

    parser.add_argument("-activation", type=str, default='swish')
    parser.add_argument("-enc_nconv", type=int, default=4)
    parser.add_argument("-dec_nconv", type=int, default=4)

    # always use default
    parser.add_argument("-n_cgs", type=int)
    parser.add_argument("-mapshuffle", type=float, default=0.0)
    parser.add_argument("--shuffle", action='store_true', default=False)
    parser.add_argument("--tqdm_flag", action='store_true', default=False)
    parser.add_argument("--det", action='store_true', default=False)
    parser.add_argument("--savemodel", action='store_true', default=True)

    # not used
    parser.add_argument("-ndata", type=int, default=200)
    parser.add_argument("-nsamples", type=int, default=200)
    parser.add_argument("-n_ensemble", type=int, default=16)
    parser.add_argument("-nevals", type=int, default=36)
    parser.add_argument("-auxcutoff", type=float, default=0.0)
    parser.add_argument("-kappa", type=float, default=0.01)    
    parser.add_argument("-nsplits", type=int, default=5)
    parser.add_argument("-cgae_reg_weight", type=float, default=0.25)
    parser.add_argument("--cross", action='store_true', default=False)
    parser.add_argument("--graph_eval", action='store_true', default=False)
    parser.add_argument("--cg_mp", action='store_true', default=False)
    parser.add_argument("--cg_radius_graph", action='store_true', default=False)
    parser.add_argument("--invariantdec", action='store_true', default=False)
    parser.add_argument("--reflectiontest", action='store_true', default=False)
    args = parser.parse_args()
    args.gpus_per_node = 4
    args.port = random.randint(49152,65535)
    #params = vars(parser.parse_args())
    executor = submitit.AutoExecutor(folder="/gpfs/scratch/eswong/GenZProt/logs/")
    executor.update_parameters(
        timeout_min=60*48,
        slurm_partition='a100-long',
        slurm_gres=f'gpu:4',
        mem= '250G',
        slurm_ntasks_per_node = 4,
        nodes=1
    )
    trainer = SLURM_Trainer(args)
    job = executor.submit(trainer)
    print(f"Submitted job_id: {job.job_id}")
    # wait until job is finished
    job.results()
    # kill job
    job.cancel()
