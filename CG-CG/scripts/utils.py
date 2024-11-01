import os 
import sys
import gc
from tqdm import tqdm 
import torch
import numpy as np
from ase import Atoms, io 
import networkx as nx
from datetime import date
import torch.autograd.profiler as profiler
from sampling import *
from sklearn.utils import shuffle
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from datetime import datetime
from datetime import timedelta
import signal, subprocess
import submitit
from pathlib import Path

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

EPS = 1e-6
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
    args.output_dir = Path(("/gpfs/scratch/eswong/CoarseGrainingVAE/logs/slurmy").replace("%j", str(job_env.job_id)))
    args.gpu = job_env.local_rank
    args.rank = job_env.global_rank

    dist.init_process_group(backend='gloo', init_method=args.dist_url, world_size=args.world_size, rank=args.rank, timeout=timedelta(seconds=1000000))
    device = torch.device('cuda:{}'.format(args.rank))
    torch.cuda.set_device(device)
    args.device = device
    dist.barrier()

    args.main = (args.rank == 0)
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
def shuffle_traj(traj, random_seed=123):
    full_idx = list(range(len(traj)))
    full_idx = shuffle(full_idx, random_state=random_seed)
    return traj[full_idx]

def annotate_job(task, job_name, N_cg):
    # get today's date and time
    today = datetime.now().strftime("%m-%d-%H-%M")
    return "{}_{}_{}_N{}".format(job_name, today, task, N_cg)

def create_dir(name):
    if not os.path.isdir(name):
        os.mkdir(name)   

def save_runtime(dtime, dir):
    hours = dtime//3600
    dtime = dtime - 3600*hours
    minutes = dtime//60
    seconds = dtime - 60*minutes
    format_time = '%d:%d:%d' %(hours,minutes,seconds)
    np.savetxt(os.path.join(dir, '{}.txt'.format(format_time)), np.ones(10))
    print("time elapsed: {}".format(format_time))
    return format_time

def check_CGgraph(dataset):
    frame_idx = np.random.randint(0, len(dataset), 20)

    for idx in frame_idx:
        a = dataset.props['CG_nbr_list'][idx]
        adj = [ tuple(pair.tolist()) for pair in a ]
        G = nx.Graph()
        G.add_edges_from(adj)
        connected = nx.is_connected(G)
        if not connected:
            print("One of the sampled CG graphs is not connected, training failed")
            return connected
        return True

class EarlyStopping():
    '''from https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/'''
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0    
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

def KL(mu1, std1, mu2, std2):
    if mu2 == None:
        return -0.5 * torch.sum(1 + torch.log(std1.pow(2)) - mu1.pow(2) - std1.pow(2), dim=-1).mean()
    else:
        return 0.5 * ( (std1.pow(2) / std2.pow(2)).sum(-1) + ((mu1 - mu2).pow(2) / std2).sum(-1) + \
            torch.log(std2.pow(2)).sum(-1) - torch.log(std1.pow(2)).sum(-1) - std1.shape[-1] ).mean()


def loop(loader, optimizer, device, model, beta, epoch, 
        gamma, eta=0.0, kappa=0.0, train=True, looptext='', tqdm_flag=True):
    
    total_loss = []
    recon_loss = []
    orient_loss = []
    norm_loss = []
    graph_loss = []
    kl_loss = []
    
    accumulation_steps = 3
    
    gc.collect()
    torch.cuda.empty_cache()

    if train:
        model.train()
        mode = '{} train'.format(looptext)
    else:
        model.train() # yes, still set to train when reconstructing
        mode = '{} valid'.format(looptext)

    if tqdm_flag:
        loader = tqdm(loader, position=0, file=sys.stdout,
                         leave=True, desc='({} epoch #{})'.format(mode, epoch))
    
    for i, batch in enumerate(loader):

        batch = batch_to(batch, device)

        S_mu, S_sigma, H_prior_mu, H_prior_sigma, xyz, xyz_recon = model(batch)

        # loss
        if S_mu is not None:
            loss_kl = KL(S_mu, S_sigma, H_prior_mu, H_prior_sigma) 
            kl_loss.append(loss_kl.item())
        else:
            loss_kl = 0.0
            kl_loss.append(0.0)

        loss_recon = (xyz_recon - xyz).pow(2).mean()

        # add graph loss 
        edge_list = batch['bond_edge_list'].to("cpu")
        xyz = batch['nxyz'][:, 1:].to("cpu")

        if gamma != 0.0:
            gen_dist = ((xyz_recon[edge_list[:, 0 ]] - xyz_recon[edge_list[:, 1 ]]).pow(2).sum(-1) + EPS).sqrt()
            data_dist = ((xyz[edge_list[:, 0 ]] - xyz[edge_list[:, 1 ]]).pow(2).sum(-1) + EPS).sqrt().to(xyz_recon.device)
            loss_graph = (gen_dist - data_dist).pow(2).mean()
        else:
            loss_graph = torch.Tensor([0.0]).to(device)

        # add orientation loss 
        cg_xyz = batch['CG_nxyz'][:, 1:]
        mapping = batch['CG_mapping']

        loss =  loss_recon + loss_kl * beta+ loss_graph * gamma 

        memory = torch.cuda.memory_allocated(device) / (1024 ** 2)

        if loss.item() >= gamma * 200.0 or torch.isnan(loss):
            print(loss.item())
            del loss, loss_graph, loss_kl, loss_recon, S_mu, S_sigma, H_prior_mu, H_prior_sigma
            continue 
        
        # optimize 
        if train:
            loss.backward()

            # perfrom gradient clipping
            if (i + 1) % accumulation_steps == 0: 
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
                optimizer.step()
                optimizer.zero_grad()

        else:
            loss.backward()

        # garbage collection
        gc.collect()
        torch.cuda.empty_cache()

        # logging 
        recon_loss.append(loss_recon.item())
        # orient_loss.append(loss_dx_orient.item())
        # norm_loss.append(loss_dx_norm.item())
        graph_loss.append(loss_graph.item())
        total_loss.append(loss.item())
        
        mean_kl = np.array(kl_loss).mean()
        mean_recon = np.array(recon_loss).mean()
        # mean_orient = np.array(orient_loss).mean()
        # mean_norm = np.array(norm_loss).mean()
        mean_graph = np.array(graph_loss).mean()
        mean_total_loss = np.array(total_loss).mean()

        postfix = ['total={:.3f}'.format(mean_total_loss),
                    'KL={:.4f}'.format(mean_kl) , 
                   'recon={:.4f}'.format(mean_recon),
                   'graph={:.4f}'.format(mean_graph) , 
                   'memory ={:.4f} Mb'.format(memory) 
                   ]
        
        if tqdm_flag:
            loader.set_postfix_str(' '.join(postfix))
        
        gc.collect()
        torch.cuda.empty_cache()

        del loss, loss_graph, loss_kl, loss_recon, S_mu, S_sigma, H_prior_mu, H_prior_sigma
        
    for result in postfix:
        print(result)
    
    return mean_total_loss, mean_kl, mean_recon, mean_graph, xyz, xyz_recon 

def get_all_true_reconstructed_structures(loader, device, args, model, atomic_nums=None, n_cg=10, atomwise_z=False, tqdm_flag=True, reflection=False):
    model = model.to(device)
    model.eval()

    true_xyzs = []
    recon_xyzs = []
    cg_xyzs = []

    heavy_ged = []
    all_ged = []

    all_valid_ratios = []
    heavy_valid_ratios = []

    if tqdm_flag:
        loader = tqdm(loader, position=0, leave=True) 

    for batch in loader:
        batch = batch_to(batch, device)

        atomic_nums = batch['nxyz'][:, 0].detach().cpu()

        if reflection: 
            xyz = batch['nxyz'][:,1:]
            xyz[:, 1] *= -1 # reflect around x-z plane
            cgxyz = batch['CG_nxyz'][:,1:]
            cgxyz[:, 1] *= -1 

        S_mu, S_sigma, H_prior_mu, H_prior_sigma, xyz, xyz_recon = model(batch)

        true_xyzs.append(xyz.detach().cpu())
        recon_xyzs.append(xyz_recon.detach().cpu())
        cg_xyzs.append(batch['CG_nxyz'][:, 1:].detach().cpu())

        recon  = xyz_recon.detach().cpu()
        data = xyz.detach().cpu()

        recon_x_split =  torch.split(recon, batch['num_atoms'].tolist())
        data_x_split =  torch.split(data, batch['num_atoms'].tolist())
        atomic_nums_split = torch.split(atomic_nums, batch['num_atoms'].tolist())

        del S_mu, S_sigma, H_prior_mu, H_prior_sigma, xyz, xyz_recon, batch

        memory = torch.cuda.memory_allocated(device) / (1024 ** 2)
        postfix = ['memory ={:.4f} Mb'.format(memory)]

        for i, x in enumerate(data_x_split):

            z = atomic_nums_split[i].numpy()

            ref_atoms = Atoms(numbers=z, positions=x.numpy())
            recon_atoms = Atoms(numbers=z, positions=recon_x_split[i].numpy())

            # compute ged diff 
            all_rmsds, heavy_rmsds, valid_ratio, valid_hh_ratio, graph_val_ratio, graph_hh_val_ratio = eval_sample_qualities(args, ref_atoms, [recon_atoms])

            heavy_ged.append(graph_val_ratio)
            all_ged.append(graph_hh_val_ratio)

            all_valid_ratios.append(valid_hh_ratio)
            heavy_valid_ratios.append(valid_ratio)
        
        if tqdm_flag:
            loader.set_postfix_str(' '.join(postfix))

    true_xyzs = torch.cat(true_xyzs).numpy()
    recon_xyzs = torch.cat(recon_xyzs).numpy()
    cg_xyzs = torch.cat(cg_xyzs).numpy()
    all_valid_ratio = np.array(all_valid_ratios).mean()
    heavy_valid_ratio = np.array(heavy_valid_ratios).mean()

    all_ged = np.array(all_ged).mean()
    heavy_ged = np.array(heavy_ged).mean()
    
    return true_xyzs, recon_xyzs, cg_xyzs, all_valid_ratio, heavy_valid_ratio, all_ged, heavy_ged

def dump_numpy2xyz(xyzs, atomic_nums, path):
    trajs = [Atoms(positions=xyz, numbers=atomic_nums.ravel()) for xyz in xyzs]
    io.write(path, trajs)