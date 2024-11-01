import os
from venv import logger
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import numpy as np 
import networkx as nx
import itertools
from .data import * 
#from .utils import * 
from torch_scatter import scatter_mean, scatter_add
from moleculekit.molecule import Molecule
import glob 
import sys
import mdtraj as md
import mdshare
import pyemma
from sklearn.utils import shuffle
import random
import tqdm
from .cgae import *
# io
from ase import Atoms, io
def dump_numpy2xyz(xyzs, atomic_nums, path):
    trajs = [Atoms(positions=xyz, numbers=atomic_nums) for xyz in xyzs]
    io.write(path, trajs)

seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
atomic_num_dict = {'C':6, 'H':1, 'O':8, 'N':7, 'S':16, 'Se': 34}

PROTEINFILES = {'covid': {'traj_paths': "../data/DESRES-Trajectory_sarscov2-11441075-no-water-zinc-glueCA/sarscov2-11441075-no-water-zinc-glueCA/sarscov2-11441075-no-water-zinc-glueCA-00*.dcd", 
                              'pdb_path': '../data/DESRES-Trajectory_sarscov2-11441075-no-water-zinc-glueCA/sarscov2-11441075-no-water-zinc-glueCA/DESRES-Trajectory_sarscov2-11441075-no-water-zinc-glueCA.pdb', 
                              'file_type': 'dcd'},

                'chignolin': {'traj_paths': "../data/chig/filtered/e1*/*.xtc", 
                              'pdb_path': '../data/chig/filtered/filtered.pdb', 
                              'file_type': 'xtc'}, 
                'dipeptide': 
                            {'pdb_path': '../data/adp/alanine-dipeptide-nowater.pdb', 
                            'traj_paths': '../data/adp/alanine-dipeptide-*-250ns-nowater.xtc',
                            'file_type': 'xtc'
                             },
                'pentapeptide': 
                            {'pdb_path': '../data/adp/pentapeptide-impl-solv.pdb',
                             'traj_paths': '../data/adp/pentapeptide-*-500ns-impl-solv.xtc',
                             'file_type': 'xtc' 
                            },
                'RBCG-eIF4E':
                            {'pdb_path': '../data/Ca-eIF4E/with_bonds.pdb',
                            'traj_paths': '../data/Ca-eIF4E/*.xtc',
                            'file_type': 'xtc'
                            },
                'eIF4E':
                            {'pdb_path': '../data/eIF4E/eIF4E.gro',
                            'traj_paths': '../data/eIF4E/*.xtc',
                            'file_type': 'xtc'
                            },
                'SBCG-eIF4E':
                            {'pdb_path': '/gpfs/home/eswong/CoarseGrainingVAE/data/SBCG-eIF4E/SBCG50_top.pdb',
                            'traj_paths': '/gpfs/home/eswong/CoarseGrainingVAE/data/SBCG-eIF4E/Ca-eIF4E_SBCG_connect.pdb',
                            'file_type': 'pdb'
                            },
                'RBCG-sarscov2': {'pdb_path': '/gpfs/home/eswong/CoarseGrainingVAE/data/sarscov2/RBCG/sarscov2_top_RBCG_connect.pdb',
                             'traj_paths': '/gpfs/home/eswong/CoarseGrainingVAE/data/sarscov2/RBCG/*.xtc',
                             'file_type': 'xtc'}, 
                'sarscov2': {'pdb_path': '/gpfs/home/eswong/CoarseGrainingVAE/data/sarscov2/FG/sarscov2_cleaned.pdb',
                             'traj_paths': '/gpfs/home/eswong/CoarseGrainingVAE/data/sarscov2/FG/*.xtc',
                             'file_type': 'xtc'},
                'SBCG100-sarscov2': {'pdb_path':'/gpfs/home/eswong/CoarseGrainingVAE/data/sarscov2/SBCG/sarscov2_0_0_SBCG_100_connect.pdb',
                             'traj_paths': '/gpfs/home/eswong/CoarseGrainingVAE/data/sarscov2/SBCG/*100_connect.pdb',
                             'file_type': 'pdb'}, 
                'SBCG35-sarscov2': {'pdb_path':'/gpfs/home/eswong/CoarseGrainingVAE/data/sarscov2/SBCG/sarscov2_0_0_SBCG_35_connect.pdb',
                                    'traj_paths': '/gpfs/home/eswong/CoarseGrainingVAE/data/sarscov2/SBCG/*35_connect.pdb',
                                    'file_type': 'pdb'}, 
                'PED3_RBCG':
                            {'pdb_path': '../data/PED/RBCG/PED00003e001.pdb',
                            'traj_paths': '../data/PED/RBCG/PED00003e001.pdb',
                            'file_type': 'pdb'
                            },
                'PED17_RBCG':
                            {'pdb_path': '../data/PED/RBCG/PED00017e001_RBCG_top.pdb',
                            'traj_paths': '../data/PED/RBCG/PED00017*.xtc',
                            'file_type': 'xtc'
                            },
                'PED24_RBCG':
                            {'pdb_path': '../data/PED/RBCG/PED00024e001.pdb',
                            'traj_paths': '../data/PED/RBCG/PED00024e001.pdb',
                            'file_type': 'pdb'
                            },
                'PED180_RBCG':
                            {'pdb_path': '../data/PED/RBCG/PED00180e005_RBCG_top.pdb',
                            'traj_paths': '../data/PED/RBCG/PED00180*.xtc',
                            'file_type': 'xtc'
                            },
                'PED3':
                            {'pdb_path': '../data/PED/FG/PED00003e001_top.pdb',
                            'traj_paths': '../data/PED/FG/PED00003e001.xtc',
                            'file_type': 'pxtcdb'
                            },
                'PED17':
                            {'pdb_path': '../data/PED/FG/PED00017e001_top.pdb',
                            'traj_paths': '../data/PED/FG/PED00017*.xtc',
                            'file_type': 'xtc'
                            },
                'PED24':
                            {'pdb_path': '../data/PED/FG/PED00024e001_top.pdb',
                            'traj_paths': '../data/PED/FG/PED00024e001.xtc',
                            'file_type': 'xtc'
                            },
                'PED180':
                            {'pdb_path': '../data/PED/FG/PED00180e005_top.pdb',
                            'traj_paths': '../data/PED/FG/PED00180*.xtc',
                            'file_type': 'xtc'
                            }
                }
# PED_PDBs = glob.glob(f'/gpfs/home/eswong/CoarseGrainingVAE/data/PED/*.pdb')
# for PDBfile in PED_PDBs:
#     ID = PDBfile.split('/')[-1].split('.')[0][3:]
#     dct = {ID: {'pdb_path': PDBfile,
#             'traj_paths': PDBfile,
#             'file_type': 'pdb'
#                     }
#                     }
#     PROTEINFILES.update(dct)

def get_backbone(top):
    backbone_index = []
    for atom in top.atoms:
        if atom.is_backbone:
            backbone_index.append(atom.index)
    return np.array(backbone_index)

def random_rotate_xyz_cg(xyz, cg_xyz ): 
    atoms = Atoms(positions=xyz, numbers=list( range(xyz.shape[0]) ))
    cgatoms = Atoms(positions=cg_xyz, numbers=list( range(cg_xyz.shape[0]) ))
    
    # generate rotation paramters 
    vec = np.random.randn(3)
    nvec = vec / np.sqrt( np.sum(vec ** 2) )
    angle = random.randrange(-180, 180)
    
    # rotate 
    atoms.rotate(angle, nvec)
    cgatoms.rotate(angle, nvec)
    
    return atoms.positions, cgatoms.positions

def random_rotation(xyz): 
    atoms = Atoms(positions=xyz, numbers=list( range(xyz.shape[0]) ))
    vec = np.random.randn(3)
    nvec = vec / np.sqrt( np.sum(vec ** 2) )
    angle = random.randrange(-180, 180)
    atoms.rotate(angle, nvec)
    return atoms.positions

def backbone_partition(traj, n_cgs, skip=100):
    atomic_nums, protein_index = get_atomNum(traj)
    #indices = traj.top.select_atom_indices('minimal')
    indices = get_backbone(traj.top)

    if indices.shape[0] < n_cgs:
        raise ValueError("N_cg = {} is larger than N_backbone = {}".format(n_cgs, indices.shape[0]) )

    if len(indices) == n_cgs:
        partition = list(range(1, n_cgs))
    else:
        partition = random.sample(range(indices.shape[0]), n_cgs - 1 )
        partition = np.array(partition)
        partition = np.sort(partition)
        segment_sizes = (partition[1:] - partition[:-1]).tolist() + [indices.shape[0] - partition[-1]] + [partition[0]]

    mapping = np.zeros(indices.shape[0])
    mapping[partition] = 1
    mapping = np.cumsum(mapping)

    backbone_cgxyz = scatter_mean(torch.Tensor(traj.xyz[:, indices]), 
                          index=torch.LongTensor(mapping), dim=1).numpy()

    mappings = []
    for i in protein_index:
        dist = traj.xyz[::skip, [i], ] - backbone_cgxyz[::skip]
        map_index = np.argmin( np.sqrt( np.sum(dist ** 2, -1)).mean(0) )
        mappings.append(map_index)

    cg_coord = None
    mapping = np.array(mappings)

    return mapping 


def get_diffpool_data(N_cg, trajs, n_data, edgeorder=1, shift=False, pdb=None, rotate=False):
    props = {}

    num_cgs = []
    num_atoms = []

    z_data = []
    xyz_data = []
    bond_data = []
    angle_data = []
    dihedral_data = []
    hyperedge_data = []

    # todo: not quite generalizable to different proteins
    if pdb is not None:
        mol = Molecule(pdb, guess=['bonds', 'angles', 'dihedrals'] )  
        dihedrals = torch.LongTensor(mol.dihedrals.astype(int))
        angles = torch.LongTensor(mol.angles.astype(int))
    else:
        dihedrals = None
        angles = None

    for traj in trajs:
        atomic_nums, protein_index = get_atomNum(traj)
        n_atoms = len(atomic_nums)
        if len(protein_index) != 0:
            frames = traj.xyz[:, protein_index, :] * 10.0 # from nm to Angstrom
        else:
            print("not protein")
            frames = traj.xyz[:, :, :] * 10.0
        print(traj.top)
        bondgraph = traj.top.to_bondgraph()
        bond_edges = torch.LongTensor( [[e[0].index, e[1].index] for e in bondgraph.edges] )# list of edge list 
        print(bond_edges)
        hyper_edges = get_high_order_edge(bond_edges, edgeorder, n_atoms)

        for xyz in frames[:n_data]: 
            # if shift:
            #     xyz = xyz - np.random.randn(1, 3)
            # if rotate:
            #     xyz = random_rotation(xyz)
            z_data.append(torch.Tensor(atomic_nums))
            coord = torch.Tensor(xyz)

            xyz_data.append(coord)
            bond_data.append(bond_edges)
            hyperedge_data.append(hyper_edges)

            angle_data.append(angles)
            dihedral_data.append(dihedrals)

            num_cgs.append(torch.LongTensor([N_cg]))
            num_atoms.append(torch.LongTensor([n_atoms]))

    #z_data, xyz_data, num_atoms, num_cgs, bond_data, hyperedge_data, angle_data, dihedral_data = shuffle( z_data, xyz_data, num_atoms, num_cgs, bond_data, hyperedge_data, angle_data, dihedral_data)


    props = {'z': z_data[:n_data],
         'xyz': xyz_data[:n_data],
         'num_atoms': num_atoms[:n_data], 
         'num_CGs':num_cgs[:n_data],
         'bond': bond_data[:n_data],
         'hyperedge': hyperedge_data[:n_data],
        }

    return props

def load_protein_traj(label, ntraj=200): 
    
    traj_files = glob.glob(PROTEINFILES[label]['traj_paths'])#[:ntraj]
    pdb_file = PROTEINFILES[label]['pdb_path']
    file_type = PROTEINFILES[label]['file_type']
    # sort the files
    traj_files = sorted(traj_files)
    print(traj_files)
    
    if file_type == 'xtc':
        trajs = [md.load_xtc(file,
                    top=pdb_file) for file in traj_files]
    elif file_type == 'dcd':
        trajs = [md.load_dcd(file,
                    top=pdb_file) for file in traj_files]
    elif file_type == 'pdb':
        trajs = [md.load(file) for file in traj_files]
    else:
        raise ValueError("file type {} not recognized".format(file_type))
    if len(trajs) == 1:
        traj = trajs[0]
    else:
        traj = md.join(trajs)
                   
    return traj


def learn_map(args, traj, reg_weight, n_cgs, n_atoms,
              n_data=1000, n_epochs=1500, 
              lr=4e-3, batch_size=32, prot_id=None):
    # get number of frames in trajectory
    n_frames = len(traj)
    props = get_diffpool_data(n_cgs, [traj], n_data=n_frames, edgeorder=1)
    dataset = DiffPoolDataset(props)
    dataset.generate_neighbor_list(8.0)
    train_index, test_index = train_test_split(list(range(len(traj)))[:n_data], test_size=0.1)
    trainset = get_subset_by_indices(train_index,dataset)
    testset = get_subset_by_indices(test_index,dataset)

    train_sampler = DistributedSampler(trainset, shuffle=True, num_replicas=args.world_size, rank=args.rank)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=DiffPool_collate, pin_memory=True, sampler=train_sampler)
    testloader = DataLoader(testset, batch_size=batch_size, collate_fn=DiffPool_collate, shuffle=True, pin_memory=True)
    
    ae = cgae(n_atoms, n_cgs).to(args.device)
    ae = DDP(ae, device_ids=[args.device])
    optimizer = torch.optim.Adam(list(ae.parameters()), lr=lr)

    tau = 1.0

    for epoch in range(n_epochs):
        all_loss = []
        all_reg = []
        all_recon = [] 

        trainloader.sampler.set_epoch(epoch)

        for i, batch in enumerate(trainloader):
        
            batch = batch_to(batch, args.device)
            xyz = batch['xyz']

            shift = xyz.mean(1)
            xyz = xyz - shift.unsqueeze(1)

            xyz, xyz_recon, M, cg_xyz = ae(xyz, tau)
            xyz_recon = torch.einsum('bnj,ni->bij', cg_xyz, ae.module.decode)
            X_lift = torch.einsum('bij,ni->bnj', cg_xyz, M)

            loss_reg = (xyz - X_lift).pow(2).sum(-1).mean()
            loss_recon = (xyz - xyz_recon).pow(2).mean() 
            loss = loss_recon + reg_weight * loss_reg

            all_reg.append(loss_reg.item())
            all_recon.append(loss_recon.item())
            all_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_loss.append(loss.item())

        if tau >= 0.025:
            tau -= 0.001

        if (epoch % 50 == 0) and (args.rank == 0):
            print(epoch, tau, np.array(all_recon).mean(), np.array(all_reg).mean())
    # inference on whole dataset to get cg_xyz
    
    del trainloader
    del testloader
    del trainset
    del testset
    del dataset
    del props
    del traj
    del train_sampler

#if args.device == 0:
    # check memory usage
    # clear memory from dataloaders
    # clear as much memory as possible
    torch.cuda.empty_cache()
    #assign_map = ae.state_dict()['assign_map']
    if args.rank == 0:
        torch.save(ae.module.assign_map.argmax(dim=-1).detach().cpu(), f'/gpfs/home/eswong/CoarseGrainingVAE/mapping_{args.dataset_label}_ncgs{n_cgs}.pt')
    return ae.module.assign_map.argmax(dim=-1).detach().cpu()
        # Detach the tensor from the computation graph
        #local_max_indices = local_max_indices.detach()
    #else:
        # Wait until the rank 0 process has saved the file
        # delay the barrier until the file is saved

        #dist.barrier()
        #return None


def get_cg_and_xyz(traj, args, config, cg_method='backbone', n_cgs=None, mapshuffle=0.0, mapping=None, prot_id=None):
    atomic_nums, protein_index = get_atomNum(traj)
    n_atoms = len(atomic_nums)
    skip = 200
    # get alpha carbon only 
    if len(protein_index) != 0:
        frames = traj.xyz[:, protein_index, :] * 10.0 
    else:
        frames = traj.xyz[:, :, :] * 10.0

    if cg_method in ['minimal', 'alpha']:
        mappings = []
        print("Note, using CG method {}, user-specified N_cg will be overwritten".format(cg_method))

        indices = traj.top.select_atom_indices(cg_method)
        for i in protein_index:
            dist = traj.xyz[::skip, [i], ] - traj.xyz[::skip, indices, :]
            map_index = np.argmin( np.sqrt( np.sum(dist ** 2, -1)).mean(0) )
            mappings.append(map_index)

        cg_coord = traj.xyz[:, indices, :] * 10.0
        mapping = np.array(mappings)

        n_cgs = len(indices)
        frames, cg_coord = shuffle(frames, cg_coord)

    elif cg_method =='newman':

        if n_cgs is None:
            raise ValueError("need to provided number of CG sites")

        protein_top = traj.top.subset(protein_index)
        g = protein_top.to_bondgraph()
        paritions = get_partition(g, n_cgs)
        mapping = parition2mapping(paritions, n_atoms)
        mapping = np.array(mapping)
        cg_coord = None

        # randomly shuffle map 
        perm_percent = mapshuffle

        if mapshuffle > 0.0:
            ran_idx = random.sample(range(mapping.shape[0]), int(perm_percent * mapping.shape[0])  )
            idx2map = mapping[ran_idx]
            mapping[ran_idx] = shuffle(idx2map)

        frames = shuffle(frames)

    elif cg_method == 'backbonepartition': 
        mapping = backbone_partition(traj, n_cgs)
        cg_coord = None

    elif cg_method == 'cgae':
        
        if mapping == None:
            print("learning CG mapping")
            mapping = learn_map(args, traj, reg_weight=config.cgae_reg_weight, n_cgs=n_cgs, n_atoms=n_atoms, batch_size=32, prot_id=prot_id)
            print(mapping)
        else:
            mapping = mapping 
        cg_coord = None
        # use the mapping to get the CG coordinates and output CG structure to xyz
        #cg_xyz = scatter_mean(torch.Tensor(frames), index=torch.LongTensor(mapping), dim=1).numpy()
        

    elif cg_method == 'seqpartition':
        partition = random.sample(range(n_atoms), n_cgs - 1 )
        partition = np.sort(partition)
        mapping = np.zeros(n_atoms)
        mapping[partition] = 1
        mapping = np.cumsum(mapping)

        cg_coord = None
        frames = shuffle(frames)

    elif cg_method =='random':

        mapping = get_random_mapping(n_cgs, n_atoms)
        cg_coord = None
        frames = shuffle(frames)

    else:
        raise ValueError("{} coarse-graining option not available".format(cg_method))


    # print coarse graining summary 
    print("CG method: {}".format(cg_method))
    print("Number of CG sites: {}".format(mapping.max() + 1))

    #assert len(list(set(mapping.tolist()))) == n_cgs

    mapping = torch.LongTensor( mapping)
    
    return mapping, frames, cg_coord


def get_atomNum(traj):
    
    atomic_nums = [atom.element.number for atom in traj.top.atoms]
    
    protein_index = traj.top.select("protein")
    protein_top = traj.top.subset(protein_index)

    if len(protein_index) != 0:
        atomic_nums = [atom.element.number for atom in protein_top.atoms]
    
    return np.array(atomic_nums), protein_index

def compute_nbr_list(frame, cutoff):
    
    dist = (frame[None, ...] - frame[:, None, :]).pow(2).sum(-1).sqrt()
    nbr_list = torch.nonzero(dist < cutoff).numpy()
    
    return nbr_list

def parition2mapping(partitions, n_nodes):
    # generate mapping 
    mapping = np.zeros(n_nodes)
    
    for k, group in enumerate(partitions):
        for node in group:
            mapping[node] = k
            
    return mapping.astype(int)

def get_partition(G, n_partitions):
    
    # adj = [tuple(pair) for pair in nbr_list]
    # G = nx.Graph()
    # G.add_edges_from(adj)

    G = nx.convert_node_labels_to_integers(G)
    comp = nx.community.girvan_newman(G)

    for communities in itertools.islice(comp, n_partitions-1):
            partitions = tuple(sorted(c) for c in communities)
        
    return partitions 

def compute_mapping(atomic_nums, traj, cutoff, n_atoms, n_cgs, skip):

    # get bond graphs 
    g = traj.top.to_bondgraph()
    paritions = get_partition(g, n_cgs)
    mapping = parition2mapping(paritions, n_atoms)

    return mapping

def get_mapping(label, cutoff, n_atoms, n_cgs, skip=200):

    peptide = get_peptide_top(label)

    files = mdshare.fetch(DATALABELS[label]['xtc'], working_directory='data')

    atomic_nums, traj = get_traj(peptide, files, n_frames=20000)
    peptide_element = [atom.element.symbol for atom in peptide.top.atoms]

    if len(traj) < skip:
        skip = len(traj)

    mappings = compute_mapping(atomic_nums, traj,  cutoff,  n_atoms, n_cgs, skip)

    return mappings.long()

def get_random_mapping(n_cg, n_atoms):

    mapping = torch.LongTensor(n_atoms).random_(0, n_cg)
    i = 1
    while len(mapping.unique()) != n_cg and i <= 10000000:
        i += 1
        mapping = torch.LongTensor(n_atoms).random_(0, n_cg)

    return mapping

def get_peptide_top(label):

    pdb = mdshare.fetch(DATALABELS[label]['pdb'], working_directory='data')
    peptide = md.load("data/{}".format(DATALABELS[label]['pdb']))

    return peptide

def get_traj(pdb, files, n_frames, shuffle=False):
    feat = pyemma.coordinates.featurizer(pdb)
    traj = pyemma.coordinates.load(files, features=feat)
    traj = np.concatenate(traj)

    peptide_element = [atom.element.symbol for atom in pdb.top.atoms]

    if shuffle: 
        traj = shuffle(traj)
        
    traj_reshape = traj.reshape(-1, len(peptide_element),  3)[:n_frames] * 10.0 # Change from nanometer to Angstrom 
    atomic_nums = np.array([atomic_num_dict[el] for el in peptide_element] )
    
    return atomic_nums, traj_reshape

# need a function to get mapping, and CG coordinates simultanesouly. We can have alpha carbon as the CG site


def get_high_order_edge(edges, order, natoms):

    # get adj 
    adj = torch.zeros(natoms, natoms)
    adj[edges[:,0], edges[:,1]] = 1
    adj[edges[:,1], edges[:,0]] = 1

    # get higher edges 
    edges = torch.triu(get_higher_order_adj_matrix(adj, order=order)).nonzero()

    return edges 

def build_dataset(mapping, traj, atom_cutoff, cg_cutoff, atomic_nums, top, order=1, cg_traj=None, split=None):
    
    CG_nxyz_data = []
    nxyz_data = []

    num_atoms_list = []
    num_CGs_list = []
    CG_mapping_list = []
    bond_edge_list = []
    bondgraph = top.to_bondgraph()

    edges = torch.LongTensor( [[e[0].index, e[1].index] for e in bondgraph.edges] )# list of edge list 
    edges = get_high_order_edge(edges, order, atomic_nums.shape[0])
    for xyz in traj:

        #xyz = random_rotation(xyz)
        nxyz = torch.cat((torch.Tensor(atomic_nums[..., None]), torch.Tensor(xyz) ), dim=-1)
        nxyz_data.append(nxyz)
        num_atoms_list.append(torch.LongTensor( [len(nxyz)]))
        bond_edge_list.append(edges)
    cg_xyzs = []
    # Aggregate CG coorinates 
    for i, nxyz in enumerate(nxyz_data):
        xyz = torch.Tensor(nxyz[:, 1:]) 
        if cg_traj is not None:
            CG_xyz = torch.Tensor( cg_traj[i] )
        else:
            CG_xyz = scatter_mean(xyz, mapping, dim=0)
        # save CG coordinates
        cg_xyzs.append(CG_xyz)
        
        CG_nxyz = torch.cat((torch.LongTensor(list(range(len(CG_xyz))))[..., None], CG_xyz), dim=-1)
        CG_nxyz_data.append(CG_nxyz)

        num_CGs_list.append(torch.LongTensor( [len(CG_nxyz)]) )
        CG_mapping_list.append(mapping)
    # get cg_xyzs in numpy array
    cg_xyzs = torch.cat(cg_xyzs).numpy()
    # # reshape to be 3d
    cg_xyzs = cg_xyzs.reshape(-1, len(CG_nxyz_data[0]), 3)
    cg_nums = [1] * len(CG_nxyz_data[0])
    dump_numpy2xyz(cg_xyzs, cg_nums, f'/gpfs/home/eswong/CoarseGrainingVAE/data/SBCG-eIF4E_{split}_cg_{len(CG_nxyz_data[0])}.xyz')

    props = {'nxyz': nxyz_data,
             'CG_nxyz': CG_nxyz_data,
             'num_atoms': num_atoms_list, 
             'num_CGs':num_CGs_list,
             'CG_mapping': CG_mapping_list, 
             'bond_edge_list':  bond_edge_list
            }

    dataset = CGDataset(props.copy())
    #dataset.generate_neighbor_list(atom_cutoff=atom_cutoff, cg_cutoff=cg_cutoff)
    
    return dataset

def get_peptide_dataset(atom_cutoff,  cg_cutoff, label, mapping, n_frames=20000, n_cg=6):

    pdb = mdshare.fetch(DATALABELS[label]['pdb'], working_directory='data')
    files = mdshare.fetch(DATALABELS[label]['xtc'], working_directory='data')
    pdb = md.load("data/{}".format(DATALABELS[label]['pdb']))
    
    atomic_nums, traj_reshape = get_traj(pdb, files, n_frames)

    dataset = build_dataset(mapping, traj_reshape, atom_cutoff, cg_cutoff, atomic_nums, pdb.top)

    return atomic_nums, dataset