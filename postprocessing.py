import mdtraj as md
#import MDAnalysis as mda
import numpy as np
import argparse

def clean_traj(traj):
    """
    Clean a trajectory by removing the hydrogens and stripping first and last residue
    """
    # get number of residues
    residues = traj.topology.n_residues
    last_residue = residues - 2
    noH_traj = traj.topology.select("not element H")
    traj = traj.atom_slice(noH_traj)
    traj = traj.atom_slice(traj.topology.select(f"resid 1 to {last_residue}"))
    return traj

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--recon', type=str)
    parser.add_argument('--truth', type=str)
    parser.add_argument('--action', type=str)
    parser.add_argument('--input', type=str) 
    args = parser.parse_args()
    if args.action == "clean":
        traj = md.load(args.input)
        traj = clean_traj(traj)
        traj.save(args.input.split('.')[0] + '_clean.pdb')
    if args.action == "rmsd":
        traj1 = md.load(args.recon)
        traj2 = md.load(args.truth)
        rmsd = get_avg_rmsd(traj1, traj2)
        print(rmsd)